#pragma once

// ============================================================================
// Strawman naive block multi-query: duplicate a 128-thread query executor.
// ----------------------------------------------------------------------------
// Q=1 is the original single-query block baseline: one query uses 128 threads
// and one resource set. Q>1 naively places Q such executors in one CUDA block:
//
//   blockDim.x = 128 * Q
//
// Q is capped at 8 so the block stays within CUDA's 1024-thread limit.
// Each 128-thread tile owns one query slot and W independent IO lanes.
// This intentionally measures naive block-internal multi-query parallelism,
// without Quiver's cross-query scheduler.
// ============================================================================

#include <cooperative_groups.h>

#include "../kernel_helpers.cuh"
#include "../inkernel_nav.cuh"
#include "../policies/dispatch.cuh"

namespace quiver {
namespace strawmen {

namespace cg = cooperative_groups;

constexpr int NAIVE_BLOCK_MULTI_QUERY_MAX_Q = 8;
constexpr int NAIVE_BLOCK_MULTI_QUERY_MAX_W = 16;
constexpr int NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY = 128;

__device__ __forceinline__ void publish_queries_completed_naive_m128(
    PersistentKernelComm* comm, int new_done)
{
    int* completed = (int*)&comm->queries_completed;
    int observed = *(volatile int32_t*)completed;
    while (observed < new_done) {
        int old = atomicCAS_system(completed, observed, new_done);
        if (old == observed) break;
        observed = old;
    }
    __threadfence_system();
}

__device__ __forceinline__ void copy_node_to_shm_naive_m128(
    const cg::thread_block_tile<NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY>& tile,
    int ltid, const uint8_t* src, uint8_t* dst, int node_size)
{
    int word_count = node_size >> 2;
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
    for (int i = ltid; i < word_count;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        dst32[i] = src32[i];
    }

    int byte_start = word_count << 2;
    for (int i = byte_start + ltid; i < node_size;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        dst[i] = src[i];
    }
    tile.sync();
}

__device__ __forceinline__ void sort_new_neighbors_by_warp_naive_m128(
    const cg::thread_block_tile<NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY>& tile,
    int ltid, int sz, int deg, uint32_t* tmp_id, float* tmp_dist)
{
    int local_idx = ltid;
    float dist = (local_idx < deg) ? tmp_dist[sz + local_idx] : INFINITY;
    uint32_t id = (local_idx < deg) ? tmp_id[sz + local_idx]
                                    : (0xffffffffu - (uint32_t)local_idx);
    warp_sort_pair_32(dist, id);
    if (local_idx < deg) {
        tmp_id[sz + local_idx] = id;
        tmp_dist[sz + local_idx] = dist;
    }
    tile.sync();
}

template <int LEN, int DEG>
__device__ __forceinline__ void merge_new_neighbor_stage_const_naive_m128(
    const cg::thread_block_tile<NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY>& tile,
    int ltid, int sz, int* mv_pos, uint32_t* mv_id, float* mv_dist,
    uint32_t* tmp_id, float* tmp_dist)
{
    int array_id = ltid / LEN;
    int start = array_id * LEN;
    int mid = min(start + LEN / 2, DEG);
    int end = min(start + LEN, DEG);
    if (ltid >= start && ltid < mid) {
        int id = lower_bound(tmp_dist + sz + mid, tmp_id + sz + mid,
                             0, end - mid,
                             tmp_dist[sz + ltid], tmp_id[sz + ltid]);
        mv_pos[ltid] = id + ltid;
    }
    if (ltid >= mid && ltid < end) {
        int id = lower_bound(tmp_dist + sz, tmp_id + sz,
                             start, mid,
                             tmp_dist[sz + ltid], tmp_id[sz + ltid]);
        mv_pos[ltid] = id + ltid - mid;
    }
    tile.sync();
    if (ltid < DEG) {
        mv_id[mv_pos[ltid]] = tmp_id[sz + ltid];
        mv_dist[mv_pos[ltid]] = tmp_dist[sz + ltid];
    }
    tile.sync();
    if (ltid < DEG) {
        tmp_id[sz + ltid] = mv_id[ltid];
        tmp_dist[sz + ltid] = mv_dist[ltid];
    }
    tile.sync();
}

__device__ __forceinline__ void persistent_merge_data_r128_ef30_naive_m128(
    const cg::thread_block_tile<NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY>& tile,
    int ltid, int sz, Data* ctx,
    uint32_t* neighbor_id, float* neighbor_dist,
    int* mv_pos, uint32_t* mv_id, float* mv_dist,
    uint32_t* tmp_id, float* tmp_dist)
{
    constexpr int DEG = 128;
    constexpr int KEEP = 60;

    sort_new_neighbors_by_warp_naive_m128(tile, ltid, sz, DEG, tmp_id, tmp_dist);
    merge_new_neighbor_stage_const_naive_m128<64, DEG>(
        tile, ltid, sz, mv_pos, mv_id, mv_dist, tmp_id, tmp_dist);
    merge_new_neighbor_stage_const_naive_m128<128, DEG>(
        tile, ltid, sz, mv_pos, mv_id, mv_dist, tmp_id, tmp_dist);

    if (ltid < DEG) {
        int id = lower_bound(tmp_dist, tmp_id, 0, sz,
                             tmp_dist[sz + ltid], tmp_id[sz + ltid]);
        if (id != sz &&
            ((tmp_id[id] ^ tmp_id[sz + ltid]) & 0x7fffffff) == 0) {
            id++;
        }
        mv_pos[sz + ltid] = id + ltid;
    }
    for (int i = ltid; i < sz;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        int id = lower_bound(tmp_dist + sz, tmp_id + sz, 0, DEG,
                             tmp_dist[i], tmp_id[i]);
        mv_pos[i] = id + i;
    }
    tile.sync();

    for (int i = ltid; i < sz + DEG;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        mv_id[mv_pos[i]] = tmp_id[i];
        mv_dist[mv_pos[i]] = tmp_dist[i];
    }
    tile.sync();

    int target = min(sz + DEG, KEEP);
    for (int i = ltid; i < target;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        neighbor_id[i] = mv_id[i];
        neighbor_dist[i] = mv_dist[i];
    }
    if (ltid == 0) ctx->size = target;
    tile.sync();
}

__device__ void persistent_merge_data_naive_m128(
    const cg::thread_block_tile<NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY>& tile,
    int ltid,
    uint8_t* buffer, int32_t node_id,
    int num_chunks, float* pq_dists, uint8_t* compressed_data,
    int nodes_per_page, int node_size, int data_size,
    int max_m, int ef_search,
    uint32_t* neighbor_id, float* neighbor_dist, Data* ctx,
    int* mv_pos, uint32_t* mv_id, float* mv_dist,
    uint32_t* tmp_id, float* tmp_dist,
    uint8_t* node_scratch)
{
    if (node_id == -1) return;

    int sz = ctx->size;
    int node_offset = node_id % nodes_per_page * node_size;
    uint8_t* node_base = buffer + node_offset;
    copy_node_to_shm_naive_m128(tile, ltid, node_base, node_scratch, node_size);
    node_base = node_scratch;
    int* buffer_u = (int*)(node_base + data_size);

    for (int i = ltid; i < sz;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        tmp_id[i] = neighbor_id[i];
        tmp_dist[i] = neighbor_dist[i];
    }
    tile.sync();

    int deg = buffer_u[0];

    if (ltid >= deg) {
        if (ltid < max_m) {
            tmp_id[sz + ltid] = 0xffffffff - ltid;
            tmp_dist[sz + ltid] = INFINITY;
        }
    } else {
        int nodeid_v = tmp_id[sz + ltid] = buffer_u[ltid + 1];
        float dist = 0;
        uint8_t* pq_vec = compressed_data + (long)nodeid_v * num_chunks;
#pragma unroll 32
        for (int j = 0; j < num_chunks; j++) {
            dist += pq_dists[j * PQSearchData::num_pivots + pq_vec[j]];
        }
        tmp_dist[sz + ltid] = dist;
    }
    tile.sync();

    if (deg == 128 && ef_search == 30) {
        persistent_merge_data_r128_ef30_naive_m128(
            tile, ltid, sz, ctx, neighbor_id, neighbor_dist,
            mv_pos, mv_id, mv_dist, tmp_id, tmp_dist);
        return;
    }

    for (int len = 2; len < 2 * deg; len *= 2) {
        int array_id = ltid / len;
        int start = array_id * len;
        int mid = min(start + len / 2, deg);
        int end = min(start + len, deg);
        if (ltid >= start && ltid < mid) {
            int id = lower_bound(tmp_dist + sz + mid, tmp_id + sz + mid,
                                 0, end - mid,
                                 tmp_dist[sz + ltid], tmp_id[sz + ltid]);
            mv_pos[ltid] = id + ltid;
        }
        if (ltid >= mid && ltid < end) {
            int id = lower_bound(tmp_dist + sz, tmp_id + sz,
                                 start, mid,
                                 tmp_dist[sz + ltid], tmp_id[sz + ltid]);
            mv_pos[ltid] = id + ltid - mid;
        }
        tile.sync();
        __threadfence_block();
        if (ltid < deg) {
            mv_id[mv_pos[ltid]] = tmp_id[sz + ltid];
            mv_dist[mv_pos[ltid]] = tmp_dist[sz + ltid];
        }
        tile.sync();
        if (ltid < deg) {
            tmp_id[sz + ltid] = mv_id[ltid];
            tmp_dist[sz + ltid] = mv_dist[ltid];
        }
        tile.sync();
    }

    if (ltid < deg) {
        int id = lower_bound(tmp_dist, tmp_id, 0, sz,
                             tmp_dist[sz + ltid], tmp_id[sz + ltid]);
        if (id != sz &&
            ((tmp_id[id] ^ tmp_id[sz + ltid]) & 0x7fffffff) == 0) {
            id++;
        }
        mv_pos[sz + ltid] = id + ltid;
    }
    for (int i = ltid; i < sz;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        int id = lower_bound(tmp_dist + sz, tmp_id + sz, 0, deg,
                             tmp_dist[i], tmp_id[i]);
        mv_pos[i] = id + i;
    }
    tile.sync();

    for (int i = ltid; i < sz + deg;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        mv_id[mv_pos[i]] = tmp_id[i];
        mv_dist[mv_pos[i]] = tmp_dist[i];
    }
    tile.sync();

    int target = min(sz + deg, 2 * ef_search);
    for (int i = ltid; i < target;
         i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        neighbor_id[i] = mv_id[i];
        neighbor_dist[i] = mv_dist[i];
    }
    if (ltid == 0) ctx->size = target;
    tile.sync();
}

template <class T>
__device__ int32_t persistent_visit_select_naive_m128(
    int ltid, float* qdata, uint8_t* node_data, int32_t node_id,
    int num_dims, int topk,
    int* nns, float* distances, int* found_cnt,
    int max_m, int ef_search,
    uint32_t* neighbor_id, float* neighbor_dist, Data* ctx)
{
    (void)max_m;
    int lane = threadIdx.x & 31;
    int32_t next = -1;

    if (ltid < 32) {
        int sz = ctx->size;
        if (sz != 0) {
            int target_r = (sz + 31) / 32 * 32;
            int count = 0;
            for (int i = lane; i < target_r; i += 32) {
                bool flag = (i != 0) && (i < sz) &&
                    (((neighbor_id[i] ^ neighbor_id[i - 1]) & 0x7fffffff) == 0);
                unsigned mask = __ballot_sync(0xffffffff, flag);
                int mv = i - count - __popc(mask & ((1u << lane) - 1));
                uint32_t t_id = neighbor_id[i];
                float t_dist = neighbor_dist[i];
                __syncwarp();
                if (i < sz && !flag) {
                    neighbor_id[mv] = t_id;
                    neighbor_dist[mv] = t_dist;
                }
                __syncwarp();
                count += __popc(mask);
                __syncwarp();
            }
            sz -= count;
            int id = sz + 1;
            for (int i = lane; i < sz; i += 32) {
                if ((neighbor_id[i] & 0x80000000u) == 0) id = min(id, i);
            }
            __syncwarp();
#pragma unroll
            for (int off = 16; off > 0; off /= 2) {
                int _id = __shfl_down_sync(0xffffffff, id, off);
                id = min(_id, id);
            }
            id = __shfl_sync(0xffffffff, id, 0);
            if (lane == 0) {
                if (id < ef_search && id < sz) {
                    next = neighbor_id[id] & 0x7fffffff;
                    neighbor_id[id] |= 0x80000000u;
                }
                ctx->size = min(sz, ef_search);
            }
            next = __shfl_sync(0xffffffff, next, 0);
        }
    } else if (ltid < 64 && node_id != -1) {
        T* buffer_u = (T*)node_data;
        float dist = square_sum_32(qdata, buffer_u, num_dims);
        int fc = found_cnt[0];
        if (fc < topk || dist < distances[topk - 1]) {
            if (topk <= 32) {
                retset_push_topk_le32(distances, nns, found_cnt[0],
                                      topk, dist, node_id);
            } else {
                retset_push_32(distances, nns, found_cnt[0],
                               topk, dist, node_id);
            }
        }
    }

    return next;
}

template <class T>
__device__ __forceinline__ int inkernel_nav_entry_naive_m128(
    const cg::thread_block_tile<NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY>& tile,
    int ltid,
    const float* qdata,
    const T* nav_data,
    const int* nav_graph,
    const int* nav_mapping,
    int num_dims,
    int max_m,
    int init_ef,
    int entry_root,
    uint32_t* pq_id,
    float* pq_dist,
    int* s_idx_scratch)
{
    (void)init_ef;
    const int warp_id = ltid >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kNumWarps = NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY / 32;
    const int L = kInkernelNavL;
    const int R = kInkernelNavR;

    if (warp_id == 0) {
        float d0 = square_sum_32<const T>(
            qdata, nav_data + (long)num_dims * entry_root, num_dims);
        if (lane == 0) {
            pq_id[0] = (uint32_t)entry_root;
            pq_dist[0] = d0;
        } else if (lane < L + R) {
            pq_id[lane] = 0xffffffffu;
            pq_dist[lane] = INFINITY;
        }
    }
    tile.sync();

    int idx = 0;
    while (idx != -1) {
        const int u = (int)(pq_id[idx] & 0x7fffffffu);
        if (ltid == 0) {
            pq_id[idx] |= 0x80000000u;
        }

        const int* edge = nav_graph + (long)max_m * u;
        const int per_warp = (R + kNumWarps - 1) / kNumWarps;
        const int my_start = warp_id * per_warp;
        const int my_end = min(my_start + per_warp, R);
        for (int i = my_start; i < my_end; ++i) {
            int v = edge[i];
            if (v == -1) break;
            float d = square_sum_32<const T>(
                qdata, nav_data + (long)num_dims * v, num_dims);
            if (lane == 0) {
                pq_id[i + L] = (uint32_t)v;
                pq_dist[i + L] = d;
            }
        }
        tile.sync();

        if (warp_id == 0) {
#pragma unroll
            for (int i = 0; i < L; ++i) {
                float dist = (lane < L + R) ? pq_dist[lane] : INFINITY;
                int pos = lane;
                if (lane < i) {
                    dist = INFINITY;
                } else if (i != 0 && lane < L + R &&
                           ((pq_id[lane] ^ pq_id[i - 1]) & 0x7fffffffu) == 0) {
                    pq_id[i - 1] |= pq_id[lane];
                    pq_dist[lane] = INFINITY;
                    dist = INFINITY;
                }
                __syncwarp();

#pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    int other_pos = __shfl_down_sync(0xffffffff, pos, offset);
                    float other_dist =
                        __shfl_down_sync(0xffffffff, dist, offset);
                    if (other_dist < dist) {
                        dist = other_dist;
                        pos = other_pos;
                    }
                }
                __syncwarp();

                if (lane == 0 && i != pos) {
                    float old_dist = pq_dist[i];
                    float new_dist = pq_dist[pos];
                    if (old_dist != new_dist) {
                        pq_dist[i] = new_dist;
                        pq_dist[pos] = old_dist;
                        uint32_t old_id = pq_id[i];
                        pq_id[i] = pq_id[pos];
                        pq_id[pos] = old_id;
                    }
                }
                __syncwarp();
            }

            uint32_t my_id = (lane < L + R) ? pq_id[lane] : 0x80000000u;
            int val = __ballot_sync(0xffffffff, !(my_id & 0x80000000u));
            if (lane == 0) {
                int x = __clz(__brev(val));
                *s_idx_scratch = (x < L) ? x : -1;
            }
        }
        tile.sync();
        idx = *s_idx_scratch;
    }

    int raw = (int)(pq_id[0] & 0x7fffffffu);
    return nav_mapping[raw];
}

template <class Dispatch, class T>
__global__ void persistent_search_kernel_naive_block_multi_query(
    PersistentKernelArgs args)
{
    static_assert(Dispatch::is_persistent,
                  "Naive block multi-query only supports StreamingDispatch.");

    extern __shared__ uint8_t shm_pool[];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY> tile =
        cg::tiled_partition<NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY>(block);

    const int tid = threadIdx.x;
    const int slot = tid / NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY;
    const int ltid = tid % NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY;
    const int bid = blockIdx.x;
    const int W = args.pipe_w;
    const int Q = args.queries_per_block;
    const int ef_plus_m = args.ef_search + args.max_m;
    const int frontier_bytes =
        (sizeof(int) * 3 + sizeof(float) * 2) * ef_plus_m;
    const int node_scratch_offset = (frontier_bytes + 7) & ~7;
    const int stride_bytes = node_scratch_offset + args.node_size;
    const int k_total_queries = args.total_queries;
    const int aef = args.aligned_ef;

    if (slot >= Q) return;
    if (W < 1 || W > NAIVE_BLOCK_MULTI_QUERY_MAX_W ||
        args.max_m != NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
        return;
    }

    __shared__ int32_t s_slot_qid[NAIVE_BLOCK_MULTI_QUERY_MAX_Q];
    __shared__ int32_t s_slot_entry[NAIVE_BLOCK_MULTI_QUERY_MAX_Q];
    __shared__ int32_t s_slot_stage[NAIVE_BLOCK_MULTI_QUERY_MAX_Q];
    __shared__ int32_t s_slot_got[NAIVE_BLOCK_MULTI_QUERY_MAX_Q];
    __shared__ int32_t s_slot_alive[NAIVE_BLOCK_MULTI_QUERY_MAX_Q];
    __shared__ int32_t s_nav_idx[NAIVE_BLOCK_MULTI_QUERY_MAX_Q];
    __shared__ int32_t s_lane_status[
        NAIVE_BLOCK_MULTI_QUERY_MAX_Q * NAIVE_BLOCK_MULTI_QUERY_MAX_W];
    __shared__ int32_t s_lane_node[
        NAIVE_BLOCK_MULTI_QUERY_MAX_Q * NAIVE_BLOCK_MULTI_QUERY_MAX_W];
    __shared__ int32_t s_lane_req[
        NAIVE_BLOCK_MULTI_QUERY_MAX_Q * NAIVE_BLOCK_MULTI_QUERY_MAX_W];

    if (ltid == 0) {
        s_slot_qid[slot] = -1;
        s_slot_entry[slot] = -1;
        s_slot_stage[slot] = -1;
        s_slot_got[slot] = 0;
        s_slot_alive[slot] = 1;
        s_nav_idx[slot] = 0;
    }
    tile.sync();

    auto slot_shm_base = [&]() -> uint8_t* {
        return shm_pool + (size_t)slot * stride_bytes;
    };
    auto slot_node_scratch = [&]() -> uint8_t* {
        return slot_shm_base() + node_scratch_offset;
    };
    auto slot_ctl = [&]() -> IOControl* {
        return args.comm->controls + ((bid * Q + slot) * W);
    };
    auto slot_buf = [&]() -> uint8_t* {
        return args.comm->buffers + (long)(bid * Q + slot) * W * shared::PAGE_SIZE;
    };
    auto slot_nid = [&]() -> uint32_t* {
        return args.d_neighbors_id + (long)(bid * Q + slot) * aef;
    };
    auto slot_ndist = [&]() -> float* {
        return args.d_neighbors_dist + (long)(bid * Q + slot) * aef;
    };
    auto slot_ctx = [&]() -> Data* {
        return args.d_ctx + (bid * Q + slot);
    };
    auto slot_nns = [&]() -> int* {
        return args.d_nns + (bid * Q + slot) * args.topk;
    };
    auto slot_dist = [&]() -> float* {
        return args.d_distances + (bid * Q + slot) * args.topk;
    };
    auto slot_fc = [&]() -> int* {
        return args.d_found_cnt + (bid * Q + slot);
    };
    auto slot_pq = [&]() -> float* {
        return args.pq_dists
            + (long)((bid * Q + slot) + args.pq_block_offset)
              * PQSearchData::num_pivots * args.num_chunks;
    };
    auto lane_idx = [&](int w) -> int {
        return slot * NAIVE_BLOCK_MULTI_QUERY_MAX_W + w;
    };

    auto try_bind_new_query = [&]() -> int {
        if (ltid == 0) {
            int qid = atomicAdd(args.d_next_query, 1);
            s_slot_qid[slot] = qid;
            int got = 0;
            while (qid < k_total_queries) {
                int fc = *(volatile int32_t*)&args.comm->feed_count;
                if (qid < fc) {
                    got = 1;
                    break;
                }
                if (*(volatile int32_t*)&args.comm->terminate) {
                    break;
                }
            }
            s_slot_got[slot] = got;
            if (got) __threadfence_system();
        }
        tile.sync();
        if (!s_slot_got[slot]) return 0;

        const int qid = s_slot_qid[slot];
        float* qdata = args.d_all_qdata + (long)qid * args.num_dims;
        int32_t entry_from_nav = -1;
        if (args.nav_data_dev != nullptr) {
            uint8_t* nav_base = slot_shm_base();
            uint32_t* nav_pq_id = (uint32_t*)nav_base;
            float* nav_pq_dist =
                (float*)(nav_pq_id + kInkernelNavL + kInkernelNavR);
            entry_from_nav = inkernel_nav_entry_naive_m128<T>(
                tile, ltid, qdata, (const T*)args.nav_data_dev,
                args.nav_graph_dev, args.nav_mapping_dev, args.nav_data_len,
                args.nav_max_m, args.nav_init_ef, args.nav_start,
                nav_pq_id, nav_pq_dist, &s_nav_idx[slot]);
        }

        {
            shared::PQSearchData* pqd = args.pq_data;
            float* dist_vec = slot_pq();
            const int num_pivots = PQSearchData::num_pivots;
            const int num_chunks = pqd->num_chunks;
            const int dim = pqd->dim;
            for (int i = ltid; i < num_pivots * num_chunks;
                 i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
                dist_vec[i] = 0;
            }
            for (int i = ltid; i < dim;
                 i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
                qdata[i] -= pqd->centroid[i];
            }
            tile.sync();
            for (int i = 0; i < dim; i++) {
                int idx = pqd->chunk_id[i];
                for (int j = ltid; j < num_pivots;
                     j += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
                    float dd = qdata[i] - pqd->pivots_t[i * num_pivots + j];
                    dist_vec[idx * num_pivots + j] += dd * dd;
                }
                tile.sync();
            }
            for (int i = ltid; i < dim;
                 i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
                qdata[i] += pqd->centroid[i];
            }
        }
        tile.sync();

        if (ltid == 0) {
            if (args.nav_data_dev != nullptr) {
                s_slot_entry[slot] = entry_from_nav;
            } else if (args.d_entry_nodes) {
                s_slot_entry[slot] = args.d_entry_nodes[qid];
            } else {
                s_slot_entry[slot] = args.enter_point;
            }
            *slot_fc() = 0;
            Data* ctx = slot_ctx();
            ctx->size = 0;
            ctx->visited_cnt = 0;
            for (int w = 0; w < W; ++w) {
                slot_ctl()[w].status = IO_IDLE;
                s_lane_status[lane_idx(w)] = IO_IDLE;
                s_lane_node[lane_idx(w)] = -1;
                s_lane_req[lane_idx(w)] = -1;
            }
        }
        for (int i = ltid; i < args.topk;
             i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
            slot_dist()[i] = INFINITY;
            slot_nns()[i] = -1;
        }
        tile.sync();

        if (ltid == 0) issue_request(&slot_ctl()[0], s_slot_entry[slot]);
        tile.sync();
        if (ltid == 0) s_slot_stage[slot] = 0;
        tile.sync();
        return 1;
    };

    auto finalize_query = [&]() {
        const int qid = s_slot_qid[slot];
        for (int i = ltid; i < args.topk;
             i += NAIVE_BLOCK_MULTI_QUERY_THREADS_PER_QUERY) {
            args.d_out_nns[(long)qid * args.topk + i] = slot_nns()[i];
            args.d_out_distances[(long)qid * args.topk + i] = slot_dist()[i];
        }
        if (ltid == 0) {
            args.d_out_found_cnt[qid] = *slot_fc();
        }
        __threadfence_system();
        tile.sync();
        if (ltid == 0) {
            int new_done = atomicAdd(args.d_queries_done, 1) + 1;
            publish_queries_completed_naive_m128(args.comm, new_done);
            s_slot_stage[slot] = -1;
        }
        tile.sync();
    };

    auto process_ready_lane = [&](int w, int32_t src_node) -> int32_t {
        uint8_t* base = slot_shm_base();
        int* mv_pos = (int*)base;
        uint32_t* mv_id = (uint32_t*)(mv_pos + ef_plus_m);
        float* mv_dist = (float*)(mv_id + ef_plus_m);
        uint32_t* tmp_id = (uint32_t*)(mv_dist + ef_plus_m);
        float* tmp_dist = (float*)(tmp_id + ef_plus_m);

        float* qdata = args.d_all_qdata + (long)s_slot_qid[slot] * args.num_dims;

        persistent_merge_data_naive_m128(
            tile, ltid, slot_buf() + (long)w * shared::PAGE_SIZE, src_node,
            args.num_chunks, slot_pq(), args.compressed_data,
            args.nodes_per_page, args.node_size, args.data_size,
            args.max_m, args.ef_search,
            slot_nid(), slot_ndist(), slot_ctx(),
            mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
            slot_node_scratch());

        return persistent_visit_select_naive_m128<T>(
            ltid, qdata, slot_node_scratch(), src_node,
            args.num_dims, args.topk,
            slot_nns(), slot_dist(), slot_fc(),
            args.max_m, args.ef_search,
            slot_nid(), slot_ndist(), slot_ctx());
    };

    auto wait_io_and_step = [&]() -> int {
        IOControl* ctl = slot_ctl();

        if (s_slot_stage[slot] == 0) {
            int do_exit = 0;
            if (ltid == 0) {
                while (true) {
                    int32_t st = *(volatile int32_t*)&ctl[0].status;
                    if (st == IO_READY) {
                        s_lane_node[lane_idx(0)] = ctl[0].node_id;
                        ctl[0].status = IO_IDLE;
                        break;
                    }
                    if (*(volatile int32_t*)&args.comm->terminate &&
                        *(volatile int32_t*)args.d_next_query >= k_total_queries) {
                        do_exit = 1;
                        break;
                    }
                }
                s_slot_alive[slot] = do_exit ? 0 : 1;
            }
            tile.sync();
            if (!s_slot_alive[slot]) return 0;

            __threadfence_system();
            int32_t next_node =
                process_ready_lane(0, s_lane_node[lane_idx(0)]);
            if (ltid == 0) s_lane_req[lane_idx(0)] = next_node;
            tile.sync();

            if (ltid < 32) {
                const int lane = ltid & 31;
                for (int w = 1; w < W; ++w) {
                    int32_t req = persistent_select_next_no_compact(
                        args.ef_search, slot_nid(), slot_ctx());
                    if (lane == 0) s_lane_req[lane_idx(w)] = req;
                }
            }
            tile.sync();

            if (ltid == 0) {
                bool any_active = false;
                for (int w = 0; w < W; ++w) {
                    int32_t req = s_lane_req[lane_idx(w)];
                    if (req != -1) {
                        issue_request(&ctl[w], req);
                        any_active = true;
                    }
                }
                s_slot_stage[slot] = any_active ? 1 : 2;
            }
            tile.sync();
            return 1;
        }

        if (ltid == 0) {
            bool any_active = false;
            bool any_ready = false;
            for (int w = 0; w < W; ++w) {
                int32_t st = *(volatile int32_t*)&ctl[w].status;
                s_lane_status[lane_idx(w)] = st;
                s_lane_req[lane_idx(w)] = -1;
                if (st == IO_READY) {
                    s_lane_node[lane_idx(w)] = ctl[w].node_id;
                    any_ready = true;
                }
                if (st != IO_IDLE) any_active = true;
            }
            if (!any_ready && !any_active) {
                s_slot_stage[slot] = 2;
            }
            s_slot_alive[slot] = any_ready ? 1 : 0;
        }
        tile.sync();

        if (s_slot_stage[slot] == 2) return 1;
        if (!s_slot_alive[slot]) return 1;

        __threadfence_system();
        for (int w = 0; w < W; ++w) {
            if (s_lane_status[lane_idx(w)] != IO_READY) continue;
            if (ltid == 0) ctl[w].status = IO_IDLE;
            tile.sync();
            int32_t next_node =
                process_ready_lane(w, s_lane_node[lane_idx(w)]);
            if (ltid == 0) s_lane_req[lane_idx(w)] = next_node;
            tile.sync();
        }

        if (ltid < 32) {
            const int lane = ltid & 31;
            for (int w = 0; w < W; ++w) {
                if (s_lane_status[lane_idx(w)] == IO_READY) continue;
                if (*(volatile int32_t*)&ctl[w].status != IO_IDLE) continue;
                int32_t req = persistent_select_next_no_compact(
                    args.ef_search, slot_nid(), slot_ctx());
                if (lane == 0) s_lane_req[lane_idx(w)] = req;
            }
        }
        tile.sync();

        if (ltid == 0) {
            bool any_active = false;
            for (int w = 0; w < W; ++w) {
                int32_t req = s_lane_req[lane_idx(w)];
                if (req != -1) {
                    issue_request(&ctl[w], req);
                    any_active = true;
                } else if (ctl[w].status != IO_IDLE) {
                    any_active = true;
                }
            }
            if (!any_active) s_slot_stage[slot] = 2;
        }
        tile.sync();
        return 1;
    };

    while (true) {
        if (s_slot_stage[slot] == -1) {
            int do_exit = 0;
            if (ltid == 0) {
                int qp = *(volatile int32_t*)args.d_next_query;
                if (qp >= k_total_queries &&
                    *(volatile int32_t*)&args.comm->terminate) {
                    do_exit = 1;
                }
                s_slot_alive[slot] = do_exit ? 0 : 1;
            }
            tile.sync();
            if (!s_slot_alive[slot]) return;

            int ok = try_bind_new_query();
            if (!ok) return;
            continue;
        }

        int alive = wait_io_and_step();
        if (!alive) return;
        if (s_slot_stage[slot] == 2) finalize_query();
    }
}

}  // namespace strawmen
}  // namespace quiver
