#pragma once

// Kernel helper functions extracted from quiver/kernel.cuh (Phase 1 refactor).
// These were identical to the FlashANNS counterparts but kept internal to
// Quiver per the refactor plan (src/flashanns/ stays frozen as a physical
// baseline and does not share code with Quiver).

#include "../io_control.cuh"
#include "../../shared/common/gpu_primitives.cuh"
#include "../../shared/common/search_state.cuh"
#include "../../shared/index/pq_device.cuh"

namespace quiver {

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

using shared::Data;
using shared::PQSearchData;
using shared::lower_bound;
using shared::retset_push_32;
using shared::square_sum_32;

__device__ __forceinline__ void copy_node_to_shm(
    const uint8_t* src, uint8_t* dst, int node_size)
{
    int word_count = node_size >> 2;
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
    for (int i = threadIdx.x; i < word_count; i += blockDim.x) {
        dst32[i] = src32[i];
    }

    int byte_start = word_count << 2;
    for (int i = byte_start + threadIdx.x; i < node_size; i += blockDim.x) {
        dst[i] = src[i];
    }
    __syncthreads();
}

__device__ __forceinline__ void drain_io_lanes_after_stop(
    IOControl* ctl, int W, int* drain_done)
{
    while (true) {
        if (threadIdx.x == 0) {
            int all_idle = 1;
            for (int w = 0; w < W; ++w) {
                int32_t st = *(volatile int32_t*)&ctl[w].status;
                if (st == IO_READY) {
                    ctl[w].status = IO_IDLE;
                } else if (st != IO_IDLE) {
                    all_idle = 0;
                }
            }
            *drain_done = all_idle;
        }
        __syncthreads();
        if (*drain_done) break;
    }
}

__device__ __noinline__ void retset_push_topk_le32(
    float* distance, int* idx, int& size,
    int max_size, float value, int value_idx)
{
    int lane = threadIdx.x & 31;
    int old_size = size;
    int capped_max = min(max_size, 32);
    bool valid = lane < old_size;
    float old_d = valid ? distance[lane] : INFINITY;
    int old_i = valid ? idx[lane] : 0;

    unsigned le_mask = __ballot_sync(0xffffffff, valid && old_d <= value);
    int insert_pos = __popc(le_mask);
    int new_size = (old_size < capped_max) ? (old_size + 1) : capped_max;

    float prev_d = __shfl_up_sync(0xffffffff, old_d, 1);
    int prev_i = __shfl_up_sync(0xffffffff, old_i, 1);
    __syncwarp();

    if (lane < new_size) {
        float out_d;
        int out_i;
        if (lane < insert_pos) {
            out_d = old_d;
            out_i = old_i;
        } else if (lane == insert_pos) {
            out_d = value;
            out_i = value_idx;
        } else {
            out_d = prev_d;
            out_i = prev_i;
        }
        distance[lane] = out_d;
        idx[lane] = out_i;
    }

    if (lane == 0) {
        size = new_size;
    }
}

__device__ __forceinline__ bool pair_less(
    float lhs_dist, uint32_t lhs_id,
    float rhs_dist, uint32_t rhs_id)
{
    uint32_t lhs_key = lhs_id & 0x7fffffffu;
    uint32_t rhs_key = rhs_id & 0x7fffffffu;
    return lhs_dist < rhs_dist ||
        (lhs_dist == rhs_dist && lhs_key < rhs_key);
}

__device__ __forceinline__ void warp_sort_pair_32(float& dist, uint32_t& id)
{
    int lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            float other_dist = __shfl_xor_sync(0xffffffff, dist, j);
            uint32_t other_id = __shfl_xor_sync(0xffffffff, id, j);
            bool current_less = pair_less(dist, id, other_dist, other_id);
            bool other_less = pair_less(other_dist, other_id, dist, id);
            bool ascending = (lane & k) == 0;
            bool lower_lane = (lane & j) == 0;
            bool take_other = lower_lane ? (ascending ? other_less : current_less)
                                         : (ascending ? current_less : other_less);
            if (take_other) {
                dist = other_dist;
                id = other_id;
            }
        }
    }
}

__device__ __forceinline__ void sort_new_neighbors_by_warp(
    int tid, int sz, int deg, uint32_t* tmp_id, float* tmp_dist)
{
    int local_idx = tid;
    float dist = (local_idx < deg) ? tmp_dist[sz + local_idx] : INFINITY;
    uint32_t id = (local_idx < deg) ? tmp_id[sz + local_idx]
                                    : (0xffffffffu - (uint32_t)local_idx);
    warp_sort_pair_32(dist, id);
    if (local_idx < deg) {
        tmp_id[sz + local_idx] = id;
        tmp_dist[sz + local_idx] = dist;
    }
    __syncthreads();
}

template <int LEN, int DEG>
__device__ __forceinline__ void merge_new_neighbor_stage_const(
    int tid, int sz, int* mv_pos, uint32_t* mv_id, float* mv_dist,
    uint32_t* tmp_id, float* tmp_dist)
{
    int array_id = tid / LEN;
    int start = array_id * LEN;
    int mid = min(start + LEN / 2, DEG);
    int end = min(start + LEN, DEG);
    if (tid >= start && tid < mid) {
        int id = lower_bound(tmp_dist + sz + mid, tmp_id + sz + mid,
                             0, end - mid,
                             tmp_dist[sz + tid], tmp_id[sz + tid]);
        mv_pos[tid] = id + tid;
    }
    if (tid >= mid && tid < end) {
        int id = lower_bound(tmp_dist + sz, tmp_id + sz,
                             start, mid,
                             tmp_dist[sz + tid], tmp_id[sz + tid]);
        mv_pos[tid] = id + tid - mid;
    }
    __syncthreads();
    if (tid < DEG) {
        mv_id[mv_pos[tid]] = tmp_id[sz + tid];
        mv_dist[mv_pos[tid]] = tmp_dist[sz + tid];
    }
    __syncthreads();
    if (tid < DEG) {
        tmp_id[sz + tid] = mv_id[tid];
        tmp_dist[sz + tid] = mv_dist[tid];
    }
    __syncthreads();
}

__device__ __forceinline__ void persistent_merge_data_r128_ef30(
    int tid, int sz, Data* ctx,
    uint32_t* neighbor_id, float* neighbor_dist,
    int* mv_pos, uint32_t* mv_id, float* mv_dist,
    uint32_t* tmp_id, float* tmp_dist)
{
    constexpr int DEG = 128;
    constexpr int KEEP = 60;

    sort_new_neighbors_by_warp(tid, sz, DEG, tmp_id, tmp_dist);
    merge_new_neighbor_stage_const<64, DEG>(tid, sz, mv_pos, mv_id, mv_dist,
                                            tmp_id, tmp_dist);
    merge_new_neighbor_stage_const<128, DEG>(tid, sz, mv_pos, mv_id, mv_dist,
                                             tmp_id, tmp_dist);

    if (tid < DEG) {
        int id = lower_bound(tmp_dist, tmp_id, 0, sz,
                             tmp_dist[sz + tid], tmp_id[sz + tid]);
        if (id != sz && ((tmp_id[id] ^ tmp_id[sz + tid]) & 0x7fffffff) == 0)
            id++;
        mv_pos[sz + tid] = id + tid;
    }
    for (int i = tid; i < sz; i += blockDim.x) {
        int id = lower_bound(tmp_dist + sz, tmp_id + sz, 0, DEG,
                             tmp_dist[i], tmp_id[i]);
        mv_pos[i] = id + i;
    }
    __syncthreads();

    for (int i = tid; i < sz + DEG; i += blockDim.x) {
        mv_id[mv_pos[i]] = tmp_id[i];
        mv_dist[mv_pos[i]] = tmp_dist[i];
    }
    __syncthreads();

    int target = min(sz + DEG, KEEP);
    for (int i = tid; i < target; i += blockDim.x) {
        neighbor_id[i] = mv_id[i];
        neighbor_dist[i] = mv_dist[i];
    }
    if (tid == 0) ctx->size = target;
    __syncthreads();
}

__device__ void persistent_merge_data(
    uint8_t* buffer, int32_t node_id,
    int num_chunks, float* pq_dists, uint8_t* compressed_data,
    int nodes_per_page, int node_size, int data_size,
    int max_m, int ef_search,
    uint32_t* neighbor_id, float* neighbor_dist, Data* ctx,
    int* mv_pos, uint32_t* mv_id, float* mv_dist,
    uint32_t* tmp_id, float* tmp_dist,
    uint8_t* node_scratch = nullptr
#ifdef QUIVER_LATENCY_PROBE
    , int64_t* probe_pq_done_out = nullptr
#endif
    )
{
    int tid = threadIdx.x;
    if (node_id == -1) return;

    int sz = ctx->size;
    int node_offset = node_id % nodes_per_page * node_size;
    uint8_t* node_base = buffer + node_offset;
    if (node_scratch != nullptr) {
        copy_node_to_shm(node_base, node_scratch, node_size);
        node_base = node_scratch;
    }
    int* buffer_u = (int*)(node_base + data_size);

    for (int i = tid; i < sz; i += blockDim.x) {
        tmp_id[i]   = neighbor_id[i];
        tmp_dist[i] = neighbor_dist[i];
    }
    __syncthreads();

    int deg = buffer_u[0];

    if (tid >= deg) {
        if (tid < max_m) {
            tmp_id[sz + tid]   = 0xffffffff - tid;
            tmp_dist[sz + tid] = INFINITY;
        }
    } else {
        int nodeid_v = tmp_id[sz + tid] = buffer_u[tid + 1];
        float dist = 0;
        uint8_t* pq_vec = compressed_data + (long)nodeid_v * num_chunks;
#pragma unroll 32
        for (int j = 0; j < num_chunks; j++)
            dist += pq_dists[j * PQSearchData::num_pivots + pq_vec[j]];
        tmp_dist[sz + tid] = dist;
    }
    __syncthreads();

#ifdef QUIVER_LATENCY_PROBE
    if (probe_pq_done_out && tid == 0) *probe_pq_done_out = globaltimer_ns();
#endif

    if (deg == 128 && ef_search == 30) {
        persistent_merge_data_r128_ef30(
            tid, sz, ctx, neighbor_id, neighbor_dist,
            mv_pos, mv_id, mv_dist, tmp_id, tmp_dist);
        return;
    }

    for (int len = 2; len < 2 * deg; len *= 2) {
        int array_id = tid / len;
        int start = array_id * len;
        int mid = min(start + len / 2, deg);
        int end = min(start + len, deg);
        if (tid >= start && tid < mid) {
            int id = lower_bound(tmp_dist + sz + mid, tmp_id + sz + mid,
                                 0, end - mid,
                                 tmp_dist[sz + tid], tmp_id[sz + tid]);
            mv_pos[tid] = id + tid;
        }
        if (tid >= mid && tid < end) {
            int id = lower_bound(tmp_dist + sz, tmp_id + sz,
                                 start, mid,
                                 tmp_dist[sz + tid], tmp_id[sz + tid]);
            mv_pos[tid] = id + tid - mid;
        }
        __syncthreads();
        __threadfence_block();
        if (tid < deg) {
            mv_id[mv_pos[tid]]   = tmp_id[sz + tid];
            mv_dist[mv_pos[tid]] = tmp_dist[sz + tid];
        }
        __syncthreads();
        if (tid < deg) {
            tmp_id[sz + tid]   = mv_id[tid];
            tmp_dist[sz + tid] = mv_dist[tid];
        }
        __syncthreads();
    }

    if (tid < deg) {
        int id = lower_bound(tmp_dist, tmp_id, 0, sz,
                             tmp_dist[sz + tid], tmp_id[sz + tid]);
        if (id != sz && ((tmp_id[id] ^ tmp_id[sz + tid]) & 0x7fffffff) == 0) id++;
        mv_pos[sz + tid] = id + tid;
    }
    for (int i = tid; i < sz; i += blockDim.x) {
        int id = lower_bound(tmp_dist + sz, tmp_id + sz, 0, deg,
                             tmp_dist[i], tmp_id[i]);
        mv_pos[i] = id + i;
    }
    __syncthreads();

    for (int i = tid; i < sz + deg; i += blockDim.x) {
        mv_id[mv_pos[i]]   = tmp_id[i];
        mv_dist[mv_pos[i]] = tmp_dist[i];
    }
    __syncthreads();

    int target = min(sz + deg, 2 * ef_search);
    for (int i = tid; i < target; i += blockDim.x) {
        neighbor_id[i]   = mv_id[i];
        neighbor_dist[i] = mv_dist[i];
    }
    if (tid == 0) ctx->size = target;
    __syncthreads();
}

template <class T>
__device__ int32_t persistent_visit_select(
    float* qdata, uint8_t* buffer, int32_t node_id,
    int num_dims, int topk,
    int* nns, float* distances, int* found_cnt,
    int nodes_per_page, int node_size,
    int max_m, int ef_search,
    uint32_t* neighbor_id, float* neighbor_dist, Data* ctx)
{
    int tid = threadIdx.x % 32;
    if (node_id != -1) {
        int buf_off = node_id % nodes_per_page * node_size;
        T* buffer_u = (T*)(buffer + buf_off);
        float dist = square_sum_32(qdata, buffer_u, num_dims);
        int fc = found_cnt[0];
        if (fc < topk || dist < distances[topk - 1]) {
            if (topk <= 32) {
                retset_push_topk_le32(distances, nns, found_cnt[0], topk, dist, node_id);
            } else {
                retset_push_32(distances, nns, found_cnt[0], topk, dist, node_id);
            }
        }
    }
    int sz = ctx->size;
    if (sz == 0) return -1;

    int target_r = (sz + 31) / 32 * 32;
    int count = 0;
    for (int i = tid; i < target_r; i += 32) {
        bool flag = (i != 0) && (i < sz) &&
            (((neighbor_id[i] ^ neighbor_id[i - 1]) & 0x7fffffff) == 0);
        unsigned mask = __ballot_sync(0xffffffff, flag);
        int mv = i - count - __popc(mask & ((1u << tid) - 1));
        uint32_t t_id   = neighbor_id[i];
        float    t_dist = neighbor_dist[i];
        __syncwarp();
        if (i < sz && !flag) { neighbor_id[mv] = t_id; neighbor_dist[mv] = t_dist; }
        __syncwarp();
        count += __popc(mask);
        __syncwarp();
    }
    sz -= count;
    int id = sz + 1;
    for (int i = tid; i < sz; i += 32)
        if ((neighbor_id[i] & 0x80000000u) == 0) id = min(id, i);
    __syncwarp();
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        int _id = __shfl_down_sync(0xffffffff, id, off);
        id = min(_id, id);
    }
    id = __shfl_sync(0xffffffff, id, 0);
    int32_t next = -1;
    if (tid == 0) {
        if (id < ef_search && id < sz) {
            next = neighbor_id[id] & 0x7fffffff;
            neighbor_id[id] |= 0x80000000u;
        }
        ctx->size = min(sz, ef_search);
    }
    return __shfl_sync(0xffffffff, next, 0);
}

template <class T>
__device__ int32_t persistent_visit_select_dualwarp(
    float* qdata, uint8_t* node_data, int32_t node_id,
    int num_dims, int topk,
    int* nns, float* distances, int* found_cnt,
    int nodes_per_page, int node_size,
    int max_m, int ef_search,
    uint32_t* neighbor_id, float* neighbor_dist, Data* ctx)
{
    (void)nodes_per_page;
    (void)node_size;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int32_t next = -1;

    if (tid < 32) {
        int sz = ctx->size;
        if (sz != 0) {
            int target_r = (sz + 31) / 32 * 32;
            int count = 0;
            for (int i = lane; i < target_r; i += 32) {
                bool flag = (i != 0) && (i < sz) &&
                    (((neighbor_id[i] ^ neighbor_id[i - 1]) & 0x7fffffff) == 0);
                unsigned mask = __ballot_sync(0xffffffff, flag);
                int mv = i - count - __popc(mask & ((1u << lane) - 1));
                uint32_t t_id   = neighbor_id[i];
                float    t_dist = neighbor_dist[i];
                __syncwarp();
                if (i < sz && !flag) { neighbor_id[mv] = t_id; neighbor_dist[mv] = t_dist; }
                __syncwarp();
                count += __popc(mask);
                __syncwarp();
            }
            sz -= count;
            int id = sz + 1;
            for (int i = lane; i < sz; i += 32)
                if ((neighbor_id[i] & 0x80000000u) == 0) id = min(id, i);
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
    } else if (tid < 64 && node_id != -1) {
        T* buffer_u = (T*)node_data;
        float dist = square_sum_32(qdata, buffer_u, num_dims);
        int fc = found_cnt[0];
        if (fc < topk || dist < distances[topk - 1]) {
            if (topk <= 32) {
                retset_push_topk_le32(distances, nns, found_cnt[0], topk, dist, node_id);
            } else {
                retset_push_32(distances, nns, found_cnt[0], topk, dist, node_id);
            }
        }
    }

    return next;
}

__device__ int32_t persistent_select_next_no_compact(
    int ef_search, uint32_t* neighbor_id, Data* ctx)
{
    int tid = threadIdx.x % 32;
    int sz = min(ctx->size, ef_search);
    if (sz == 0) return -1;

    int id = sz + 1;
    for (int i = tid; i < sz; i += 32)
        if ((neighbor_id[i] & 0x80000000u) == 0) id = min(id, i);
    __syncwarp();
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        int _id = __shfl_down_sync(0xffffffff, id, off);
        id = min(_id, id);
    }
    id = __shfl_sync(0xffffffff, id, 0);

    int32_t next = -1;
    if (tid == 0) {
        if (id < sz) {
            next = neighbor_id[id] & 0x7fffffff;
            neighbor_id[id] |= 0x80000000u;
        }
        ctx->size = sz;
    }
    return __shfl_sync(0xffffffff, next, 0);
}

__device__ __forceinline__ int count_topk_gt_hits(
    const int* nns, int topk, const int* gt, int gt_width)
{
    int hits = 0;
    for (int i = 0; i < topk; ++i) {
        int id = nns[i];
        if (id < 0) continue;
        for (int j = 0; j < gt_width; ++j) {
            if (id == gt[j]) {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

__device__ int32_t persistent_select_next(
    int max_m, int ef_search,
    uint32_t* neighbor_id, float* neighbor_dist, Data* ctx)
{
    int tid = threadIdx.x % 32;
    int sz = ctx->size;
    if (sz == 0) return -1;
    int target_r = (sz + 31) / 32 * 32;
    int count = 0;
    for (int i = tid; i < target_r; i += 32) {
        bool flag = (i != 0) && (i < sz) &&
            (((neighbor_id[i] ^ neighbor_id[i - 1]) & 0x7fffffff) == 0);
        unsigned mask = __ballot_sync(0xffffffff, flag);
        int mv = i - count - __popc(mask & ((1u << tid) - 1));
        uint32_t t_id   = neighbor_id[i];
        float    t_dist = neighbor_dist[i];
        __syncwarp();
        if (i < sz && !flag) { neighbor_id[mv] = t_id; neighbor_dist[mv] = t_dist; }
        __syncwarp();
        count += __popc(mask);
        __syncwarp();
    }
    sz -= count;
    int id = sz + 1;
    for (int i = tid; i < sz; i += 32)
        if ((neighbor_id[i] & 0x80000000u) == 0) id = min(id, i);
    __syncwarp();
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        int _id = __shfl_down_sync(0xffffffff, id, off);
        id = min(_id, id);
    }
    id = __shfl_sync(0xffffffff, id, 0);
    int32_t next = -1;
    if (tid == 0) {
        if (id < ef_search && id < sz) {
            next = neighbor_id[id] & 0x7fffffff;
            neighbor_id[id] |= 0x80000000u;
        }
        ctx->size = min(sz, ef_search);
    }
    return __shfl_sync(0xffffffff, next, 0);
}

__device__ __forceinline__ void issue_request(IOControl* ctl, int32_t node_id)
{
    ctl->node_id = node_id;
    // Publish node_id before status so CPU acquire-load of IO_REQUESTED sees
    // the request payload. A second fence after status is unnecessary here.
    __threadfence_system();
    ctl->status = IO_REQUESTED;
}

} // namespace quiver
