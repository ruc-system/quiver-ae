#pragma once

// ============================================================================
// Strawman PQ-in-global-memory / explicit context spill-load.
// ----------------------------------------------------------------------------
// This strawman keeps Kernel-Side Query Switching, but removes the
// occupancy-aware context placement assumption that per-slot context can stay
// resident in shared memory across switches.
//
// Resident state for every query slot lives in global memory:
//   - PQ LUT
//   - frontier neighbor_id / neighbor_dist
//   - Data ctx
//   - top-k nns / distances / found_cnt
//
// On every slot scheduling event, the block explicitly loads that slot context
// into a single shared-memory workspace, processes the ready IO lane(s), then
// explicitly writes the full context back to global memory before switching
// away.  The workspace is one copy per block, not Q copies per block.
// ============================================================================

#include "../kernel_helpers.cuh"
#include "../inkernel_nav.cuh"
#include "../policies/dispatch.cuh"

namespace quiver {
namespace strawmen {

constexpr int PQ_IN_GMEM_MAX_Q = 8;

__host__ __device__ __forceinline__ size_t pq_in_gmem_align8(size_t x)
{
    return (x + 7) & ~size_t{7};
}

__host__ __device__ __forceinline__ size_t pq_in_gmem_workspace_bytes(
    int ef_search,
    int max_m,
    int node_size,
    int aligned_ef,
    int topk,
    int num_chunks)
{
    const size_t frontier_bytes =
        (sizeof(int) * 3 + sizeof(float) * 2) * (size_t)(ef_search + max_m);
    const size_t node_scratch_offset = pq_in_gmem_align8(frontier_bytes);
    const size_t stride_bytes = node_scratch_offset + (size_t)node_size;
    size_t workspace_bytes = pq_in_gmem_align8(stride_bytes);
    workspace_bytes += (size_t)aligned_ef * sizeof(uint32_t);
    workspace_bytes = pq_in_gmem_align8(workspace_bytes);
    workspace_bytes += (size_t)aligned_ef * sizeof(float);
    workspace_bytes = pq_in_gmem_align8(workspace_bytes);
    workspace_bytes += (size_t)topk * sizeof(int);
    workspace_bytes = pq_in_gmem_align8(workspace_bytes);
    workspace_bytes += (size_t)topk * sizeof(float);
    workspace_bytes = pq_in_gmem_align8(workspace_bytes);
    workspace_bytes += (size_t)PQSearchData::num_pivots
                       * (size_t)num_chunks
                       * sizeof(float);
    return workspace_bytes;
}

template <class Dispatch, class T>
__global__ void persistent_search_kernel_pq_in_gmem(PersistentKernelArgs args)
{
    static_assert(Dispatch::is_persistent,
                  "Strawman PQ-in-global-memory only meaningful with StreamingDispatch.");

    extern __shared__ uint8_t shm_pool[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int W   = args.pipe_w;
    const int Q   = args.queries_per_block;
    const int ef_plus_m = args.ef_search + args.max_m;
    const int frontier_bytes = (sizeof(int) * 3 + sizeof(float) * 2) * ef_plus_m;
    const int node_scratch_offset = (frontier_bytes + 7) & ~7;
    const int stride_bytes = node_scratch_offset + args.node_size;
    const int k_total_queries = args.total_queries;
    const int aef = args.aligned_ef;
    const int pq_lut_size = PQSearchData::num_pivots * args.num_chunks;  // floats

    __shared__ int32_t s_slot_qid[PQ_IN_GMEM_MAX_Q];
    __shared__ int32_t s_slot_entry[PQ_IN_GMEM_MAX_Q];
    __shared__ int32_t s_slot_stage[PQ_IN_GMEM_MAX_Q];
    __shared__ int32_t s_slot_next[PQ_IN_GMEM_MAX_Q];
    __shared__ int32_t s_active_mask;
    __shared__ int     s_pick;
    __shared__ int32_t s_kernel_exit;
    __shared__ int32_t s_snap_st[16];
    __shared__ int32_t s_snap_nd[16];
    __shared__ int32_t s_lane_req[16];
    __shared__ Data    s_work_ctx;
    __shared__ int32_t s_work_found_cnt;

    if (tid == 0) {
        s_active_mask = 0;
        s_kernel_exit = 0;
        for (int q = 0; q < Q; ++q) {
            s_slot_qid[q]   = -1;
            s_slot_entry[q] = -1;
            s_slot_stage[q] = -1;
            s_slot_next[q]  = -1;
        }
    }
    __syncthreads();

    auto work_shm_base = [&]() -> uint8_t* {
        return shm_pool;
    };
    auto work_node_scratch = [&]() -> uint8_t* {
        return work_shm_base() + node_scratch_offset;
    };
    auto work_nid = [&]() -> uint32_t* {
        size_t off = pq_in_gmem_align8((size_t)stride_bytes);
        return (uint32_t*)(shm_pool + off);
    };
    auto work_ndist = [&]() -> float* {
        size_t off = pq_in_gmem_align8((size_t)stride_bytes);
        off += (size_t)aef * sizeof(uint32_t);
        off = pq_in_gmem_align8(off);
        return (float*)(shm_pool + off);
    };
    auto work_nns = [&]() -> int* {
        size_t off = pq_in_gmem_align8((size_t)stride_bytes);
        off += (size_t)aef * sizeof(uint32_t);
        off = pq_in_gmem_align8(off);
        off += (size_t)aef * sizeof(float);
        off = pq_in_gmem_align8(off);
        return (int*)(shm_pool + off);
    };
    auto work_dist = [&]() -> float* {
        size_t off = pq_in_gmem_align8((size_t)stride_bytes);
        off += (size_t)aef * sizeof(uint32_t);
        off = pq_in_gmem_align8(off);
        off += (size_t)aef * sizeof(float);
        off = pq_in_gmem_align8(off);
        off += (size_t)args.topk * sizeof(int);
        off = pq_in_gmem_align8(off);
        return (float*)(shm_pool + off);
    };
    auto work_pq = [&]() -> float* {
        size_t off = pq_in_gmem_align8((size_t)stride_bytes);
        off += (size_t)aef * sizeof(uint32_t);
        off = pq_in_gmem_align8(off);
        off += (size_t)aef * sizeof(float);
        off = pq_in_gmem_align8(off);
        off += (size_t)args.topk * sizeof(int);
        off = pq_in_gmem_align8(off);
        off += (size_t)args.topk * sizeof(float);
        off = pq_in_gmem_align8(off);
        return (float*)(shm_pool + off);
    };

    auto slot_ctl = [&](int q) -> IOControl* {
        return args.comm->controls + ((bid * Q + q) * W);
    };
    auto slot_buf = [&](int q) -> uint8_t* {
        return args.comm->buffers + (long)(bid * Q + q) * W * shared::PAGE_SIZE;
    };
    auto slot_nid = [&](int q) -> uint32_t* {
        return args.d_neighbors_id + (long)(bid * Q + q) * aef;
    };
    auto slot_ndist = [&](int q) -> float* {
        return args.d_neighbors_dist + (long)(bid * Q + q) * aef;
    };
    auto slot_ctx = [&](int q) -> Data* {
        return args.d_ctx + (bid * Q + q);
    };
    auto slot_nns = [&](int q) -> int* {
        return args.d_nns + (bid * Q + q) * args.topk;
    };
    auto slot_dist = [&](int q) -> float* {
        return args.d_distances + (bid * Q + q) * args.topk;
    };
    auto slot_fc = [&](int q) -> int* {
        return args.d_found_cnt + (bid * Q + q);
    };
    auto slot_pq = [&](int q) -> float* {
        return args.pq_dists
            + (long)((bid * Q + q) + args.pq_block_offset)
              * PQSearchData::num_pivots * args.num_chunks;
    };

    auto load_slot_context = [&](int q) {
        uint32_t* g_nid = slot_nid(q);
        float* g_ndist = slot_ndist(q);
        int* g_nns = slot_nns(q);
        float* g_dist = slot_dist(q);
        float* g_pq = slot_pq(q);
        uint32_t* s_nid = work_nid();
        float* s_ndist = work_ndist();
        int* s_nns = work_nns();
        float* s_dist = work_dist();
        float* s_pq = work_pq();

        for (int i = tid; i < aef; i += blockDim.x) {
            s_nid[i] = g_nid[i];
            s_ndist[i] = g_ndist[i];
        }
        for (int i = tid; i < args.topk; i += blockDim.x) {
            s_nns[i] = g_nns[i];
            s_dist[i] = g_dist[i];
        }
        for (int i = tid; i < pq_lut_size; i += blockDim.x) {
            s_pq[i] = g_pq[i];
        }
        if (tid == 0) {
            Data* g_ctx = slot_ctx(q);
            s_work_ctx.size = g_ctx->size;
            s_work_ctx.visited_cnt = g_ctx->visited_cnt;
            s_work_found_cnt = *slot_fc(q);
        }
        __syncthreads();
    };

    auto store_slot_context = [&](int q) {
        uint32_t* g_nid = slot_nid(q);
        float* g_ndist = slot_ndist(q);
        int* g_nns = slot_nns(q);
        float* g_dist = slot_dist(q);
        float* g_pq = slot_pq(q);
        uint32_t* s_nid = work_nid();
        float* s_ndist = work_ndist();
        int* s_nns = work_nns();
        float* s_dist = work_dist();
        float* s_pq = work_pq();

        for (int i = tid; i < aef; i += blockDim.x) {
            g_nid[i] = s_nid[i];
            g_ndist[i] = s_ndist[i];
        }
        for (int i = tid; i < args.topk; i += blockDim.x) {
            g_nns[i] = s_nns[i];
            g_dist[i] = s_dist[i];
        }
        for (int i = tid; i < pq_lut_size; i += blockDim.x) {
            g_pq[i] = s_pq[i];
        }
        if (tid == 0) {
            Data* g_ctx = slot_ctx(q);
            g_ctx->size = s_work_ctx.size;
            g_ctx->visited_cnt = s_work_ctx.visited_cnt;
            *slot_fc(q) = s_work_found_cnt;
        }
        __syncthreads();
    };

    auto try_bind_new_query = [&](int q) -> bool {
        int qid;
        if (tid == 0) {
            qid = atomicAdd(args.d_next_query, 1);
            s_slot_qid[q] = qid;
        }
        __syncthreads();
        qid = s_slot_qid[q];

        const int total = k_total_queries;
        if (tid == 0) {
            int got = 0;
            while (qid < total) {
                int fc = *(volatile int32_t*)&args.comm->feed_count;
                if (qid < fc) { got = 1; break; }
                if (*(volatile int32_t*)&args.comm->terminate) { got = 0; break; }
            }
            if (qid >= total) got = 0;
            if (got) __threadfence_system();
            s_slot_stage[q] = got ? 0 : -2;
        }
        __syncthreads();

        if (s_slot_stage[q] == -2) {
            __syncthreads();
            return false;
        }

        float* qdata = args.d_all_qdata + (long)qid * args.num_dims;

        int32_t entry_from_nav = -1;
        if (args.nav_data_dev != nullptr) {
            uint8_t* nav_base = work_shm_base();
            int*      nav_mv_pos  = (int*)nav_base;
            uint32_t* nav_pq_id   = (uint32_t*)(nav_mv_pos + ef_plus_m);
            float*    nav_pq_dist = (float*)   (nav_pq_id  + ef_plus_m);
            __shared__ int s_nav_idx;
            entry_from_nav = inkernel_nav_entry<T>(
                qdata,
                (const T*)args.nav_data_dev,
                args.nav_graph_dev,
                args.nav_mapping_dev,
                args.nav_data_len,
                args.nav_max_m,
                args.nav_init_ef,
                args.nav_start,
                nav_pq_id,
                nav_pq_dist,
                &s_nav_idx);
        }

        {
            shared::PQSearchData* pqd = args.pq_data;
            float* dist_vec = work_pq();
            const int num_pivots = PQSearchData::num_pivots;
            const int chunks_local = pqd->num_chunks;
            const int dim = pqd->dim;
            for (int i = tid; i < num_pivots * chunks_local; i += blockDim.x) {
                dist_vec[i] = 0;
            }
            for (int i = tid; i < dim; i += blockDim.x) {
                qdata[i] -= pqd->centroid[i];
            }
            __syncthreads();
            for (int i = 0; i < dim; i++) {
                int idx = pqd->chunk_id[i];
                for (int j = tid; j < num_pivots; j += blockDim.x) {
                    float dd = qdata[i] - pqd->pivots_t[i * num_pivots + j];
                    dist_vec[idx * num_pivots + j] += dd * dd;
                }
                __syncthreads();
            }
            for (int i = tid; i < dim; i += blockDim.x) {
                qdata[i] += pqd->centroid[i];
            }
        }
        __syncthreads();

        int32_t entry;
        if (args.nav_data_dev != nullptr) {
            entry = entry_from_nav;
        } else if (args.d_entry_nodes) {
            entry = args.d_entry_nodes[qid];
        } else {
            entry = args.enter_point;
        }

        if (tid == 0) {
            s_slot_entry[q] = entry;
            s_work_ctx.size = 0;
            s_work_ctx.visited_cnt = 0;
            s_work_found_cnt = 0;
            for (int w = 0; w < W; ++w) slot_ctl(q)[w].status = IO_IDLE;
        }
        for (int i = tid; i < args.topk; i += blockDim.x) {
            work_dist()[i] = INFINITY;
            work_nns()[i]  = -1;
        }
        __syncthreads();

        store_slot_context(q);

        if (tid == 0) issue_request(&slot_ctl(q)[0], entry);
        __syncthreads();
        return true;
    };

    auto finalize_slot = [&](int q) {
        int qid = s_slot_qid[q];
        load_slot_context(q);
        for (int i = tid; i < args.topk; i += blockDim.x) {
            args.d_out_nns[(long)qid * args.topk + i]       = work_nns()[i];
            args.d_out_distances[(long)qid * args.topk + i] = work_dist()[i];
        }
        if (tid == 0) {
            args.d_out_found_cnt[qid] = s_work_found_cnt;
            int new_done = atomicAdd(args.d_queries_done, 1) + 1;
            *(volatile int32_t*)&args.comm->queries_completed = new_done;
            __threadfence_system();
            s_slot_stage[q] = -1;
        }
        __syncthreads();
    };

    auto try_process_slot_hop = [&](int q) -> bool {
        IOControl* ctl = slot_ctl(q);

        if (s_slot_stage[q] == 0) {
            if (tid == 0) {
                int32_t st = *(volatile int32_t*)&ctl[0].status;
                s_pick = (st == IO_READY) ? q : -1;
            }
            __syncthreads();
            if (s_pick != q) return false;

            if (tid == 0) {
                s_snap_nd[0] = ctl[0].node_id;
                ctl[0].status = IO_IDLE;
            }
            __syncthreads();
            __threadfence_system();

            load_slot_context(q);

            uint8_t* base = work_shm_base();
            int*      mv_pos  = (int*)base;
            uint32_t* mv_id   = (uint32_t*)(mv_pos  + ef_plus_m);
            float*    mv_dist = (float*)   (mv_id   + ef_plus_m);
            uint32_t* tmp_id  = (uint32_t*)(mv_dist + ef_plus_m);
            float*    tmp_dist= (float*)   (tmp_id  + ef_plus_m);
            float* qdata = args.d_all_qdata + (long)s_slot_qid[q] * args.num_dims;

            persistent_merge_data(
                slot_buf(q), s_slot_entry[q],
                args.num_chunks, work_pq(), args.compressed_data,
                args.nodes_per_page, args.node_size, args.data_size,
                args.max_m, args.ef_search,
                work_nid(), work_ndist(), &s_work_ctx,
                mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
                work_node_scratch());

            int32_t boot_req = persistent_visit_select_dualwarp<T>(
                qdata, work_node_scratch(), s_slot_entry[q],
                args.num_dims, args.topk,
                work_nns(), work_dist(), &s_work_found_cnt,
                args.nodes_per_page, args.node_size,
                args.max_m, args.ef_search,
                work_nid(), work_ndist(), &s_work_ctx);
            if (tid == 0) s_lane_req[0] = boot_req;
            __syncthreads();

            if (tid < 32) {
                for (int w = 1; w < W; w++)
                    s_lane_req[w] = persistent_select_next_no_compact(
                        args.ef_search, work_nid(), &s_work_ctx);
            }
            __syncthreads();

            store_slot_context(q);

            if (tid == 0) {
                bool any_issued = false;
                for (int w = 0; w < W; w++) {
                    if (s_lane_req[w] != -1) {
                        issue_request(&ctl[w], s_lane_req[w]);
                        any_issued = true;
                    }
                }
                s_slot_stage[q] = any_issued ? 1 : 2;
            }
            __syncthreads();
            return true;
        }

        if (tid == 0) {
            s_pick = -1;
            bool any_active = false;
            for (int w = 0; w < W; w++) {
                int32_t st = *(volatile int32_t*)&ctl[w].status;
                s_snap_st[w] = st;
                if (st == IO_READY) s_pick = w;
                if (st != IO_IDLE) any_active = true;
            }
            if (s_pick == -1 && !any_active)
                s_pick = -2;
            if (s_pick >= 0) {
                for (int w = 0; w < W; w++) {
                    if (s_snap_st[w] == IO_READY)
                        s_snap_nd[w] = ctl[w].node_id;
                }
            }
        }
        __syncthreads();

        if (s_pick == -2) {
            if (tid == 0) s_slot_stage[q] = 2;
            __syncthreads();
            return false;
        }
        if (s_pick == -1) return false;

        __threadfence_system();

        load_slot_context(q);

        uint8_t* base = work_shm_base();
        int*      mv_pos  = (int*)base;
        uint32_t* mv_id   = (uint32_t*)(mv_pos  + ef_plus_m);
        float*    mv_dist = (float*)   (mv_id   + ef_plus_m);
        uint32_t* tmp_id  = (uint32_t*)(mv_dist + ef_plus_m);
        float*    tmp_dist= (float*)   (tmp_id  + ef_plus_m);
        float* qdata = args.d_all_qdata + (long)s_slot_qid[q] * args.num_dims;

        for (int w = 0; w < W; w++) {
            if (s_snap_st[w] != IO_READY) continue;
            if (tid == 0) ctl[w].status = IO_IDLE;

            persistent_merge_data(
                slot_buf(q) + w * shared::PAGE_SIZE, s_snap_nd[w],
                args.num_chunks, work_pq(), args.compressed_data,
                args.nodes_per_page, args.node_size, args.data_size,
                args.max_m, args.ef_search,
                work_nid(), work_ndist(), &s_work_ctx,
                mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
                work_node_scratch());

            int32_t lane_req = persistent_visit_select_dualwarp<T>(
                qdata, work_node_scratch(), s_snap_nd[w],
                args.num_dims, args.topk,
                work_nns(), work_dist(), &s_work_found_cnt,
                args.nodes_per_page, args.node_size,
                args.max_m, args.ef_search,
                work_nid(), work_ndist(), &s_work_ctx);
            if (tid == 0) s_lane_req[w] = lane_req;
            __syncthreads();
        }

        if (tid < 32) {
            for (int w = 0; w < W; w++) {
                if (s_snap_st[w] != IO_READY &&
                    *(volatile int32_t*)&ctl[w].status == IO_IDLE) {
                    s_lane_req[w] = persistent_select_next_no_compact(
                        args.ef_search, work_nid(), &s_work_ctx);
                }
            }
        }
        __syncthreads();

        store_slot_context(q);

        if (tid == 0) {
            bool any_active = false;
            for (int w = 0; w < W; w++) {
                if (s_snap_st[w] == IO_READY && s_lane_req[w] != -1) {
                    issue_request(&ctl[w], s_lane_req[w]);
                    any_active = true;
                } else if (s_snap_st[w] != IO_READY &&
                           ctl[w].status == IO_IDLE &&
                           s_lane_req[w] != -1) {
                    issue_request(&ctl[w], s_lane_req[w]);
                    any_active = true;
                } else if (ctl[w].status != IO_IDLE) {
                    any_active = true;
                }
            }
            if (!any_active) s_slot_stage[q] = 2;
        }
        __syncthreads();
        return true;
    };

    int last_slot = -1;

    while (true) {
        __syncthreads();

        for (int q = 0; q < Q; ++q) {
            if (s_slot_stage[q] == -1) {
                if (tid == 0) {
                    int qid_preview = *(volatile int32_t*)args.d_next_query;
                    if (qid_preview >= k_total_queries &&
                        *(volatile int32_t*)&args.comm->terminate) {
                        s_kernel_exit = 1;
                    }
                }
                __syncthreads();
                if (s_kernel_exit) break;

                bool ok = try_bind_new_query(q);
                if (!ok) {
                    if (tid == 0) s_slot_stage[q] = -3;
                    __syncthreads();
                }
            }
        }

        int any_active = 0;
        if (tid == 0) {
            for (int q = 0; q < Q; ++q) {
                if (s_slot_stage[q] >= 0) { any_active = 1; break; }
            }
            s_active_mask = any_active;
        }
        __syncthreads();
        if (!s_active_mask) {
            if (tid == 0) {
                if (*(volatile int32_t*)&args.comm->terminate ||
                    *(volatile int32_t*)args.d_next_query >= k_total_queries) {
                    s_kernel_exit = 1;
                }
            }
            __syncthreads();
            if (s_kernel_exit) return;
            continue;
        }

        bool made_progress = false;
        for (int probe = 0; probe < Q; ++probe) {
            int q = (last_slot + 1 + probe) % Q;
            if (s_slot_stage[q] < 0 || s_slot_stage[q] == 2) continue;
            if (try_process_slot_hop(q)) {
                last_slot = q;
                made_progress = true;
                if (s_slot_stage[q] == 2) finalize_slot(q);
                break;
            }
        }

        if (!made_progress) {
            for (int q = 0; q < Q; ++q) {
                if (s_slot_stage[q] == 2) finalize_slot(q);
            }
        }
    }
}

}  // namespace strawmen
}  // namespace quiver
