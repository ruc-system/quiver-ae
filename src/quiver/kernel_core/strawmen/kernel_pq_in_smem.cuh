#pragma once

// ============================================================================
// Strawman PQ-in-shared-memory: PQ LUT in shared memory
// ----------------------------------------------------------------------------
// Strawman optimisation: "PQ distance table is per-query hot data, hit 256 ×
// num_chunks × deg times per hop. Putting it in shared memory should be
// strictly better than global memory."
//
// Why this fails for ANN (in contrast to e.g. ML serving):
//   - SIFT-100M num_chunks = 64  →  PQ LUT = 256 × 64 × 4B = 64 KB / slot.
//   - A100 dynamic shared memory cap (per block) = 164 KB (with opt-in via
//     cudaFuncSetAttribute). H100 is 228 KB but still per-block.
//   - With Q slots in one block:
//        Q=1: 64 KB PQ + ~3 KB frontier = 67 KB     (occupancy ~2 / SM)
//        Q=2: 128 KB + 6 KB = 134 KB                 (occupancy 1, severe drop)
//        Q=3: 192 KB + 10 KB = 202 KB > 164 KB       cudaFuncSetAttribute fails
//        Q=4: 256 KB + 13 KB = 269 KB > 164 KB       cudaFuncSetAttribute fails
//   - The host launcher should report the cudaFuncSetAttribute error and
//     refuse to launch for Q>=3. For Q=1,2 the kernel runs but throughput
//     drops because (a) occupancy collapses and (b) PQ access pattern is
//     not actually shared-friendly (each thread reads its own pivot row).
//
// Everything else (scheduler, IO state machine, streaming admission) is
// IDENTICAL to kernel_qi.cuh. The only ablated dimension is the memory
// class chosen for `pq_dists`.
//
// Implementation note: persistent_merge_data() and persistent_visit_select()
// take pq_dists as a `float*` parameter and don't care which memory space
// it lives in. We simply route slot_pq(q) to a per-slot region inside
// shm_pool instead of args.pq_dists.
// ============================================================================

#include "../kernel_helpers.cuh"
#include "../inkernel_nav.cuh"
#include "../policies/dispatch.cuh"

namespace quiver {
namespace strawmen {

constexpr int PQ_IN_SMEM_MAX_Q = 8;

template <class Dispatch, class T>
__global__ void persistent_search_kernel_pq_in_smem(PersistentKernelArgs args)
{
    static_assert(Dispatch::is_persistent,
                  "Strawman PQ-in-shared-memory only meaningful with StreamingDispatch.");

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
    const int num_chunks = args.num_chunks;
    const int pq_lut_size = PQSearchData::num_pivots * num_chunks;  // floats

    // Layout of shm_pool:
    //   [ frontier_slot0 (stride_bytes) | frontier_slot1 | ... | frontier_slotQ-1 ]
    //   [ pq_lut_slot0 (4 * pq_lut_size) | pq_lut_slot1 | ... | pq_lut_slotQ-1 ]
    const size_t frontier_total_bytes = (size_t)Q * stride_bytes;

    __shared__ int32_t s_slot_qid[PQ_IN_SMEM_MAX_Q];
    __shared__ int32_t s_slot_entry[PQ_IN_SMEM_MAX_Q];
    __shared__ int32_t s_slot_stage[PQ_IN_SMEM_MAX_Q];
    __shared__ int32_t s_slot_next[PQ_IN_SMEM_MAX_Q];
    __shared__ int32_t s_active_mask;
    __shared__ int     s_pick;
    __shared__ int32_t s_kernel_exit;
    __shared__ int32_t s_snap_st[16];
    __shared__ int32_t s_snap_nd[16];
    __shared__ int32_t s_lane_req[16];
#ifdef QUIVER_LIGHT_BREAKDOWN
    __shared__ int64_t s_breakdown_lane_issue_ts[PQ_IN_SMEM_MAX_Q][16];
#endif

    if (tid == 0) {
        s_active_mask = 0;
        s_kernel_exit = 0;
        for (int q = 0; q < Q; ++q) {
            s_slot_qid[q]   = -1;
            s_slot_entry[q] = -1;
            s_slot_stage[q] = -1;
            s_slot_next[q]  = -1;
#ifdef QUIVER_LIGHT_BREAKDOWN
            for (int w = 0; w < 16; ++w)
                s_breakdown_lane_issue_ts[q][w] = 0;
#endif
        }
    }
    __syncthreads();

    const int aef = args.aligned_ef;

    auto slot_shm_base = [&](int q) -> uint8_t* {
        return shm_pool + (size_t)q * stride_bytes;
    };
    auto slot_node_scratch = [&](int q) -> uint8_t* {
        return slot_shm_base(q) + node_scratch_offset;
    };
    auto slot_pq_smem = [&](int q) -> float* {
        return (float*)(shm_pool + frontier_total_bytes
                        + (size_t)q * pq_lut_size * sizeof(float));
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

    // ------------------------------------------------------------------
    // try_bind_new_query: identical to kernel_qi.cuh, EXCEPT that the
    // residual-PQ initialisation writes into shared memory (slot_pq_smem)
    // rather than the global pq_dists array. The global array is left
    // unused by the PQ-in-shared-memory strawman (host can leave it allocated for ABI compatibility).
    // ------------------------------------------------------------------
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

#ifdef QUIVER_LIGHT_BREAKDOWN
        const int breakdown_sample_idx = quiver_breakdown_sample_index(
            qid, args.total_queries, args.breakdown_sample_count);
        if (tid == 0 && breakdown_sample_idx >= 0 &&
            args.d_breakdown_traces != nullptr) {
            auto& tr = args.d_breakdown_traces[breakdown_sample_idx];
            tr.query_id = qid;
            tr.valid = 1;
            tr.query_start_ns = globaltimer_ns();
            tr.query_done_ns = 0;
            tr.useful_compute_ns = 0;
            tr.resumption_delay_ns = 0;
            tr.ssd_wait_ns = 0;
            tr.compute_interval_count = 0;
            tr.wait_interval_count = 0;
            tr.overlap_interval_count = 0;
            for (int w = 0; w < W; ++w)
                s_breakdown_lane_issue_ts[q][w] = 0;
        }
#endif

        float* qdata = args.d_all_qdata + (long)qid * args.num_dims;

        int32_t entry_from_nav = -1;
        if (args.nav_data_dev != nullptr) {
            uint8_t* nav_base = slot_shm_base(q);
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
            float* dist_vec = slot_pq_smem(q);   // <-- in shared memory now
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
            *slot_fc(q) = 0;
            Data* ctx = slot_ctx(q);
            ctx->size = 0;
            ctx->visited_cnt = 0;
            for (int w = 0; w < W; ++w) slot_ctl(q)[w].status = IO_IDLE;
        }
        for (int i = tid; i < args.topk; i += blockDim.x) {
            slot_dist(q)[i] = INFINITY;
            slot_nns(q)[i]  = -1;
        }
        __syncthreads();

        if (tid == 0) {
#ifdef QUIVER_LIGHT_BREAKDOWN
            if (breakdown_sample_idx >= 0)
                s_breakdown_lane_issue_ts[q][0] = globaltimer_ns();
#endif
            issue_request(&slot_ctl(q)[0], entry);
        }
        __syncthreads();
        return true;
    };

    auto finalize_slot = [&](int q) {
        int qid = s_slot_qid[q];
        for (int i = tid; i < args.topk; i += blockDim.x) {
            args.d_out_nns[(long)qid * args.topk + i]       = slot_nns(q)[i];
            args.d_out_distances[(long)qid * args.topk + i] = slot_dist(q)[i];
        }
        if (tid == 0) {
            args.d_out_found_cnt[qid] = *slot_fc(q);
            int new_done = atomicAdd(args.d_queries_done, 1) + 1;
            *(volatile int32_t*)&args.comm->queries_completed = new_done;
            __threadfence_system();
            s_slot_stage[q] = -1;
#ifdef QUIVER_LIGHT_BREAKDOWN
            const int breakdown_sample_idx = quiver_breakdown_sample_index(
                qid, args.total_queries, args.breakdown_sample_count);
            if (breakdown_sample_idx >= 0 &&
                args.d_breakdown_traces != nullptr)
                args.d_breakdown_traces[breakdown_sample_idx].query_done_ns =
                    globaltimer_ns();
#endif
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

#ifdef QUIVER_LIGHT_BREAKDOWN
            const int breakdown_sample_idx = quiver_breakdown_sample_index(
                s_slot_qid[q], args.total_queries,
                args.breakdown_sample_count);
            QuiverLightBreakdownTrace* breakdown_trace =
                breakdown_sample_idx >= 0 &&
                        args.d_breakdown_traces != nullptr
                    ? &args.d_breakdown_traces[breakdown_sample_idx]
                    : nullptr;
            int64_t breakdown_compute_start_ns = 0;
            if (tid == 0 && breakdown_trace != nullptr) {
                const int64_t ready = globaltimer_ns();
                quiver_breakdown_add_ssd_wait_interval(
                    breakdown_trace, s_breakdown_lane_issue_ts[q][0], ready);
                s_breakdown_lane_issue_ts[q][0] = 0;
                breakdown_compute_start_ns = globaltimer_ns();
                breakdown_trace->resumption_delay_ns +=
                    breakdown_compute_start_ns - ready;
            }
#endif

            if (tid == 0) {
                s_snap_nd[0] = ctl[0].node_id;
                ctl[0].status = IO_IDLE;
            }
            __syncthreads();
            __threadfence_system();

            uint8_t* base = slot_shm_base(q);
            int*      mv_pos  = (int*)base;
            uint32_t* mv_id   = (uint32_t*)(mv_pos  + ef_plus_m);
            float*    mv_dist = (float*)   (mv_id   + ef_plus_m);
            uint32_t* tmp_id  = (uint32_t*)(mv_dist + ef_plus_m);
            float*    tmp_dist= (float*)   (tmp_id  + ef_plus_m);
            float* qdata = args.d_all_qdata + (long)s_slot_qid[q] * args.num_dims;

            persistent_merge_data(
                slot_buf(q), s_slot_entry[q],
                args.num_chunks, slot_pq_smem(q), args.compressed_data,
                args.nodes_per_page, args.node_size, args.data_size,
                args.max_m, args.ef_search,
                slot_nid(q), slot_ndist(q), slot_ctx(q),
                mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
                slot_node_scratch(q));

            int32_t boot_req = persistent_visit_select_dualwarp<T>(
                qdata, slot_node_scratch(q), s_slot_entry[q],
                args.num_dims, args.topk,
                slot_nns(q), slot_dist(q), slot_fc(q),
                args.nodes_per_page, args.node_size,
                args.max_m, args.ef_search,
                slot_nid(q), slot_ndist(q), slot_ctx(q));
            if (tid == 0) s_lane_req[0] = boot_req;
            __syncthreads();

            if (tid < 32) {
                for (int w = 1; w < W; w++)
                    s_lane_req[w] = persistent_select_next_no_compact(
                        args.ef_search, slot_nid(q), slot_ctx(q));
            }
            __syncthreads();

#ifdef QUIVER_LIGHT_BREAKDOWN
            if (tid == 0 && breakdown_trace != nullptr)
                quiver_breakdown_add_compute_interval(
                    breakdown_trace, breakdown_compute_start_ns,
                    globaltimer_ns());
#endif

            if (tid == 0) {
                bool any_issued = false;
                for (int w = 0; w < W; w++) {
                    if (s_lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                        if (breakdown_trace != nullptr)
                            s_breakdown_lane_issue_ts[q][w] = globaltimer_ns();
#endif
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

#ifdef QUIVER_LIGHT_BREAKDOWN
        const int breakdown_sample_idx = quiver_breakdown_sample_index(
            s_slot_qid[q], args.total_queries, args.breakdown_sample_count);
        QuiverLightBreakdownTrace* breakdown_trace =
            breakdown_sample_idx >= 0 && args.d_breakdown_traces != nullptr
                ? &args.d_breakdown_traces[breakdown_sample_idx]
                : nullptr;
        int64_t breakdown_compute_start_ns = 0;
        if (tid == 0 && breakdown_trace != nullptr) {
            const int64_t ready = globaltimer_ns();
            for (int w = 0; w < W; ++w) {
                if (s_snap_st[w] == IO_READY &&
                    s_breakdown_lane_issue_ts[q][w] > 0) {
                    quiver_breakdown_add_ssd_wait_interval(
                        breakdown_trace, s_breakdown_lane_issue_ts[q][w],
                        ready);
                    s_breakdown_lane_issue_ts[q][w] = 0;
                }
            }
            breakdown_compute_start_ns = globaltimer_ns();
            breakdown_trace->resumption_delay_ns +=
                breakdown_compute_start_ns - ready;
        }
#endif

        __threadfence_system();

        uint8_t* base = slot_shm_base(q);
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
                args.num_chunks, slot_pq_smem(q), args.compressed_data,
                args.nodes_per_page, args.node_size, args.data_size,
                args.max_m, args.ef_search,
                slot_nid(q), slot_ndist(q), slot_ctx(q),
                mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
                slot_node_scratch(q));

            int32_t lane_req = persistent_visit_select_dualwarp<T>(
                qdata, slot_node_scratch(q), s_snap_nd[w],
                args.num_dims, args.topk,
                slot_nns(q), slot_dist(q), slot_fc(q),
                args.nodes_per_page, args.node_size,
                args.max_m, args.ef_search,
                slot_nid(q), slot_ndist(q), slot_ctx(q));
            if (tid == 0) s_lane_req[w] = lane_req;
            __syncthreads();
        }

#ifdef QUIVER_LIGHT_BREAKDOWN
        if (tid == 0 && breakdown_trace != nullptr)
            quiver_breakdown_add_compute_interval(
                breakdown_trace, breakdown_compute_start_ns, globaltimer_ns());
#endif

        if (tid < 32) {
            for (int w = 0; w < W; w++) {
                if (s_snap_st[w] != IO_READY &&
                    *(volatile int32_t*)&ctl[w].status == IO_IDLE) {
                    s_lane_req[w] = persistent_select_next_no_compact(
                        args.ef_search, slot_nid(q), slot_ctx(q));
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            bool any_active = false;
            for (int w = 0; w < W; w++) {
                if (s_snap_st[w] == IO_READY && s_lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                    if (breakdown_trace != nullptr)
                        s_breakdown_lane_issue_ts[q][w] = globaltimer_ns();
#endif
                    issue_request(&ctl[w], s_lane_req[w]);
                    any_active = true;
                } else if (s_snap_st[w] != IO_READY &&
                           ctl[w].status == IO_IDLE &&
                           s_lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                    if (breakdown_trace != nullptr)
                        s_breakdown_lane_issue_ts[q][w] = globaltimer_ns();
#endif
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

    // ============================================================
    // Main scheduler loop — IDENTICAL to kernel_qi.cuh.
    // The only difference is what slot_pq points to (smem here, gmem there).
    // ============================================================
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
