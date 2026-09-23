#pragma once

// Quiver Q-interleave persistent kernel (Q slots per block).
// Extracted from quiver/kernel.cuh (Phase 1 refactor).
// Phase 3.2: parameterized on Dispatch policy.
// Phase 4b: StaticDispatch path implemented. Both dispatch policies supported.
// Phase 5: multi-lane (pipe_w > 1) support. Each slot can hold W IO lanes,
//   matching kernel_q1's pipe loop semantics. Boot fills lane 0 first, then
//   select_next fills lanes 1..W-1. Steady-state processes all ready lanes
//   per scheduler iteration.

#include "kernel_helpers.cuh"
#include "policies/dispatch.cuh"
#include "inkernel_nav.cuh"

namespace quiver {

// ================================================================
// Q-interleave persistent kernel: each block holds Q parallel query slots.
// When one slot is waiting for IO, the block processes another slot's ready
// hop.  Supports arbitrary pipe_w (W IO lanes per slot); in-flight IO count
// = num_blocks * Q * W.  W=1 keeps IO non-speculative; W>=2 adds per-query
// IO overlap (speculative prefetch) to reduce hop latency.
//
// Shared-memory layout (Q copies of the per-query working arrays):
//   shm_pool = [ slot0_mv_pos, slot0_mv_id, slot0_mv_dist,
//                slot0_tmp_id, slot0_tmp_dist,
//                slot1_mv_pos, ... ]
//
// Per-slot state is packed into parallel __shared__ arrays indexed by slot q.
//
// Dispatch policy:
//   - StreamingDispatch: each slot repeatedly binds new qids via atomicAdd +
//     feed_count spin-wait, kernel runs until total_queries reached.
//   - StaticDispatch: each slot binds ONE qid statically
//     (qid = batch_offset + bid * Q + q), runs it to completion, then retires.
//     Kernel returns when all Q slots have retired. Host must launch
//     ceil(total / (num_blocks * Q)) batches.
// ================================================================

constexpr int QI_MAX_Q = 8;  // compile-time upper bound; runtime Q <= QI_MAX_Q

template <class Dispatch, class T>
__global__ void persistent_search_kernel_qi(PersistentKernelArgs args)
{
    extern __shared__ uint8_t shm_pool[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
#ifdef QUIVER_FIXED_PIPE_WIDTH
    constexpr int W = QUIVER_FIXED_PIPE_WIDTH;
#else
    const int W = args.pipe_w;
#endif
#ifdef QUIVER_FIXED_QUERIES_PER_BLOCK
    constexpr int Q = QUIVER_FIXED_QUERIES_PER_BLOCK;
#else
    const int Q = args.queries_per_block;
#endif
    const int ef_plus_m = args.ef_search + args.max_m;
    const int frontier_bytes = (sizeof(int) * 3 + sizeof(float) * 2) * ef_plus_m;
    const int node_scratch_offset = (frontier_bytes + 7) & ~7;
    const int stride_bytes = node_scratch_offset + args.node_size;
    // Cache total_queries into a local const so every control-flow read uses a
    // register rather than hitting the args struct through PCIe constant path.
    const int k_total_queries = args.total_queries;

    // Per-slot shared-memory pointers (Q copies laid out contiguously)
    // For slot q, the base is shm_pool + q * stride_bytes.
    // We keep a single shared handle and index by q when needed.

    __shared__ int32_t s_slot_qid[QI_MAX_Q];
    __shared__ int32_t s_slot_entry[QI_MAX_Q];
    // Slot stage encoding:
    //   -1 : EMPTY/DONE (needs new query or kernel exit)
    //    0 : BOOT_WAIT  (issued IO for entry node, waiting for IO_READY)
    //    1 : RUNNING    (at least one hop completed; in pipe_loop)
    __shared__ int32_t s_slot_stage[QI_MAX_Q];
    __shared__ int32_t s_slot_next[QI_MAX_Q];  // next node id returned by visit/select (-1 = converged)
    __shared__ int32_t s_active_mask;
    __shared__ int     s_pick;  // slot index chosen this iteration (-1 if none)
    __shared__ int32_t s_kernel_exit;
    // Multi-lane pipe state (reused across Q slots; only one slot processed
    // at a time by the scheduler).
    __shared__ int32_t s_snap_st[16];   // per-lane status snapshot
    __shared__ int32_t s_snap_nd[16];   // per-lane node_id latch
    __shared__ int32_t s_lane_req[16];  // per-lane next IO request
    __shared__ int32_t s_slot_step[QI_MAX_Q];
    __shared__ int32_t s_slot_gpruning_no_improve_count[QI_MAX_Q];
    __shared__ int32_t s_slot_last_found_cnt[QI_MAX_Q];
    __shared__ float   s_slot_last_top1_dist[QI_MAX_Q];
    __shared__ float   s_slot_last_topk_dist[QI_MAX_Q];
    __shared__ int32_t s_drain_done;

#ifdef QUIVER_LATENCY_PROBE
    __shared__ int32_t s_probe_hop_count[QI_MAX_Q];
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
    __shared__ int64_t s_breakdown_lane_issue_ts[QI_MAX_Q][16];
    __shared__ int64_t s_breakdown_wait_start_ns;
    __shared__ int64_t s_breakdown_pure_wait_ns;
    __shared__ int32_t s_breakdown_window_idx;
    __shared__ int32_t s_breakdown_window_trigger_qid;
    __shared__ int32_t s_breakdown_window_waiting;
    __shared__ int64_t s_breakdown_window_start_ns;
    __shared__ int64_t s_breakdown_window_deadline_ns;
#endif

    if (tid == 0) {
        s_active_mask = 0;
        s_kernel_exit = 0;
        for (int q = 0; q < Q; ++q) {
            s_slot_qid[q]   = -1;
            s_slot_entry[q] = -1;
            s_slot_stage[q] = -1;  // EMPTY
            s_slot_next[q]  = -1;
            s_slot_step[q]  = 0;
            s_slot_gpruning_no_improve_count[q] = 0;
            s_slot_last_found_cnt[q] = 0;
            s_slot_last_top1_dist[q] = INFINITY;
            s_slot_last_topk_dist[q] = INFINITY;
        }
#ifdef QUIVER_LATENCY_PROBE
        for (int q = 0; q < Q; ++q) s_probe_hop_count[q] = 0;
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
        for (int q = 0; q < Q; ++q)
            for (int w = 0; w < 16; ++w)
                s_breakdown_lane_issue_ts[q][w] = 0;
        s_breakdown_wait_start_ns = 0;
        s_breakdown_pure_wait_ns = 0;
        s_breakdown_window_idx = -1;
        s_breakdown_window_trigger_qid = -1;
        s_breakdown_window_waiting = 0;
        s_breakdown_window_start_ns = 0;
        s_breakdown_window_deadline_ns = 0;
#endif
    }
    __syncthreads();

    const int aef = args.aligned_ef;

    auto slot_shm_base = [&](int q) -> uint8_t* {
        return shm_pool + (size_t)q * stride_bytes;
    };
    auto slot_node_scratch = [&](int q) -> uint8_t* {
        return slot_shm_base(q) + node_scratch_offset;
    };

    // Per-slot global state accessors
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

#ifdef QUIVER_LATENCY_PROBE
    int64_t probe_block_start = 0;
    int64_t probe_no_ready_wait_start = 0;
    int64_t probe_no_ready_wait_ns = 0;
    int32_t probe_no_ready_waiting = 0;
    int32_t probe_no_ready_spins = 0;
    int32_t probe_query_count = 0;
    const int32_t probe_block_trace_idx = args.probe_block_trace_offset + bid;
    if (tid == 0) probe_block_start = globaltimer_ns();
#endif

    // ============================================================
    // Helper: grab new query for an empty slot; issue boot IO.
    // Returns true if slot became active; false if no more queries (slot stays empty).
    //
    // Dispatch policy branches:
    //   - StreamingDispatch: atomicAdd(d_next_query) + spin-wait feed_count.
    //   - StaticDispatch: qid = batch_offset + bid * Q + q (one-shot per slot
    //     per kernel launch).
    // ============================================================
    auto try_bind_new_query = [&](int q) -> bool {
#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_sched_start = 0;
        if (tid == 0) probe_t_sched_start = globaltimer_ns();
#endif
        int qid;
        if constexpr (Dispatch::is_persistent) {
            // StreamingDispatch: dynamic admission
            if (tid == 0) {
                qid = atomicAdd(args.d_next_query, 1);
                s_slot_qid[q] = qid;
            }
            __syncthreads();
            qid = s_slot_qid[q];

            // Wait until CPU feeds this qid, or detect termination.
            // Use args.total_queries (kernel-arg constant, in constant memory)
            // to avoid PCIe reads through comm that occasionally race with
            // other threads and leave the loop in an inconsistent state.
            const int total = k_total_queries;
            if (tid == 0) {
                int got = 0;
                while (qid < total) {
                    int fc = *(volatile int32_t*)&args.comm->feed_count;
                    if (qid < fc) { got = 1; break; }
                    if (*(volatile int32_t*)&args.comm->terminate) { got = 0; break; }
                }
                if (qid >= total) got = 0;
                // Cross-stream acquire fence: pairs with host-side
                // cudaMemcpyAsync(h2d_stream) + feed_count bump.  Without
                // this, subsequent reads of d_all_qdata[qid] /
                // d_entry_nodes[qid] in this block may hit stale L2 lines
                // (host DMA landed on a different stream, no implicit
                // dependency to this kernel's stream).  See detailed
                // comment in dispatch.cuh::StreamingDispatch.
                if (got) __threadfence_system();
                s_slot_stage[q] = got ? 0 : -2;  // 0 = BOOT_WAIT; -2 = exit signal
            }
            __syncthreads();

            if (s_slot_stage[q] == -2) {
                // Kernel should start shutting down this slot (no more queries).
                __syncthreads();
                return false;
            }
        } else {
            // StaticDispatch: static one-shot binding per (bid, slot_q).
            if (tid == 0) {
                int q_static = StaticDispatch::static_qid_for_slot(args, bid, q);
                s_slot_qid[q] = q_static;
                s_slot_stage[q] = (q_static >= 0) ? 0 : -2;  // 0 = BOOT_WAIT
            }
            __syncthreads();
            qid = s_slot_qid[q];

            if (qid < 0) {
                // Out-of-range qid (final batch has fewer queries than
                // num_blocks * Q): this slot has no work this launch.
                if (tid == 0) {
                    s_slot_stage[q] = -2;
                }
                __syncthreads();
                return false;
            }
        }

        // Initialize per-slot state for qid
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
            if (s_breakdown_window_idx < 0 &&
                breakdown_sample_idx < args.breakdown_cta_window_count &&
                args.breakdown_cta_window_ns > 0) {
                const int64_t now = tr.query_start_ns;
                s_breakdown_window_idx = breakdown_sample_idx;
                s_breakdown_window_trigger_qid = qid;
                s_breakdown_window_start_ns = now;
                s_breakdown_window_deadline_ns = now + args.breakdown_cta_window_ns;
                s_breakdown_pure_wait_ns = 0;
                s_breakdown_wait_start_ns = 0;
                s_breakdown_window_waiting = 0;
            }
        }
#endif
#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_query_start = 0;
        if (tid == 0) probe_t_query_start = globaltimer_ns();
#endif
        float* qdata = args.d_all_qdata + (long)qid * args.num_dims;

        // ---- In-kernel nav (must run BEFORE PQ init_query, which mutates
        // qdata in place by subtracting the centroid).  Reuses the per-slot
        // shared scratch mv_id / mv_dist as the nav priority queue buffer
        // (L + R = 32 entries, fits in 256 B, well under our per-slot stride).
        //
        // Block-level: ALL 128 threads enter; the 4 warps split the R=28
        // neighbor-distance computations.  s_nav_idx does NOT need to be
        // a per-slot array because try_bind_new_query is
        // called sequentially over the Q slots (only one slot is binding
        // at any given moment within a block).
        int32_t entry_from_nav = -1;
        if (args.nav_data_dev != nullptr) {
            uint8_t* nav_base = slot_shm_base(q);
            int*      nav_mv_pos  = (int*)nav_base;
            uint32_t* nav_pq_id   = (uint32_t*)(nav_mv_pos + ef_plus_m);
            float*    nav_pq_dist = (float*)   (nav_pq_id  + ef_plus_m);
            __shared__ int s_nav_idx;     // cross-warp broadcast of next-pivot slot
            int mapped = inkernel_nav_entry<T>(
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
            entry_from_nav = mapped;
        }

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_nav_done = 0;
        if (tid == 0) probe_t_nav_done = globaltimer_ns();
#endif

        // Inlined PQSearchData::init_query but targeting slot_pq(q) explicitly,
        // since the library helper indexes pq_dists by blockIdx.x which doesn't
        // match Q-interleave per-slot layout (bid * Q + q).
        {
            shared::PQSearchData* pqd = args.pq_data;
            float* dist_vec = slot_pq(q);
            const int num_pivots = PQSearchData::num_pivots;
            const int num_chunks = pqd->num_chunks;
            const int dim = pqd->dim;
            for (size_t i = tid; i < (size_t)num_pivots * num_chunks; i += blockDim.x) {
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
            for (size_t i = tid; i < (size_t)dim; i += blockDim.x) {
                qdata[i] += pqd->centroid[i];
            }
        }
        __syncthreads();

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_pq_done = 0;
        if (tid == 0) probe_t_pq_done = globaltimer_ns();
#endif

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
            s_slot_step[q] = 0;
            s_slot_gpruning_no_improve_count[q] = 0;
            s_slot_last_found_cnt[q] = 0;
            s_slot_last_top1_dist[q] = INFINITY;
            s_slot_last_topk_dist[q] = INFINITY;
            // Reset IOControl status for all W lanes of this slot
            for (int w = 0; w < W; ++w) slot_ctl(q)[w].status = IO_IDLE;
        }
        for (int i = tid; i < args.topk; i += blockDim.x) {
            slot_dist(q)[i] = INFINITY;
            slot_nns(q)[i]  = -1;
        }
        __syncthreads();

        // Issue boot IO on lane 0.
        if (tid == 0) {
#ifdef QUIVER_LIGHT_BREAKDOWN
            if (breakdown_sample_idx >= 0)
                s_breakdown_lane_issue_ts[q][0] = globaltimer_ns();
#endif
            issue_request(&slot_ctl(q)[0], entry);
        }
        __syncthreads();

#ifdef QUIVER_LATENCY_PROBE
        {
            int64_t probe_t_boot_issue = 0;
            if (tid == 0) probe_t_boot_issue = globaltimer_ns();
            int qid_bound = s_slot_qid[q];
            if (qid_bound >= 0 && qid_bound < args.probe_max_traces && tid == 0) {
                auto* tr = &args.d_probe_traces[qid_bound];
                tr->query_id = qid_bound;
                tr->t_sched_start  = probe_t_sched_start;
                tr->t_query_start  = probe_t_query_start;
                tr->t_nav_done     = probe_t_nav_done;
                tr->t_pq_done      = probe_t_pq_done;
                tr->t_boot_issue   = probe_t_boot_issue;
                tr->boot_spin_iters = 0;
            }
            if (tid == 0) s_probe_hop_count[q] = 0;
        }
#endif
        if (tid == 0) {
#ifdef QUIVER_LATENCY_PROBE
            probe_query_count++;
#endif
        }
        return true;
    };

    // ============================================================
    // Helper: finalize a converged slot: write results to global, mark done.
    //
    // Dispatch policy note:
    //   - StreamingDispatch: slot returns to EMPTY (-1), ready to bind new qid.
    //   - StaticDispatch: slot goes to RETIRED (-3), never picks up a new qid
    //     within this kernel launch (next qid for this (bid, q) will only
    //     come from a future launch after host bumps batch_offset).
    // ============================================================
    auto finalize_slot = [&](int q) {
        int qid = s_slot_qid[q];
#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_pipe_done = 0;
        if (tid == 0) probe_t_pipe_done = globaltimer_ns();
#endif
        for (int i = tid; i < args.topk; i += blockDim.x) {
            args.d_out_nns[(long)qid * args.topk + i]       = slot_nns(q)[i];
            args.d_out_distances[(long)qid * args.topk + i] = slot_dist(q)[i];
        }
        if (tid == 0) {
            if (args.d_finish_step != nullptr)
                args.d_finish_step[qid] = s_slot_step[q];
            args.d_out_found_cnt[qid] = *slot_fc(q);
        }
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            int new_done = atomicAdd(args.d_queries_done, 1) + 1;
            *(volatile int32_t*)&args.comm->queries_completed = new_done;
            __threadfence_system();
            if constexpr (Dispatch::is_persistent) {
                s_slot_stage[q] = -1;  // EMPTY: ready to bind new qid
            } else {
                s_slot_stage[q] = -3;  // RETIRED: one-shot in StaticDispatch
            }
        }
        __syncthreads();

#ifdef QUIVER_LIGHT_BREAKDOWN
        if (tid == 0 && args.d_breakdown_traces != nullptr) {
            const int idx = quiver_breakdown_sample_index(
                qid, args.total_queries, args.breakdown_sample_count);
            if (idx >= 0)
                args.d_breakdown_traces[idx].query_done_ns = globaltimer_ns();
        }
#endif

#ifdef QUIVER_LATENCY_PROBE
        {
            int64_t probe_t_query_done = 0;
            if (tid == 0) probe_t_query_done = globaltimer_ns();
            if (qid >= 0 && qid < args.probe_max_traces && tid == 0) {
                auto* tr = &args.d_probe_traces[qid];
                tr->t_pipe_done  = probe_t_pipe_done;
                tr->t_query_done = probe_t_query_done;
                tr->num_hops     = s_probe_hop_count[q];
            }
        }
#endif
    };

    // ============================================================
    // Helper: for a slot at BOOT_WAIT or RUNNING, process ready hops
    // across all W lanes.  Returns true if work was done.
    //
    // BOOT_WAIT (stage 0): only lane 0 has pending IO.  After processing
    //   the boot data, fill lanes 1..W-1 via select_next, then issue all.
    // RUNNING (stage 1): scan all W lanes, process all IO_READY lanes
    //   serially (merge+visit), select_next for idle lanes, issue all.
    //   Mirrors kernel_q1's pipe loop semantics.
    // ============================================================
    auto try_process_slot_hop = [&](int q) -> bool {
        IOControl* ctl = slot_ctl(q);

        // ---- BOOT_WAIT (stage 0): only lane 0 has pending IO ----
        if (s_slot_stage[q] == 0) {
            if (tid == 0) {
                int32_t st = *(volatile int32_t*)&ctl[0].status;
                s_pick = (st == IO_READY) ? q : -1;
            }
            __syncthreads();
            if (s_pick != q) return false;

#ifdef QUIVER_LIGHT_BREAKDOWN
            const int breakdown_qid = s_slot_qid[q];
            const int breakdown_idx = quiver_breakdown_sample_index(
                breakdown_qid, args.total_queries, args.breakdown_sample_count);
            QuiverLightBreakdownTrace* breakdown_trace =
                breakdown_idx >= 0 && args.d_breakdown_traces != nullptr
                    ? &args.d_breakdown_traces[breakdown_idx]
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

#ifdef QUIVER_LATENCY_PROBE
            int64_t probe_t_boot_ready = 0;
            if (tid == 0) probe_t_boot_ready = globaltimer_ns();
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

#ifdef QUIVER_LATENCY_PROBE
            int64_t probe_t_boot_pq_done = 0;
#endif
            persistent_merge_data(
                slot_buf(q), s_slot_entry[q],
                args.num_chunks, slot_pq(q), args.compressed_data,
                args.nodes_per_page, args.node_size, args.data_size,
                args.max_m, args.ef_search,
                slot_nid(q), slot_ndist(q), slot_ctx(q),
                mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
                slot_node_scratch(q)
#ifdef QUIVER_LATENCY_PROBE
                , &probe_t_boot_pq_done
#endif
                );

#ifdef QUIVER_LATENCY_PROBE
            int64_t probe_t_boot_merge_done = 0;
            if (tid == 0) probe_t_boot_merge_done = globaltimer_ns();
#endif

            int32_t boot_req = persistent_visit_select_dualwarp<T>(
                qdata, slot_node_scratch(q), s_slot_entry[q],
                args.num_dims, args.topk,
                slot_nns(q), slot_dist(q), slot_fc(q),
                args.nodes_per_page, args.node_size,
                args.max_m, args.ef_search,
                slot_nid(q), slot_ndist(q), slot_ctx(q));
            if (tid == 0) s_lane_req[0] = boot_req;
            __syncthreads();

#ifdef QUIVER_LIGHT_BREAKDOWN
            if (tid == 0 && breakdown_trace != nullptr)
                quiver_breakdown_add_compute_interval(
                    breakdown_trace, breakdown_compute_start_ns,
                    globaltimer_ns());
#endif

            if (tid == 0) {
                ++s_slot_step[q];
                const bool topk_improved =
                    *slot_fc(q) > s_slot_last_found_cnt[q] ||
                    slot_dist(q)[0] < s_slot_last_top1_dist[q] ||
                    slot_dist(q)[args.topk - 1] < s_slot_last_topk_dist[q];
                if (s_slot_step[q] < args.early_exit_gpruning_warmup_steps) {
                    s_slot_gpruning_no_improve_count[q] = 0;
                } else {
                    s_slot_gpruning_no_improve_count[q] =
                        topk_improved ? 0 : s_slot_gpruning_no_improve_count[q] + 1;
                }
                s_slot_last_found_cnt[q] = *slot_fc(q);
                s_slot_last_top1_dist[q] = slot_dist(q)[0];
                s_slot_last_topk_dist[q] = slot_dist(q)[args.topk - 1];
                if (args.early_exit_policy == static_cast<int>(EarlyExitPolicy::kGPruning) &&
                    s_slot_step[q] >= args.early_exit_gpruning_warmup_steps &&
                    args.early_exit_gpruning_patience > 0 &&
                    s_slot_gpruning_no_improve_count[q] >= args.early_exit_gpruning_patience) {
                    if (args.d_early_exit_stop_step != nullptr)
                        args.d_early_exit_stop_step[s_slot_qid[q]] = s_slot_step[q];
                    s_slot_stage[q] = 2;
                }
            }
            __syncthreads();
            if (s_slot_stage[q] == 2) {
                drain_io_lanes_after_stop(ctl, W, &s_drain_done);
                return true;
            }

            // Fill remaining lanes 1..W-1 via select_next
            if (tid < 32) {
                for (int w = 1; w < W; w++)
                    s_lane_req[w] = persistent_select_next_no_compact(
                        args.ef_search, slot_nid(q), slot_ctx(q));
            }
            __syncthreads();

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

#ifdef QUIVER_LATENCY_PROBE
            {
                int64_t probe_t_boot_all_issued = 0;
                if (tid == 0) probe_t_boot_all_issued = globaltimer_ns();
                int qid_boot = s_slot_qid[q];
                if (qid_boot >= 0 && qid_boot < args.probe_max_traces && tid == 0) {
                    auto* tr = &args.d_probe_traces[qid_boot];
                    tr->t_boot_ready      = probe_t_boot_ready;
                    tr->t_boot_pq_done    = probe_t_boot_pq_done;
                    tr->t_boot_merge_done = probe_t_boot_merge_done;
                    tr->t_boot_compute_done = probe_t_boot_all_issued;
                    tr->t_boot_all_issued = probe_t_boot_all_issued;
                    tr->t_pipe_start      = probe_t_boot_all_issued;
                }
            }
#endif
            return true;
        }

        // ---- RUNNING (stage 1): scan all W lanes ----
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
                s_pick = -2;  // all lanes idle: converged
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
        const int breakdown_qid = s_slot_qid[q];
        const int breakdown_idx = quiver_breakdown_sample_index(
            breakdown_qid, args.total_queries, args.breakdown_sample_count);
        QuiverLightBreakdownTrace* breakdown_trace =
            breakdown_idx >= 0 && args.d_breakdown_traces != nullptr
                ? &args.d_breakdown_traces[breakdown_idx]
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

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_ready_found = 0;
        if (tid == 0) probe_t_ready_found = globaltimer_ns();
#endif

        __threadfence_system();

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_compute_start = 0;
        if (tid == 0) probe_t_compute_start = globaltimer_ns();
#endif

        uint8_t* base = slot_shm_base(q);
        int*      mv_pos  = (int*)base;
        uint32_t* mv_id   = (uint32_t*)(mv_pos  + ef_plus_m);
        float*    mv_dist = (float*)   (mv_id   + ef_plus_m);
        uint32_t* tmp_id  = (uint32_t*)(mv_dist + ef_plus_m);
        float*    tmp_dist= (float*)   (tmp_id  + ef_plus_m);
        float* qdata = args.d_all_qdata + (long)s_slot_qid[q] * args.num_dims;

        // Process ALL ready lanes serially (merge_data + visit_select
        // modify shared neighbor arrays, so lanes cannot be parallel).
        for (int w = 0; w < W; w++) {
            if (s_snap_st[w] != IO_READY) continue;
            if (tid == 0) ctl[w].status = IO_IDLE;
#ifdef QUIVER_LIGHT_BREAKDOWN
            int64_t breakdown_lane_compute_start_ns = 0;
            if (tid == 0 && breakdown_trace != nullptr)
                breakdown_lane_compute_start_ns = globaltimer_ns();
#endif

            persistent_merge_data(
                slot_buf(q) + w * shared::PAGE_SIZE, s_snap_nd[w],
                args.num_chunks, slot_pq(q), args.compressed_data,
                args.nodes_per_page, args.node_size, args.data_size,
                args.max_m, args.ef_search,
                slot_nid(q), slot_ndist(q), slot_ctx(q),
                mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
                slot_node_scratch(q)
#ifdef QUIVER_LATENCY_PROBE
                , nullptr
#endif
                );

            int32_t lane_req = persistent_visit_select_dualwarp<T>(
                qdata, slot_node_scratch(q), s_snap_nd[w],
                args.num_dims, args.topk,
                slot_nns(q), slot_dist(q), slot_fc(q),
                args.nodes_per_page, args.node_size,
                args.max_m, args.ef_search,
                slot_nid(q), slot_ndist(q), slot_ctx(q));
            if (tid == 0) s_lane_req[w] = lane_req;
            __syncthreads();

#ifdef QUIVER_LIGHT_BREAKDOWN
            if (tid == 0 && breakdown_trace != nullptr)
                quiver_breakdown_add_compute_interval(
                    breakdown_trace, breakdown_lane_compute_start_ns,
                    globaltimer_ns());
#endif

            if (tid == 0) {
                ++s_slot_step[q];
                const bool topk_improved =
                    *slot_fc(q) > s_slot_last_found_cnt[q] ||
                    slot_dist(q)[0] < s_slot_last_top1_dist[q] ||
                    slot_dist(q)[args.topk - 1] < s_slot_last_topk_dist[q];
                if (s_slot_step[q] < args.early_exit_gpruning_warmup_steps) {
                    s_slot_gpruning_no_improve_count[q] = 0;
                } else {
                    s_slot_gpruning_no_improve_count[q] =
                        topk_improved ? 0 : s_slot_gpruning_no_improve_count[q] + 1;
                }
                s_slot_last_found_cnt[q] = *slot_fc(q);
                s_slot_last_top1_dist[q] = slot_dist(q)[0];
                s_slot_last_topk_dist[q] = slot_dist(q)[args.topk - 1];
                if (args.early_exit_policy == static_cast<int>(EarlyExitPolicy::kGPruning) &&
                    s_slot_step[q] >= args.early_exit_gpruning_warmup_steps &&
                    args.early_exit_gpruning_patience > 0 &&
                    s_slot_gpruning_no_improve_count[q] >= args.early_exit_gpruning_patience) {
                    if (args.d_early_exit_stop_step != nullptr)
                        args.d_early_exit_stop_step[s_slot_qid[q]] = s_slot_step[q];
                    s_slot_stage[q] = 2;
                }
            }
            __syncthreads();
            if (s_slot_stage[q] == 2) break;

        }

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_compute_done = 0;
        if (tid == 0) probe_t_compute_done = globaltimer_ns();
#endif

        if (s_slot_stage[q] == 2) {
            drain_io_lanes_after_stop(ctl, W, &s_drain_done);
            return true;
        }

        // select_next for IDLE lanes that were not ready this iteration
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

        // Issue all new IO requests; check convergence
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

#ifdef QUIVER_LATENCY_PROBE
        {
            int qid_hop = s_slot_qid[q];
            if (qid_hop >= 0 && qid_hop < args.probe_max_traces && tid == 0) {
                int hc = s_probe_hop_count[q];
                if (hc < PROBE_MAX_HOPS) {
                    auto& h = args.d_probe_traces[qid_hop].hops[hc];
                    h.ready_found_ts     = probe_t_ready_found;
                    h.compute_start_ts   = probe_t_compute_start;
                    h.compute_done_ts    = probe_t_compute_done;
                    h.ready_lanes        = 0;
                    h.spin_iters         = 0;
                    for (int w = 0; w < W; w++)
                        if (s_snap_st[w] == IO_READY) h.ready_lanes++;
                }
                s_probe_hop_count[q] = hc + 1;
            }
        }
#endif
        return true;
    };

    // ============================================================
    // MAIN SCHEDULER LOOP
    // Round-robin over Q slots; prefer those with IO_READY.
    // ============================================================
    int last_slot = -1;
#ifdef QUIVER_LATENCY_PROBE
    auto probe_end_no_ready_wait = [&](int64_t now) {
        if (tid == 0 && probe_no_ready_waiting) {
            if (now > probe_no_ready_wait_start) {
                probe_no_ready_wait_ns += now - probe_no_ready_wait_start;
            }
            probe_no_ready_waiting = 0;
            probe_no_ready_wait_start = 0;
        }
    };

    auto probe_write_block_trace = [&]() {
        if (tid == 0 && args.d_probe_block_traces != nullptr &&
            probe_block_trace_idx >= 0 &&
            probe_block_trace_idx < args.probe_max_block_traces) {
            auto& tr = args.d_probe_block_traces[probe_block_trace_idx];
            tr.block_id = probe_block_trace_idx;
            tr.query_count = probe_query_count;
            tr.t_block_start = probe_block_start;
            tr.t_block_done = globaltimer_ns();
            tr.no_ready_wait_ns = probe_no_ready_wait_ns;
            tr.no_ready_spins = probe_no_ready_spins;
        }
    };
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
    auto breakdown_cta_finish_window = [&](int64_t now) {
        if (tid != 0 || s_breakdown_window_idx < 0) return;
        const int idx = s_breakdown_window_idx;
        if (idx < 0 || idx >= args.breakdown_cta_window_count ||
            args.d_breakdown_cta_windows == nullptr) {
            s_breakdown_window_idx = -1;
            return;
        }
        int64_t end_ns = now;
        if (s_breakdown_window_deadline_ns > 0 &&
            end_ns > s_breakdown_window_deadline_ns) {
            end_ns = s_breakdown_window_deadline_ns;
        }
        if (s_breakdown_window_waiting) {
            if (end_ns > s_breakdown_wait_start_ns) {
                s_breakdown_pure_wait_ns += end_ns - s_breakdown_wait_start_ns;
            }
            s_breakdown_window_waiting = 0;
            s_breakdown_wait_start_ns = 0;
        }
        if (end_ns > s_breakdown_window_start_ns) {
            auto& window = args.d_breakdown_cta_windows[idx];
            window.valid = 1;
            window.block_id = bid;
            window.trigger_query_id = s_breakdown_window_trigger_qid;
            window.query_count_delta = 0;
            window.window_start_ns = s_breakdown_window_start_ns;
            window.window_end_ns = end_ns;
            window.pure_io_wait_ns = s_breakdown_pure_wait_ns;
        }
        s_breakdown_window_idx = -1;
        s_breakdown_window_trigger_qid = -1;
        s_breakdown_window_start_ns = 0;
        s_breakdown_window_deadline_ns = 0;
        s_breakdown_pure_wait_ns = 0;
        s_breakdown_wait_start_ns = 0;
        s_breakdown_window_waiting = 0;
    };

    auto breakdown_cta_end_wait = [&](int64_t now) {
        if (tid == 0 && s_breakdown_window_idx >= 0 &&
            s_breakdown_window_waiting) {
            int64_t wait_end = now;
            if (s_breakdown_window_deadline_ns > 0 &&
                wait_end > s_breakdown_window_deadline_ns) {
                wait_end = s_breakdown_window_deadline_ns;
            }
            if (wait_end > s_breakdown_wait_start_ns) {
                s_breakdown_pure_wait_ns += wait_end - s_breakdown_wait_start_ns;
            }
            s_breakdown_window_waiting = 0;
            s_breakdown_wait_start_ns = 0;
        }
    };

    auto breakdown_cta_start_wait_if_needed = [&](bool still_active) {
        if (tid == 0 && s_breakdown_window_idx >= 0 && still_active) {
            const int64_t now = globaltimer_ns();
            if (s_breakdown_window_deadline_ns > 0 &&
                now >= s_breakdown_window_deadline_ns) {
                breakdown_cta_finish_window(now);
            } else if (!s_breakdown_window_waiting) {
                s_breakdown_wait_start_ns = now;
                s_breakdown_window_waiting = 1;
            }
        }
    };

    auto breakdown_cta_tick_window = [&]() {
        if (tid == 0 && s_breakdown_window_idx >= 0) {
            const int64_t now = globaltimer_ns();
            if (s_breakdown_window_deadline_ns > 0 &&
                now >= s_breakdown_window_deadline_ns) {
                breakdown_cta_finish_window(now);
            }
        }
    };
#endif

    while (true) {
        // Ensure all threads see a consistent view of s_slot_stage[] before
        // branching on it (the subsequent if-guarded paths contain their own
        // __syncthreads(); divergent reads of stage would deadlock those).
        __syncthreads();
#ifdef QUIVER_LIGHT_BREAKDOWN
        breakdown_cta_tick_window();
#endif

        // 1. Try to bring empty slots online (bind new query + issue boot IO).
        for (int q = 0; q < Q; ++q) {
            if (s_slot_stage[q] == -1) {  // EMPTY
                if constexpr (Dispatch::is_persistent) {
                    if (tid == 0) {
                        // Check if we should exit entirely
                        int qid_preview = *(volatile int32_t*)args.d_next_query;
                        if (qid_preview >= k_total_queries &&
                            *(volatile int32_t*)&args.comm->terminate) {
                            s_kernel_exit = 1;
                        }
                    }
                    __syncthreads();
                    if (s_kernel_exit) break;
                }

                bool ok = try_bind_new_query(q);
                if (!ok) {
                    // No more queries available for this slot.  Mark as
                    // permanently retired (stage = -3) so we don't try again.
                    if (tid == 0) s_slot_stage[q] = -3;
                    __syncthreads();
                } else {
#ifdef QUIVER_LATENCY_PROBE
                    probe_end_no_ready_wait(globaltimer_ns());
#endif
                }
            }
        }

        // 2. If no active slots, terminate.
        // stages >= 0 are active; stages -1/-2/-3 are not.
        // For StaticDispatch: -3 (retired) is a terminal state per launch;
        //   -1 will be set to -3 immediately by try_bind in next iteration
        //   (since static_qid returns same qid, but caller sees "bound once").
        //   In practice finalize_slot directly sets -3 for StaticDispatch.
        int any_active = 0;
        if (tid == 0) {
            for (int q = 0; q < Q; ++q) {
                if (s_slot_stage[q] >= 0) { any_active = 1; break; }
            }
            s_active_mask = any_active;
        }
        __syncthreads();
        if (!s_active_mask) {
            if constexpr (Dispatch::is_persistent) {
                // StreamingDispatch: exit only if terminate signal or
                // d_next_query drained. Otherwise spin for more queries.
                if (tid == 0) {
                    if (*(volatile int32_t*)&args.comm->terminate ||
                        *(volatile int32_t*)args.d_next_query >= k_total_queries) {
                        s_kernel_exit = 1;
                    }
                }
                __syncthreads();
                if (s_kernel_exit) {
#ifdef QUIVER_LATENCY_PROBE
                    probe_write_block_trace();
#endif
                    return;
                }
                continue;  // spin briefly, wait for more queries
            } else {
                // StaticDispatch: no empty slots means all Q slots have
                // retired (-3) or were out-of-range (-2 -> -3).
                // One-shot kernel launch is done.
#ifdef QUIVER_LATENCY_PROBE
                probe_write_block_trace();
#endif
                return;
            }
        }

        // 3. Scan Q slots for IO_READY; process the first ready one found.
        //    Start scan from (last_slot+1) % Q to round-robin.
        bool made_progress = false;
        for (int probe = 0; probe < Q; ++probe) {
            int q = (last_slot + 1 + probe) % Q;
            if (s_slot_stage[q] < 0 || s_slot_stage[q] == 2) continue;
#ifdef QUIVER_LATENCY_PROBE
            int64_t probe_try_start = 0;
            if (tid == 0) probe_try_start = globaltimer_ns();
#endif
            if (try_process_slot_hop(q)) {
#ifdef QUIVER_LATENCY_PROBE
                probe_end_no_ready_wait(probe_try_start);
#endif
                last_slot = q;
                made_progress = true;
                // If slot converged during processing, finalize
                if (s_slot_stage[q] == 2) {
                    finalize_slot(q);
                }
                break;  // restart scheduling from next slot
            }
        }

        if (!made_progress) {
            // No slot had IO_READY; also check for converged slots to finalize.
            for (int q = 0; q < Q; ++q) {
                if (s_slot_stage[q] == 2) {
                    finalize_slot(q);
                }
            }
#ifdef QUIVER_LIGHT_BREAKDOWN
            bool still_active = false;
            if (tid == 0) {
                for (int q = 0; q < Q; ++q) {
                    if (s_slot_stage[q] >= 0 && s_slot_stage[q] != 2) {
                        still_active = true;
                        break;
                    }
                }
            }
            breakdown_cta_start_wait_if_needed(still_active);
#endif
#ifdef QUIVER_LATENCY_PROBE
            if (tid == 0) {
                bool probe_active = false;
                for (int q = 0; q < Q; ++q) {
                    if (s_slot_stage[q] >= 0 && s_slot_stage[q] != 2) {
                        probe_active = true;
                        break;
                    }
                }
                if (probe_active && !probe_no_ready_waiting) {
                    probe_no_ready_wait_start = globaltimer_ns();
                    probe_no_ready_waiting = 1;
                }
                if (probe_active) probe_no_ready_spins++;
            }
#endif
            // Spin briefly: just continue the outer while loop.
        }
#ifdef QUIVER_LIGHT_BREAKDOWN
        else {
            breakdown_cta_end_wait(globaltimer_ns());
        }
#endif
    } // scheduler loop
}

} // namespace quiver
