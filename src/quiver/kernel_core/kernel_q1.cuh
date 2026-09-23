#pragma once

// Quiver Q=1 fast-path persistent kernel.
// Extracted from quiver/kernel.cuh (Phase 1 refactor).
// Phase 3.1a: Dispatch is now a template policy parameter.
// Behavior is unchanged for StreamingDispatch (the only instantiation used today).

#include "kernel_helpers.cuh"
#include "policies/dispatch.cuh"
#include "inkernel_nav.cuh"

namespace quiver {

// ================================================================
// Persistent kernel: one launch per search batch.
// Each block loops over queries sequentially.
// Within each query, ALL ready lanes are processed per iteration
// to avoid lane starvation with fast backends.
// ================================================================
template <class Dispatch, class T>
__global__ void persistent_search_kernel(PersistentKernelArgs args)
{
    extern __shared__ uint8_t shm_pool[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
#ifdef QUIVER_FIXED_PIPE_WIDTH
    constexpr int W = QUIVER_FIXED_PIPE_WIDTH;
#else
    const int W = args.pipe_w;
#endif

    int*      mv_pos  = (int*)shm_pool;
    uint32_t* mv_id   = (uint32_t*)(mv_pos  + args.ef_search + args.max_m);
    float*    mv_dist = (float*)   (mv_id   + args.ef_search + args.max_m);
    uint32_t* tmp_id  = (uint32_t*)(mv_dist + args.ef_search + args.max_m);
    float*    tmp_dist= (float*)   (tmp_id  + args.ef_search + args.max_m);
    const int scratch_offset =
        (int)(((sizeof(int) * 3 + sizeof(float) * 2) *
               (args.ef_search + args.max_m) + 7) & ~7);
    uint8_t* node_scratch = shm_pool + scratch_offset;

    __shared__ int32_t lane_req[16];
    __shared__ int     s_query_id;
    __shared__ int     s_ready_w;
    __shared__ int32_t snap_st[16];
    __shared__ int32_t snap_nd[16];
    __shared__ int     s_stop_requested;
    __shared__ int     s_step_recall_reached;
    __shared__ int     s_drain_done;
#ifdef QUIVER_LIGHT_BREAKDOWN
    __shared__ int64_t breakdown_lane_issue_ts[16];
    __shared__ int64_t breakdown_wait_start_ns;
    __shared__ int64_t breakdown_pure_wait_ns;
#endif

    const int aef = args.aligned_ef;
    uint32_t* my_nid    = args.d_neighbors_id   + (long)bid * aef;
    float*    my_ndist  = args.d_neighbors_dist  + (long)bid * aef;
    Data*     my_ctx    = args.d_ctx             + bid;
    int*      my_nns    = args.d_nns             + bid * args.topk;
    float*    my_dist   = args.d_distances       + bid * args.topk;
    int*      my_fc     = args.d_found_cnt       + bid;
    float*    my_pq     = args.pq_dists
                        + (long)(bid + args.pq_block_offset)
                          * PQSearchData::num_pivots * args.num_chunks;
    IOControl* my_ctl   = args.comm->controls + bid * W;
    uint8_t*   my_buf   = args.comm->buffers
                        + (long)bid * W * shared::PAGE_SIZE;

#ifdef QUIVER_LIGHT_BREAKDOWN
    if (tid == 0) {
        for (int w = 0; w < 16; ++w) breakdown_lane_issue_ts[w] = 0;
        breakdown_wait_start_ns = 0;
        breakdown_pure_wait_ns = 0;
    }
    __syncthreads();
#endif

    // ============================================================
    // OUTER LOOP: one query per iteration.
    // Dispatch policy decides how qid is acquired and when to exit.
    // StreamingDispatch: atomicAdd + spin-wait on feed_count (streaming admission).
    // ============================================================
    while (true) {
#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_before_sched = globaltimer_ns();
#endif
        // tid 0 writes qid (or -1 for exit) into s_query_id; broadcast via __syncthreads.
        if (tid == 0) {
            Dispatch::acquire_next_qid_tid0(args, bid, /*slot*/0, &s_query_id);
        }
        __syncthreads();
        const int qid = s_query_id;
        if (qid == -1) return;  // Dispatch signaled terminate

#ifdef QUIVER_LIGHT_BREAKDOWN
        const int breakdown_sample_idx = quiver_breakdown_sample_index(
            qid, args.total_queries, args.breakdown_sample_count);
        QuiverLightBreakdownTrace* breakdown_trace =
            breakdown_sample_idx >= 0 && args.d_breakdown_traces != nullptr
                ? &args.d_breakdown_traces[breakdown_sample_idx]
                : nullptr;
        if (tid == 0 && breakdown_trace != nullptr) {
            breakdown_trace->query_id = qid;
            breakdown_trace->valid = 1;
            breakdown_trace->query_start_ns = globaltimer_ns();
            breakdown_trace->query_done_ns = 0;
            breakdown_trace->useful_compute_ns = 0;
            breakdown_trace->resumption_delay_ns = 0;
            breakdown_trace->ssd_wait_ns = 0;
            breakdown_trace->compute_interval_count = 0;
            breakdown_trace->wait_interval_count = 0;
            breakdown_trace->overlap_interval_count = 0;
            breakdown_pure_wait_ns = 0;
            for (int w = 0; w < W; ++w) breakdown_lane_issue_ts[w] = 0;
        }
#endif

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_query_start = globaltimer_ns();
        const bool do_trace = (qid < args.probe_max_traces);
        QuiverProbeQueryTrace* my_trace = do_trace
            ? &args.d_probe_traces[qid] : nullptr;
#endif

        float* qdata = args.d_all_qdata + (long)qid * args.num_dims;
        const bool do_step_recall =
            args.d_step_recall_gt != nullptr &&
            args.d_step_recall_step != nullptr &&
            args.d_step_recall_hits != nullptr;
        const bool do_early_exit =
            args.early_exit_policy != static_cast<int>(EarlyExitPolicy::kDefault);
        const int* step_recall_gt = do_step_recall
            ? args.d_step_recall_gt +
                  (long)(qid % args.step_recall_gt_count) *
                      args.step_recall_gt_width
            : nullptr;
        int step_recall_step = 0;
        int gpruning_no_improve_count = 0;
        int last_found_cnt = 0;
        float last_top1_dist = INFINITY;
        float last_topk_dist = INFINITY;
        if (tid == 0) {
            s_stop_requested = 0;
            s_step_recall_reached = 0;
        }
        __syncthreads();

        // ---- In-kernel nav (before init_query mutates qdata) ----
        // Block-level: ALL 128 threads enter; the 4 warps split the R=28
        // neighbor-distance computations evenly, warp 0 does the L=4
        // selection.  See inkernel_nav.cuh for the contract.
        int32_t entry_from_nav = -1;
        if (args.nav_data_dev != nullptr) {
            // Reuse mv_id / mv_dist shm scratch for the nav priority
            // queue (L + R = 32 entries, 256 B).  mv_* are not used
            // until after boot IO completes.
            uint32_t* nav_pq_id   = mv_id;
            float*    nav_pq_dist = mv_dist;
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
        int64_t probe_t_nav_done = globaltimer_ns();
#endif

        args.pq_data->init_query(qdata, args.pq_block_offset);
        __syncthreads();

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_pq_done = globaltimer_ns();
#endif

        if (tid == 0) {
            my_fc[0] = 0;
            my_ctx->size = 0;
            my_ctx->visited_cnt = 0;
        }
        for (int i = tid; i < args.topk; i += blockDim.x) {
            my_dist[i] = INFINITY;
            my_nns[i]  = -1;
        }
        if (tid < W) my_ctl[tid].status = IO_IDLE;
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
#ifdef QUIVER_LIGHT_BREAKDOWN
            if (breakdown_trace != nullptr)
                breakdown_lane_issue_ts[0] = globaltimer_ns();
#endif
            issue_request(&my_ctl[0], entry);
        }
        __syncthreads();

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_boot_issue = globaltimer_ns();
        int probe_boot_spins = 0;
#endif

        // ---- BOOT: wait for SubLane 0 ----
        while (true) {
            if (tid == 0)
                s_ready_w = (*(volatile int32_t*)&my_ctl[0].status == IO_READY) ? 0 : -1;
            __syncthreads();
            if (s_ready_w == 0) break;
#ifdef QUIVER_LIGHT_BREAKDOWN
            if (tid == 0 && breakdown_trace != nullptr &&
                breakdown_wait_start_ns == 0)
                breakdown_wait_start_ns = globaltimer_ns();
#endif
#ifdef QUIVER_LATENCY_PROBE
            if (tid == 0) probe_boot_spins++;
#endif
        }
#ifdef QUIVER_LIGHT_BREAKDOWN
        int64_t breakdown_compute_start_ns = 0;
        if (tid == 0 && breakdown_trace != nullptr) {
            const int64_t now = globaltimer_ns();
            if (breakdown_wait_start_ns > 0) {
                breakdown_pure_wait_ns += now - breakdown_wait_start_ns;
                breakdown_wait_start_ns = 0;
            }
            quiver_breakdown_add_ssd_wait_interval(
                breakdown_trace, breakdown_lane_issue_ts[0], now);
            breakdown_lane_issue_ts[0] = 0;
            breakdown_compute_start_ns = now;
        }
#endif
        __threadfence_system();

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_boot_ready = globaltimer_ns();
#endif

        if (tid == 0) my_ctl[0].status = IO_IDLE;

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_boot_pq_done = 0;
#endif
        persistent_merge_data(
            my_buf, entry,
            args.num_chunks, my_pq, args.compressed_data,
            args.nodes_per_page, args.node_size, args.data_size,
            args.max_m, args.ef_search,
            my_nid, my_ndist, my_ctx,
            mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
            node_scratch
#ifdef QUIVER_LATENCY_PROBE
            , &probe_t_boot_pq_done
#endif
            );

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_boot_merge_done = globaltimer_ns();
#endif

        int32_t boot_req = persistent_visit_select_dualwarp<T>(
            qdata, node_scratch, entry,
            args.num_dims, args.topk,
            my_nns, my_dist, my_fc,
            args.nodes_per_page, args.node_size,
            args.max_m, args.ef_search,
            my_nid, my_ndist, my_ctx);
        if (tid == 0) lane_req[0] = boot_req;
        __syncthreads();

        if (tid == 0) step_recall_step = 1;
        __syncthreads();

        if (tid == 0 && do_early_exit) {
            const bool topk_improved =
                my_fc[0] > last_found_cnt ||
                my_dist[0] < last_top1_dist ||
                my_dist[args.topk - 1] < last_topk_dist;
            if (step_recall_step < args.early_exit_gpruning_warmup_steps) {
                gpruning_no_improve_count = 0;
            } else {
                gpruning_no_improve_count =
                    topk_improved ? 0 : gpruning_no_improve_count + 1;
            }
            last_found_cnt = my_fc[0];
            last_top1_dist = my_dist[0];
            last_topk_dist = my_dist[args.topk - 1];
            if (args.early_exit_policy == static_cast<int>(EarlyExitPolicy::kGPruning) &&
                       step_recall_step >= args.early_exit_gpruning_warmup_steps &&
                       args.early_exit_gpruning_patience > 0 &&
                       gpruning_no_improve_count >= args.early_exit_gpruning_patience) {
                s_stop_requested = 1;
                if (args.d_early_exit_stop_step != nullptr)
                    args.d_early_exit_stop_step[qid] = step_recall_step;
                lane_req[0] = -1;
            } else if ((args.early_exit_policy == static_cast<int>(EarlyExitPolicy::kDarth) ||
                        args.early_exit_policy == static_cast<int>(EarlyExitPolicy::kDarthTrace)) &&
                args.early_exit_darth_interval > 0 &&
                args.darth_comm_slots != nullptr &&
                (step_recall_step % args.early_exit_darth_interval) == 0) {
                DarthCommSlot* slot = &args.darth_comm_slots[bid];
                int epoch = slot->request_epoch + 1;
                slot->qid = qid;
                slot->step = step_recall_step;
                slot->found_cnt = my_fc[0];
                slot->no_improve_count = gpruning_no_improve_count;
                slot->top1_dist = my_dist[0];
                slot->topk_dist = my_dist[args.topk - 1];
                slot->gap_ratio = (my_dist[0] > 0.0f && my_dist[args.topk - 1] < INFINITY)
                    ? my_dist[args.topk - 1] / my_dist[0]
                    : 0.0f;
                int copy_k = min(args.topk, DARTH_FEATURE_TOPK_MAX);
                for (int i = 0; i < copy_k; ++i) slot->top_ids[i] = my_nns[i];
                for (int i = copy_k; i < DARTH_FEATURE_TOPK_MAX; ++i) slot->top_ids[i] = -1;
                __threadfence_system();
                slot->request_epoch = epoch;
                slot->request_count++;
                long long spins = 0;
                while (*(volatile int32_t*)&slot->decision_epoch < epoch) {
                    ++spins;
                }
                __threadfence_system();
                slot->wait_iters_total += spins;
                if (*(volatile int32_t*)&slot->stop != 0) {
                    s_stop_requested = 1;
                    if (args.d_early_exit_stop_step != nullptr)
                        args.d_early_exit_stop_step[qid] = step_recall_step;
                    lane_req[0] = -1;
                }
            }
        }
        __syncthreads();

        if (do_step_recall && tid == 0 && !s_stop_requested) {
            int hits = count_topk_gt_hits(
                my_nns, args.topk, step_recall_gt, args.step_recall_gt_width);
            if (hits >= args.step_recall_required_hits) {
                s_step_recall_reached = 1;
                s_stop_requested = 1;
                args.d_step_recall_step[qid] = step_recall_step;
                args.d_step_recall_hits[qid] = hits;
                lane_req[0] = -1;
            }
        }
        __syncthreads();

        if (tid < 32) {
            for (int w = 1; w < W; w++)
                lane_req[w] = persistent_select_next_no_compact(
                    args.ef_search, my_nid, my_ctx);
        }
        __syncthreads();

#ifdef QUIVER_LIGHT_BREAKDOWN
        if (tid == 0 && breakdown_trace != nullptr) {
            quiver_breakdown_add_compute_interval(
                breakdown_trace, breakdown_compute_start_ns, globaltimer_ns());
        }
#endif

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_boot_compute_done = globaltimer_ns();
        int probe_hop_count = 0;
#endif

        if (tid == 0 && !s_stop_requested) {
            for (int w = 0; w < W; w++) {
                if (lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                    if (breakdown_trace != nullptr)
                        breakdown_lane_issue_ts[w] = globaltimer_ns();
#endif
                    issue_request(&my_ctl[w], lane_req[w]);
                }
            }
        }
        __syncthreads();

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_boot_all_issued = globaltimer_ns();
#endif

        // ---- PIPE LOOP: process all ready lanes per iteration ----
#ifdef QUIVER_LATENCY_PROBE
        int probe_spin_since_last_ready = 0;
#endif
        while (!s_stop_requested) {
            // Fast spin: only thread 0 polls status until it finds at least
            // one IO_READY lane (or all converged).  This avoids block-wide
            // barriers during SSD-wait spin, which is the common case.
            if (tid == 0) {
                while (true) {
                    s_ready_w = -1;
                    bool any_active = false;
                    for (int w = 0; w < W; w++) {
                        int32_t st = *(volatile int32_t*)&my_ctl[w].status;
                        snap_st[w] = st;
                        if (st == IO_READY) s_ready_w = w;
                        if (st != IO_IDLE) any_active = true;
                    }
                    if (s_ready_w == -2) break;  // unreachable, placeholder
                    if (s_ready_w == -1 && !any_active) {
                        s_ready_w = -2;  // converged
                        break;
                    }
                    if (s_ready_w != -1) break;  // found ready lane
#ifdef QUIVER_LIGHT_BREAKDOWN
                    if (breakdown_trace != nullptr &&
                        breakdown_wait_start_ns == 0)
                        breakdown_wait_start_ns = globaltimer_ns();
#endif
#ifdef QUIVER_LATENCY_PROBE
                    probe_spin_since_last_ready++;
#endif
                }
                // Latch node_ids for ready lanes before broadcasting
                if (s_ready_w != -2) {
                    for (int w = 0; w < W; w++) {
                        if (snap_st[w] == IO_READY)
                            snap_nd[w] = my_ctl[w].node_id;
                    }
                }
            }
            __syncthreads();

            if (s_ready_w == -2) break;

#ifdef QUIVER_LIGHT_BREAKDOWN
            int64_t breakdown_ready_ns = 0;
            int64_t breakdown_pipe_compute_start_ns = 0;
            if (tid == 0 && breakdown_trace != nullptr) {
                breakdown_ready_ns = globaltimer_ns();
                if (breakdown_wait_start_ns > 0) {
                    breakdown_pure_wait_ns +=
                        breakdown_ready_ns - breakdown_wait_start_ns;
                    breakdown_wait_start_ns = 0;
                }
                for (int w = 0; w < W; ++w) {
                    if (snap_st[w] == IO_READY &&
                        breakdown_lane_issue_ts[w] > 0) {
                        quiver_breakdown_add_ssd_wait_interval(
                            breakdown_trace, breakdown_lane_issue_ts[w],
                            breakdown_ready_ns);
                        breakdown_lane_issue_ts[w] = 0;
                    }
                }
                breakdown_pipe_compute_start_ns = globaltimer_ns();
                breakdown_trace->resumption_delay_ns +=
                    breakdown_pipe_compute_start_ns - breakdown_ready_ns;
            }
#endif

#ifdef QUIVER_LATENCY_PROBE
            int64_t probe_t_ready_found = globaltimer_ns();
#endif

            // System fence: ensures CPU-written buffer data is visible to GPU
            // before reading. Required once per iteration (covers all ready lanes).
            __threadfence_system();

#ifdef QUIVER_LATENCY_PROBE
            int64_t probe_t_compute_start = globaltimer_ns();
            int probe_ready_count = 0;
#endif

            // Process ALL ready lanes serially. merge_data needs all 128 threads
            // (bitonic sort), and visit_select modifies shared neighbor arrays
            // (my_nid/my_ndist), so lanes cannot be processed in parallel.
            for (int w = 0; w < W; w++) {
                if (snap_st[w] != IO_READY) continue;

#ifdef QUIVER_LATENCY_PROBE
                if (tid == 0) probe_ready_count++;
#endif

                if (tid == 0) my_ctl[w].status = IO_IDLE;

                persistent_merge_data(
                    my_buf + w * shared::PAGE_SIZE, snap_nd[w],
                    args.num_chunks, my_pq, args.compressed_data,
                    args.nodes_per_page, args.node_size, args.data_size,
                    args.max_m, args.ef_search,
                    my_nid, my_ndist, my_ctx,
                    mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
                    node_scratch
#ifdef QUIVER_LATENCY_PROBE
                    , nullptr
#endif
                    );

                int32_t next_req = persistent_visit_select_dualwarp<T>(
                    qdata, node_scratch, snap_nd[w],
                    args.num_dims, args.topk,
                    my_nns, my_dist, my_fc,
                    args.nodes_per_page, args.node_size,
                    args.max_m, args.ef_search,
                    my_nid, my_ndist, my_ctx);
                if (tid == 0) lane_req[w] = next_req;
                __syncthreads();

                if (tid == 0) ++step_recall_step;
                __syncthreads();

                if (tid == 0 && do_early_exit) {
                    const bool topk_improved =
                        my_fc[0] > last_found_cnt ||
                        my_dist[0] < last_top1_dist ||
                        my_dist[args.topk - 1] < last_topk_dist;
                    if (step_recall_step < args.early_exit_gpruning_warmup_steps) {
                        gpruning_no_improve_count = 0;
                    } else {
                        gpruning_no_improve_count =
                            topk_improved ? 0 : gpruning_no_improve_count + 1;
                    }
                    last_found_cnt = my_fc[0];
                    last_top1_dist = my_dist[0];
                    last_topk_dist = my_dist[args.topk - 1];
                    if (args.early_exit_policy == static_cast<int>(EarlyExitPolicy::kGPruning) &&
                               step_recall_step >= args.early_exit_gpruning_warmup_steps &&
                               args.early_exit_gpruning_patience > 0 &&
                               gpruning_no_improve_count >= args.early_exit_gpruning_patience) {
                        s_stop_requested = 1;
                        if (args.d_early_exit_stop_step != nullptr)
                            args.d_early_exit_stop_step[qid] = step_recall_step;
                        lane_req[w] = -1;
                    } else if ((args.early_exit_policy == static_cast<int>(EarlyExitPolicy::kDarth) ||
                                args.early_exit_policy == static_cast<int>(EarlyExitPolicy::kDarthTrace)) &&
                        args.early_exit_darth_interval > 0 &&
                        args.darth_comm_slots != nullptr &&
                        (step_recall_step % args.early_exit_darth_interval) == 0) {
                        DarthCommSlot* slot = &args.darth_comm_slots[bid];
                        int epoch = slot->request_epoch + 1;
                        slot->qid = qid;
                        slot->step = step_recall_step;
                        slot->found_cnt = my_fc[0];
                        slot->no_improve_count = gpruning_no_improve_count;
                        slot->top1_dist = my_dist[0];
                        slot->topk_dist = my_dist[args.topk - 1];
                        slot->gap_ratio =
                            (my_dist[0] > 0.0f && my_dist[args.topk - 1] < INFINITY)
                                ? my_dist[args.topk - 1] / my_dist[0]
                                : 0.0f;
                        int copy_k = min(args.topk, DARTH_FEATURE_TOPK_MAX);
                        for (int i = 0; i < copy_k; ++i) slot->top_ids[i] = my_nns[i];
                        for (int i = copy_k; i < DARTH_FEATURE_TOPK_MAX; ++i)
                            slot->top_ids[i] = -1;
                        __threadfence_system();
                        slot->request_epoch = epoch;
                        slot->request_count++;
                        long long spins = 0;
                        while (*(volatile int32_t*)&slot->decision_epoch < epoch) {
                            ++spins;
                        }
                        __threadfence_system();
                        slot->wait_iters_total += spins;
                        if (*(volatile int32_t*)&slot->stop != 0) {
                            s_stop_requested = 1;
                            if (args.d_early_exit_stop_step != nullptr)
                                args.d_early_exit_stop_step[qid] = step_recall_step;
                            lane_req[w] = -1;
                        }
                    }
                }
                __syncthreads();

                if (do_step_recall && tid == 0 && !s_stop_requested) {
                    int hits = count_topk_gt_hits(
                        my_nns, args.topk, step_recall_gt,
                        args.step_recall_gt_width);
                    if (hits >= args.step_recall_required_hits) {
                        s_step_recall_reached = 1;
                        s_stop_requested = 1;
                        args.d_step_recall_step[qid] = step_recall_step;
                        args.d_step_recall_hits[qid] = hits;
                        lane_req[w] = -1;
                    }
                }
                __syncthreads();

            }

            if (s_stop_requested) {
                drain_io_lanes_after_stop(my_ctl, W, &s_drain_done);
                break;
            }

#ifdef QUIVER_LIGHT_BREAKDOWN
            if (tid == 0 && breakdown_trace != nullptr) {
                quiver_breakdown_add_compute_interval(
                    breakdown_trace, breakdown_pipe_compute_start_ns,
                    globaltimer_ns());
            }
#endif

#ifdef QUIVER_LATENCY_PROBE
            int64_t probe_t_compute_done = globaltimer_ns();
#endif

            // Phase 1: select_next for IDLE lanes (warp 0 uses shared neighbor
            // arrays).  Must happen BEFORE issuing ready-lane requests so both
            // kinds of requests can be issued in a single barrier-free step.
            if (tid < 32) {
                for (int w = 0; w < W; w++) {
                    if (snap_st[w] != IO_READY &&
                        *(volatile int32_t*)&my_ctl[w].status == IO_IDLE) {
                        lane_req[w] = persistent_select_next_no_compact(
                            args.ef_search, my_nid, my_ctx);
                    }
                }
            }
            __syncthreads();

            // Phase 2: issue all new IO requests in one pass (both for
            // processed ready-lanes and for freshly selected idle-lanes).
            if (tid == 0) {
                for (int w = 0; w < W; w++) {
                    if (snap_st[w] == IO_READY && lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                        if (breakdown_trace != nullptr)
                            breakdown_lane_issue_ts[w] = globaltimer_ns();
#endif
                        issue_request(&my_ctl[w], lane_req[w]);
                    } else if (snap_st[w] != IO_READY &&
                               my_ctl[w].status == IO_IDLE &&
                               lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                        if (breakdown_trace != nullptr)
                            breakdown_lane_issue_ts[w] = globaltimer_ns();
#endif
                        issue_request(&my_ctl[w], lane_req[w]);
                    }
                }
            }
            __syncthreads();

#ifdef QUIVER_LATENCY_PROBE
            if (tid == 0) {
                if (do_trace && probe_hop_count < PROBE_MAX_HOPS) {
                    QuiverProbeHop& h = my_trace->hops[probe_hop_count];
                    h.ready_found_ts = probe_t_ready_found;
                    h.compute_start_ts = probe_t_compute_start;
                    h.compute_done_ts = probe_t_compute_done;
                    h.ready_lanes = probe_ready_count;
                    h.spin_iters = probe_spin_since_last_ready;
                }
                probe_spin_since_last_ready = 0;
                probe_hop_count++;
            }
#endif
        } // pipe loop

#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_pipe_done = globaltimer_ns();
#endif

        if (do_step_recall && tid == 0 && !s_step_recall_reached) {
            int hits = count_topk_gt_hits(
                my_nns, args.topk, step_recall_gt, args.step_recall_gt_width);
            args.d_step_recall_step[qid] = -1;
            args.d_step_recall_hits[qid] = hits;
        }
        __syncthreads();

        // ---- FINISH: write results to mapped-pinned output buffers ----
        if (tid == 0 && args.d_finish_step != nullptr) {
            args.d_finish_step[qid] = step_recall_step;
        }
        __syncthreads();

        for (int i = tid; i < args.topk; i += blockDim.x) {
            args.d_out_nns[qid * args.topk + i]       = my_nns[i];
            args.d_out_distances[qid * args.topk + i]  = my_dist[i];
        }
        if (tid == 0) {
            args.d_out_found_cnt[qid] = my_fc[0];
        }
        // Fence all threads' result writes so they are host-visible
        // before tid 0 bumps the completion counter.
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            int new_done = atomicAdd(args.d_queries_done, 1) + 1;
            *(volatile int32_t*)&args.comm->queries_completed = new_done;
            __threadfence_system();
#ifdef QUIVER_LIGHT_BREAKDOWN
            if (breakdown_trace != nullptr) {
                const int64_t done = globaltimer_ns();
                breakdown_trace->query_done_ns = done;
                if (args.d_breakdown_cta_windows != nullptr &&
                    breakdown_sample_idx < args.breakdown_cta_window_count) {
                    auto& window =
                        args.d_breakdown_cta_windows[breakdown_sample_idx];
                    window.valid = 1;
                    window.block_id = bid;
                    window.trigger_query_id = qid;
                    window.query_count_delta = 1;
                    window.window_start_ns = breakdown_trace->query_start_ns;
                    window.window_end_ns = done;
                    window.pure_io_wait_ns = breakdown_pure_wait_ns;
                }
            }
#endif
        }
#ifdef QUIVER_LATENCY_PROBE
        int64_t probe_t_query_done = 0;
        if (tid == 0) probe_t_query_done = globaltimer_ns();
#endif
        __syncthreads();

#ifdef QUIVER_LATENCY_PROBE
        if (tid == 0 && do_trace) {
            my_trace->query_id = qid;
            my_trace->num_hops = min(probe_hop_count, PROBE_MAX_HOPS);
            my_trace->t_sched_start = probe_t_before_sched;
            my_trace->t_query_start = probe_t_query_start;
            my_trace->t_nav_done = probe_t_nav_done;
            my_trace->t_pq_done = probe_t_pq_done;
            my_trace->t_boot_issue = probe_t_boot_issue;
            my_trace->t_boot_ready = probe_t_boot_ready;
            my_trace->t_boot_pq_done = probe_t_boot_pq_done;
            my_trace->t_boot_merge_done = probe_t_boot_merge_done;
            my_trace->t_boot_compute_done = probe_t_boot_compute_done;
            my_trace->t_boot_all_issued = probe_t_boot_all_issued;
            my_trace->t_pipe_start = probe_t_boot_all_issued;
            my_trace->t_pipe_done = probe_t_pipe_done;
            my_trace->t_query_done = probe_t_query_done;
            my_trace->boot_spin_iters = probe_boot_spins;
        }
#endif

        // StaticDispatch: one query per block per launch; exit after finishing.
        // StreamingDispatch: loop back and acquire the next qid.
        if constexpr (!Dispatch::is_persistent) {
            return;
        }
    } // query loop
}

} // namespace quiver
