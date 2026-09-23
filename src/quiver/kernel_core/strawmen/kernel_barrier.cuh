#pragma once

// ============================================================================
// Strawman barrier: per-hop block barrier (strawman Q-mini-batch)
// ----------------------------------------------------------------------------
// This is a deliberate strawman baseline used in the §Key Idea defence.
// It is structurally identical to kernel_qi.cuh (Q-interleave persistent kernel
// with StreamingDispatch) EXCEPT the scheduler loop:
//
//   - Quiver scheduler  : event-driven (tid0 picks ANY slot whose lane 0 is
//                         IO_READY -> __syncthreads -> all 128 threads serve
//                         that one slot's hop). Slots make hop-progress
//                         independently, no cross-slot barrier.
//
//   - Strawman barrier       : "fair round-robin" -- each outer iteration enforces
//                         a cross-slot wait so that EVERY active slot has
//                         IO_READY before any slot's hop is processed. Then
//                         all active slots advance ONE hop in series within
//                         the same iteration. This is exactly the natural
//                         translation of "block holds Q slots" by someone who
//                         has not thought about IO-fan-in dynamics; it amounts
//                         to a Q-wide mini-batch within the block.
//

//   1. Block-wide tail latency: slow IOs gate every active slot.
//   2. SSD pipeline becomes "pulse-shaped": all slots issue together,
//      all wait, all fire -> the CPU poller sees burst REQUESTED, but
//      the SSD queue drains between bursts.
//   3. Per-hop interleave saved by the event-driven design is lost.
//
// Everything else (try_bind, finalize, RETIRED handling, state machine
// values, per-slot resource layout) is unchanged so that the only variable
// being ablated is the scheduling discipline.
//
// ONLY the StreamingDispatch instantiation is supported here -- the strawman
// is meant to be compared against Quiver's StreamingDispatch baseline.
// ============================================================================

#include "../kernel_helpers.cuh"
#include "../inkernel_nav.cuh"
#include "../policies/dispatch.cuh"

namespace quiver {
namespace strawmen {

constexpr int BARRIER_MAX_Q = 8;

template <class Dispatch, class T>
__global__ void persistent_search_kernel_barrier(PersistentKernelArgs args)
{
    static_assert(Dispatch::is_persistent,
                  "Strawman barrier only meaningful with StreamingDispatch (the "
                  "ablation isolates 'per-hop barrier' variable while keeping "
                  "the streaming admission protocol identical to Quiver).");

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

    __shared__ int32_t s_slot_qid[BARRIER_MAX_Q];
    __shared__ int32_t s_slot_entry[BARRIER_MAX_Q];
    // Slot stage encoding (same as kernel_qi.cuh):
    //   -3 RETIRED, -2 EXIT_SIGNAL, -1 EMPTY, 0 BOOT_WAIT, 1 RUNNING, 2 CONVERGED
    __shared__ int32_t s_slot_stage[BARRIER_MAX_Q];
    __shared__ int32_t s_slot_next[BARRIER_MAX_Q];
    __shared__ int32_t s_kernel_exit;
    __shared__ int32_t s_all_ready;     // Barrier-strawman cross-slot flag

    if (tid == 0) {
        s_kernel_exit = 0;
        for (int q = 0; q < Q; ++q) {
            s_slot_qid[q]   = -1;
            s_slot_entry[q] = -1;
            s_slot_stage[q] = -1;  // EMPTY
            s_slot_next[q]  = -1;
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

    // ------------------------------------------------------------------
    // try_bind_new_query: identical to kernel_qi.cuh (kept verbatim so the
    // only variable being ablated is the scheduler).
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

        if (tid == 0) issue_request(&slot_ctl(q)[0], entry);
        __syncthreads();
        return true;
    };

    // ------------------------------------------------------------------
    // finalize_slot: identical to kernel_qi.cuh
    // ------------------------------------------------------------------
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
            s_slot_stage[q] = -1;  // EMPTY: ready to bind new qid (streaming)
        }
        __syncthreads();
    };

    // ------------------------------------------------------------------
    // process_slot_hop_force: process one hop for slot q. Caller must
    // have already verified s_slot_stage[q] is BOOT_WAIT or RUNNING and
    // that lane 0 is IO_READY (the barrier strawman's per-hop barrier guarantees
    // both). Differs from kernel_qi.cuh's try_process_slot_hop in that it
    // unconditionally processes (no early-out on !IO_READY).
    // ------------------------------------------------------------------
    auto process_slot_hop_force = [&](int q) {
        int32_t nid;
        if (tid == 0) {
            nid = slot_ctl(q)[0].node_id;
            s_slot_next[q] = nid;
            slot_ctl(q)[0].status = IO_IDLE;
        }
        __syncthreads();
        nid = s_slot_next[q];

        __threadfence_system();

        uint8_t* base = slot_shm_base(q);
        int*      mv_pos  = (int*)base;
        uint32_t* mv_id   = (uint32_t*)(mv_pos  + ef_plus_m);
        float*    mv_dist = (float*)   (mv_id   + ef_plus_m);
        uint32_t* tmp_id  = (uint32_t*)(mv_dist + ef_plus_m);
        float*    tmp_dist= (float*)   (tmp_id  + ef_plus_m);

        float* qdata = args.d_all_qdata + (long)s_slot_qid[q] * args.num_dims;

        int32_t src_node = (s_slot_stage[q] == 0) ? s_slot_entry[q] : nid;

        persistent_merge_data(
            slot_buf(q), src_node,
            args.num_chunks, slot_pq(q), args.compressed_data,
            args.nodes_per_page, args.node_size, args.data_size,
            args.max_m, args.ef_search,
            slot_nid(q), slot_ndist(q), slot_ctx(q),
            mv_pos, mv_id, mv_dist, tmp_id, tmp_dist,
            slot_node_scratch(q));

        int32_t next_node = -1;
        next_node = persistent_visit_select_dualwarp<T>(
            qdata, slot_node_scratch(q), src_node,
            args.num_dims, args.topk,
            slot_nns(q), slot_dist(q), slot_fc(q),
            args.nodes_per_page, args.node_size,
            args.max_m, args.ef_search,
            slot_nid(q), slot_ndist(q), slot_ctx(q));
        __syncthreads();

        if (tid == 0) s_slot_next[q] = next_node;
        __syncthreads();
        next_node = s_slot_next[q];

        if (tid == 0) {
            if (next_node != -1) {
                issue_request(&slot_ctl(q)[0], next_node);
                s_slot_stage[q] = 1;  // RUNNING
            } else {
                s_slot_stage[q] = 2;  // CONVERGED
            }
        }
        __syncthreads();
    };

    // ============================================================
    // STRAWMAN BARRIER SCHEDULER LOOP — per-hop block barrier.
    //
    // Each outer iteration:
    //   (1) bind any EMPTY slot to a new query (issue boot IO)
    //   (2) check exit conditions
    //   (3) **CROSS-SLOT BARRIER**: spin until every active slot's lane 0
    //       has reached IO_READY  (this is the strawman "wait for all" pattern)
    //   (4) for each active slot: process exactly one hop (in q order)
    //   (5) finalize any CONVERGED slot
    //
    // Compared with kernel_qi.cuh, step (3) and the in-order step (4)
    // are the only structural changes. They turn the block into a Q-wide
    // mini-batch synchronous scheduler.
    // ============================================================
    while (true) {
        __syncthreads();

        // --- (1) bring empty slots online ---
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
                    if (tid == 0) s_slot_stage[q] = -3;  // RETIRED
                    __syncthreads();
                }
            }
        }

        // --- (2) exit if no active slots ---
        int any_active = 0;
        if (tid == 0) {
            for (int q = 0; q < Q; ++q) {
                if (s_slot_stage[q] >= 0) { any_active = 1; break; }
            }
            s_kernel_exit |=
                (!any_active) &&
                (*(volatile int32_t*)&args.comm->terminate ||
                 *(volatile int32_t*)args.d_next_query >= k_total_queries);
            s_all_ready = any_active;  // overload as "have any active to wait for"
        }
        __syncthreads();
        if (s_kernel_exit) return;
        if (!s_all_ready) continue;  // no active slots; loop to bind more

        // --- (3) CROSS-SLOT BARRIER (the strawman's defining sin) ---
        //
        // Wait until EVERY active slot whose stage in {BOOT_WAIT, RUNNING}
        // has lane 0 IO_READY. Slots with stage CONVERGED are skipped (they
        // will be finalized in step 5 of this iteration).
        //
        // Critically: the block does no useful compute while waiting --
        // tid 0 spin-polls all active lanes in turn, the rest of the
        // block is parked on the next __syncthreads.
        if (tid == 0) {
            int spin = 0;
            while (true) {
                bool all_ok = true;
                for (int q = 0; q < Q; ++q) {
                    int st_slot = s_slot_stage[q];
                    if (st_slot != 0 && st_slot != 1) continue;  // not waiting
                    int32_t io = *(volatile int32_t*)&slot_ctl(q)[0].status;
                    if (io != IO_READY) { all_ok = false; break; }
                }
                if (all_ok) break;
                // Cheap escape hatch in case of terminate while spinning.
                if (++spin >= (1 << 18)) {
                    if (*(volatile int32_t*)&args.comm->terminate) {
                        s_kernel_exit = 1;
                        break;
                    }
                    spin = 0;
                }
            }
        }
        __syncthreads();
        if (s_kernel_exit) return;

        // --- (4) process every active slot's one hop, in q order ---
        for (int q = 0; q < Q; ++q) {
            if (s_slot_stage[q] != 0 && s_slot_stage[q] != 1) continue;
            process_slot_hop_force(q);
        }

        // --- (5) finalize any CONVERGED slot ---
        for (int q = 0; q < Q; ++q) {
            if (s_slot_stage[q] == 2) finalize_slot(q);
        }
    }
}

}  // namespace strawmen
}  // namespace quiver
