#pragma once

#include "io_control.cuh"
#include "shared/common/gpu_primitives.cuh"
#include "shared/common/search_state.cuh"
#include "shared/index/pq_device.cuh"

namespace flashanns {

using shared::Data;
using shared::PQSearchData;
using shared::lower_bound;
using shared::retset_push_32;
using shared::square_sum_32;

__device__ void persistent_merge_data(
    uint8_t* buffer, int32_t node_id,
    int num_chunks, float* pq_dists, uint8_t* compressed_data,
    int nodes_per_page, int node_size, int data_size,
    int max_m, int ef_search,
    uint32_t* neighbor_id, float* neighbor_dist, Data* ctx,
    int* mv_pos, uint32_t* mv_id, float* mv_dist,
    uint32_t* tmp_id, float* tmp_dist
#ifdef FLASHANNS_LATENCY_PROBE
    , int64_t* out_pq_done
#endif
    )
{
    int tid = threadIdx.x;
    if (node_id == -1) {
#ifdef FLASHANNS_LATENCY_PROBE
        if (tid == 0) { *out_pq_done = 0; }
#endif
        return;
    }

    int sz = ctx->size;
    int edge_offset = node_id % nodes_per_page * node_size + data_size;
    int* buffer_u = (int*)(buffer + edge_offset);

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

#ifdef FLASHANNS_LATENCY_PROBE
    if (tid == 0) *out_pq_done = globaltimer_ns();
#endif

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
        retset_push_32(distances, nns, found_cnt[0], topk, dist, node_id);
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
    __threadfence_system();
    ctl->status = IO_REQUESTED;
    __threadfence_system();
}

// ================================================================
// Persistent kernel — FlashANNS design (VLDB 2026).
//
// Each kernel launch processes exactly one batch of B = num_blocks
// queries. Block bid is statically bound to query bid for the entire
// launch:
//     query id = blockIdx.x
//     block bid works on query bid, runs its full beam-search (boot +
//     pipe loop), writes results, then returns.
//
// There is no intra-launch work-stealing: once bid's query converges,
// that block is idle for the remainder of this launch. The host loop
// issues the next launch (cudaDeviceSynchronize + D2H + next batch).
//
// Within each query, ALL ready IO lanes are processed per pipe iteration
// to avoid lane starvation with fast backends.
// ================================================================
template <class T>
__global__ void persistent_search_kernel(PersistentKernelArgs args)
{
    extern __shared__ uint8_t shm_pool[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
#ifdef FLASHANNS_FIXED_PIPE_WIDTH
    constexpr int W = FLASHANNS_FIXED_PIPE_WIDTH;
#else
    const int W = args.pipe_w;
#endif

    int*      mv_pos  = (int*)shm_pool;
    uint32_t* mv_id   = (uint32_t*)(mv_pos  + args.ef_search + args.max_m);
    float*    mv_dist = (float*)   (mv_id   + args.ef_search + args.max_m);
    uint32_t* tmp_id  = (uint32_t*)(mv_dist + args.ef_search + args.max_m);
    float*    tmp_dist= (float*)   (tmp_id  + args.ef_search + args.max_m);

    __shared__ int32_t lane_req[16];
    __shared__ int     s_ready_w;
    __shared__ int32_t snap_st[16];
    __shared__ int32_t snap_nd[16];

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
    __shared__ int32_t s_breakdown_cta_window_active;
    __shared__ int64_t s_breakdown_cta_window_start_ns;
    __shared__ int64_t s_breakdown_cta_window_deadline_ns;
    __shared__ int64_t s_breakdown_cta_window_pure_wait_ns;
    __shared__ int64_t s_breakdown_cta_window_wait_start_ns;
    __shared__ int32_t s_breakdown_cta_window_waiting;
    __shared__ int32_t s_breakdown_cta_window_idx;
    __shared__ int32_t s_breakdown_cta_window_trigger_qid;
    __shared__ int64_t s_breakdown_lane_issue_ts[16];

    if (tid == 0) {
        s_breakdown_cta_window_active = 0;
        s_breakdown_cta_window_start_ns = 0;
        s_breakdown_cta_window_deadline_ns = 0;
        s_breakdown_cta_window_pure_wait_ns = 0;
        s_breakdown_cta_window_wait_start_ns = 0;
        s_breakdown_cta_window_waiting = 0;
        s_breakdown_cta_window_idx = -1;
        s_breakdown_cta_window_trigger_qid = -1;
        for (int w = 0; w < 16; ++w) s_breakdown_lane_issue_ts[w] = 0;
    }
    __syncthreads();

    auto breakdown_cta_finish_window = [&](int64_t now) {
        if (tid != 0 || !s_breakdown_cta_window_active) return;
        const int idx = s_breakdown_cta_window_idx;
        if (args.d_breakdown_cta_windows == nullptr || idx < 0 ||
            idx >= args.breakdown_cta_window_count) {
            s_breakdown_cta_window_active = 0;
            return;
        }
        int64_t end_ns = now;
        if (s_breakdown_cta_window_deadline_ns > 0 &&
            end_ns > s_breakdown_cta_window_deadline_ns) {
            end_ns = s_breakdown_cta_window_deadline_ns;
        }
        if (s_breakdown_cta_window_waiting) {
            const int64_t wait_end = end_ns;
            if (wait_end > s_breakdown_cta_window_wait_start_ns) {
                s_breakdown_cta_window_pure_wait_ns +=
                    wait_end - s_breakdown_cta_window_wait_start_ns;
            }
            s_breakdown_cta_window_waiting = 0;
            s_breakdown_cta_window_wait_start_ns = 0;
        }
        if (end_ns > s_breakdown_cta_window_start_ns) {
            auto& tr = args.d_breakdown_cta_windows[idx];
            tr.valid = 1;
            tr.block_id = bid;
            tr.trigger_query_id = s_breakdown_cta_window_trigger_qid;
            tr.window_start_ns = s_breakdown_cta_window_start_ns;
            tr.window_end_ns = end_ns;
            tr.pure_io_wait_ns = s_breakdown_cta_window_pure_wait_ns;
        }
        s_breakdown_cta_window_active = 0;
        s_breakdown_cta_window_start_ns = 0;
        s_breakdown_cta_window_deadline_ns = 0;
        s_breakdown_cta_window_pure_wait_ns = 0;
        s_breakdown_cta_window_wait_start_ns = 0;
        s_breakdown_cta_window_waiting = 0;
        s_breakdown_cta_window_idx = -1;
        s_breakdown_cta_window_trigger_qid = -1;
    };

    auto breakdown_cta_end_wait = [&](int64_t now) {
        if (tid == 0 && s_breakdown_cta_window_active &&
            s_breakdown_cta_window_waiting) {
            int64_t wait_end = now;
            if (s_breakdown_cta_window_deadline_ns > 0 &&
                wait_end > s_breakdown_cta_window_deadline_ns) {
                wait_end = s_breakdown_cta_window_deadline_ns;
            }
            if (wait_end > s_breakdown_cta_window_wait_start_ns) {
                s_breakdown_cta_window_pure_wait_ns +=
                    wait_end - s_breakdown_cta_window_wait_start_ns;
            }
            s_breakdown_cta_window_waiting = 0;
            s_breakdown_cta_window_wait_start_ns = 0;
        }
    };

    auto breakdown_cta_start_wait = [&]() {
        if (tid == 0 && s_breakdown_cta_window_active) {
            const int64_t now = globaltimer_ns();
            if (s_breakdown_cta_window_deadline_ns > 0 &&
                now >= s_breakdown_cta_window_deadline_ns) {
                breakdown_cta_finish_window(now);
            } else if (!s_breakdown_cta_window_waiting) {
                s_breakdown_cta_window_wait_start_ns = now;
                s_breakdown_cta_window_waiting = 1;
            }
        }
    };

    auto breakdown_cta_tick_window = [&]() {
        if (tid == 0 && s_breakdown_cta_window_active) {
            const int64_t now = globaltimer_ns();
            if (s_breakdown_cta_window_deadline_ns > 0 &&
                now >= s_breakdown_cta_window_deadline_ns) {
                breakdown_cta_finish_window(now);
            }
        }
    };
#endif


#ifdef FLASHANNS_LATENCY_PROBE
    __shared__ int64_t probe_pq_done_ts;
#endif

    // ============================================================
    // One query per block — static binding (qid = blockIdx.x).
    // Out-of-range blocks (bid >= total_queries when the final batch is
    // smaller than num_blocks) exit immediately.
    // ============================================================
    const int qid = bid;
    if (qid >= args.comm->total_queries) return;

#ifdef QUIVER_LIGHT_BREAKDOWN
    const int global_qid = args.batch_query_offset + qid;
    const int breakdown_sample_idx = shared::breakdown::sample_index(
        global_qid, args.total_search_queries, args.breakdown_sample_count);
    const bool do_tech =
        breakdown_sample_idx >= 0 && args.d_breakdown_traces != nullptr;
    FlashLightBreakdownTrace* breakdown_trace =
        do_tech ? &args.d_breakdown_traces[breakdown_sample_idx] : nullptr;
    if (tid == 0 && do_tech) {
        breakdown_trace->query_id = global_qid;
        breakdown_trace->valid = 1;
        breakdown_trace->compute_interval_count = 0;
        breakdown_trace->wait_interval_count = 0;
        breakdown_trace->query_start_ns = globaltimer_ns();
        breakdown_trace->query_done_ns = 0;
        breakdown_trace->useful_compute_ns = 0;
        breakdown_trace->resumption_delay_ns = 0;
    }
    if (tid == 0 && W == 2 && do_tech &&
        args.d_breakdown_cta_windows != nullptr &&
        args.breakdown_cta_window_ns > 0 &&
        breakdown_sample_idx < args.breakdown_cta_window_count) {
        const int64_t now = globaltimer_ns();
        s_breakdown_cta_window_active = 1;
        s_breakdown_cta_window_idx = breakdown_sample_idx;
        s_breakdown_cta_window_trigger_qid = global_qid;
        s_breakdown_cta_window_start_ns = now;
        s_breakdown_cta_window_deadline_ns =
            now + args.breakdown_cta_window_ns;
        s_breakdown_cta_window_pure_wait_ns = 0;
        s_breakdown_cta_window_wait_start_ns = 0;
        s_breakdown_cta_window_waiting = 0;
    }
    __syncthreads();
#endif

    {  // Single-iteration scope (replaces the former work-stealing loop).
       // Kept as a block so all subsequent "break / continue" references and
       // existing indentation still compile; the block runs exactly once.
#ifdef FLASHANNS_LATENCY_PROBE
        int64_t probe_t_query_start = globaltimer_ns();
        int64_t probe_t_init_done = 0;
        int64_t probe_t_boot_io_done = 0, probe_t_boot_pq_done = 0;
        int64_t probe_t_boot_merge_done = 0, probe_t_boot_sort_done = 0;
        int64_t probe_t_boot_fill_done = 0;
        int64_t probe_pipe_io_ns = 0, probe_pipe_pq_ns = 0;
        int64_t probe_pipe_merge_ns = 0, probe_pipe_sort_ns = 0;
        int64_t probe_pipe_issue_ns = 0;
        int32_t probe_hop_count = 0;
        int32_t probe_spin_since_last_ready = 0;
#endif

        float* qdata = args.d_all_qdata + (long)qid * args.num_dims;

        args.pq_data->init_query(qdata, args.pq_block_offset);
        __syncthreads();

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

#ifdef FLASHANNS_LATENCY_PROBE
        if (tid == 0) probe_t_init_done = globaltimer_ns();
#endif

        int32_t entry = args.d_entry_nodes
                      ? args.d_entry_nodes[qid]
                      : args.enter_point;
        if (tid == 0) {
#ifdef QUIVER_LIGHT_BREAKDOWN
            if (do_tech) s_breakdown_lane_issue_ts[0] = globaltimer_ns();
#endif
            issue_request(&my_ctl[0], entry);
        }
        __syncthreads();

        // ---- BOOT: wait for SubLane 0 ----
        while (true) {
            if (tid == 0)
                s_ready_w = (*(volatile int32_t*)&my_ctl[0].status == IO_READY) ? 0 : -1;
            __syncthreads();
            if (s_ready_w == 0) break;
#ifdef QUIVER_LIGHT_BREAKDOWN
            breakdown_cta_start_wait();
#endif
        }
#ifdef QUIVER_LIGHT_BREAKDOWN
        breakdown_cta_end_wait(globaltimer_ns());
#endif
        __threadfence_system();

#ifdef FLASHANNS_LATENCY_PROBE
        if (tid == 0) probe_t_boot_io_done = globaltimer_ns();
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
        int64_t breakdown_t_boot_ready = 0;
        int64_t breakdown_t_boot_compute_start = 0;
        if (tid == 0 && do_tech) {
            breakdown_t_boot_ready = globaltimer_ns();
            if (s_breakdown_lane_issue_ts[0] > 0) {
                flash_breakdown_add_wait_interval(
                    breakdown_trace, s_breakdown_lane_issue_ts[0],
                    breakdown_t_boot_ready);
                s_breakdown_lane_issue_ts[0] = 0;
            }
        }
#endif

        if (tid == 0) my_ctl[0].status = IO_IDLE;

#ifdef QUIVER_LIGHT_BREAKDOWN
        if (tid == 0 && do_tech) {
            breakdown_t_boot_compute_start = globaltimer_ns();
            if (breakdown_t_boot_compute_start > breakdown_t_boot_ready) {
                breakdown_trace->resumption_delay_ns +=
                    breakdown_t_boot_compute_start - breakdown_t_boot_ready;
            }
        }
#endif

        persistent_merge_data(
            my_buf, entry,
            args.num_chunks, my_pq, args.compressed_data,
            args.nodes_per_page, args.node_size, args.data_size,
            args.max_m, args.ef_search,
            my_nid, my_ndist, my_ctx,
            mv_pos, mv_id, mv_dist, tmp_id, tmp_dist
#ifdef FLASHANNS_LATENCY_PROBE
            , &probe_pq_done_ts
#endif
            );
#ifdef FLASHANNS_LATENCY_PROBE
        if (tid == 0) {
            probe_t_boot_pq_done = probe_pq_done_ts;
            probe_t_boot_merge_done = globaltimer_ns();
        }
#endif

        if (tid < 32) {
            lane_req[0] = persistent_visit_select<T>(
                qdata, my_buf, entry,
                args.num_dims, args.topk,
                my_nns, my_dist, my_fc,
                args.nodes_per_page, args.node_size,
                args.max_m, args.ef_search,
                my_nid, my_ndist, my_ctx);
        }
        __syncthreads();

#ifdef QUIVER_LIGHT_BREAKDOWN
        if (tid == 0 && do_tech) {
            flash_breakdown_add_compute_interval(
                breakdown_trace, breakdown_t_boot_compute_start,
                globaltimer_ns());
        }
#endif

#ifdef FLASHANNS_LATENCY_PROBE
        if (tid == 0) probe_t_boot_sort_done = globaltimer_ns();
#endif

        if (tid < 32) {
            for (int w = 1; w < W; w++)
                lane_req[w] = persistent_select_next(
                    args.max_m, args.ef_search, my_nid, my_ndist, my_ctx);
        }
        __syncthreads();

        if (tid == 0) {
            for (int w = 0; w < W; w++) {
                if (lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                    if (do_tech)
                        s_breakdown_lane_issue_ts[w] = globaltimer_ns();
#endif
                    issue_request(&my_ctl[w], lane_req[w]);
                }
            }
        }
        __syncthreads();

#ifdef FLASHANNS_LATENCY_PROBE
        if (tid == 0) probe_t_boot_fill_done = globaltimer_ns();
        if (tid == 0 && probe_hop_count < FLASH_PROBE_MAX_HOPS) {
            FlashProbeHop& h = args.d_probe_traces[qid].hops[probe_hop_count];
            h.t_wait_start = probe_t_init_done;
            h.t_ready_found = probe_t_boot_io_done;
            h.t_compute_done = probe_t_boot_sort_done;
            h.t_issue_done = probe_t_boot_fill_done;
            h.ready_lanes = 1;
            h.spin_iters = 0;
            probe_hop_count++;
        }
        int64_t probe_t_last_compute_end = globaltimer_ns();
#endif
        // ---- PIPE LOOP: process all ready lanes per iteration ----
        while (true) {
#ifdef QUIVER_LIGHT_BREAKDOWN
            breakdown_cta_tick_window();
#endif
            if (tid < W) {
                snap_st[tid] = *(volatile int32_t*)&my_ctl[tid].status;
                snap_nd[tid] = my_ctl[tid].node_id;
            }
            __syncthreads();

            if (tid == 0) {
                s_ready_w = -1;
                bool any_active = false;
                for (int w = 0; w < W; w++) {
                    if (snap_st[w] == IO_READY) s_ready_w = w;
                    if (snap_st[w] != IO_IDLE)  any_active = true;
                }
                if (s_ready_w == -1 && !any_active)
                    s_ready_w = -2;
            }
            __syncthreads();

            if (s_ready_w == -2) break;
            if (s_ready_w == -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                breakdown_cta_start_wait();
#endif
#ifdef FLASHANNS_LATENCY_PROBE
                if (tid == 0) probe_spin_since_last_ready++;
#endif
                continue;
            }
#ifdef QUIVER_LIGHT_BREAKDOWN
            breakdown_cta_end_wait(globaltimer_ns());
#endif

#ifdef FLASHANNS_LATENCY_PROBE
            int64_t probe_t_wait_start = probe_t_last_compute_end;
            int64_t probe_t_ready_found = 0;
            int32_t probe_ready_count = 0;
            if (tid == 0) {
                probe_t_ready_found = globaltimer_ns();
                probe_pipe_io_ns += probe_t_ready_found - probe_t_wait_start;
                for (int w = 0; w < W; w++) {
                    if (snap_st[w] == IO_READY) probe_ready_count++;
                }
            }
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
            int64_t breakdown_t_ready_found = 0;
            if (tid == 0 && do_tech) {
                breakdown_t_ready_found = globaltimer_ns();
                for (int w = 0; w < W; w++) {
                    if (snap_st[w] == IO_READY &&
                        s_breakdown_lane_issue_ts[w] > 0) {
                        flash_breakdown_add_wait_interval(
                            breakdown_trace, s_breakdown_lane_issue_ts[w],
                            breakdown_t_ready_found);
                        s_breakdown_lane_issue_ts[w] = 0;
                    }
                }
            }
#endif

            __threadfence_system();

#ifdef QUIVER_LIGHT_BREAKDOWN
            int64_t breakdown_t_compute_start = 0;
            if (tid == 0 && do_tech) {
                breakdown_t_compute_start = globaltimer_ns();
                if (breakdown_t_compute_start > breakdown_t_ready_found) {
                    breakdown_trace->resumption_delay_ns +=
                        breakdown_t_compute_start - breakdown_t_ready_found;
                }
            }
#endif

            for (int w = 0; w < W; w++) {
                if (snap_st[w] != IO_READY) continue;

                if (tid == 0) my_ctl[w].status = IO_IDLE;
#ifdef QUIVER_LIGHT_BREAKDOWN
                int64_t breakdown_t_lane_compute_start = 0;
                if (tid == 0 && do_tech)
                    breakdown_t_lane_compute_start = globaltimer_ns();
#endif

#ifdef FLASHANNS_LATENCY_PROBE
                int64_t probe_t_lane_merge_start = globaltimer_ns();
#endif
                persistent_merge_data(
                    my_buf + w * shared::PAGE_SIZE, snap_nd[w],
                    args.num_chunks, my_pq, args.compressed_data,
                    args.nodes_per_page, args.node_size, args.data_size,
                    args.max_m, args.ef_search,
                    my_nid, my_ndist, my_ctx,
                    mv_pos, mv_id, mv_dist, tmp_id, tmp_dist
#ifdef FLASHANNS_LATENCY_PROBE
                    , &probe_pq_done_ts
#endif
                    );
#ifdef FLASHANNS_LATENCY_PROBE
                int64_t probe_t_lane_merge_end = globaltimer_ns();
                if (tid == 0) {
                    probe_pipe_pq_ns += (probe_pq_done_ts - probe_t_lane_merge_start);
                    probe_pipe_merge_ns += (probe_t_lane_merge_end - probe_pq_done_ts);
                }
#endif

                if (tid < 32)
                    lane_req[w] = persistent_visit_select<T>(
                        qdata, my_buf + w * shared::PAGE_SIZE, snap_nd[w],
                        args.num_dims, args.topk,
                        my_nns, my_dist, my_fc,
                        args.nodes_per_page, args.node_size,
                        args.max_m, args.ef_search,
                        my_nid, my_ndist, my_ctx);
                __syncthreads();

#ifdef QUIVER_LIGHT_BREAKDOWN
                if (tid == 0 && do_tech) {
                    flash_breakdown_add_compute_interval(
                        breakdown_trace, breakdown_t_lane_compute_start,
                        globaltimer_ns());
                }
#endif

#ifdef FLASHANNS_LATENCY_PROBE
                if (tid == 0) probe_pipe_sort_ns += globaltimer_ns() - probe_t_lane_merge_end;
#endif
            }

#ifdef FLASHANNS_LATENCY_PROBE
            int64_t probe_t_issue_start = globaltimer_ns();
#endif

            if (tid == 0) {
                for (int w = 0; w < W; w++) {
                    if (snap_st[w] == IO_READY && lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                        if (do_tech)
                            s_breakdown_lane_issue_ts[w] = globaltimer_ns();
#endif
                        issue_request(&my_ctl[w], lane_req[w]);
                    }
                }
            }

            if (tid < 32) {
                for (int w = 0; w < W; w++) {
                    if (snap_st[w] != IO_READY &&
                        *(volatile int32_t*)&my_ctl[w].status == IO_IDLE) {
                        lane_req[w] = persistent_select_next(
                            args.max_m, args.ef_search,
                            my_nid, my_ndist, my_ctx);
                    }
                }
            }
            __syncthreads();

            if (tid == 0) {
                for (int w = 0; w < W; w++) {
                    if (snap_st[w] != IO_READY &&
                        my_ctl[w].status == IO_IDLE &&
                        lane_req[w] != -1) {
#ifdef QUIVER_LIGHT_BREAKDOWN
                        if (do_tech)
                            s_breakdown_lane_issue_ts[w] = globaltimer_ns();
#endif
                        issue_request(&my_ctl[w], lane_req[w]);
                    }
                }
            }
            __syncthreads();

#ifdef FLASHANNS_LATENCY_PROBE
            if (tid == 0) probe_pipe_issue_ns += globaltimer_ns() - probe_t_issue_start;
            int64_t probe_t_issue_done = globaltimer_ns();
            if (tid == 0 && probe_hop_count < FLASH_PROBE_MAX_HOPS) {
                FlashProbeHop& h = args.d_probe_traces[qid].hops[probe_hop_count];
                h.t_wait_start = probe_t_wait_start;
                h.t_ready_found = probe_t_ready_found;
                h.t_compute_done = probe_t_issue_start;
                h.t_issue_done = probe_t_issue_done;
                h.ready_lanes = probe_ready_count;
                h.spin_iters = probe_spin_since_last_ready;
            }
            if (tid == 0) {
                probe_hop_count++;
                probe_spin_since_last_ready = 0;
            }
            probe_t_last_compute_end = probe_t_issue_done;
#endif
        } // pipe loop

#ifdef FLASHANNS_LATENCY_PROBE
        int64_t probe_t_pipe_done = globaltimer_ns();
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
        breakdown_cta_finish_window(globaltimer_ns());
#endif

        // ---- FINISH: write results ----
        for (int i = tid; i < args.topk; i += blockDim.x) {
            args.d_out_nns[qid * args.topk + i]       = my_nns[i];
            args.d_out_distances[qid * args.topk + i]  = my_dist[i];
        }
        if (tid == 0) {
            args.d_out_found_cnt[qid] = my_fc[0];
            atomicAdd(args.d_queries_done, 1);
        }
        __syncthreads();

#ifdef QUIVER_LIGHT_BREAKDOWN
        if (tid == 0 && do_tech) {
            breakdown_trace->query_done_ns = globaltimer_ns();
        }
#endif

#ifdef FLASHANNS_LATENCY_PROBE
        if (tid == 0) {
            FlashProbeQueryTrace& tr = args.d_probe_traces[qid];
            tr.query_id         = qid;
            tr.num_hops         = min(probe_hop_count, FLASH_PROBE_MAX_HOPS);
            tr.t_query_start    = probe_t_query_start;
            tr.t_init_done      = probe_t_init_done;
            tr.t_boot_io_done   = probe_t_boot_io_done;
            tr.t_boot_pq_done   = probe_t_boot_pq_done;
            tr.t_boot_merge_done = probe_t_boot_merge_done;
            tr.t_boot_sort_done = probe_t_boot_sort_done;
            tr.t_boot_fill_done = probe_t_boot_fill_done;
            tr.t_pipe_done      = probe_t_pipe_done;
            tr.t_query_done     = globaltimer_ns();
            tr.pipe_io_ns       = probe_pipe_io_ns;
            tr.pipe_pq_ns       = probe_pipe_pq_ns;
            tr.pipe_merge_ns    = probe_pipe_merge_ns;
            tr.pipe_sort_ns     = probe_pipe_sort_ns;
            tr.pipe_issue_ns    = probe_pipe_issue_ns;
        }
#endif
    } // single-query scope
}

} // namespace flashanns
