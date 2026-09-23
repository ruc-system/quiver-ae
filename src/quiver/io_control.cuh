#pragma once

#include <cstdint>
#include "../shared/common/light_breakdown.hpp"
#include "../shared/common/search_state.cuh"
#include "../shared/index/pq_search.hpp"
#include "../shared/io/loader.hpp"

namespace quiver {

enum IOStatus : int32_t {
    IO_IDLE      = 0,
    IO_REQUESTED = 1,
    IO_READY     = 2,
    IO_SUBMITTED = 3,  // CPU has submitted to SPDK, waiting for callback
};

struct __align__(64) IOControl {
    volatile int32_t status;
    int32_t node_id;
    int32_t pad[14];
};

struct __align__(64) PersistentKernelComm {
    // _pad0: DO NOT REMOVE. Keeps queries_completed at offset=4 within the
    // cacheline. Moving it to offset=0 measurably degrades GPU->CPU PCIe
    // visibility latency for the polling loop in run_persistent_search.
    int32_t _pad0;
    volatile int32_t queries_completed;
    int32_t total_queries;       // total queries in dataset (for output sizing)
    int32_t num_blocks;
    int32_t pipe_w;
    volatile int32_t feed_count; // CPU writes: how many queries GPU is allowed to process
    volatile int32_t terminate;  // CPU writes: 1 = all done, GPU should exit
    int32_t pad[9];

    IOControl* controls;   // [num_blocks * queries_per_block * pipe_w], pinned+mapped
    uint8_t*   buffers;    // [num_blocks * queries_per_block * pipe_w * PAGE_SIZE], DMA-able+registered
};

enum class EarlyExitPolicy : int {
    kDefault = 0,
    kDarth = 1,
    kGPruning = 2,
    kDarthTrace = 3,
};

static constexpr int DARTH_FEATURE_TOPK_MAX = 32;

struct __align__(128) DarthCommSlot {
    volatile int32_t request_epoch;
    volatile int32_t decision_epoch;
    volatile int32_t stop;
    volatile int32_t request_count;
    volatile long long wait_iters_total;
    int32_t qid;
    int32_t step;
    int32_t found_cnt;
    int32_t no_improve_count;
    float top1_dist;
    float topk_dist;
    float gap_ratio;
    float predicted_recall;
    int32_t top_ids[DARTH_FEATURE_TOPK_MAX];
};

struct PersistentKernelArgs {
    PersistentKernelComm* comm;

    int*     d_next_query;     // device-memory atomic counter (avoids PCIe atomics)
    int*     d_queries_done;   // device-memory completion counter

    float*   d_all_qdata;      // [total_queries * num_dims], device
    int32_t* d_entry_nodes;    // [total_queries], device (nullptr → use enter_point or in-kernel nav)
    int32_t  enter_point;

    // In-kernel nav graph view. When nav_data_dev != nullptr, each slot
    // computes its own entry point inside the persistent kernel during
    // try_bind_new_query, replacing the pre-kernel get_entry_kernel +
    // nav_translate_kernel launches (which cost a launch per CPU feed
    // batch).  When nav_data_dev == nullptr, kernel falls back to
    // d_entry_nodes (if set) or args.enter_point (global root).
    //
    // Data type of nav_data_dev matches the base graph (UINT8 or FLOAT),
    // carried via the kernel template parameter T.
    uint8_t* nav_data_dev;     // device pointer (nullptr = in-kernel nav disabled)
    int32_t* nav_graph_dev;    // [num_node * nav_max_m] adjacency
    int32_t* nav_mapping_dev;  // [num_node] nav->base node id translation
    int      nav_num_node;
    int      nav_data_len;     // dims used by nav (typically == num_dims)
    int      nav_max_m;
    int      nav_start;        // nav graph root
    int      nav_init_ef;      // initial ef (== min(ef_search, 5))

    shared::PQSearchData* pq_data;  // device pointer (for init_query)
    int      num_chunks;
    float*   pq_dists;         // device: [N * 256 * num_chunks], N=num_blocks*queries_per_block
    uint8_t* compressed_data;  // device: PQ compressed vectors

    int nodes_per_page;
    int node_size;
    int data_size;
    int num_dims;
    int max_m;
    int ef_search;
    int aligned_ef;
    int topk;
    int pipe_w;
    int queries_per_block;     // Q: multi-query interleaving per block
    int num_blocks;            // actual GPU block count
    int pq_block_offset;       // offset into shared PQ tables (for multi-runner)
    int total_queries;         // cached copy of comm->total_queries (avoids PCIe read races)
    int batch_offset;          // StaticDispatch: qid = batch_offset + blockIdx.x.
                               // Unused (0) for StreamingDispatch. Updated by host
                               // between per-batch kernel launches in static mode.

    uint32_t* d_neighbors_id;    // [N * aligned_ef]
    float*    d_neighbors_dist;  // [N * aligned_ef]
    shared::Data* d_ctx;         // [N]
    int*      d_nns;             // [N * topk]
    float*    d_distances;       // [N * topk]
    int*      d_found_cnt;       // [N]

    int*   d_out_nns;            // [total_queries * topk]
    float* d_out_distances;      // [total_queries * topk]
    int*   d_out_found_cnt;      // [total_queries]

    const int* d_step_recall_gt;  // [step_recall_gt_count * step_recall_gt_width], optional
    int step_recall_gt_width;
    int step_recall_gt_count;
    int step_recall_required_hits;
    int* d_step_recall_step;      // [total_queries], -1 when threshold is not reached
    int* d_step_recall_hits;      // [total_queries], final or threshold-reaching hits

    int early_exit_policy;          // EarlyExitPolicy encoded as int; kDefault preserves baseline.
    int early_exit_darth_interval;  // DARTH-style predictor invocation interval in steps.
    float early_exit_darth_threshold;
    int early_exit_gpruning_warmup_steps;
    int early_exit_gpruning_patience;
    DarthCommSlot* darth_comm_slots;  // [num_blocks], mapped pinned, optional
    int* d_early_exit_stop_step;      // [total_queries], -1 when not early-stopped
    int* d_finish_step;               // [total_queries], natural or early-stop finish step

#ifdef QUIVER_LIGHT_BREAKDOWN
    struct QuiverLightBreakdownTrace* d_breakdown_traces;
    struct QuiverBreakdownCtaWindowTrace* d_breakdown_cta_windows;
    int breakdown_sample_count;
    int breakdown_cta_window_count;
    int64_t breakdown_cta_window_ns;
#endif

#ifdef QUIVER_LATENCY_PROBE
    struct QuiverProbeQueryTrace* d_probe_traces;       // [probe_max_traces]
    int probe_max_traces;
    struct QuiverProbeBlockTrace* d_probe_block_traces; // [probe_max_block_traces]
    int probe_max_block_traces;
    int probe_block_trace_offset;
#endif
};

#ifdef QUIVER_LIGHT_BREAKDOWN

static constexpr int QUIVER_LIGHT_BREAKDOWN_MAX_COMPUTE_INTERVALS = 128;
static constexpr int QUIVER_LIGHT_BREAKDOWN_MAX_WAIT_INTERVALS = 256;
static constexpr int QUIVER_LIGHT_BREAKDOWN_MAX_OVERLAP_INTERVALS = 256;

__host__ __device__ __forceinline__ int quiver_breakdown_sample_query_id(
    int sample_idx, int total_queries, int sample_count) {
    return shared::breakdown::sample_query_id(
        sample_idx, total_queries, sample_count);
}

__host__ __device__ __forceinline__ int quiver_breakdown_sample_index(
    int qid, int total_queries, int sample_count) {
    return shared::breakdown::sample_index(qid, total_queries, sample_count);
}

struct QuiverLightBreakdownTrace {
    int32_t query_id;
    int32_t valid;
    int32_t compute_interval_count;
    int32_t wait_interval_count;
    int32_t overlap_interval_count;
    int32_t _pad;
    int64_t query_start_ns;
    int64_t query_done_ns;
    int64_t useful_compute_ns;
    int64_t resumption_delay_ns;
    int64_t ssd_wait_ns;
    int64_t compute_start_ns[QUIVER_LIGHT_BREAKDOWN_MAX_COMPUTE_INTERVALS];
    int64_t compute_end_ns[QUIVER_LIGHT_BREAKDOWN_MAX_COMPUTE_INTERVALS];
    int64_t wait_start_ns[QUIVER_LIGHT_BREAKDOWN_MAX_WAIT_INTERVALS];
    int64_t wait_end_ns[QUIVER_LIGHT_BREAKDOWN_MAX_WAIT_INTERVALS];
    int64_t overlap_start_ns[QUIVER_LIGHT_BREAKDOWN_MAX_OVERLAP_INTERVALS];
    int64_t overlap_end_ns[QUIVER_LIGHT_BREAKDOWN_MAX_OVERLAP_INTERVALS];
};

struct QuiverBreakdownCtaWindowTrace {
    int32_t valid;
    int32_t block_id;
    int32_t trigger_query_id;
    int32_t query_count_delta;
    int64_t window_start_ns;
    int64_t window_end_ns;
    int64_t pure_io_wait_ns;
    int64_t _pad;
};

__device__ __forceinline__ int64_t quiver_breakdown_compute_overlap_ns(
    const QuiverLightBreakdownTrace* tr, int64_t start_ns, int64_t end_ns) {
    if (tr == nullptr || end_ns <= start_ns) return 0;
    int64_t overlap = 0;
    const int n =
        tr->compute_interval_count <
                QUIVER_LIGHT_BREAKDOWN_MAX_COMPUTE_INTERVALS
            ? tr->compute_interval_count
            : QUIVER_LIGHT_BREAKDOWN_MAX_COMPUTE_INTERVALS;
    for (int i = 0; i < n; ++i) {
        const int64_t s =
            tr->compute_start_ns[i] > start_ns ? tr->compute_start_ns[i]
                                               : start_ns;
        const int64_t e =
            tr->compute_end_ns[i] < end_ns ? tr->compute_end_ns[i] : end_ns;
        if (e > s) overlap += e - s;
    }
    return overlap;
}

__device__ __forceinline__ void quiver_breakdown_add_compute_interval(
    QuiverLightBreakdownTrace* tr, int64_t start_ns, int64_t end_ns) {
    if (tr == nullptr || end_ns <= start_ns) return;
    tr->useful_compute_ns += end_ns - start_ns;
    const int idx = tr->compute_interval_count;
    if (idx < QUIVER_LIGHT_BREAKDOWN_MAX_COMPUTE_INTERVALS) {
        tr->compute_start_ns[idx] = start_ns;
        tr->compute_end_ns[idx] = end_ns;
    }
    tr->compute_interval_count = idx + 1;
}

__device__ __forceinline__ void quiver_breakdown_add_ssd_wait_interval(
    QuiverLightBreakdownTrace* tr, int64_t issue_ns, int64_t ready_ns) {
    if (tr == nullptr || ready_ns <= issue_ns) return;
    const int idx = tr->wait_interval_count;
    if (idx < QUIVER_LIGHT_BREAKDOWN_MAX_WAIT_INTERVALS) {
        tr->wait_start_ns[idx] = issue_ns;
        tr->wait_end_ns[idx] = ready_ns;
    }
    tr->wait_interval_count = idx + 1;
}

__device__ __forceinline__ void quiver_breakdown_add_overlap_interval(
    QuiverLightBreakdownTrace* tr, int64_t start_ns, int64_t end_ns) {
    if (tr == nullptr || end_ns <= start_ns) return;
    const int idx = tr->overlap_interval_count;
    if (idx < QUIVER_LIGHT_BREAKDOWN_MAX_OVERLAP_INTERVALS) {
        tr->overlap_start_ns[idx] = start_ns;
        tr->overlap_end_ns[idx] = end_ns;
    }
    tr->overlap_interval_count = idx + 1;
}

#endif  // QUIVER_LIGHT_BREAKDOWN

// ================================================================
// Latency probe diagnostics (compile-time gated)
// ================================================================
#ifdef QUIVER_LATENCY_PROBE

static constexpr int PROBE_MAX_QUERY_TRACES = 20000;
static constexpr int PROBE_MAX_HOPS = 64;

struct QuiverProbeHop {
    int64_t ready_found_ts;
    int64_t compute_start_ts;
    int64_t compute_done_ts;
    int32_t ready_lanes;
    int32_t spin_iters;
};

struct QuiverProbeQueryTrace {
    int32_t query_id;
    int32_t num_hops;
    int64_t t_sched_start;
    int64_t t_query_start;
    int64_t t_nav_done;
    int64_t t_pq_done;
    int64_t t_boot_issue;
    int64_t t_boot_ready;
    int64_t t_boot_pq_done;
    int64_t t_boot_merge_done;
    int64_t t_boot_compute_done;
    int64_t t_boot_all_issued;
    int64_t t_pipe_start;
    int64_t t_pipe_done;
    int64_t t_query_done;
    int32_t boot_spin_iters;
    int32_t _pad;
    QuiverProbeHop hops[PROBE_MAX_HOPS];
};

struct QuiverProbeBlockTrace {
    int32_t block_id;
    int32_t query_count;
    int64_t t_block_start;
    int64_t t_block_done;
    int64_t no_ready_wait_ns;
    int32_t no_ready_spins;
    int32_t _pad;
};

#endif // QUIVER_LATENCY_PROBE

} // namespace quiver
