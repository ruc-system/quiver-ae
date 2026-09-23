#pragma once

#include <cstdint>
#include "../shared/common/light_breakdown.hpp"
#include "shared/common/search_state.cuh"
#include "shared/index/pq_search.hpp"
#include "shared/io/loader.hpp"

namespace flashanns {

enum IOStatus : int32_t {
    IO_IDLE      = 0,
    IO_REQUESTED = 1,
    IO_READY     = 2,
};

struct __align__(64) IOControl {
    volatile int32_t status;
    int32_t node_id;
    int32_t pad[14];
};

struct __align__(64) PersistentKernelComm {
    volatile int32_t next_query;
    volatile int32_t queries_completed;
    int32_t total_queries;  // FlashANNS: size of the CURRENT batch (= num_blocks)
                            //            — reset by the host each launch.
                            //            Not the total number of queries in the
                            //            whole search.
    int32_t num_blocks;
    int32_t pipe_w;
    int32_t pad[11];

    IOControl* controls;   // [num_blocks * pipe_w], pinned+mapped
    uint8_t*   buffers;    // [num_blocks * pipe_w * PAGE_SIZE], DMA-able+registered
};

struct PersistentKernelArgs {
    PersistentKernelComm* comm;

    int*     d_next_query;     // unused in FlashANNS (qid = blockIdx.x); kept
                               //   only for PersistentKernelArgs ABI compat.
    int*     d_queries_done;   // device-memory completion counter

    float*   d_all_qdata;      // [total_queries * num_dims], device
    int32_t* d_entry_nodes;    // [total_queries], device (nullptr → use enter_point)
    int32_t  enter_point;

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

    uint32_t* d_neighbors_id;    // [N * aligned_ef]
    float*    d_neighbors_dist;  // [N * aligned_ef]
    shared::Data* d_ctx;         // [N]
    int*      d_nns;             // [N * topk]
    float*    d_distances;       // [N * topk]
    int*      d_found_cnt;       // [N]

    int*   d_out_nns;            // [total_queries * topk]
    float* d_out_distances;      // [total_queries * topk]
    int*   d_out_found_cnt;      // [total_queries]

#ifdef FLASHANNS_LATENCY_PROBE
    struct FlashProbeQueryTrace* d_probe_traces;  // [num_queries], per-query timestamp trace
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
    struct FlashLightBreakdownTrace* d_breakdown_traces;  // [breakdown_sample_count]
    struct FlashBreakdownCtaWindowTrace* d_breakdown_cta_windows;  // [batch_size]
    int breakdown_sample_count;
    int breakdown_cta_window_count;
    int64_t breakdown_cta_window_ns;
    int total_search_queries;
    int batch_query_offset;
#endif
};

// ================================================================
// Latency probe diagnostics (compile-time gated)
//
// Per-query timestamp trace (globaltimer_ns based).
// Boot phase: absolute timestamps at stage boundaries.
// Pipe phase: accumulated durations (ns) across all hops.
// ================================================================
#if defined(FLASHANNS_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

#endif

#ifdef FLASHANNS_LATENCY_PROBE

static constexpr int FLASH_PROBE_MAX_HOPS = 64;

struct FlashProbeHop {
    int64_t t_wait_start;
    int64_t t_ready_found;
    int64_t t_compute_done;
    int64_t t_issue_done;
    int32_t ready_lanes;
    int32_t spin_iters;
};

struct FlashProbeQueryTrace {
    int32_t query_id;
    int32_t num_hops;
    int64_t t_query_start;
    int64_t t_init_done;
    int64_t t_boot_io_done;
    int64_t t_boot_pq_done;
    int64_t t_boot_merge_done;
    int64_t t_boot_sort_done;
    int64_t t_boot_fill_done;
    int64_t t_pipe_done;
    int64_t t_query_done;
    int64_t pipe_io_ns;
    int64_t pipe_pq_ns;
    int64_t pipe_merge_ns;
    int64_t pipe_sort_ns;
    int64_t pipe_issue_ns;
    FlashProbeHop hops[FLASH_PROBE_MAX_HOPS];
};

#endif // FLASHANNS_LATENCY_PROBE

#ifdef QUIVER_LIGHT_BREAKDOWN
static constexpr int FLASH_BREAKDOWN_MAX_COMPUTE_INTERVALS = 128;
static constexpr int FLASH_BREAKDOWN_MAX_WAIT_INTERVALS = 256;

struct FlashLightBreakdownTrace {
    int32_t query_id;
    int32_t valid;
    int32_t compute_interval_count;
    int32_t wait_interval_count;
    int64_t query_start_ns;
    int64_t query_done_ns;
    int64_t useful_compute_ns;
    int64_t resumption_delay_ns;
    int64_t compute_start_ns[FLASH_BREAKDOWN_MAX_COMPUTE_INTERVALS];
    int64_t compute_end_ns[FLASH_BREAKDOWN_MAX_COMPUTE_INTERVALS];
    int64_t wait_start_ns[FLASH_BREAKDOWN_MAX_WAIT_INTERVALS];
    int64_t wait_end_ns[FLASH_BREAKDOWN_MAX_WAIT_INTERVALS];
};

struct FlashBreakdownCtaWindowTrace {
    int32_t valid;
    int32_t block_id;
    int32_t trigger_query_id;
    int32_t _pad0;
    int64_t window_start_ns;
    int64_t window_end_ns;
    int64_t pure_io_wait_ns;
    int64_t _pad1;
};

__device__ __forceinline__ void flash_breakdown_add_compute_interval(
    FlashLightBreakdownTrace* tr, int64_t start_ns, int64_t end_ns)
{
    if (tr == nullptr || end_ns <= start_ns) return;
    tr->useful_compute_ns += end_ns - start_ns;
    const int idx = tr->compute_interval_count;
    if (idx < FLASH_BREAKDOWN_MAX_COMPUTE_INTERVALS) {
        tr->compute_start_ns[idx] = start_ns;
        tr->compute_end_ns[idx] = end_ns;
    }
    tr->compute_interval_count = idx + 1;
}

__device__ __forceinline__ void flash_breakdown_add_wait_interval(
    FlashLightBreakdownTrace* tr, int64_t start_ns, int64_t end_ns)
{
    if (tr == nullptr || end_ns <= start_ns) return;
    const int idx = tr->wait_interval_count;
    if (idx < FLASH_BREAKDOWN_MAX_WAIT_INTERVALS) {
        tr->wait_start_ns[idx] = start_ns;
        tr->wait_end_ns[idx] = end_ns;
    }
    tr->wait_interval_count = idx + 1;
}
#endif

} // namespace flashanns
