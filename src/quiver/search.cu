#include "search.cuh"
#include "kernel.cuh"
#include "io_control.cuh"
#include "../shared/common/cuda_utils.cuh"
#include "../shared/common/logging.hpp"
#include "../shared/common/runtime.hpp"
#include "../shared/common/run_report.hpp"
#include "../shared/common/search_state.cuh"
#ifdef QUIVER_LIGHT_BREAKDOWN
#include "../shared/common/ae_csv.hpp"
#include "../shared/common/light_breakdown.hpp"
#endif
#include "../shared/index/nav_kernel.cuh"
#include "../shared/io/runtime_stats.hpp"
#include "../../bin/darth_predictor_generated.hpp"

#ifdef QUIVER_LATENCY_PROBE
#include "probe_report.cuh"
#endif

#include <thread>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <vector>
#include <nvtx3/nvToolsExt.h>

namespace quiver {

#ifdef QUIVER_LIGHT_BREAKDOWN
static int64_t quiver_breakdown_cta_window_ns_from_env() {
    const char* value = std::getenv("QUIVER_BREAKDOWN_CTA_WINDOW_US");
    const long long us =
        value != nullptr ? std::strtoll(value, nullptr, 10) : 1000;
    return std::max(1LL, us) * 1000;
}
#endif

struct PollStats {
    std::atomic<long> io_found{0};
    std::atomic<long> poll_iters{0};
#ifdef QUIVER_LATENCY_PROBE
    std::mutex io_latency_mu;
    std::vector<double> end_to_end_io_us;
#endif
};

#ifdef QUIVER_LATENCY_PROBE
static inline int64_t host_steady_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static int quiver_io_sample_cases_from_env() {
    const char* env = std::getenv("QUIVER_IO_SAMPLE_CASES");
    if (env == nullptr || env[0] == '\0') return 256;
    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || value < 0) return 256;
    return static_cast<int>(value);
}

static bool should_sample_batch_case(int batch_idx, int num_batches,
                                     int sample_cases) {
    if (sample_cases <= 0 || num_batches <= 0) return false;
    if (sample_cases >= num_batches) return true;
    return ((batch_idx + 1) * sample_cases / num_batches) >
           (batch_idx * sample_cases / num_batches);
}
#endif

static void print_batch_latency_summary(const char* label,
                                        const std::vector<double>& latencies_ms) {
    if (latencies_ms.empty()) {
        std::cout << "[" << label << "] BatchLatency(ms): no samples\n";
        return;
    }
    std::vector<double> sorted = latencies_ms;
    std::sort(sorted.begin(), sorted.end());
    double avg = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double p90 = sorted[sorted.size() * 90 / 100];
    double p99 = sorted[sorted.size() * 99 / 100];
    double max = sorted.back();
    std::cout << "[" << label << "] BatchLatency(ms): count=" << sorted.size()
              << " avg=" << std::fixed << std::setprecision(3) << avg
              << " p90=" << p90
              << " p99=" << p99
              << " max=" << max << "\n";
}

static int required_hits_for_threshold(float threshold, int topk) {
    return std::max(1, std::min(topk, (int)std::ceil(threshold * topk - 1e-6f)));
}

static double percentile_sorted(const std::vector<int>& sorted, double pct) {
    if (sorted.empty()) return 0.0;
    size_t idx = (size_t)(sorted.size() * pct / 100.0);
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return (double)sorted[idx];
}

static void write_step_recall_csv(
    const PerStepRecallOptions& options,
    const std::vector<int>& steps,
    const std::vector<int>& hits,
    int topk) {
    std::ofstream out(options.csv_path);
    if (!out.is_open()) {
        ERROR("Failed to open step recall CSV {}", options.csv_path);
        std::exit(1);
    }
    out << "query_id,step,hits,recall,reached\n";
    int reached = 0;
    std::vector<int> reached_steps;
    reached_steps.reserve(steps.size());
    for (size_t q = 0; q < steps.size(); ++q) {
        const bool ok = steps[q] >= 0;
        if (ok) {
            ++reached;
            reached_steps.push_back(steps[q]);
        }
        const double recall = (double)hits[q] / (double)topk;
        out << q << "," << steps[q] << "," << hits[q] << ","
            << std::fixed << std::setprecision(6) << recall << ","
            << (ok ? 1 : 0) << "\n";
    }
    out.close();

    std::sort(reached_steps.begin(), reached_steps.end());
    std::cout << "[Quiver] Step recall CSV: " << options.csv_path << "\n"
              << "[Quiver] Step recall reached=" << reached
              << " unreached=" << (steps.size() - (size_t)reached)
              << " threshold=" << options.threshold
              << " required_hits=" << required_hits_for_threshold(options.threshold, topk)
              << std::endl;
    if (!reached_steps.empty()) {
        std::cout << "[Quiver] Step recall steps p50="
                  << percentile_sorted(reached_steps, 50)
                  << " p90=" << percentile_sorted(reached_steps, 90)
                  << " p99=" << percentile_sorted(reached_steps, 99)
                  << " max=" << reached_steps.back() << std::endl;
    }
}

static int count_feature_gt_hits(
    const DarthCommSlot& feature,
    int topk,
    const int* gt,
    int gt_width) {
    int hits = 0;
    const int k = std::min(topk, DARTH_FEATURE_TOPK_MAX);
    for (int i = 0; i < k; ++i) {
        const int id = feature.top_ids[i];
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

// ================================================================
// CPU poll loop: direct-submit mode
// SPDK callback writes IO_READY directly to IOControl.status,
// eliminating the poll_task() → write IO_READY roundtrip.
// ================================================================
static void cpu_poll_loop(
    int tid, int slot_start, int slot_end,
    IOControl* controls, uint8_t* buffers,
    int nodes_per_page,
    std::shared_ptr<shared::IndexLoader> loader,
    int poll_submit_batch_size,
    std::atomic<bool>& done,
#ifdef QUIVER_LATENCY_PROBE
    std::atomic<bool>& sample_current_batch,
#endif
    PollStats& stats)
{
    int n = slot_end - slot_start;
    std::vector<shared::DirectIoRequest> req_buf;
    std::vector<int> idx_buf;
    req_buf.reserve(poll_submit_batch_size);
    idx_buf.reserve(poll_submit_batch_size);
#ifdef QUIVER_LATENCY_PROBE
    std::vector<int> sampled_idx_buf;
    sampled_idx_buf.reserve(poll_submit_batch_size);
    std::vector<int64_t> submit_ns(n, 0);
    std::vector<int64_t> complete_ns(n, 0);
#endif

    long local_found = 0, local_iters = 0;

    while (!done.load(std::memory_order_relaxed)) {
        local_iters++;
        req_buf.clear();
        idx_buf.clear();
#ifdef QUIVER_LATENCY_PROBE
        sampled_idx_buf.clear();
#endif
        for (int i = 0; i < n; i++) {
            int gi = slot_start + i;
            int32_t st = __atomic_load_n(&controls[gi].status, __ATOMIC_ACQUIRE);
#ifdef QUIVER_LATENCY_PROBE
            const bool sample_io = sample_current_batch.load(
                std::memory_order_relaxed);
            const int64_t observed_complete_ns =
                __atomic_load_n(&complete_ns[i], __ATOMIC_ACQUIRE);
            if (observed_complete_ns > 0 && submit_ns[i] > 0) {
                {
                    std::lock_guard<std::mutex> lock(stats.io_latency_mu);
                    stats.end_to_end_io_us.push_back(
                        static_cast<double>(observed_complete_ns - submit_ns[i]) /
                        1000.0);
                }
                submit_ns[i] = 0;
                __atomic_store_n(&complete_ns[i], 0, __ATOMIC_RELEASE);
            }
#endif
            if (st == IO_REQUESTED) {
                int32_t expected = IO_REQUESTED;
                if (!__atomic_compare_exchange_n(&controls[gi].status,
                        &expected, (int32_t)IO_SUBMITTED,
                        /*weak=*/false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                    continue;

                local_found++;
                int blk = controls[gi].node_id / nodes_per_page;
#ifdef QUIVER_LATENCY_PROBE
                volatile int64_t* sampled_complete_ns =
                    sample_io ? &complete_ns[i] : nullptr;
#endif
                req_buf.push_back({
                    blk,
                    buffers + (int64_t)gi * shared::PAGE_SIZE,
                    &controls[gi].status,
                    (int32_t)IO_READY
#ifdef QUIVER_LATENCY_PROBE
                    ,
                    sampled_complete_ns
#endif
                });
                idx_buf.push_back(i);
#ifdef QUIVER_LATENCY_PROBE
                if (sampled_complete_ns != nullptr) {
                    sampled_idx_buf.push_back(i);
                }
#endif
                if ((int)req_buf.size() >= poll_submit_batch_size) {
#ifdef QUIVER_LATENCY_PROBE
                    const int64_t now_ns = host_steady_now_ns();
                    for (int li : sampled_idx_buf) {
                        submit_ns[li] = now_ns;
                        __atomic_store_n(&complete_ns[li], 0,
                                         __ATOMIC_RELEASE);
                    }
#endif
                    loader->submit_direct(req_buf, tid);
                    req_buf.clear();
                    idx_buf.clear();
#ifdef QUIVER_LATENCY_PROBE
                    sampled_idx_buf.clear();
#endif
                }
            }
        }
        if (!req_buf.empty()) {
#ifdef QUIVER_LATENCY_PROBE
            const int64_t now_ns = host_steady_now_ns();
            for (int li : sampled_idx_buf) {
                submit_ns[li] = now_ns;
                __atomic_store_n(&complete_ns[li], 0, __ATOMIC_RELEASE);
            }
#endif
            loader->submit_direct(req_buf, tid);
        }
    }
#ifdef QUIVER_LATENCY_PROBE
    for (int i = 0; i < n; ++i) {
        const int64_t observed_complete_ns =
            __atomic_load_n(&complete_ns[i], __ATOMIC_ACQUIRE);
        if (observed_complete_ns > 0 && submit_ns[i] > 0) {
            std::lock_guard<std::mutex> lock(stats.io_latency_mu);
            stats.end_to_end_io_us.push_back(
                static_cast<double>(observed_complete_ns - submit_ns[i]) /
                1000.0);
        }
    }
#endif
    stats.io_found.fetch_add(local_found);
    stats.poll_iters.fetch_add(local_iters);
}

static void wait_static_batch_io_quiescent(IOControl* h_controls, int num_io_slots)
{
    while (true) {
        bool pending = false;
        for (int i = 0; i < num_io_slots; i++) {
            int32_t status = __atomic_load_n(&h_controls[i].status, __ATOMIC_ACQUIRE);
            if (status == IO_REQUESTED || status == IO_SUBMITTED) {
                pending = true;
                break;
            }
        }
        if (!pending) {
            return;
        }
        std::this_thread::yield();
    }
}

static void reset_static_batch_state(
    PersistentKernelComm* h_comm,
    IOControl* h_controls,
    int num_io_slots,
    int batch_query_limit,
    int num_blocks,
    int pipe_w,
    int* d_queries_done)
{
    wait_static_batch_io_quiescent(h_controls, num_io_slots);
    CHECK_CUDA(cudaMemset(d_queries_done, 0, sizeof(int)));
    for (int i = 0; i < num_io_slots; i++) {
        h_controls[i].node_id = -1;
        __atomic_store_n(&h_controls[i].status, (int32_t)IO_IDLE, __ATOMIC_RELEASE);
    }

    h_comm->total_queries      = batch_query_limit;
    h_comm->num_blocks         = num_blocks;
    h_comm->pipe_w             = pipe_w;
    h_comm->feed_count         = 0;
    h_comm->terminate          = 0;
    h_comm->queries_completed  = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

static void set_device_int(int* d_value, int value)
{
    CHECK_CUDA(cudaMemcpy(d_value, &value, sizeof(int), cudaMemcpyHostToDevice));
}

// ================================================================
// Public entry point — simple single-stream persistent search
// ================================================================
void run_persistent_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
    int mini_batch,
    std::shared_ptr<shared::IndexLoader> loader,
    shared::PQSearch* pq,
    shared::NavGraph* nav,
    int enter_point,
    uint8_t* starter,
    const float* qdata,
    int num_queries,
    int topk,
    int ef_search,
    int* nns,
    float* distances,
    int* found_cnt,
    int queries_per_block,
    int repeat,
    const PerStepRecallOptions& per_step_recall,
    const EarlyExitOptions& early_exit)
{
#ifdef QUIVER_FIXED_PIPE_WIDTH
    pipe_w = QUIVER_FIXED_PIPE_WIDTH;
#endif
#ifdef QUIVER_FIXED_QUERIES_PER_BLOCK
    queries_per_block = QUIVER_FIXED_QUERIES_PER_BLOCK;
#endif
    int num_dims       = (int)layout.num_dims;
    int max_m          = (int)layout.max_m0;
    int nodes_per_page = (int)layout.nodes_per_page;
    int node_size      = (int)layout.node_size;
    int data_size      = (int)layout.data_size;
    int aligned_ef     = (max_m + ef_search + 31) / 32 * 32;

    // Q-interleave: each block holds `queries_per_block` parallel query slots.
    // Total task slots (per-query state) = num_blocks * Q.
    // Total IO slots (per-lane IO control + DMA buf) = task_slots * pipe_w.
    const int Q = queries_per_block > 0 ? queries_per_block : 1;
    int B = mini_batch > 0 ? mini_batch : num_queries;
    int num_blocks = B;
    int num_task_slots = num_blocks * Q;
    int num_io_slots   = num_task_slots * pipe_w;

    if (early_exit.enabled() &&
        early_exit.policy != EarlyExitPolicy::kGPruning &&
        Q != 1) {
        ERROR("Early-exit policy currently supports Q-interleave only for gpruning");
        std::exit(1);
    }

    int poll_submit_batch_size = 256;
    const char* psb_env = getenv("CPU_POLL_SUBMIT_BATCH_SIZE");
    if (psb_env) poll_submit_batch_size = std::atoi(psb_env);

    // Prefeed depth (env QUIVER_PREFEED): allow CPU to over-feed GPU by this
    // many queries before throttling on completions.  Default 0 keeps strict
    // 1:1 flow control for minimum latency.  Larger values trade latency for
    // throughput when PCIe feed_count visibility delay is the bottleneck.
    int prefeed_depth_cfg = 0;
    const char* pf_env_early = getenv("QUIVER_PREFEED");
    if (pf_env_early) prefeed_depth_cfg = std::atoi(pf_env_early);

    INFO("Persistent kernel: B={}, pipe_w={}, num_blocks={}, Q={}, task_slots={}, io_slots={}, T={}, poll_submit_batch_size={}, prefeed={}",
         B, pipe_w, num_blocks, Q, num_task_slots, num_io_slots,
         thread_cnt, poll_submit_batch_size, prefeed_depth_cfg);

    // ---- 1. PQ init ----
    pq->init_device(num_dims, (int)layout.num_data, num_task_slots, ef_search);

    // ---- 2. IOControl + DMA buffers (single set) ----
    IOControl* h_controls;
    CHECK_CUDA(cudaHostAlloc(&h_controls, sizeof(IOControl) * num_io_slots,
                             cudaHostAllocMapped));
    for (int i = 0; i < num_io_slots; i++) h_controls[i].status = IO_IDLE;

    uint8_t* h_buffers =
        loader->create_buffer((int64_t)shared::PAGE_SIZE * num_io_slots);
    CHECK_CUDA(cudaHostRegister(h_buffers,
                                (size_t)shared::PAGE_SIZE * num_io_slots,
                                cudaHostRegisterDefault));

    // ---- 3. Comm struct (pinned+mapped) ----
    PersistentKernelComm* h_comm;
    CHECK_CUDA(cudaHostAlloc(&h_comm, sizeof(PersistentKernelComm),
                              cudaHostAllocMapped));
    h_comm->pipe_w   = pipe_w;
    h_comm->controls = h_controls;
    h_comm->buffers  = h_buffers;
    h_comm->feed_count = 0;
    h_comm->terminate  = 0;
    h_comm->queries_completed = 0;

    // ---- 4. Device allocations (sized for all queries) ----
    // Note: streaming path computes nav entry inside the persistent kernel
    // (see PersistentKernelArgs::nav_data_dev and inkernel_nav.cuh), so
    // d_entry_nodes is unused here and we don't allocate a d_entry buffer.
    float*    d_qdata;
    uint32_t* d_nid;
    float*    d_ndist;
    shared::Data* d_ctx;
    int*      d_nns_dev;
    float*    d_dist;
    int*      d_fc;
    int*      d_out_nns;
    float*    d_out_dist;
    int*      d_out_fc;
    int*      d_next_query;
    int*      d_queries_done;
    int*      d_step_recall_gt = nullptr;
    int*      d_step_recall_step = nullptr;
    int*      d_step_recall_hits = nullptr;
    int*      d_early_exit_stop_step = nullptr;
    int*      d_finish_step = nullptr;
    DarthCommSlot* h_darth_comm_slots = nullptr;

    CHECK_CUDA(cudaMalloc(&d_qdata, sizeof(float) * num_queries * num_dims));
    CHECK_CUDA(cudaMalloc(&d_nid,   sizeof(uint32_t) * aligned_ef * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_ndist, sizeof(float)    * aligned_ef * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_ctx, sizeof(shared::Data) * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_nns_dev,   sizeof(int)  * topk * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_dist,  sizeof(float)    * topk * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_fc,    sizeof(int)      * num_task_slots));
    CHECK_CUDA(cudaHostAlloc(&d_out_nns,  sizeof(int)   * topk * num_queries, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostAlloc(&d_out_dist, sizeof(float) * topk * num_queries, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostAlloc(&d_out_fc,   sizeof(int)   * num_queries, cudaHostAllocMapped));
    CHECK_CUDA(cudaMalloc(&d_next_query,   sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_queries_done, sizeof(int)));

    CHECK_CUDA(cudaMalloc(&d_finish_step, sizeof(int) * num_queries));
    CHECK_CUDA(cudaMemset(d_finish_step, 0xff, sizeof(int) * num_queries));
    CHECK_CUDA(cudaMalloc(&d_early_exit_stop_step, sizeof(int) * num_queries));
    CHECK_CUDA(cudaMemset(d_early_exit_stop_step, 0xff, sizeof(int) * num_queries));

    if (per_step_recall.enabled()) {
        if (Q != 1 || pipe_w != 1) {
            ERROR("Per-step recall currently supports only queries_per_block=1 and pipe_width=1");
            std::exit(1);
        }
        if (topk > per_step_recall.ground_truth_width) {
            ERROR("Per-step recall topk={} exceeds ground truth width={}",
                  topk, per_step_recall.ground_truth_width);
            std::exit(1);
        }
        if (per_step_recall.threshold <= 0.0f || per_step_recall.threshold > 1.0f) {
            ERROR("Per-step recall threshold must be in (0, 1], got {}",
                  per_step_recall.threshold);
            std::exit(1);
        }
        const size_t gt_elems =
            (size_t)per_step_recall.ground_truth_count *
            (size_t)per_step_recall.ground_truth_width;
        CHECK_CUDA(cudaMalloc(&d_step_recall_gt, sizeof(int) * gt_elems));
        CHECK_CUDA(cudaMemcpy(d_step_recall_gt, per_step_recall.ground_truth,
                              sizeof(int) * gt_elems, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&d_step_recall_step, sizeof(int) * num_queries));
        CHECK_CUDA(cudaMalloc(&d_step_recall_hits, sizeof(int) * num_queries));
        CHECK_CUDA(cudaMemset(d_step_recall_step, 0xff, sizeof(int) * num_queries));
        CHECK_CUDA(cudaMemset(d_step_recall_hits, 0, sizeof(int) * num_queries));
    }

    if (early_exit.policy == EarlyExitPolicy::kDarth ||
        early_exit.policy == EarlyExitPolicy::kDarthTrace) {
        CHECK_CUDA(cudaHostAlloc(&h_darth_comm_slots,
                                 sizeof(DarthCommSlot) * num_blocks,
                                 cudaHostAllocMapped));
        std::memset(h_darth_comm_slots, 0, sizeof(DarthCommSlot) * num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            h_darth_comm_slots[i].stop = 0;
            for (int j = 0; j < DARTH_FEATURE_TOPK_MAX; ++j) {
                h_darth_comm_slots[i].top_ids[j] = -1;
            }
        }
    }

#ifdef QUIVER_LATENCY_PROBE
    QuiverProbeQueryTrace* d_probe_traces = nullptr;
    int probe_num_traces = std::min(num_queries, (int)PROBE_MAX_QUERY_TRACES);
    CHECK_CUDA(cudaMalloc(&d_probe_traces, sizeof(QuiverProbeQueryTrace) * probe_num_traces));
    CHECK_CUDA(cudaMemset(d_probe_traces, 0, sizeof(QuiverProbeQueryTrace) * probe_num_traces));
    QuiverProbeBlockTrace* d_probe_block_traces = nullptr;
#endif

#ifdef QUIVER_LIGHT_BREAKDOWN
    QuiverLightBreakdownTrace* d_breakdown_traces = nullptr;
    QuiverBreakdownCtaWindowTrace* d_breakdown_cta_windows = nullptr;
    const int breakdown_sample_count = std::min(
        num_queries,
        (int)shared::breakdown::QUIVER_LIGHT_BREAKDOWN_SAMPLE_QUERIES);
    const int breakdown_cta_window_count = breakdown_sample_count;
    CHECK_CUDA(cudaMalloc(
        &d_breakdown_traces,
        sizeof(QuiverLightBreakdownTrace) * breakdown_sample_count));
    CHECK_CUDA(cudaMemset(
        d_breakdown_traces, 0,
        sizeof(QuiverLightBreakdownTrace) * breakdown_sample_count));
    CHECK_CUDA(cudaMalloc(
        &d_breakdown_cta_windows,
        sizeof(QuiverBreakdownCtaWindowTrace) * breakdown_cta_window_count));
    CHECK_CUDA(cudaMemset(
        d_breakdown_cta_windows, 0,
        sizeof(QuiverBreakdownCtaWindowTrace) * breakdown_cta_window_count));
#endif

    // ---- 5. Build kernel args ----
    PersistentKernelArgs args{};
    args.comm             = h_comm;
    args.d_next_query     = d_next_query;
    args.d_queries_done   = d_queries_done;
    args.enter_point      = enter_point;
    args.pq_data          = pq->get_device_ptr();
    args.num_chunks       = pq->device_data.num_chunks;
    args.pq_dists         = pq->device_data.pq_dists;
    args.compressed_data  = pq->device_data.compressed_data;
    args.pq_block_offset  = 0;
    args.nodes_per_page   = nodes_per_page;
    args.node_size        = node_size;
    args.data_size        = data_size;
    args.num_dims         = num_dims;
    args.max_m            = max_m;
    args.ef_search        = ef_search;
    args.aligned_ef       = aligned_ef;
    args.topk             = topk;
    args.pipe_w           = pipe_w;
    args.queries_per_block = Q;
    args.d_neighbors_id   = d_nid;
    args.d_neighbors_dist = d_ndist;
    args.d_ctx            = d_ctx;
    args.d_nns            = d_nns_dev;
    args.d_distances      = d_dist;
    args.d_found_cnt      = d_fc;
    args.d_all_qdata      = d_qdata;
    // In-kernel nav: each slot computes its own entry at query-bind time
    // (see inkernel_nav.cuh / kernel_q{1,i}.cuh), so d_entry_nodes is
    // always nullptr and nav_* below carries the nav graph pointers.
    // This eliminates per-feed nav launch overhead (previously
    // O(feed_events) kernel launches on h2d_stream via the old external
    // get_entry_kernel + nav_translate_kernel path).
    args.d_entry_nodes   = nullptr;
    if (nav != nullptr) {
        args.nav_data_dev    = nav->data_dev;
        args.nav_graph_dev   = nav->graph_dev;
        args.nav_mapping_dev = nav->mapping_dev;
        args.nav_num_node    = nav->num_node;
        args.nav_data_len    = nav->data_len;
        args.nav_max_m       = nav->max_m;
        args.nav_start       = nav->start;
        args.nav_init_ef     = std::min(ef_search, 5);
    } else {
        args.nav_data_dev    = nullptr;
        args.nav_graph_dev   = nullptr;
        args.nav_mapping_dev = nullptr;
        args.nav_num_node    = 0;
        args.nav_data_len    = 0;
        args.nav_max_m       = 0;
        args.nav_start       = 0;
        args.nav_init_ef     = 0;
    }
    args.num_blocks       = num_blocks;
    args.total_queries    = num_queries;
    args.d_out_nns        = d_out_nns;
    args.d_out_distances  = d_out_dist;
    args.d_out_found_cnt  = d_out_fc;
    args.d_step_recall_gt = d_step_recall_gt;
    args.step_recall_gt_width = per_step_recall.ground_truth_width;
    args.step_recall_gt_count = per_step_recall.ground_truth_count;
    args.step_recall_required_hits =
        per_step_recall.enabled()
            ? required_hits_for_threshold(per_step_recall.threshold, topk)
            : 0;
    args.d_step_recall_step = d_step_recall_step;
    args.d_step_recall_hits = d_step_recall_hits;
    args.early_exit_policy = static_cast<int>(early_exit.policy);
    args.early_exit_darth_interval = early_exit.darth_interval;
    args.early_exit_darth_threshold = early_exit.darth_threshold;
    args.early_exit_gpruning_warmup_steps = early_exit.gpruning_warmup_steps;
    args.early_exit_gpruning_patience = early_exit.gpruning_patience;
    args.darth_comm_slots = h_darth_comm_slots;
    args.d_early_exit_stop_step = d_early_exit_stop_step;
    args.d_finish_step = d_finish_step;

#ifdef QUIVER_LIGHT_BREAKDOWN
    args.d_breakdown_traces = d_breakdown_traces;
    args.d_breakdown_cta_windows = d_breakdown_cta_windows;
    args.breakdown_sample_count = breakdown_sample_count;
    args.breakdown_cta_window_count = breakdown_cta_window_count;
    args.breakdown_cta_window_ns =
        quiver_breakdown_cta_window_ns_from_env();
#endif

#ifdef QUIVER_LATENCY_PROBE
    args.d_probe_traces      = d_probe_traces;
    args.probe_max_traces    = probe_num_traces;
    args.d_probe_block_traces = d_probe_block_traces;
    args.probe_max_block_traces = 0;
    args.probe_block_trace_offset = 0;
#endif

    // Q-interleave: Q copies of frontier working arrays plus one staged node
    // buffer per slot. The staged node keeps hot SPDK page reads in shared
    // memory for merge and exact-distance phases.
    int frontier_bytes = (int)((sizeof(int) * 3 + sizeof(float) * 2)
                               * (ef_search + max_m));
    int slot_smem = ((frontier_bytes + 7) & ~7) + node_size;
    int smem = slot_smem * Q;

    // ---- 6. Pin host arrays ----
    CHECK_CUDA(cudaHostRegister((void*)qdata,
               sizeof(float) * num_queries * num_dims, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(nns,
               sizeof(int) * num_queries * topk, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(distances,
               sizeof(float) * num_queries * topk, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(found_cnt,
               sizeof(int) * num_queries, cudaHostRegisterDefault));

    // ---- 7. Start CPU IO poll threads ----
    std::atomic<bool> poll_done(false);
#ifdef QUIVER_LATENCY_PROBE
    std::atomic<bool> sample_current_batch(false);
#endif
    std::vector<std::thread> pollers;
    PollStats poll_stats;

    int slots_per_thread = (num_io_slots + thread_cnt - 1) / thread_cnt;

    for (int t = 0; t < thread_cnt; t++) {
        int s0 = t * slots_per_thread;
        int s1 = std::min(s0 + slots_per_thread, num_io_slots);
        pollers.emplace_back([&, t, s0, s1]() {
            shared::bind_core(t * 2 + 21);
            cpu_poll_loop(t, s0, s1,
                          h_controls, h_buffers,
                          nodes_per_page, loader,
                          poll_submit_batch_size,
                          poll_done,
#ifdef QUIVER_LATENCY_PROBE
                          sample_current_batch,
#endif
                          poll_stats);
        });
    }

    std::atomic<bool> darth_done(false);
    std::atomic<long> darth_predict_calls(0);
    std::atomic<long> darth_stop_count(0);
    std::atomic<long long> darth_predict_ns(0);
    std::thread darth_predictor_thread;
    if (h_darth_comm_slots != nullptr) {
        darth_predictor_thread = std::thread([&, topk]() {
            std::vector<int> seen(num_blocks, 0);
            std::ofstream trace;
            const bool write_trace =
                early_exit.darth_trace_csv_path != nullptr &&
                early_exit.darth_trace_csv_path[0] != '\0';
            if (write_trace) {
                trace.open(early_exit.darth_trace_csv_path);
                if (!trace.is_open()) {
                    ERROR("Failed to open DARTH trace CSV {}",
                          early_exit.darth_trace_csv_path);
                    std::exit(1);
                }
                trace << "qid,step,found_cnt,top1_dist,topk_dist,gap_ratio,"
                         "no_improve_count,predicted_recall,recall,stop\n";
            }

            while (!darth_done.load(std::memory_order_acquire)) {
                bool progressed = false;
                for (int i = 0; i < num_blocks; ++i) {
                    DarthCommSlot& slot = h_darth_comm_slots[i];
                    int epoch = __atomic_load_n(
                        (int32_t*)&slot.request_epoch, __ATOMIC_ACQUIRE);
                    if (epoch <= seen[i]) continue;
                    seen[i] = epoch;
                    progressed = true;

                    const auto t_pred0 = std::chrono::steady_clock::now();
                    float predicted = predict_darth_recall(slot);
                    const auto t_pred1 = std::chrono::steady_clock::now();
                    darth_predict_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t_pred1 - t_pred0).count(),
                        std::memory_order_relaxed);

                    const bool stop =
                        early_exit.policy == EarlyExitPolicy::kDarth &&
                        predicted >= early_exit.darth_threshold;
                    slot.predicted_recall = predicted;
                    __atomic_store_n((int32_t*)&slot.stop, stop ? 1 : 0,
                                     __ATOMIC_RELEASE);
                    __atomic_store_n((int32_t*)&slot.decision_epoch, epoch,
                                     __ATOMIC_RELEASE);
                    darth_predict_calls.fetch_add(1, std::memory_order_relaxed);
                    if (stop) darth_stop_count.fetch_add(1, std::memory_order_relaxed);

                    if (write_trace) {
                        double recall = -1.0;
                        if (early_exit.darth_ground_truth != nullptr &&
                            early_exit.darth_ground_truth_width > 0 &&
                            early_exit.darth_ground_truth_count > 0) {
                            const int gt_q =
                                slot.qid % early_exit.darth_ground_truth_count;
                            const int* gt = early_exit.darth_ground_truth +
                                (long)gt_q * early_exit.darth_ground_truth_width;
                            int hits = count_feature_gt_hits(
                                slot, topk, gt, early_exit.darth_ground_truth_width);
                            recall = (double)hits / (double)topk;
                        }
                        trace << slot.qid << "," << slot.step << ","
                              << slot.found_cnt << "," << slot.top1_dist << ","
                              << slot.topk_dist << "," << slot.gap_ratio << ","
                              << slot.no_improve_count << "," << predicted << ","
                              << std::fixed << std::setprecision(6) << recall << ","
                              << (stop ? 1 : 0) << "\n";
                    }
                }
                if (!progressed) std::this_thread::yield();
            }
            if (trace.is_open()) trace.close();
        });
    }

    // ---- 8. Streaming search: one launch, CPU-driven flow control ----
    // Per-query latency tracking.  Use raw-array + index instead of
    // std::vector<double> + push_back on the host hot loop: at 78k QPS
    // with ~200k queries we enter the completion branch ~O(10^5) times,
    // and push_back's amortized O(1) still costs 10-20 ns per call plus
    // cache-line churn.  A preallocated buffer is strictly faster.
    std::vector<double> query_start_times(num_queries, 0.0);
    std::vector<double> query_latencies_buf(num_queries, 0.0);
    int latency_count = 0;

    // Dedicated streams so the persistent search kernel and per-feed H2D
    // don't implicitly serialize through the legacy default (NULL) stream.
    //
    // Pitfall: if the persistent kernel is launched on stream 0 (default),
    // *new* operations submitted to ANY non-NULL stream implicitly wait for
    // the running kernel to finish (CUDA legacy default-stream semantics).
    // That makes cudaMemcpyAsync(h2d_stream) + cudaStreamSynchronize hang
    // in the host admission loop, which in turn prevents feed_count from
    // advancing, which makes the kernel spin forever -> full wedge.
    //
    // Fix: launch the persistent kernel on its own non-NULL kernel_stream.
    // Two non-NULL streams have NO implicit synchronization, so per-feed
    // H2D proceeds concurrently with the running kernel.  Cross-stream
    // memory visibility between H2D and kernel reads of d_qdata is
    // guaranteed by:
    //   host side: cudaMemcpyAsync(h2d_stream) + cudaStreamSynchronize;
    //   device side: __threadfence_system() in the per-query admission
    //                spin (see dispatch.cuh / kernel_qi.cuh).
    cudaStream_t h2d_stream;
    cudaStream_t kernel_stream;
    CHECK_CUDA(cudaStreamCreate(&h2d_stream));
    CHECK_CUDA(cudaStreamCreate(&kernel_stream));

    // Reset counters + IOControl (done before any query becomes admissible so
    // that t0 marks the true moment queries start arriving at the CPU).
    CHECK_CUDA(cudaMemset(d_next_query,   0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_queries_done, 0, sizeof(int)));
    for (int i = 0; i < num_io_slots; i++)
        h_controls[i].status = IO_IDLE;

    h_comm->total_queries      = num_queries;
    h_comm->num_blocks         = num_blocks;
    h_comm->feed_count         = 0;
    h_comm->terminate          = 0;
    h_comm->queries_completed  = 0;
    CHECK_CUDA(cudaDeviceSynchronize());

    // Prefeed depth: allow GPU to have extra queries ready so blocks never
    // stall waiting for CPU feed_count updates across PCIe.  Controlled by
    // environment variable QUIVER_PREFEED (default: num_blocks, i.e. 2x the
    // base concurrency).  Setting to 0 restores the original 1:1 flow control.
    const int prefeed_depth = prefeed_depth_cfg;

    // Max in-flight queries = task_slots (= num_blocks * Q) + prefeed.
    const int max_inflight_queries = num_task_slots + prefeed_depth;
    int initial_feed = std::min(max_inflight_queries, num_queries);

    // Lambda: H2D for queries in range [begin, end).  Nav is done inside
    // the persistent kernel (see inkernel_nav_entry in kernel_qi.cuh /
    // kernel_q1.cuh), so this only submits the query data H2D.
    //
    // `sync_block`:
    //   - true  : block the calling thread until the H2D is complete.
    //             Used only for the *initial* feed before we launch the
    //             persistent kernel, where blocking is fine.
    //   - false : fire-and-forget submit to h2d_stream; caller is responsible
    //             for later gating feed_count bumps on a cudaEvent recorded
    //             after this submission.  This is the steady-state path
    //             inside the host admission loop.
    auto h2d_range_submit = [&](int begin, int end, bool sync_block) {
        if (end <= begin) return;
        int k = end - begin;
        CHECK_CUDA(cudaMemcpyAsync(
            d_qdata + (size_t)begin * num_dims,
            qdata   + (size_t)begin * num_dims,
            sizeof(float) * (size_t)k * num_dims,
            cudaMemcpyHostToDevice, h2d_stream));
        if (sync_block) {
            CHECK_CUDA(cudaStreamSynchronize(h2d_stream));
        }
    };
    auto h2d_range = [&](int begin, int end) {
        h2d_range_submit(begin, end, /*sync_block=*/true);
    };

    // t0: user-visible start — initial batch of queries arrives at CPU.
    // Every subsequent feed records its own arrival timestamp (see host loop).
    // H2D + Nav for the initial batch are part of their processing and are
    // included in per-query latency via query_start_times[i] = t0.
    double t0 = shared::elapsed();

    // H2D for the initial batch only (nav happens in-kernel).  Subsequent
    // batches are streamed in by the host admission loop below.
    nvtxRangePush("H2D_initial");
    h2d_range(0, initial_feed);
    nvtxRangePop();
    double t_after_h2d = shared::elapsed();

    for (int i = 0; i < initial_feed; i++)
        query_start_times[i] = t0;
    struct AdmissionBatch {
        int target;
        double start_time;
    };
    std::vector<AdmissionBatch> admission_batches;
    std::vector<double> batch_latencies;
    admission_batches.reserve(num_queries + 1);
    if (initial_feed > 0) {
        admission_batches.push_back({initial_feed, t0});
    }
    size_t next_admission_to_record = 0;

    // Feed initial queries: GPU blocks can process queries 0..initial_feed-1
    __atomic_store_n((int32_t*)&h_comm->feed_count, initial_feed, __ATOMIC_RELEASE);

    // Launch persistent kernel (single launch for ALL queries).
    // IMPORTANT: launched on kernel_stream (non-NULL) so per-feed H2D on
    // h2d_stream doesn't implicitly wait for this long-running kernel.
    nvtxRangePush("SearchKernel");
    if (Q <= 1) {
        // Q=1 fast path: original single-query-per-block kernel
        // Phase 3.1a: kernel now templated on Dispatch policy.
        if (data_type == shared::UINT8)
            persistent_search_kernel<StreamingDispatch, uint8_t><<<num_blocks, 128, smem, kernel_stream>>>(args);
        else if (data_type == shared::INT8)
            persistent_search_kernel<StreamingDispatch, int8_t><<<num_blocks, 128, smem, kernel_stream>>>(args);
        else
            persistent_search_kernel<StreamingDispatch, float><<<num_blocks, 128, smem, kernel_stream>>>(args);
    } else {
        // Q>1: Q-interleave kernel (experimental).
        // Phase 3.2: templated on Dispatch; currently only StreamingDispatch
        // is supported (see kernel_qi.cuh static_assert).
        if (data_type == shared::UINT8)
            persistent_search_kernel_qi<StreamingDispatch, uint8_t><<<num_blocks, 128, smem, kernel_stream>>>(args);
        else if (data_type == shared::INT8)
            persistent_search_kernel_qi<StreamingDispatch, int8_t><<<num_blocks, 128, smem, kernel_stream>>>(args);
        else
            persistent_search_kernel_qi<StreamingDispatch, float><<<num_blocks, 128, smem, kernel_stream>>>(args);
    }

    // CPU completion poller: monitor queries_completed (mapped), feed new queries.
    // Maintain invariant: feed_count <= last_done + (num_blocks*Q) + prefeed_depth
    // so that at most that many queries are in-flight.
    //
    // Streaming arrival model: each newly admitted query is treated as having
    // just arrived at the CPU.  We do per-feed H2D+nav, then record
    // query_start_times[i] = t_arrival (the moment its data is handed to GPU).
    // This is the symmetric counterpart of FlashANNS's per-batch arrival model.
    //
    // Async-feed with event ring (perf fix for §2.11, 2026-04-24):
    //   Previously each feed did `cudaStreamSynchronize(h2d_stream)` before
    //   bumping feed_count.  At ~64k QPS that CPU-blocking sync happens once
    //   per-feed and dominated the host loop, cutting QPS in half.
    //   New design:
    //     1. Submit H2D(+nav) to h2d_stream without blocking (`sync_block=false`)
    //     2. cudaEventRecord(event, h2d_stream); push {target, event} into
    //        pending ring
    //     3. Each loop iter polls the ring head via cudaEventQuery (non-blocking)
    //     4. Only when the event completes do we bump feed_count up to its
    //        target.  This preserves the "H2D must finish before GPU sees
    //        feed_count advance" invariant without CPU sync.
    //   The ring is sized generously (max_inflight + small slack) so that
    //   submitters never block for ring space.
    int last_done = 0;
    int next_to_feed = initial_feed;
    const int max_inflight = max_inflight_queries;

    // event_pool lives at function scope so cleanup at the end of this
    // function can destroy whatever events were allocated.  Only the
    // per-feed async path populates it.
    std::vector<cudaEvent_t> event_pool;

    // Event ring for async feed gating.  Size chosen so we practically never
    // run out of free events; each feed consumes exactly one, and feeds are
    // gated by max_inflight_queries, so a ring of 2x that is plenty.
    const int event_ring_cap = std::max(64, 2 * max_inflight_queries);
    event_pool.resize(event_ring_cap);
    for (int i = 0; i < event_ring_cap; ++i) {
        CHECK_CUDA(cudaEventCreateWithFlags(&event_pool[i],
                                            cudaEventDisableTiming));
    }
    struct PendingFeed {
        int target;           // new_feed_target at submission time
        cudaEvent_t ev;       // gating event on h2d_stream
    };
    std::vector<PendingFeed> pending_ring(event_ring_cap);
    int pending_head = 0;
    int pending_count = 0;
    int ev_alloc_cursor = 0;  // round-robin allocator into event_pool

    // Admit coalescing threshold: only submit a new H2D when either
    //   (a) the outstanding admit gap reaches this many queries, or
    //   (b) we've reached the final batch (must flush), or
    //   (c) there is nothing already in flight (avoid starving the kernel).
    // This cuts cudaMemcpyAsync / cudaEventRecord / cudaEventQuery call
    // frequency by ~admit_min_batch x (each costs ~us of driver time).
    // Defaulting to 32 gives ~6k admits per 200k-query run; kernel's
    // max_inflight_queries (num_blocks*Q, typically 1000+) keeps GPU
    // blocks fed even when we skip a small admit.  Env overridable.
    int admit_min_batch = 32;
    if (const char* env = std::getenv("QUIVER_ADMIT_MIN_BATCH"))
        admit_min_batch = std::max(1, std::atoi(env));

    while (last_done < num_queries) {
        // 1) Drain any pending H2D events whose H2D has completed, bumping
        //    feed_count up to the max target whose event is ready.
        int new_feed_visible = 0;
        bool any_bumped = false;
        while (pending_count > 0) {
            cudaError_t q = cudaEventQuery(pending_ring[pending_head].ev);
            if (q == cudaSuccess) {
                new_feed_visible = pending_ring[pending_head].target;
                any_bumped = true;
                pending_head = (pending_head + 1) % event_ring_cap;
                --pending_count;
            } else if (q == cudaErrorNotReady) {
                break;
            } else {
                CHECK_CUDA(q);
            }
        }
        if (any_bumped) {
            __atomic_store_n((int32_t*)&h_comm->feed_count,
                             new_feed_visible, __ATOMIC_RELEASE);
        }

        // 2) Read completion counter (mapped pinned).
        int current_done = __atomic_load_n(
            (int32_t*)&h_comm->queries_completed, __ATOMIC_ACQUIRE);

        if (current_done > last_done) {
            double now = shared::elapsed();
            for (int i = last_done; i < current_done; i++) {
                query_latencies_buf[latency_count++] =
                    (now - query_start_times[i]) * 1000.0;
            }
            last_done = current_done;
            while (next_admission_to_record < admission_batches.size() &&
                   current_done >= admission_batches[next_admission_to_record].target) {
                batch_latencies.push_back(
                    (now - admission_batches[next_admission_to_record].start_time) *
                    1000.0);
                ++next_admission_to_record;
            }
        }

        // 3) Admit more queries if flow control allows.
        int allowed = last_done + max_inflight;
        int new_feed_target = std::min(num_queries, allowed);
        int gap = new_feed_target - next_to_feed;
        bool must_submit = (gap > 0) && (
            gap >= admit_min_batch
            || new_feed_target == num_queries
            || pending_count == 0
        );
        if (must_submit) {
            double t_arrival = shared::elapsed();
            h2d_range_submit(next_to_feed, new_feed_target,
                             /*sync_block=*/false);
            cudaEvent_t ev = event_pool[ev_alloc_cursor];
            ev_alloc_cursor = (ev_alloc_cursor + 1) % event_ring_cap;
            CHECK_CUDA(cudaEventRecord(ev, h2d_stream));
            for (int i = next_to_feed; i < new_feed_target; i++) {
                query_start_times[i] = t_arrival;
            }
            admission_batches.push_back({new_feed_target, t_arrival});
            next_to_feed = new_feed_target;
            int tail = (pending_head + pending_count) % event_ring_cap;
            pending_ring[tail] = {next_to_feed, ev};
            ++pending_count;
        }
    }

    // Drain any last pending events (should be rare: if all queries have
    // completed, all H2Ds must have been consumed by the kernel already).
    while (pending_count > 0) {
        CHECK_CUDA(cudaEventSynchronize(pending_ring[pending_head].ev));
        pending_head = (pending_head + 1) % event_ring_cap;
        --pending_count;
    }

    // Signal kernel to terminate (all queries done, some blocks may be spin-waiting)
    __atomic_store_n((int32_t*)&h_comm->terminate, 1, __ATOMIC_RELEASE);
    CHECK_CUDA(cudaDeviceSynchronize());
    darth_done.store(true, std::memory_order_release);
    if (darth_predictor_thread.joinable()) {
        darth_predictor_thread.join();
    }
    nvtxRangePop(); // SearchKernel
    double t_after_kernel = shared::elapsed();

    // Results already in mapped-pinned buffers (host-visible after kernel
    // fences).  Copy to caller's output arrays (host-to-host memcpy).
    nvtxRangePush("D2H");
    memcpy(nns, d_out_nns, sizeof(int) * topk * num_queries);
    memcpy(distances, d_out_dist, sizeof(float) * topk * num_queries);
    memcpy(found_cnt, d_out_fc, sizeof(int) * num_queries);
    nvtxRangePop(); // D2H

    if (per_step_recall.enabled()) {
        std::vector<int> h_step(num_queries);
        std::vector<int> h_hits(num_queries);
        CHECK_CUDA(cudaMemcpy(h_step.data(), d_step_recall_step,
                              sizeof(int) * num_queries, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_hits.data(), d_step_recall_hits,
                              sizeof(int) * num_queries, cudaMemcpyDeviceToHost));
        write_step_recall_csv(per_step_recall, h_step, h_hits, topk);
    }

    int early_exit_stop_count = 0;
    double early_exit_avg_stop_step = 0.0;
    double avg_finish_step = 0.0;
    std::vector<int> h_stop_step(num_queries);
    CHECK_CUDA(cudaMemcpy(h_stop_step.data(), d_early_exit_stop_step,
                          sizeof(int) * num_queries, cudaMemcpyDeviceToHost));
    long long stop_step_sum = 0;
    for (int step : h_stop_step) {
        if (step >= 0) {
            ++early_exit_stop_count;
            stop_step_sum += step;
        }
    }
    if (early_exit_stop_count > 0) {
        early_exit_avg_stop_step =
            (double)stop_step_sum / (double)early_exit_stop_count;
    }
    std::vector<int> h_finish_step(num_queries);
    CHECK_CUDA(cudaMemcpy(h_finish_step.data(), d_finish_step,
                          sizeof(int) * num_queries, cudaMemcpyDeviceToHost));
    long long finish_step_sum = 0;
    int finish_step_count = 0;
    for (int step : h_finish_step) {
        if (step >= 0) {
            ++finish_step_count;
            finish_step_sum += step;
        }
    }
    if (finish_step_count > 0) {
        avg_finish_step = (double)finish_step_sum / (double)finish_step_count;
    }

    double t1 = shared::elapsed();

    // ---- 9. Stop poll threads ----
    poll_done.store(true, std::memory_order_release);
    for (auto& p : pollers)
        p.join();

    // ---- 10. Report ----
    double search_time = t1 - t0;
    auto io_stats = loader->snapshot_stats();

    // Per-query latency summary (only the first `latency_count` entries
    // of the preallocated buffer are valid).
    std::sort(query_latencies_buf.begin(),
              query_latencies_buf.begin() + latency_count);
    double ql_avg = 0, ql_p50 = 0, ql_p90 = 0, ql_p99 = 0, ql_p999 = 0, ql_max = 0;
    if (latency_count > 0) {
        ql_avg  = std::accumulate(query_latencies_buf.begin(),
                                  query_latencies_buf.begin() + latency_count, 0.0)
                  / latency_count;
        ql_p50  = query_latencies_buf[(size_t)latency_count * 50 / 100];
        ql_p90  = query_latencies_buf[(size_t)latency_count * 90 / 100];
        ql_p99  = query_latencies_buf[(size_t)latency_count * 99 / 100];
        ql_p999 = query_latencies_buf[(size_t)(latency_count * 0.999)];
        ql_max  = query_latencies_buf[latency_count - 1];
    }

    std::ostringstream runtime_params;
    runtime_params << "num_blocks=" << num_blocks << " pipe_width=" << pipe_w
                   << " queries_per_block=" << Q
                   << " task_slots=" << num_task_slots
                   << " io_slots=" << num_io_slots
                   << " max_inflight=" << max_inflight_queries
                   << " poll_threads=" << thread_cnt
                   << " poll_submit_batch_size=" << poll_submit_batch_size
                   << " prefeed=" << prefeed_depth;
    if (early_exit.enabled()) {
        runtime_params << " early_exit_policy="
                       << (early_exit.policy == EarlyExitPolicy::kDarth ? "darth" :
                           early_exit.policy == EarlyExitPolicy::kDarthTrace ? "darth_trace" : "gpruning")
                       << " early_exit_darth_interval=" << early_exit.darth_interval
                       << " early_exit_darth_threshold=" << early_exit.darth_threshold
                       << " early_exit_gpruning_warmup_steps=" << early_exit.gpruning_warmup_steps
                       << " early_exit_gpruning_patience=" << early_exit.gpruning_patience;
        if (h_darth_comm_slots != nullptr) {
            long long wait_iters = 0;
            long long requests = 0;
            for (int i = 0; i < num_blocks; ++i) {
                wait_iters += h_darth_comm_slots[i].wait_iters_total;
                requests += h_darth_comm_slots[i].request_count;
            }
            runtime_params << " darth_predict_calls=" << darth_predict_calls.load()
                           << " darth_stops=" << darth_stop_count.load()
                           << " darth_predict_ns=" << darth_predict_ns.load()
                           << " darth_gpu_wait_iters=" << wait_iters
                           << " darth_gpu_wait_requests=" << requests;
        }
    }
    runtime_params << " spdk_backpressure="
                   << io_stats.submit_backpressure_count
                   << " spdk_nvme_errors=" << io_stats.nvme_error_count;
    shared::RunReport report;
    report.program = "Quiver";
    report.query = {shared::data_type_name(data_type), topk, ef_search, repeat};
    report.runtime_params = runtime_params.str();
    report.qps = num_queries / search_time;
    report.latency_label = "QueryLatency(ms)";
    report.latency = {ql_avg, ql_p50, ql_p90, ql_p99, ql_p999, ql_max};
    report.io_completed = io_stats.pages_submitted;
    report.search_time_s = search_time;
    report.hot_max_submitted_pages_per_sec =
        io_stats.hot_max_submitted_pages_per_sec;
    report.hot_sample_count = io_stats.hot_sample_count;
    report.num_blocks = num_blocks;
    report.queries_per_block = Q;
    report.pipe_width = pipe_w;
    shared::print_run_report(report);
    print_batch_latency_summary("Quiver", batch_latencies);

#ifdef QUIVER_LIGHT_BREAKDOWN
    {
        std::vector<QuiverLightBreakdownTrace> traces(
            breakdown_sample_count);
        CHECK_CUDA(cudaMemcpy(
            traces.data(), d_breakdown_traces,
            sizeof(QuiverLightBreakdownTrace) * breakdown_sample_count,
            cudaMemcpyDeviceToHost));
        std::vector<double> compute, io_wait, resume, cta_active_pct;
        for (int i = 0; i < breakdown_sample_count; ++i) {
            const auto& tr = traces[i];
            if (!tr.valid || tr.query_done_ns <= tr.query_start_ns) continue;
            std::vector<std::pair<int64_t, int64_t>> waits;
            const int wait_n = std::min(
                tr.wait_interval_count,
                QUIVER_LIGHT_BREAKDOWN_MAX_WAIT_INTERVALS);
            const int compute_n = std::min(
                tr.compute_interval_count,
                QUIVER_LIGHT_BREAKDOWN_MAX_COMPUTE_INTERVALS);
            for (int h = 0; h < wait_n; ++h)
                waits.push_back({tr.wait_start_ns[h], tr.wait_end_ns[h]});
            shared::ae::append_hop_intervals(
                "Quiver", i, tr.wait_start_ns, tr.wait_end_ns, wait_n,
                tr.compute_start_ns, tr.compute_end_ns, compute_n);
            const double latency_us =
                (tr.query_done_ns - tr.query_start_ns) / 1000.0;
            const auto scaled = shared::breakdown::scale_components_to_total(
                {tr.useful_compute_ns / 1000.0,
                 shared::breakdown::merged_interval_sum_us(std::move(waits)),
                 tr.resumption_delay_ns / 1000.0},
                latency_us);
            compute.push_back(scaled[0]);
            io_wait.push_back(scaled[1]);
            resume.push_back(scaled[2]);
        }
        std::vector<QuiverBreakdownCtaWindowTrace> windows(
            breakdown_cta_window_count);
        CHECK_CUDA(cudaMemcpy(
            windows.data(), d_breakdown_cta_windows,
            sizeof(QuiverBreakdownCtaWindowTrace) *
                breakdown_cta_window_count,
            cudaMemcpyDeviceToHost));
        for (const auto& window : windows) {
            if (!window.valid ||
                window.window_end_ns <= window.window_start_ns) continue;
            const double total =
                (window.window_end_ns - window.window_start_ns) / 1000.0;
            const double idle = window.pure_io_wait_ns / 1000.0;
            cta_active_pct.push_back(
                shared::breakdown::make_cta_activity_sample(
                    total, std::max(0.0, total - idle)).active_ratio_pct);
        }
        shared::breakdown::print_quiver_metric_samples(
            compute, io_wait, resume,
            /*scheduling_gap=*/{}, /*other=*/{}, /*kernel_launch=*/{},
            /*h2d=*/{}, /*d2h=*/{}, /*latency=*/{}, /*cta_window=*/{},
            /*cta_active=*/{}, /*cta_pure_io_wait=*/{}, cta_active_pct);
        shared::ae::append_closed_breakdown(
            "Quiver", compute, io_wait, resume, cta_active_pct);
    }
#endif

    {
        const char* policy_name = "none";
        if (early_exit.enabled()) {
            policy_name =
                early_exit.policy == EarlyExitPolicy::kDarth ? "darth" :
                early_exit.policy == EarlyExitPolicy::kDarthTrace ? "darth_trace" : "gpruning";
        }
        std::cout << "[Quiver] Finish summary: policy=" << policy_name
                  << " avg_finish_step=" << std::fixed << std::setprecision(2)
                  << avg_finish_step
                  << " finished=" << finish_step_count << "/" << num_queries;
        if (early_exit.enabled()) {
            std::cout << " stopped=" << early_exit_stop_count << "/"
                      << num_queries << " ("
                      << std::fixed << std::setprecision(2)
                      << (100.0 * (double)early_exit_stop_count / (double)num_queries)
                      << "%)"
                      << " avg_stop_step=" << std::setprecision(2)
                      << early_exit_avg_stop_step;
        }
        if (h_darth_comm_slots != nullptr) {
            long long wait_iters = 0;
            long long requests = 0;
            for (int i = 0; i < num_blocks; ++i) {
                wait_iters += h_darth_comm_slots[i].wait_iters_total;
                requests += h_darth_comm_slots[i].request_count;
            }
            const long calls = darth_predict_calls.load();
            const double avg_predict_us =
                calls > 0 ? (double)darth_predict_ns.load() / (double)calls / 1000.0 : 0.0;
            const double avg_wait_iters =
                requests > 0 ? (double)wait_iters / (double)requests : 0.0;
            std::cout << " predictor_calls=" << calls
                      << " avg_predict_us=" << std::setprecision(3)
                      << avg_predict_us
                      << " gpu_wait_requests=" << requests
                      << " avg_gpu_wait_iters=" << std::setprecision(1)
                      << avg_wait_iters;
        }
        std::cout << std::defaultfloat << std::endl;
    }

#ifdef QUIVER_LATENCY_PROBE
    {
        std::vector<QuiverProbeQueryTrace> h_traces(probe_num_traces);
        CHECK_CUDA(cudaMemcpy(h_traces.data(), d_probe_traces,
                              sizeof(QuiverProbeQueryTrace) * probe_num_traces,
                              cudaMemcpyDeviceToHost));
        if (std::getenv("QUIVER_PROBE_QUERY_BREAKDOWN") != nullptr) {
            print_query_latency_breakdown(h_traces.data(), probe_num_traces);
        }
        print_global_hop_probe_breakdown(h_traces.data(), probe_num_traces);
        if (std::getenv("QUIVER_PROBE_CSV") != nullptr) {
            std::printf("Quiver EndToEndIO probe is only supported in static "
                        "mode; no CDF written for streaming mode.\n");
        }
    }
#endif

    // ---- 11. Cleanup ----
#ifdef QUIVER_LATENCY_PROBE
    CHECK_CUDA(cudaFree(d_probe_traces));
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
    CHECK_CUDA(cudaFree(d_breakdown_traces));
    CHECK_CUDA(cudaFree(d_breakdown_cta_windows));
#endif
    CHECK_CUDA(cudaStreamDestroy(h2d_stream));
    CHECK_CUDA(cudaStreamDestroy(kernel_stream));
    for (auto& ev : event_pool) {
        CHECK_CUDA(cudaEventDestroy(ev));
    }
    CHECK_CUDA(cudaFree(d_qdata));
    CHECK_CUDA(cudaFree(d_nid));
    CHECK_CUDA(cudaFree(d_ndist));
    CHECK_CUDA(cudaFree(d_ctx));
    CHECK_CUDA(cudaFree(d_nns_dev));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_fc));
    if (d_step_recall_gt) CHECK_CUDA(cudaFree(d_step_recall_gt));
    if (d_step_recall_step) CHECK_CUDA(cudaFree(d_step_recall_step));
    if (d_step_recall_hits) CHECK_CUDA(cudaFree(d_step_recall_hits));
    if (d_early_exit_stop_step) CHECK_CUDA(cudaFree(d_early_exit_stop_step));
    if (d_finish_step) CHECK_CUDA(cudaFree(d_finish_step));
    if (h_darth_comm_slots) CHECK_CUDA(cudaFreeHost(h_darth_comm_slots));
    CHECK_CUDA(cudaFreeHost(d_out_nns));
    CHECK_CUDA(cudaFreeHost(d_out_dist));
    CHECK_CUDA(cudaFreeHost(d_out_fc));
    CHECK_CUDA(cudaFree(d_next_query));
    CHECK_CUDA(cudaFree(d_queries_done));
    CHECK_CUDA(cudaFreeHost(h_comm));

    CHECK_CUDA(cudaHostUnregister(h_buffers));
    loader->destroy_buffer(h_buffers);
    CHECK_CUDA(cudaFreeHost(h_controls));
    CHECK_CUDA(cudaHostUnregister((void*)qdata));
    CHECK_CUDA(cudaHostUnregister(nns));
    CHECK_CUDA(cudaHostUnregister(distances));
    CHECK_CUDA(cudaHostUnregister(found_cnt));
}

// ================================================================
// StaticDispatch ablation entry point.
// The host still exposes FlashANNS-style fixed-size batches, but each batch
// runs Quiver's streaming dispatch kernel over a pre-admitted query range.
// This keeps the stable StreamingDispatch state machine while preserving the
// static binary's CLI and reporting surface.
// ================================================================
void run_static_batch_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
    int num_blocks,
    std::shared_ptr<shared::IndexLoader> loader,
    shared::PQSearch* pq,
    shared::NavGraph* nav,
    int enter_point,
    uint8_t* starter,
    const float* qdata,
    int num_queries,
    int topk,
    int ef_search,
    int* nns,
    float* distances,
    int* found_cnt,
    int queries_per_block,
    int repeat,
    const EarlyExitOptions& early_exit)
{
#ifdef QUIVER_FIXED_PIPE_WIDTH
    pipe_w = QUIVER_FIXED_PIPE_WIDTH;
#endif
#ifdef QUIVER_FIXED_QUERIES_PER_BLOCK
    queries_per_block = QUIVER_FIXED_QUERIES_PER_BLOCK;
#endif
    const int Q = queries_per_block > 0 ? queries_per_block : 1;
    int num_dims       = (int)layout.num_dims;
    int max_m          = (int)layout.max_m0;
    int nodes_per_page = (int)layout.nodes_per_page;
    int node_size      = (int)layout.node_size;
    int data_size      = (int)layout.data_size;
    int aligned_ef     = (max_m + ef_search + 31) / 32 * 32;

    int num_task_slots = num_blocks * Q;
    int num_io_slots   = num_task_slots * pipe_w;

    if (early_exit.enabled() &&
        early_exit.policy != EarlyExitPolicy::kGPruning &&
        Q != 1) {
        ERROR("Static early-exit policy currently supports Q-interleave only for gpruning");
        std::exit(1);
    }

    int poll_submit_batch_size = 256;
    const char* psb_env = getenv("CPU_POLL_SUBMIT_BATCH_SIZE");
    if (psb_env) poll_submit_batch_size = std::atoi(psb_env);

    INFO("StaticBatch kernel: B={}, pipe_w={}, num_blocks={}, Q={}, task_slots={}, io_slots={}, T={}, poll_submit_batch_size={}",
         num_blocks, pipe_w, num_blocks, Q, num_task_slots, num_io_slots,
         thread_cnt, poll_submit_batch_size);

    // ---- 1. PQ init ----
    pq->init_device(num_dims, (int)layout.num_data, num_task_slots, ef_search);

    // ---- 2. IOControl + DMA buffers ----
    IOControl* h_controls;
    CHECK_CUDA(cudaHostAlloc(&h_controls, sizeof(IOControl) * num_io_slots,
                             cudaHostAllocMapped));
    for (int i = 0; i < num_io_slots; i++) h_controls[i].status = IO_IDLE;

    uint8_t* h_buffers =
        loader->create_buffer((int64_t)shared::PAGE_SIZE * num_io_slots);
    CHECK_CUDA(cudaHostRegister(h_buffers,
                                (size_t)shared::PAGE_SIZE * num_io_slots,
                                cudaHostRegisterDefault));

    // ---- 3. Comm struct (pinned+mapped) ----
    PersistentKernelComm* h_comm;
    CHECK_CUDA(cudaHostAlloc(&h_comm, sizeof(PersistentKernelComm),
                              cudaHostAllocMapped));
    h_comm->pipe_w   = pipe_w;
    h_comm->controls = h_controls;
    h_comm->buffers  = h_buffers;
    h_comm->feed_count = 0;
    h_comm->terminate  = 0;
    h_comm->queries_completed = 0;

    // ---- 4. Device allocations ----
    float*    d_qdata;
    uint32_t* d_nid;
    float*    d_ndist;
    shared::Data* d_ctx;
    int*      d_nns_dev;
    float*    d_dist;
    int*      d_fc;
    int*      d_out_nns;
    float*    d_out_dist;
    int*      d_out_fc;
    int*      d_next_query;
    int*      d_queries_done;
    int*      d_early_exit_stop_step = nullptr;
    int*      d_finish_step = nullptr;
    DarthCommSlot* h_darth_comm_slots = nullptr;

    CHECK_CUDA(cudaMalloc(&d_qdata, sizeof(float) * num_queries * num_dims));
    CHECK_CUDA(cudaMalloc(&d_nid,   sizeof(uint32_t) * aligned_ef * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_ndist, sizeof(float)    * aligned_ef * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_ctx, sizeof(shared::Data) * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_nns_dev,   sizeof(int)  * topk * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_dist,  sizeof(float)    * topk * num_task_slots));
    CHECK_CUDA(cudaMalloc(&d_fc,    sizeof(int)      * num_task_slots));
    CHECK_CUDA(cudaHostAlloc(&d_out_nns,  sizeof(int)   * topk * num_queries, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostAlloc(&d_out_dist, sizeof(float) * topk * num_queries, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostAlloc(&d_out_fc,   sizeof(int)   * num_queries, cudaHostAllocMapped));
    CHECK_CUDA(cudaMalloc(&d_next_query,   sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_queries_done, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_finish_step, sizeof(int) * num_queries));
    CHECK_CUDA(cudaMemset(d_finish_step, 0xff, sizeof(int) * num_queries));
    CHECK_CUDA(cudaMalloc(&d_early_exit_stop_step, sizeof(int) * num_queries));
    CHECK_CUDA(cudaMemset(d_early_exit_stop_step, 0xff, sizeof(int) * num_queries));

    if (early_exit.policy == EarlyExitPolicy::kDarth ||
        early_exit.policy == EarlyExitPolicy::kDarthTrace) {
        CHECK_CUDA(cudaHostAlloc(&h_darth_comm_slots,
                                 sizeof(DarthCommSlot) * num_blocks,
                                 cudaHostAllocMapped));
        std::memset(h_darth_comm_slots, 0, sizeof(DarthCommSlot) * num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            h_darth_comm_slots[i].stop = 0;
            for (int j = 0; j < DARTH_FEATURE_TOPK_MAX; ++j) {
                h_darth_comm_slots[i].top_ids[j] = -1;
            }
        }
    }

#ifdef QUIVER_LATENCY_PROBE
    QuiverProbeQueryTrace* d_probe_traces = nullptr;
    int probe_num_traces = std::min(num_queries, (int)PROBE_MAX_QUERY_TRACES);
    CHECK_CUDA(cudaMalloc(&d_probe_traces, sizeof(QuiverProbeQueryTrace) * probe_num_traces));
    CHECK_CUDA(cudaMemset(d_probe_traces, 0, sizeof(QuiverProbeQueryTrace) * probe_num_traces));
    const int probe_batch_stride = num_blocks * Q;
    const int probe_num_batches =
        (num_queries + probe_batch_stride - 1) / probe_batch_stride;
    const int probe_num_block_traces = probe_num_batches * num_blocks;
    QuiverProbeBlockTrace* d_probe_block_traces = nullptr;
    CHECK_CUDA(cudaMalloc(&d_probe_block_traces,
                          sizeof(QuiverProbeBlockTrace) * probe_num_block_traces));
    CHECK_CUDA(cudaMemset(d_probe_block_traces, 0,
                          sizeof(QuiverProbeBlockTrace) * probe_num_block_traces));
#endif

    // ---- 5. Build kernel args ----
    PersistentKernelArgs args{};
    args.comm             = h_comm;
    args.d_next_query     = d_next_query;
    args.d_queries_done   = d_queries_done;
    args.enter_point      = enter_point;
    args.pq_data          = pq->get_device_ptr();
    args.num_chunks       = pq->device_data.num_chunks;
    args.pq_dists         = pq->device_data.pq_dists;
    args.compressed_data  = pq->device_data.compressed_data;
    args.pq_block_offset  = 0;
    args.nodes_per_page   = nodes_per_page;
    args.node_size        = node_size;
    args.data_size        = data_size;
    args.num_dims         = num_dims;
    args.max_m            = max_m;
    args.ef_search        = ef_search;
    args.aligned_ef       = aligned_ef;
    args.topk             = topk;
    args.pipe_w           = pipe_w;
    args.queries_per_block = Q;
    args.d_neighbors_id   = d_nid;
    args.d_neighbors_dist = d_ndist;
    args.d_ctx            = d_ctx;
    args.d_nns            = d_nns_dev;
    args.d_distances      = d_dist;
    args.d_found_cnt      = d_fc;
    args.d_all_qdata      = d_qdata;
    args.d_entry_nodes    = nullptr;
    if (nav != nullptr) {
        args.nav_data_dev    = nav->data_dev;
        args.nav_graph_dev   = nav->graph_dev;
        args.nav_mapping_dev = nav->mapping_dev;
        args.nav_num_node    = nav->num_node;
        args.nav_data_len    = nav->data_len;
        args.nav_max_m       = nav->max_m;
        args.nav_start       = nav->start;
        args.nav_init_ef     = std::min(ef_search, 5);
    } else {
        args.nav_data_dev    = nullptr;
        args.nav_graph_dev   = nullptr;
        args.nav_mapping_dev = nullptr;
        args.nav_num_node    = 0;
        args.nav_data_len    = 0;
        args.nav_max_m       = 0;
        args.nav_start       = 0;
        args.nav_init_ef     = 0;
    }
    args.num_blocks       = num_blocks;
    args.total_queries    = num_queries;
    args.batch_offset     = 0;
    args.d_out_nns        = d_out_nns;
    args.d_out_distances  = d_out_dist;
    args.d_out_found_cnt  = d_out_fc;
    args.early_exit_policy = static_cast<int>(early_exit.policy);
    args.early_exit_darth_interval = early_exit.darth_interval;
    args.early_exit_darth_threshold = early_exit.darth_threshold;
    args.early_exit_gpruning_warmup_steps = early_exit.gpruning_warmup_steps;
    args.early_exit_gpruning_patience = early_exit.gpruning_patience;
    args.darth_comm_slots = h_darth_comm_slots;
    args.d_early_exit_stop_step = d_early_exit_stop_step;
    args.d_finish_step = d_finish_step;

#ifdef QUIVER_LATENCY_PROBE
    args.d_probe_traces      = d_probe_traces;
    args.probe_max_traces    = probe_num_traces;
    args.d_probe_block_traces = d_probe_block_traces;
    args.probe_max_block_traces = probe_num_block_traces;
    args.probe_block_trace_offset = 0;
#endif

    int frontier_bytes = (int)((sizeof(int) * 3 + sizeof(float) * 2)
                               * (ef_search + max_m));
    int slot_smem = ((frontier_bytes + 7) & ~7) + node_size;
    int smem = slot_smem * Q;

    // ---- 6. Pin host arrays ----
    CHECK_CUDA(cudaHostRegister((void*)qdata,
               sizeof(float) * num_queries * num_dims, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(nns,
               sizeof(int) * num_queries * topk, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(distances,
               sizeof(float) * num_queries * topk, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(found_cnt,
               sizeof(int) * num_queries, cudaHostRegisterDefault));

    // ---- 7. Start CPU IO poll threads ----
    std::atomic<bool> poll_done(false);
#ifdef QUIVER_LATENCY_PROBE
    std::atomic<bool> sample_current_batch(false);
#endif
    std::vector<std::thread> pollers;
    PollStats poll_stats;

    int slots_per_thread = (num_io_slots + thread_cnt - 1) / thread_cnt;
    for (int t = 0; t < thread_cnt; t++) {
        int s0 = t * slots_per_thread;
        int s1 = std::min(s0 + slots_per_thread, num_io_slots);
        pollers.emplace_back([&, t, s0, s1]() {
            shared::bind_core(t * 2 + 21);
            cpu_poll_loop(t, s0, s1,
                          h_controls, h_buffers,
                          nodes_per_page, loader,
                          poll_submit_batch_size,
                          poll_done,
#ifdef QUIVER_LATENCY_PROBE
                          sample_current_batch,
#endif
                          poll_stats);
        });
    }

    std::atomic<bool> darth_done(false);
    std::atomic<long> darth_predict_calls(0);
    std::atomic<long> darth_stop_count(0);
    std::atomic<long long> darth_predict_ns(0);
    std::thread darth_predictor_thread;
    if (h_darth_comm_slots != nullptr) {
        darth_predictor_thread = std::thread([&, topk]() {
            std::vector<int> seen(num_blocks, 0);
            std::ofstream trace;
            const bool write_trace =
                early_exit.darth_trace_csv_path != nullptr &&
                early_exit.darth_trace_csv_path[0] != '\0';
            if (write_trace) {
                trace.open(early_exit.darth_trace_csv_path);
                if (!trace.is_open()) {
                    ERROR("Failed to open DARTH trace CSV {}",
                          early_exit.darth_trace_csv_path);
                    std::exit(1);
                }
                trace << "qid,step,found_cnt,top1_dist,topk_dist,gap_ratio,"
                         "no_improve_count,predicted_recall,recall,stop\n";
            }
            while (!darth_done.load(std::memory_order_acquire)) {
                bool progressed = false;
                for (int i = 0; i < num_blocks; ++i) {
                    DarthCommSlot& slot = h_darth_comm_slots[i];
                    int epoch = __atomic_load_n(
                        (int32_t*)&slot.request_epoch, __ATOMIC_ACQUIRE);
                    if (epoch <= seen[i]) continue;
                    seen[i] = epoch;
                    progressed = true;

                    const auto t_pred0 = std::chrono::steady_clock::now();
                    float predicted = predict_darth_recall(slot);
                    const auto t_pred1 = std::chrono::steady_clock::now();
                    darth_predict_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t_pred1 - t_pred0).count(),
                        std::memory_order_relaxed);

                    const bool stop =
                        early_exit.policy == EarlyExitPolicy::kDarth &&
                        predicted >= early_exit.darth_threshold;
                    slot.predicted_recall = predicted;
                    __atomic_store_n((int32_t*)&slot.stop, stop ? 1 : 0,
                                     __ATOMIC_RELEASE);
                    __atomic_store_n((int32_t*)&slot.decision_epoch, epoch,
                                     __ATOMIC_RELEASE);
                    darth_predict_calls.fetch_add(1, std::memory_order_relaxed);
                    if (stop) darth_stop_count.fetch_add(1, std::memory_order_relaxed);

                    if (write_trace) {
                        double recall = -1.0;
                        if (early_exit.darth_ground_truth != nullptr &&
                            early_exit.darth_ground_truth_width > 0 &&
                            early_exit.darth_ground_truth_count > 0) {
                            const int gt_q =
                                slot.qid % early_exit.darth_ground_truth_count;
                            const int* gt = early_exit.darth_ground_truth +
                                (long)gt_q * early_exit.darth_ground_truth_width;
                            int hits = count_feature_gt_hits(
                                slot, topk, gt, early_exit.darth_ground_truth_width);
                            recall = (double)hits / (double)topk;
                        }
                        trace << slot.qid << "," << slot.step << ","
                              << slot.found_cnt << "," << slot.top1_dist << ","
                              << slot.topk_dist << "," << slot.gap_ratio << ","
                              << slot.no_improve_count << "," << predicted << ","
                              << std::fixed << std::setprecision(6) << recall << ","
                              << (stop ? 1 : 0) << "\n";
                    }
                }
                if (!progressed) std::this_thread::yield();
            }
            if (trace.is_open()) trace.close();
        });
    }

    // ---- 8. Batch search: multiple kernel launches ----
    std::vector<double> query_latencies;
    query_latencies.reserve(num_queries);

    double t0 = shared::elapsed();

    // H2D all query data at once
    nvtxRangePush("H2D");
    CHECK_CUDA(cudaMemcpy(d_qdata, qdata,
                          sizeof(float) * num_queries * num_dims,
                          cudaMemcpyHostToDevice));
    nvtxRangePop();
    double t_after_h2d = shared::elapsed();

    CHECK_CUDA(cudaDeviceSynchronize());

    // Batch stride = num_blocks * Q (each launch admits this many qids).
    const int batch_stride = num_blocks * Q;
    int num_batches = (num_queries + batch_stride - 1) / batch_stride;
    std::vector<double> batch_latencies;
    batch_latencies.reserve(num_batches);
    INFO("StaticBatch: {} batches of {} blocks x Q={} = {} queries per batch, total {} queries",
         num_batches, num_blocks, Q, batch_stride, num_queries);
#ifdef QUIVER_LATENCY_PROBE
    const int io_sample_cases = quiver_io_sample_cases_from_env();
    std::printf("Quiver static IO sampling: target_cases=%d, total_batches=%d\n",
                io_sample_cases, num_batches);
#endif

    for (int b = 0; b < num_batches; b++) {
        int batch_start = b * batch_stride;
        int batch_end   = std::min(batch_start + batch_stride, num_queries);
        int batch_size  = batch_end - batch_start;
#ifdef QUIVER_LATENCY_PROBE
        sample_current_batch.store(
            should_sample_batch_case(b, num_batches, io_sample_cases),
            std::memory_order_relaxed);
        args.probe_block_trace_offset = b * num_blocks;
#endif

        // Static batches reuse the same slot-indexed resources. Make every
        // launch start from the same empty state as a fresh streaming window.
        reset_static_batch_state(h_comm, h_controls, num_io_slots,
                                 batch_end, num_blocks, pipe_w,
                                 d_queries_done);
        args.total_queries = batch_end;
        if (Q <= 1) {
            args.batch_offset = batch_start;
        } else {
            set_device_int(d_next_query, batch_start);
            args.batch_offset = 0;  // unused by StreamingDispatch
            __atomic_store_n((int32_t*)&h_comm->feed_count, batch_end, __ATOMIC_RELEASE);
        }

        double t_batch_start = shared::elapsed();

        // Q=1 keeps the one-query-per-block StaticDispatch fast path. Q>=2
        // launches a finite streaming kernel over the pre-admitted batch range:
        // d_next_query starts at batch_start and total_queries stops at
        // batch_end, so the kernel exits once this static range completes.
        nvtxRangePush("StaticBatchKernel");
        if (Q <= 1) {
            if (data_type == shared::UINT8)
                persistent_search_kernel<StaticDispatch, uint8_t><<<num_blocks, 128, smem>>>(args);
            else if (data_type == shared::INT8)
                persistent_search_kernel<StaticDispatch, int8_t><<<num_blocks, 128, smem>>>(args);
            else
                persistent_search_kernel<StaticDispatch, float><<<num_blocks, 128, smem>>>(args);
        } else {
            if (data_type == shared::UINT8)
                persistent_search_kernel_qi<StreamingDispatch, uint8_t><<<num_blocks, 128, smem>>>(args);
            else if (data_type == shared::INT8)
                persistent_search_kernel_qi<StreamingDispatch, int8_t><<<num_blocks, 128, smem>>>(args);
            else
                persistent_search_kernel_qi<StreamingDispatch, float><<<num_blocks, 128, smem>>>(args);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        nvtxRangePop();

        double t_batch_end = shared::elapsed();
        double batch_latency_ms = (t_batch_end - t_batch_start) * 1000.0;
        batch_latencies.push_back(batch_latency_ms);
        for (int i = 0; i < batch_size; i++) {
            query_latencies.push_back(batch_latency_ms);
        }
    }
#ifdef QUIVER_LATENCY_PROBE
    sample_current_batch.store(false, std::memory_order_relaxed);
#endif

    double t_after_kernel = shared::elapsed();
    darth_done.store(true, std::memory_order_release);
    if (darth_predictor_thread.joinable()) {
        darth_predictor_thread.join();
    }

    // Results already in mapped-pinned buffers (host-visible after
    // cudaDeviceSynchronize).  Copy to caller's output arrays.
    nvtxRangePush("D2H");
    memcpy(nns, d_out_nns, sizeof(int) * topk * num_queries);
    memcpy(distances, d_out_dist, sizeof(float) * topk * num_queries);
    memcpy(found_cnt, d_out_fc, sizeof(int) * num_queries);
    nvtxRangePop();

    int early_exit_stop_count = 0;
    double early_exit_avg_stop_step = 0.0;
    double avg_finish_step = 0.0;
    std::vector<int> h_stop_step(num_queries);
    CHECK_CUDA(cudaMemcpy(h_stop_step.data(), d_early_exit_stop_step,
                          sizeof(int) * num_queries, cudaMemcpyDeviceToHost));
    long long stop_step_sum = 0;
    for (int step : h_stop_step) {
        if (step >= 0) {
            ++early_exit_stop_count;
            stop_step_sum += step;
        }
    }
    if (early_exit_stop_count > 0) {
        early_exit_avg_stop_step =
            (double)stop_step_sum / (double)early_exit_stop_count;
    }
    std::vector<int> h_finish_step(num_queries);
    CHECK_CUDA(cudaMemcpy(h_finish_step.data(), d_finish_step,
                          sizeof(int) * num_queries, cudaMemcpyDeviceToHost));
    long long finish_step_sum = 0;
    int finish_step_count = 0;
    for (int step : h_finish_step) {
        if (step >= 0) {
            ++finish_step_count;
            finish_step_sum += step;
        }
    }
    if (finish_step_count > 0) {
        avg_finish_step = (double)finish_step_sum / (double)finish_step_count;
    }

    double t1 = shared::elapsed();

    // ---- 9. Stop poll threads ----
    poll_done.store(true, std::memory_order_release);
    for (auto& p : pollers)
        p.join();

    // ---- 10. Report ----
    double search_time = t1 - t0;
    auto io_stats = loader->snapshot_stats();

    std::sort(query_latencies.begin(), query_latencies.end());
    double ql_avg = 0, ql_p50 = 0, ql_p90 = 0, ql_p99 = 0, ql_p999 = 0, ql_max = 0;
    if (!query_latencies.empty()) {
        ql_avg  = std::accumulate(query_latencies.begin(), query_latencies.end(), 0.0)
                  / query_latencies.size();
        ql_p50  = query_latencies[query_latencies.size() * 50 / 100];
        ql_p90  = query_latencies[query_latencies.size() * 90 / 100];
        ql_p99  = query_latencies[query_latencies.size() * 99 / 100];
        ql_p999 = query_latencies[(size_t)(query_latencies.size() * 0.999)];
        ql_max  = query_latencies.back();
    }

    std::ostringstream runtime_params;
    runtime_params << "num_blocks=" << num_blocks << " pipe_width=" << pipe_w
                   << " task_slots=" << num_task_slots
                   << " io_slots=" << num_io_slots
                   << " poll_threads=" << thread_cnt
                   << " poll_submit_batch_size=" << poll_submit_batch_size
                   << " batches=" << num_batches
                   << " queries_per_block=" << Q;
    if (early_exit.enabled()) {
        runtime_params << " early_exit_policy="
                       << (early_exit.policy == EarlyExitPolicy::kDarth ? "darth" :
                           early_exit.policy == EarlyExitPolicy::kDarthTrace ? "darth_trace" : "gpruning")
                       << " early_exit_darth_interval=" << early_exit.darth_interval
                       << " early_exit_darth_threshold=" << early_exit.darth_threshold
                       << " early_exit_gpruning_warmup_steps=" << early_exit.gpruning_warmup_steps
                       << " early_exit_gpruning_patience=" << early_exit.gpruning_patience;
        if (h_darth_comm_slots != nullptr) {
            long long wait_iters = 0;
            long long requests = 0;
            for (int i = 0; i < num_blocks; ++i) {
                wait_iters += h_darth_comm_slots[i].wait_iters_total;
                requests += h_darth_comm_slots[i].request_count;
            }
            runtime_params << " darth_predict_calls=" << darth_predict_calls.load()
                           << " darth_stops=" << darth_stop_count.load()
                           << " darth_predict_ns=" << darth_predict_ns.load()
                           << " darth_gpu_wait_iters=" << wait_iters
                           << " darth_gpu_wait_requests=" << requests;
        }
    }
    runtime_params << " spdk_backpressure="
                   << io_stats.submit_backpressure_count
                   << " spdk_nvme_errors=" << io_stats.nvme_error_count;
    shared::RunReport report;
    report.program = "Quiver";
    report.query = {shared::data_type_name(data_type), topk, ef_search, repeat};
    report.runtime_params = runtime_params.str();
    report.qps = num_queries / search_time;
    report.latency_label = "QueryLatency(ms)";
    report.latency = {ql_avg, ql_p50, ql_p90, ql_p99, ql_p999, ql_max};
    report.io_completed = io_stats.pages_submitted;
    report.search_time_s = search_time;
    report.hot_max_submitted_pages_per_sec =
        io_stats.hot_max_submitted_pages_per_sec;
    report.hot_sample_count = io_stats.hot_sample_count;
    report.num_blocks = num_blocks;
    report.queries_per_block = Q;
    report.pipe_width = pipe_w;
    shared::print_run_report(report);
    print_batch_latency_summary("Quiver StaticBatch", batch_latencies);

    {
        const char* policy_name = "none";
        if (early_exit.enabled()) {
            policy_name =
                early_exit.policy == EarlyExitPolicy::kDarth ? "darth" :
                early_exit.policy == EarlyExitPolicy::kDarthTrace ? "darth_trace" : "gpruning";
        }
        std::cout << "[Quiver] Finish summary: policy=" << policy_name
                  << " avg_finish_step=" << std::fixed << std::setprecision(2)
                  << avg_finish_step
                  << " finished=" << finish_step_count << "/" << num_queries;
        if (early_exit.enabled()) {
            std::cout << " stopped=" << early_exit_stop_count << "/"
                      << num_queries << " ("
                      << std::fixed << std::setprecision(2)
                      << (100.0 * (double)early_exit_stop_count / (double)num_queries)
                      << "%)"
                      << " avg_stop_step=" << std::setprecision(2)
                      << early_exit_avg_stop_step;
        }
        if (h_darth_comm_slots != nullptr) {
            long long wait_iters = 0;
            long long requests = 0;
            for (int i = 0; i < num_blocks; ++i) {
                wait_iters += h_darth_comm_slots[i].wait_iters_total;
                requests += h_darth_comm_slots[i].request_count;
            }
            const long calls = darth_predict_calls.load();
            const double avg_predict_us =
                calls > 0 ? (double)darth_predict_ns.load() / (double)calls / 1000.0 : 0.0;
            const double avg_wait_iters =
                requests > 0 ? (double)wait_iters / (double)requests : 0.0;
            std::cout << " predictor_calls=" << calls
                      << " avg_predict_us=" << std::setprecision(3)
                      << avg_predict_us
                      << " gpu_wait_requests=" << requests
                      << " avg_gpu_wait_iters=" << std::setprecision(1)
                      << avg_wait_iters;
        }
        std::cout << std::defaultfloat << std::endl;
    }

#ifdef QUIVER_LATENCY_PROBE
    {
        std::vector<QuiverProbeQueryTrace> h_traces(probe_num_traces);
        CHECK_CUDA(cudaMemcpy(h_traces.data(), d_probe_traces,
                              sizeof(QuiverProbeQueryTrace) * probe_num_traces,
                              cudaMemcpyDeviceToHost));
        if (std::getenv("QUIVER_PROBE_QUERY_BREAKDOWN") != nullptr) {
            print_query_latency_breakdown(h_traces.data(), probe_num_traces);
        }
        print_global_hop_probe_breakdown(h_traces.data(), probe_num_traces);
        std::vector<QuiverProbeBlockTrace> h_block_traces(probe_num_block_traces);
        CHECK_CUDA(cudaMemcpy(h_block_traces.data(), d_probe_block_traces,
                              sizeof(QuiverProbeBlockTrace) * probe_num_block_traces,
                              cudaMemcpyDeviceToHost));
        print_native_block_timeline_breakdown(
            h_block_traces.data(), probe_num_block_traces, Q);
        print_end_to_end_io_breakdown(poll_stats.end_to_end_io_us);
        write_probe_cdf_csv(std::getenv("QUIVER_PROBE_CSV"),
                            poll_stats.end_to_end_io_us);
    }
#endif

    // ---- 11. Cleanup ----
#ifdef QUIVER_LATENCY_PROBE
    CHECK_CUDA(cudaFree(d_probe_traces));
    CHECK_CUDA(cudaFree(d_probe_block_traces));
#endif
    CHECK_CUDA(cudaFree(d_qdata));
    CHECK_CUDA(cudaFree(d_nid));
    CHECK_CUDA(cudaFree(d_ndist));
    CHECK_CUDA(cudaFree(d_ctx));
    CHECK_CUDA(cudaFree(d_nns_dev));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_fc));
    if (d_early_exit_stop_step) CHECK_CUDA(cudaFree(d_early_exit_stop_step));
    if (d_finish_step) CHECK_CUDA(cudaFree(d_finish_step));
    if (h_darth_comm_slots) CHECK_CUDA(cudaFreeHost(h_darth_comm_slots));
    CHECK_CUDA(cudaFreeHost(d_out_nns));
    CHECK_CUDA(cudaFreeHost(d_out_dist));
    CHECK_CUDA(cudaFreeHost(d_out_fc));
    CHECK_CUDA(cudaFree(d_next_query));
    CHECK_CUDA(cudaFree(d_queries_done));
    CHECK_CUDA(cudaFreeHost(h_comm));

    CHECK_CUDA(cudaHostUnregister(h_buffers));
    loader->destroy_buffer(h_buffers);
    CHECK_CUDA(cudaFreeHost(h_controls));
    CHECK_CUDA(cudaHostUnregister((void*)qdata));
    CHECK_CUDA(cudaHostUnregister(nns));
    CHECK_CUDA(cudaHostUnregister(distances));
    CHECK_CUDA(cudaHostUnregister(found_cnt));
}

} // namespace quiver
