#include "search.cuh"
#include "kernel.cuh"
#include "io_control.cuh"
#include "shared/common/cuda_utils.cuh"
#include "shared/common/logging.hpp"
#include "shared/common/runtime.hpp"
#include "shared/common/run_report.hpp"
#include "shared/common/search_state.cuh"
#include "shared/index/nav_kernel.cuh"
#include "shared/io/runtime_stats.hpp"

#ifdef FLASHANNS_LATENCY_PROBE
#include "probe_report.cuh"
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
#include "../shared/common/light_breakdown.hpp"
#include "../shared/common/ae_csv.hpp"
#endif

#include <thread>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <sstream>
#include <vector>
#include <nvtx3/nvToolsExt.h>

namespace flashanns {

struct PollStats {
    std::atomic<long> io_found{0};
    std::atomic<long> io_completed{0};
    std::atomic<long> poll_iters{0};
    std::atomic<long> ctx_stalls{0};
#ifdef FLASHANNS_LATENCY_PROBE
    std::mutex io_latency_mu;
    std::vector<double> end_to_end_io_us;
#endif
};

#ifdef FLASHANNS_LATENCY_PROBE
static inline int64_t host_steady_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static int flashanns_io_sample_cases_from_env() {
    const char* env = std::getenv("FLASHANNS_IO_SAMPLE_CASES");
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

#ifdef QUIVER_LIGHT_BREAKDOWN
struct FlashBreakdownBatchSample {
    int batch_id = -1;
    double compute_us = 0.0;
    double io_wait_us = 0.0;
    double resume_us = 0.0;
    double kernel_launch_us = 0.0;
    double h2d_us = 0.0;
    double d2h_us = 0.0;
    double latency_us = 0.0;
};

static int64_t flashanns_breakdown_cta_window_ns_from_env() {
    const char* env = std::getenv("FLASHANNS_BREAKDOWN_CTA_WINDOW_US");
    if (env == nullptr || env[0] == '\0') {
        env = std::getenv("QUIVER_BREAKDOWN_CTA_WINDOW_US");
    }
    if (env == nullptr || env[0] == '\0') return 1000LL * 1000LL;
    char* end = nullptr;
    long long value_us = std::strtoll(env, &end, 10);
    if (end == env || value_us <= 0) return 1000LL * 1000LL;
    return value_us * 1000LL;
}
#endif

// ================================================================
// CPU poll loop: via submit_task/poll_task
// ================================================================
static void cpu_poll_loop(
    int tid, int slot_start, int slot_end,
    IOControl* controls, uint8_t* buffers,
    int nodes_per_page,
    std::shared_ptr<shared::IndexLoader> loader,
    int ctx_base, int pool_size,
    int poll_submit_batch_size,
    std::atomic<bool>& done,
#ifdef FLASHANNS_LATENCY_PROBE
    std::atomic<bool>& sample_current_batch,
#endif
    PollStats& stats)
{
    int n = slot_end - slot_start;
    struct Batch {
        int ctx_id;
        std::vector<int> local_indices;
#ifdef FLASHANNS_LATENCY_PROBE
        std::vector<int64_t> submit_ns;
        std::vector<int64_t> complete_ns;
        bool sample_io;
#endif
    };
    std::vector<bool> in_flight(n, false);
    std::vector<int> free_ctx;
    free_ctx.reserve(pool_size);
    for (int i = 0; i < pool_size; i++) free_ctx.push_back(ctx_base + i);
    std::vector<Batch> active;
    active.reserve(pool_size);
#ifdef FLASHANNS_LATENCY_PROBE
    auto record_batch_latency = [&](const Batch& b) {
        std::lock_guard<std::mutex> lock(stats.io_latency_mu);
        append_completed_io_latencies(stats.end_to_end_io_us, b.submit_ns,
                                      b.complete_ns);
    };
#endif
    std::vector<shared::IoRequest> req_buf;
    std::vector<int> idx_buf;
    req_buf.reserve(poll_submit_batch_size);
    idx_buf.reserve(poll_submit_batch_size);
    auto submit_current_batch = [&]() {
        int cid = free_ctx.back(); free_ctx.pop_back();
        Batch batch;
        batch.ctx_id = cid;
        batch.local_indices = std::move(idx_buf);
#ifdef FLASHANNS_LATENCY_PROBE
        batch.sample_io = sample_current_batch.load(std::memory_order_relaxed);
        if (batch.sample_io) {
            const int64_t submit_ns = host_steady_now_ns();
            batch.submit_ns.assign(req_buf.size(), submit_ns);
            batch.complete_ns.assign(req_buf.size(), 0);
            std::vector<shared::TracedIoRequest> traced;
            traced.reserve(req_buf.size());
            for (size_t i = 0; i < req_buf.size(); ++i) {
                traced.push_back({req_buf[i].first, req_buf[i].second,
                                  batch.submit_ns[i],
                                  &batch.complete_ns[i]});
            }
            loader->submit_traced_task(traced, tid, cid);
        } else
#endif
        {
            loader->submit_task(req_buf, tid, cid);
        }
        active.push_back(std::move(batch));
        for (int li : active.back().local_indices) in_flight[li] = true;
        req_buf.clear(); idx_buf.clear();
    };

    long local_found = 0, local_completed = 0, local_iters = 0, local_stalls = 0;

    while (!done.load(std::memory_order_relaxed)) {
        local_iters++;
        int w = 0;
        for (int r = 0; r < (int)active.size(); r++) {
            auto& b = active[r];
            if (loader->poll_task(b.ctx_id)) {
                local_completed += b.local_indices.size();
#ifdef FLASHANNS_LATENCY_PROBE
                if (b.sample_io) {
                    record_batch_latency(b);
                }
#endif
                for (int li : b.local_indices) {
                    in_flight[li] = false;
                    int gi = slot_start + li;
                    __atomic_store_n(&controls[gi].status,
                                     (int32_t)IO_READY, __ATOMIC_RELEASE);
                }
                free_ctx.push_back(b.ctx_id);
            } else {
                if (w != r) active[w] = std::move(active[r]);
                w++;
            }
        }
        active.resize(w);
        if (free_ctx.empty()) { local_stalls++; continue; }
        req_buf.clear(); idx_buf.clear();
        for (int i = 0; i < n; i++) {
            if (in_flight[i]) continue;
            int gi = slot_start + i;
            int32_t st = __atomic_load_n(&controls[gi].status, __ATOMIC_ACQUIRE);
            if (st == IO_REQUESTED) {
                local_found++;
                int blk = controls[gi].node_id / nodes_per_page;
                req_buf.push_back(
                    {blk, buffers + (int64_t)gi * shared::PAGE_SIZE});
                idx_buf.push_back(i);
                if ((int)req_buf.size() >= poll_submit_batch_size) {
                    submit_current_batch();
                    if (free_ctx.empty()) break;
                }
            }
        }
        if (!req_buf.empty() && !free_ctx.empty()) {
            submit_current_batch();
        }
    }
#ifdef FLASHANNS_LATENCY_PROBE
    for (const auto& b : active) {
        if (b.sample_io && loader->poll_task(b.ctx_id)) {
            record_batch_latency(b);
        }
    }
#endif
    stats.io_found.fetch_add(local_found);
    stats.io_completed.fetch_add(local_completed);
    stats.poll_iters.fetch_add(local_iters);
    stats.ctx_stalls.fetch_add(local_stalls);
}

// ================================================================
// Public entry point — simple single-stream persistent search
// ================================================================
void run_persistent_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
    int task_context_count,
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
    int repeat)
{
#ifdef FLASHANNS_FIXED_PIPE_WIDTH
    pipe_w = FLASHANNS_FIXED_PIPE_WIDTH;
#endif
    int num_dims       = (int)layout.num_dims;
    int max_m          = (int)layout.max_m0;
    int nodes_per_page = (int)layout.nodes_per_page;
    int node_size      = (int)layout.node_size;
    int data_size      = (int)layout.data_size;
    int aligned_ef     = (max_m + ef_search + 31) / 32 * 32;

    int B = mini_batch > 0 ? mini_batch : num_queries;
    int num_blocks = B;
    int num_io_slots = num_blocks * pipe_w;

    int poll_submit_batch_size = 256;
    const char* psb_env = getenv("CPU_POLL_SUBMIT_BATCH_SIZE");
    if (psb_env) poll_submit_batch_size = std::atoi(psb_env);

    INFO("Persistent kernel: B={}, pipe_w={}, num_blocks={}, T={}, task_contexts={}, poll_submit_batch_size={}",
         B, pipe_w, num_blocks, thread_cnt, task_context_count,
         poll_submit_batch_size);

    // ---- 1. PQ init ----
    pq->init_device(num_dims, (int)layout.num_data, num_blocks, ef_search);

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
    memset(h_comm, 0, sizeof(PersistentKernelComm));
    h_comm->pipe_w   = pipe_w;
    h_comm->controls = h_controls;
    h_comm->buffers  = h_buffers;

    // ---- 4. Device allocations (single set for num_blocks) ----
    float*    d_qdata;
    int32_t*  d_entry;
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

    CHECK_CUDA(cudaMalloc(&d_qdata, sizeof(float) * B * num_dims));
    CHECK_CUDA(cudaMalloc(&d_entry, sizeof(int32_t) * B));
    CHECK_CUDA(cudaMalloc(&d_nid,   sizeof(uint32_t) * aligned_ef * num_blocks));
    CHECK_CUDA(cudaMalloc(&d_ndist, sizeof(float)    * aligned_ef * num_blocks));
    CHECK_CUDA(cudaMalloc(&d_ctx, sizeof(shared::Data) * num_blocks));
    CHECK_CUDA(cudaMalloc(&d_nns_dev,   sizeof(int)  * topk * num_blocks));
    CHECK_CUDA(cudaMalloc(&d_dist,  sizeof(float)    * topk * num_blocks));
    CHECK_CUDA(cudaMalloc(&d_fc,    sizeof(int)      * num_blocks));
    CHECK_CUDA(cudaMalloc(&d_out_nns,  sizeof(int)   * topk * B));
    CHECK_CUDA(cudaMalloc(&d_out_dist, sizeof(float) * topk * B));
    CHECK_CUDA(cudaMalloc(&d_out_fc,   sizeof(int)   * B));
    CHECK_CUDA(cudaMalloc(&d_next_query,   sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_queries_done, sizeof(int)));

#ifdef QUIVER_LIGHT_BREAKDOWN
    const int num_batches_for_tech = (num_queries + B - 1) / B;
    FlashLightBreakdownTrace* d_breakdown_traces = nullptr;
    const int breakdown_batch_sample_count =
        std::min(num_batches_for_tech,
                 (int)shared::breakdown::QUIVER_LIGHT_BREAKDOWN_SAMPLE_BATCHES);
    const int breakdown_sample_count = breakdown_batch_sample_count;
    FlashBreakdownCtaWindowTrace* d_breakdown_cta_windows = nullptr;
    const int breakdown_cta_window_count = breakdown_sample_count;
    const int64_t breakdown_cta_window_ns =
        flashanns_breakdown_cta_window_ns_from_env();
    CHECK_CUDA(cudaMalloc(&d_breakdown_traces,
                          sizeof(FlashLightBreakdownTrace) *
                              breakdown_sample_count));
    CHECK_CUDA(cudaMemset(d_breakdown_traces, 0,
                          sizeof(FlashLightBreakdownTrace) *
                              breakdown_sample_count));
    CHECK_CUDA(cudaMalloc(&d_breakdown_cta_windows,
                          sizeof(FlashBreakdownCtaWindowTrace) *
                              breakdown_cta_window_count));
    CHECK_CUDA(cudaMemset(d_breakdown_cta_windows, 0,
                          sizeof(FlashBreakdownCtaWindowTrace) *
                              breakdown_cta_window_count));
    std::vector<FlashLightBreakdownTrace> h_breakdown_traces(
        breakdown_sample_count);
    std::vector<FlashBreakdownCtaWindowTrace> h_breakdown_cta_windows(
        breakdown_cta_window_count);
    std::vector<FlashBreakdownBatchSample> breakdown_batch_samples;
    breakdown_batch_samples.reserve(breakdown_batch_sample_count);
#endif

#ifdef FLASHANNS_LATENCY_PROBE
    FlashProbeQueryTrace* d_probe_traces = nullptr;
    CHECK_CUDA(cudaMalloc(&d_probe_traces, sizeof(FlashProbeQueryTrace) * B));
    CHECK_CUDA(cudaMemset(d_probe_traces, 0, sizeof(FlashProbeQueryTrace) * B));
    std::vector<FlashProbeQueryTrace> h_probe_traces(num_queries);
#endif

    // ---- 5. Build kernel args (template) ----
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
    args.d_neighbors_id   = d_nid;
    args.d_neighbors_dist = d_ndist;
    args.d_ctx            = d_ctx;
    args.d_nns            = d_nns_dev;
    args.d_distances      = d_dist;
    args.d_found_cnt      = d_fc;

#ifdef FLASHANNS_LATENCY_PROBE
    args.d_probe_traces = d_probe_traces;
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
    args.d_breakdown_traces = d_breakdown_traces;
    args.d_breakdown_cta_windows = d_breakdown_cta_windows;
    args.breakdown_sample_count = breakdown_sample_count;
    args.breakdown_cta_window_count = breakdown_cta_window_count;
    args.breakdown_cta_window_ns = breakdown_cta_window_ns;
    args.total_search_queries = num_queries;
    args.batch_query_offset = 0;
#endif

    int smem = (int)((sizeof(int) * 3 + sizeof(float) * 2)
                     * (ef_search + max_m));

    // ---- 6. Pin host arrays ----
    CHECK_CUDA(cudaHostRegister((void*)qdata,
               sizeof(float) * num_queries * num_dims, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(nns,
               sizeof(int) * num_queries * topk, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(distances,
               sizeof(float) * num_queries * topk, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(found_cnt,
               sizeof(int) * num_queries, cudaHostRegisterDefault));

    // ---- 7. Start CPU poll threads ----
    std::atomic<bool> poll_done(false);
#ifdef FLASHANNS_LATENCY_PROBE
    std::atomic<bool> sample_current_batch(false);
#endif
    std::vector<std::thread> pollers;
    PollStats poll_stats;

    int slots_per_thread = (num_io_slots + thread_cnt - 1) / thread_cnt;
    int base_contexts_per_thread = task_context_count / thread_cnt;
    int extra_contexts = task_context_count % thread_cnt;

    for (int t = 0; t < thread_cnt; t++) {
        int s0 = t * slots_per_thread;
        int s1 = std::min(s0 + slots_per_thread, num_io_slots);
        int pool_size = base_contexts_per_thread + (t < extra_contexts ? 1 : 0);
        int ctx_base = t * base_contexts_per_thread + std::min(t, extra_contexts);
        pollers.emplace_back([&, t, s0, s1, ctx_base, pool_size]() {
            shared::bind_core(t * 2 + 21);
            cpu_poll_loop(t, s0, s1,
                          h_controls, h_buffers,
                          nodes_per_page, loader,
                          ctx_base, pool_size,
                          poll_submit_batch_size,
                          poll_done,
#ifdef FLASHANNS_LATENCY_PROBE
                          sample_current_batch,
#endif
                          poll_stats);
        });
    }

    // ---- 8. Batch loop: per-batch H2D → nav (GPU) → kernel → D2H ----
    std::vector<double> batch_latencies;
    batch_latencies.reserve((num_queries + B - 1) / B);

    // Stage accumulators across all batches
    double acc_h2d = 0, acc_nav = 0, acc_kernel = 0, acc_d2h = 0;
#ifdef FLASHANNS_LATENCY_PROBE
    const int num_batches = (num_queries + B - 1) / B;
    const int io_sample_cases = flashanns_io_sample_cases_from_env();
    std::printf("FlashANNS IO sampling: target_cases=%d, total_batches=%d\n",
                io_sample_cases, num_batches);
#endif

    CHECK_CUDA(cudaDeviceSynchronize());
    double t0 = shared::elapsed();

    for (int batch_off = 0; batch_off < num_queries; batch_off += B) {
        int batch_size = std::min(B, num_queries - batch_off);
        int blocks_this = std::min(num_blocks, batch_size);
#ifdef FLASHANNS_LATENCY_PROBE
        const int batch_idx = batch_off / B;
        sample_current_batch.store(
            should_sample_batch_case(batch_idx, num_batches, io_sample_cases),
            std::memory_order_relaxed);
#endif

        double tb0 = shared::elapsed();

        // H2D query data
        nvtxRangePush("H2D");
        CHECK_CUDA(cudaMemcpy(d_qdata,
                              qdata + (int64_t)batch_off * num_dims,
                              sizeof(float) * batch_size * num_dims,
                              cudaMemcpyHostToDevice));
        nvtxRangePop();
        double tb_after_h2d = shared::elapsed();

        // Nav: get_entry_kernel writes to d_entry, then GPU-side translate
        nvtxRangePush("Nav");
        if (nav) {
            int init_ef = std::min(ef_search, 5);
            shared::get_entry_kernel(data_type)<<<(batch_size + 1) / 2, 64>>>(
                d_qdata, nav->data_dev, nav->graph_dev, batch_size,
                nav->num_node, nav->data_len, nav->max_m,
                init_ef, nav->start,
                d_entry, nullptr, nullptr);
            int grid = (batch_size + 255) / 256;
            shared::nav_translate_kernel<<<grid, 256>>>(
                d_entry, nav->mapping_dev, batch_size);
        }
        nvtxRangePop(); // Nav

        // Reset counters + IOControl
        CHECK_CUDA(cudaMemset(d_next_query,   0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_queries_done, 0, sizeof(int)));
        for (int i = 0; i < blocks_this * pipe_w; i++)
            h_controls[i].status = IO_IDLE;

        // Update per-batch args
        args.d_all_qdata     = d_qdata;
        args.d_entry_nodes   = nav ? d_entry : nullptr;
        args.num_blocks      = blocks_this;
        args.d_out_nns       = d_out_nns;
        args.d_out_distances = d_out_dist;
        args.d_out_found_cnt = d_out_fc;
        // NOTE: 'total_queries' here is the size of THIS batch, not the
        // entire search. FlashANNS launches one kernel per batch, and each
        // block maps 1:1 to a query within that batch (qid = blockIdx.x).
        h_comm->total_queries = batch_size;
        h_comm->num_blocks    = blocks_this;
#ifdef QUIVER_LIGHT_BREAKDOWN
        args.batch_query_offset = batch_off;
#endif
        double tb_after_nav = shared::elapsed();

        // Launch kernel
        nvtxRangePush("SearchKernel");
#ifdef QUIVER_LIGHT_BREAKDOWN
        double tb_before_kernel_launch = shared::elapsed();
#endif
        if (data_type == shared::UINT8)
            persistent_search_kernel<uint8_t><<<blocks_this, 128, smem>>>(args);
        else if (data_type == shared::INT8)
            persistent_search_kernel<int8_t><<<blocks_this, 128, smem>>>(args);
        else
            persistent_search_kernel<float><<<blocks_this, 128, smem>>>(args);
#ifdef QUIVER_LIGHT_BREAKDOWN
        double tb_after_kernel_launch = shared::elapsed();
#endif

        CHECK_CUDA(cudaDeviceSynchronize());
        nvtxRangePop(); // SearchKernel
        double tb_after_kernel = shared::elapsed();

        // D2H results
        nvtxRangePush("D2H");
        CHECK_CUDA(cudaMemcpy(nns + batch_off * topk, d_out_nns,
                              sizeof(int) * topk * batch_size,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(distances + batch_off * topk, d_out_dist,
                              sizeof(float) * topk * batch_size,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(found_cnt + batch_off, d_out_fc,
                              sizeof(int) * batch_size,
                              cudaMemcpyDeviceToHost));
        nvtxRangePop(); // D2H
#ifdef QUIVER_LIGHT_BREAKDOWN
        double tb_after_d2h = shared::elapsed();
#endif

#ifdef FLASHANNS_LATENCY_PROBE
        CHECK_CUDA(cudaMemcpy(h_probe_traces.data() + batch_off,
                              d_probe_traces,
                              sizeof(FlashProbeQueryTrace) * batch_size,
                              cudaMemcpyDeviceToHost));
#endif

        double tb1 = shared::elapsed();
#ifdef QUIVER_LIGHT_BREAKDOWN
        CHECK_CUDA(cudaMemcpy(h_breakdown_traces.data(),
                              d_breakdown_traces,
                              sizeof(FlashLightBreakdownTrace) *
                                  breakdown_sample_count,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_breakdown_cta_windows.data(),
                              d_breakdown_cta_windows,
                              sizeof(FlashBreakdownCtaWindowTrace) *
                                  breakdown_cta_window_count,
                              cudaMemcpyDeviceToHost));
        {
            double compute_sum_us = 0.0;
            double io_wait_sum_us = 0.0;
            double resume_sum_us = 0.0;
            int compute_count = 0;
            for (const auto& tr : h_breakdown_traces) {
                if (!tr.valid || tr.query_id < batch_off ||
                    tr.query_id >= batch_off + batch_size) {
                    continue;
                }
                std::vector<std::pair<int64_t, int64_t>> wait_intervals;
                const int wait_count = std::min(
                    tr.wait_interval_count,
                    FLASH_BREAKDOWN_MAX_WAIT_INTERVALS);
                wait_intervals.reserve(wait_count);
                for (int w = 0; w < wait_count; ++w) {
                    wait_intervals.push_back(
                        {tr.wait_start_ns[w], tr.wait_end_ns[w]});
                }
                const int hop_compute_count = std::min(
                    tr.compute_interval_count,
                    FLASH_BREAKDOWN_MAX_COMPUTE_INTERVALS);
                const int hop_count = std::max(wait_count, hop_compute_count);
                for (int h = 0; h < hop_count; ++h) {
                    const double wait_us =
                        h < wait_count &&
                                tr.wait_end_ns[h] > tr.wait_start_ns[h]
                            ? static_cast<double>(
                                  tr.wait_end_ns[h] - tr.wait_start_ns[h]) /
                                  1000.0
                            : 0.0;
                    const double compute_us =
                        h < hop_compute_count &&
                                tr.compute_end_ns[h] > tr.compute_start_ns[h]
                            ? static_cast<double>(
                                  tr.compute_end_ns[h] -
                                  tr.compute_start_ns[h]) /
                                  1000.0
                            : 0.0;
                    shared::ae::append_hop_sample(
                        "FlashANNS", tr.query_id, h, wait_us, compute_us);
                }
                compute_sum_us +=
                    static_cast<double>(tr.useful_compute_ns) / 1000.0;
                io_wait_sum_us += shared::breakdown::merged_interval_sum_us(
                    std::move(wait_intervals));
                resume_sum_us +=
                    static_cast<double>(tr.resumption_delay_ns) / 1000.0;
                compute_count++;
            }
            if (compute_count > 0) {
                const int batch_idx = batch_off / B;
                const double n = static_cast<double>(compute_count);
                breakdown_batch_samples.push_back(
                    {batch_idx,
                     compute_sum_us / n,
                     io_wait_sum_us / n,
                     resume_sum_us / n,
                     (tb_after_kernel_launch - tb_before_kernel_launch) * 1e6,
                     (tb_after_h2d - tb0) * 1e6,
                     (tb_after_d2h - tb_after_kernel) * 1e6,
                     (tb1 - tb0) * 1e6});
            }
        }
#endif
        batch_latencies.push_back((tb1 - tb0) * 1000.0);

        acc_h2d    += tb_after_h2d    - tb0;
        acc_nav    += tb_after_nav    - tb_after_h2d;
        acc_kernel += tb_after_kernel - tb_after_nav;
        acc_d2h    += tb1             - tb_after_kernel;
    }

    double t1 = shared::elapsed();

    // ---- 9. Stop poll threads ----
#ifdef FLASHANNS_LATENCY_PROBE
    sample_current_batch.store(false, std::memory_order_relaxed);
#endif
    poll_done.store(true, std::memory_order_release);
    for (auto& p : pollers)
        p.join();

    // ---- 10. Report ----
    double search_time = t1 - t0;
    auto io_stats = loader->snapshot_stats();

    // Per-batch latency summary + per-query approximation (batch_lat / batch_size)
    std::vector<double> sorted_bl = batch_latencies;
    std::sort(sorted_bl.begin(), sorted_bl.end());
    double bl_avg = 0, bl_p50 = 0, bl_p90 = 0, bl_p99 = 0, bl_p999 = 0, bl_max = 0;
    if (!sorted_bl.empty()) {
        bl_avg  = std::accumulate(sorted_bl.begin(), sorted_bl.end(), 0.0) / sorted_bl.size();
        bl_p50  = sorted_bl[sorted_bl.size() * 50 / 100];
        bl_p90  = sorted_bl[sorted_bl.size() * 90 / 100];
        bl_p99  = sorted_bl[sorted_bl.size() * 99 / 100];
        bl_p999 = sorted_bl[(size_t)(sorted_bl.size() * 0.999)];
        bl_max  = sorted_bl.back();
    }

    std::ostringstream runtime_params;
    runtime_params << "batch=" << B << " pipe_width=" << pipe_w
                   << " poll_threads=" << thread_cnt
                   << " spdk_task_contexts=" << task_context_count
                   << " poll_submit_batch_size=" << poll_submit_batch_size;
    shared::RunReport report;
    report.program = "FlashANNS";
    report.query = {shared::data_type_name(data_type), topk, ef_search, repeat};
    report.runtime_params = runtime_params.str();
    report.qps = num_queries / search_time;
    report.latency_label = "BatchLatency(ms)";
    report.latency = {bl_avg, bl_p50, bl_p90, bl_p99, bl_p999, bl_max};
    report.io_completed = io_stats.pages_submitted;
    report.search_time_s = search_time;
    report.hot_max_submitted_pages_per_sec =
        io_stats.hot_max_submitted_pages_per_sec;
    report.hot_sample_count = io_stats.hot_sample_count;
    report.num_blocks = B;
    report.pipe_width = pipe_w;
    shared::print_run_report(report);

#ifdef QUIVER_LIGHT_BREAKDOWN
    {
        std::vector<double> compute_samples;
        std::vector<double> io_wait_samples;
        std::vector<double> resume_samples;
        std::vector<double> scheduling_gap_samples;
        std::vector<double> other_samples;
        std::vector<double> kernel_launch_samples;
        std::vector<double> h2d_samples;
        std::vector<double> d2h_samples;
        std::vector<double> cta_window_samples;
        std::vector<double> cta_active_samples;
        std::vector<double> cta_pure_io_wait_samples;
        std::vector<double> cta_active_ratio_samples;
        compute_samples.reserve(breakdown_batch_samples.size());
        io_wait_samples.reserve(breakdown_batch_samples.size());
        resume_samples.reserve(breakdown_batch_samples.size());
        scheduling_gap_samples.reserve(breakdown_batch_samples.size());
        other_samples.reserve(breakdown_batch_samples.size());
        kernel_launch_samples.reserve(breakdown_batch_samples.size());
        h2d_samples.reserve(breakdown_batch_samples.size());
        d2h_samples.reserve(breakdown_batch_samples.size());
        cta_window_samples.reserve(h_breakdown_cta_windows.size());
        cta_active_samples.reserve(h_breakdown_cta_windows.size());
        cta_pure_io_wait_samples.reserve(h_breakdown_cta_windows.size());
        cta_active_ratio_samples.reserve(h_breakdown_cta_windows.size());

        for (const auto& sample : breakdown_batch_samples) {
            const double body_us =
                std::max(0.0, sample.latency_us - sample.kernel_launch_us -
                                  sample.h2d_us - sample.d2h_us);
            const std::vector<double> scaled_body =
                shared::breakdown::scale_components_to_total(
                    {sample.compute_us, sample.io_wait_us, sample.resume_us},
                    body_us);
            const double compute_us =
                scaled_body.size() == 3 ? scaled_body[0] : 0.0;
            const double io_wait_us =
                scaled_body.size() == 3 ? scaled_body[1] : 0.0;
            const double resume_us =
                scaled_body.size() == 3 ? scaled_body[2] : 0.0;
            const double other_us =
                std::max(0.0, sample.latency_us - compute_us -
                                  io_wait_us - resume_us -
                                  sample.kernel_launch_us - sample.h2d_us -
                                  sample.d2h_us);

            compute_samples.push_back(compute_us);
            io_wait_samples.push_back(io_wait_us);
            resume_samples.push_back(resume_us);
            scheduling_gap_samples.push_back(0.0);
            other_samples.push_back(other_us);
            kernel_launch_samples.push_back(sample.kernel_launch_us);
            h2d_samples.push_back(sample.h2d_us);
            d2h_samples.push_back(sample.d2h_us);
        }
        if (pipe_w == 2) {
            for (const auto& tr : h_breakdown_cta_windows) {
                if (!tr.valid || tr.window_end_ns <= tr.window_start_ns)
                    continue;
                const double window_us =
                    static_cast<double>(tr.window_end_ns -
                                        tr.window_start_ns) /
                    1000.0;
                const double pure_wait_us =
                    static_cast<double>(tr.pure_io_wait_ns) / 1000.0;
                const double active_us =
                    window_us > pure_wait_us ? window_us - pure_wait_us : 0.0;
                const auto cta_activity =
                    shared::breakdown::make_cta_activity_sample(
                        window_us, active_us);
                cta_window_samples.push_back(window_us);
                cta_active_samples.push_back(cta_activity.active_us);
                cta_pure_io_wait_samples.push_back(cta_activity.idle_us);
                cta_active_ratio_samples.push_back(
                    cta_activity.active_ratio_pct);
            }
        }
        if (!compute_samples.empty()) {
            shared::breakdown::print_closed_metric_samples(
                "FlashANNS", compute_samples, io_wait_samples,
                resume_samples, scheduling_gap_samples, other_samples,
                kernel_launch_samples, h2d_samples, d2h_samples,
                cta_window_samples, cta_active_samples,
                cta_pure_io_wait_samples, cta_active_ratio_samples);
            shared::ae::append_closed_breakdown(
                "FlashANNS", compute_samples, io_wait_samples, resume_samples,
                cta_active_ratio_samples, kernel_launch_samples, h2d_samples,
                d2h_samples);
        }
    }
#endif

#ifdef FLASHANNS_LATENCY_PROBE
    print_query_latency_breakdown(
        h_probe_traces.data(), num_queries,
        acc_h2d, acc_nav, acc_d2h, search_time);
    print_block_timeline_breakdown(h_probe_traces.data(), num_queries);
    print_hop_latency_breakdown(h_probe_traces.data(), num_queries);
    print_end_to_end_io_breakdown(poll_stats.end_to_end_io_us);
    write_end_to_end_io_cdf_csv(std::getenv("FLASHANNS_PROBE_CSV"),
                                poll_stats.end_to_end_io_us);
#endif

    // ---- 11. Cleanup ----
#ifdef FLASHANNS_LATENCY_PROBE
    CHECK_CUDA(cudaFree(d_probe_traces));
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
    CHECK_CUDA(cudaFree(d_breakdown_traces));
    CHECK_CUDA(cudaFree(d_breakdown_cta_windows));
#endif
    CHECK_CUDA(cudaFree(d_qdata));
    CHECK_CUDA(cudaFree(d_entry));
    CHECK_CUDA(cudaFree(d_nid));
    CHECK_CUDA(cudaFree(d_ndist));
    CHECK_CUDA(cudaFree(d_ctx));
    CHECK_CUDA(cudaFree(d_nns_dev));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_fc));
    CHECK_CUDA(cudaFree(d_out_nns));
    CHECK_CUDA(cudaFree(d_out_dist));
    CHECK_CUDA(cudaFree(d_out_fc));
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

} // namespace flashanns
