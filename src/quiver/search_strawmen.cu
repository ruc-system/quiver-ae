// Strawman host launchers for §Key Idea ablation.
//
// Builds only when QUIVER_BUILD_STRAWMEN=ON. The host scaffolding mirrors
// run_persistent_search (search.cu) verbatim so that the only ablated
// variable is the GPU kernel itself. Each strawman replaces a single design
// dimension: scheduler discipline, warp granularity, or PQ memory class
// while keeping streaming admission, per-slot resource layout, and IO state
// machine identical to Quiver.

#include "search.cuh"

#ifdef QUIVER_BUILD_STRAWMEN

#include "io_control.cuh"
#include "kernel_core/kernel_helpers.cuh"
#include "kernel_core/policies/dispatch.cuh"
#include "kernel_core/strawmen/kernel_barrier.cuh"
#include "kernel_core/strawmen/kernel_naive_block_multi_query.cuh"
#include "kernel_core/strawmen/kernel_pq_in_gmem.cuh"
#include "kernel_core/strawmen/kernel_pq_in_smem.cuh"

#include "../shared/common/cuda_utils.cuh"
#include "../shared/common/logging.hpp"
#include "../shared/common/runtime.hpp"
#include "../shared/common/run_report.hpp"
#include "../shared/common/search_state.cuh"
#ifdef QUIVER_LIGHT_BREAKDOWN
#include "../shared/common/ae_csv.hpp"
#include "../shared/common/light_breakdown.hpp"
#endif
#include "../shared/io/runtime_stats.hpp"

#include <thread>
#include <atomic>
#include <algorithm>
#ifdef QUIVER_LIGHT_BREAKDOWN
#include <cstring>
#endif
#include <numeric>
#include <nvtx3/nvToolsExt.h>

namespace quiver {

namespace {

struct StrawmanPollStats {
    std::atomic<long> io_found{0};
    std::atomic<long> poll_iters{0};
};

// Verbatim copy of search.cu's cpu_poll_loop minus the latency-probe path.
// We do not share the production version because keeping a strawman-local
// copy avoids changing search.cu's symbol visibility.
static void strawman_poll_loop(
    int tid, int slot_start, int slot_end,
    IOControl* controls, uint8_t* buffers,
    int nodes_per_page,
    std::shared_ptr<shared::IndexLoader> loader,
    int poll_submit_batch_size,
    std::atomic<bool>& done,
    StrawmanPollStats& stats)
{
    int n = slot_end - slot_start;
    std::vector<shared::DirectIoRequest> req_buf;
    req_buf.reserve(poll_submit_batch_size);
    long local_found = 0, local_iters = 0;

    while (!done.load(std::memory_order_relaxed)) {
        local_iters++;
        req_buf.clear();
        for (int i = 0; i < n; i++) {
            int gi = slot_start + i;
            int32_t st = __atomic_load_n(&controls[gi].status, __ATOMIC_ACQUIRE);
            if (st == IO_REQUESTED) {
                int32_t expected = IO_REQUESTED;
                if (!__atomic_compare_exchange_n(&controls[gi].status,
                        &expected, (int32_t)IO_SUBMITTED,
                        false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                    continue;
                local_found++;
                int blk = controls[gi].node_id / nodes_per_page;
                req_buf.push_back({
                    blk,
                    buffers + (int64_t)gi * shared::PAGE_SIZE,
                    &controls[gi].status,
                    (int32_t)IO_READY
                });
                if ((int)req_buf.size() >= poll_submit_batch_size) {
                    loader->submit_direct(req_buf, tid);
                    req_buf.clear();
                }
            }
        }
        if (!req_buf.empty()) loader->submit_direct(req_buf, tid);
    }
    stats.io_found.fetch_add(local_found);
    stats.poll_iters.fetch_add(local_iters);
}

// Single launcher used by all strawmen. The kernel is passed as a typed
// function pointer so we can keep one host-scaffolding implementation and
// add additional strawmen later by exposing additional `strawman_kernel_fn`s without
// duplicating the 250-line host body.
using strawman_kernel_fn = void (*)(PersistentKernelArgs);

void run_strawman_launcher(
    const char* program_name,
    strawman_kernel_fn kernel,
    const shared::Layout& layout,
    int pipe_w,
    int thread_cnt,
    int mini_batch,
    std::shared_ptr<shared::IndexLoader> loader,
    shared::PQSearch* pq,
    shared::NavGraph* nav,
    int enter_point,
    uint8_t* /*starter*/,
    const float* qdata,
    int num_queries,
    int topk,
    int ef_search,
    int* nns,
    float* distances,
    int* found_cnt,
    int queries_per_block,
    shared::DataType data_type,
    int repeat,
    // PQ-in-shared-memory hook: extra dynamic smem per slot (e.g. PQ LUT size).  Default 0 ⇒
    // strawman uses ONLY the per-slot frontier scratch (= barrier / Quiver layout).
    // When > 0 the launcher (a) calls cudaFuncSetAttribute to opt into
    // CC8.0+ dynamic smem above 48 KB, and (b) reports launch errors cleanly
    // when smem exceeds the per-block hardware cap (rather than crashing).
    size_t extra_smem_per_slot = 0,
    // Thread-topology hook: override block thread count. Default 128 (the
    // standard-layout strawmen use the standard Quiver block). Naive block
    // multi-query sets this to 128*Q so each slot owns a full query executor.
    int threads_per_block_override = 128,
    // Dynamic-smem hook: override the default Q-copy frontier layout.  Used by
    // pq-in-gmem, where a single block-wide workspace is reused across slots
    // and the resident per-slot context is explicitly loaded from global memory.
    size_t dynamic_smem_override = 0)
{
    int num_dims       = (int)layout.num_dims;
    int max_m          = (int)layout.max_m0;
    int nodes_per_page = (int)layout.nodes_per_page;
    int node_size      = (int)layout.node_size;
    int data_size      = (int)layout.data_size;
    int aligned_ef     = (max_m + ef_search + 31) / 32 * 32;

    const int Q = queries_per_block > 0 ? queries_per_block : 1;
    int B = mini_batch > 0 ? mini_batch : num_queries;
    int num_blocks = B;
    int num_task_slots = num_blocks * Q;
    int num_io_slots   = num_task_slots * pipe_w;

    int poll_submit_batch_size = 256;
    const char* psb_env = getenv("CPU_POLL_SUBMIT_BATCH_SIZE");
    if (psb_env) poll_submit_batch_size = std::atoi(psb_env);

    int prefeed_depth_cfg = 0;
    const char* pf_env_early = getenv("QUIVER_PREFEED");
    if (pf_env_early) prefeed_depth_cfg = std::atoi(pf_env_early);

    INFO("[strawman {}] B={}, pipe_w={}, num_blocks={}, Q={}, task_slots={}, io_slots={}, T={}, poll_submit_batch_size={}, prefeed={}",
         program_name, B, pipe_w, num_blocks, Q, num_task_slots, num_io_slots,
         thread_cnt, poll_submit_batch_size, prefeed_depth_cfg);

    pq->init_device(num_dims, (int)layout.num_data, num_task_slots, ef_search);

    IOControl* h_controls;
    CHECK_CUDA(cudaHostAlloc(&h_controls, sizeof(IOControl) * num_io_slots,
                             cudaHostAllocMapped));
    for (int i = 0; i < num_io_slots; i++) h_controls[i].status = IO_IDLE;

    uint8_t* h_buffers =
        loader->create_buffer((int64_t)shared::PAGE_SIZE * num_io_slots);
    CHECK_CUDA(cudaHostRegister(h_buffers,
                                (size_t)shared::PAGE_SIZE * num_io_slots,
                                cudaHostRegisterDefault));

    PersistentKernelComm* h_comm;
    CHECK_CUDA(cudaHostAlloc(&h_comm, sizeof(PersistentKernelComm),
                              cudaHostAllocMapped));
    h_comm->pipe_w   = pipe_w;
    h_comm->controls = h_controls;
    h_comm->buffers  = h_buffers;
    h_comm->feed_count = 0;
    h_comm->terminate  = 0;

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
    args.d_out_nns        = d_out_nns;
    args.d_out_distances  = d_out_dist;
    args.d_out_found_cnt  = d_out_fc;

    size_t frontier_smem_per_slot =
        (((sizeof(int) * 3 + sizeof(float) * 2) *
          (size_t)(ef_search + max_m) + 7) & ~size_t{7}) +
        (size_t)node_size;
    size_t smem_total = dynamic_smem_override > 0
        ? dynamic_smem_override
        : (frontier_smem_per_slot + extra_smem_per_slot) * (size_t)Q;

    if (extra_smem_per_slot > 0 || dynamic_smem_override > 0) {
        // Probe per-block dynamic smem cap so we can report a clean error
        // BEFORE attempting cudaFuncSetAttribute (which on overflow returns
        // cudaErrorInvalidValue with a less informative message).
        int device = 0;
        CHECK_CUDA(cudaGetDevice(&device));
        int max_optin_smem = 0;
        CHECK_CUDA(cudaDeviceGetAttribute(
            &max_optin_smem,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        INFO("[strawman {}] smem layout: frontier={} B/slot, extra={} B/slot, "
             "override={} B, total={} KB/block; HW per-block cap (optin) = {} KB",
             program_name, frontier_smem_per_slot, extra_smem_per_slot,
             dynamic_smem_override,
             smem_total / 1024, max_optin_smem / 1024);
        if ((int)smem_total > max_optin_smem) {
            ERROR("[strawman {}] LAUNCH ABORTED: requested {} KB dynamic smem "
                  "exceeds per-block opt-in cap {} KB. This can be the expected "
                  "failure mode for strawmen that deliberately move large query "
                  "context into dynamic shared memory.",
                  program_name, smem_total / 1024, max_optin_smem / 1024);
            // Skip kernel launch but still tear down so the binary exits 0
            // cleanly with the diagnostic above (the failure IS the result).
            CHECK_CUDA(cudaFree(d_qdata));
            CHECK_CUDA(cudaFree(d_nid));
            CHECK_CUDA(cudaFree(d_ndist));
            CHECK_CUDA(cudaFree(d_ctx));
            CHECK_CUDA(cudaFree(d_nns_dev));
            CHECK_CUDA(cudaFree(d_dist));
            CHECK_CUDA(cudaFree(d_fc));
            CHECK_CUDA(cudaFreeHost(d_out_nns));
            CHECK_CUDA(cudaFreeHost(d_out_dist));
            CHECK_CUDA(cudaFreeHost(d_out_fc));
            CHECK_CUDA(cudaFree(d_next_query));
            CHECK_CUDA(cudaFree(d_queries_done));
            CHECK_CUDA(cudaFreeHost(h_comm));
            CHECK_CUDA(cudaHostUnregister(h_buffers));
            loader->destroy_buffer(h_buffers);
            CHECK_CUDA(cudaFreeHost(h_controls));
            printf("%s: LAUNCH_ABORTED (smem overflow), requested_smem_KB=%zu, HW_optin_cap_KB=%d\n",
                   program_name, smem_total / 1024, max_optin_smem / 1024);
            return;
        }
        cudaError_t e = cudaFuncSetAttribute(
            (const void*)kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_total);
        if (e != cudaSuccess) {
            ERROR("[strawman {}] cudaFuncSetAttribute failed for {} KB smem: {}",
                  program_name, smem_total / 1024, cudaGetErrorString(e));
            CHECK_CUDA(e);
        }
    }
    int smem = (int)smem_total;

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
    args.d_breakdown_traces = d_breakdown_traces;
    args.d_breakdown_cta_windows = d_breakdown_cta_windows;
    args.breakdown_sample_count = breakdown_sample_count;
    args.breakdown_cta_window_count = breakdown_cta_window_count;
    args.breakdown_cta_window_ns = 1000 * 1000;
#endif

    CHECK_CUDA(cudaHostRegister((void*)qdata,
               sizeof(float) * num_queries * num_dims, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(nns,
               sizeof(int) * num_queries * topk, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(distances,
               sizeof(float) * num_queries * topk, cudaHostRegisterDefault));
    CHECK_CUDA(cudaHostRegister(found_cnt,
               sizeof(int) * num_queries, cudaHostRegisterDefault));

    std::atomic<bool> poll_done(false);
    std::vector<std::thread> pollers;
    StrawmanPollStats poll_stats;
    int slots_per_thread = (num_io_slots + thread_cnt - 1) / thread_cnt;
    for (int t = 0; t < thread_cnt; t++) {
        int s0 = t * slots_per_thread;
        int s1 = std::min(s0 + slots_per_thread, num_io_slots);
        pollers.emplace_back([&, t, s0, s1]() {
            shared::bind_core(t * 2 + 21);
            strawman_poll_loop(t, s0, s1,
                               h_controls, h_buffers,
                               nodes_per_page, loader,
                               poll_submit_batch_size, poll_done, poll_stats);
        });
    }

    std::vector<double> query_start_times(num_queries, 0.0);
    std::vector<double> query_latencies_buf(num_queries, 0.0);
    int latency_count = 0;

    cudaStream_t h2d_stream;
    cudaStream_t kernel_stream;
    CHECK_CUDA(cudaStreamCreate(&h2d_stream));
    CHECK_CUDA(cudaStreamCreate(&kernel_stream));

    double t0 = shared::elapsed();

    CHECK_CUDA(cudaMemset(d_next_query,   0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_queries_done, 0, sizeof(int)));
    for (int i = 0; i < num_io_slots; i++) h_controls[i].status = IO_IDLE;

    h_comm->total_queries      = num_queries;
    h_comm->num_blocks         = num_blocks;
    h_comm->feed_count         = 0;
    h_comm->terminate          = 0;
    h_comm->queries_completed  = 0;

    CHECK_CUDA(cudaDeviceSynchronize());

    const int prefeed_depth = prefeed_depth_cfg;
    const int max_inflight_queries = num_task_slots + prefeed_depth;
    int initial_feed = std::min(max_inflight_queries, num_queries);

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

    nvtxRangePush("H2D_initial");
    h2d_range_submit(0, initial_feed, /*sync_block=*/true);
    nvtxRangePop();
    double t_after_h2d = shared::elapsed();

    for (int i = 0; i < initial_feed; i++) query_start_times[i] = t0;

    __atomic_store_n((int32_t*)&h_comm->feed_count, initial_feed, __ATOMIC_RELEASE);

    nvtxRangePush("StrawmanKernel");
    int threads_per_block = threads_per_block_override;
    INFO("[strawman {}] launching kernel: <<<{}, {}, smem={} B>>>",
         program_name, num_blocks, threads_per_block, smem);
    kernel<<<num_blocks, threads_per_block, smem, kernel_stream>>>(args);

    int last_done = 0;
    int next_to_feed = initial_feed;
    const int event_ring_cap = std::max(64, 2 * max_inflight_queries);
    std::vector<cudaEvent_t> event_pool(event_ring_cap);
    for (int i = 0; i < event_ring_cap; ++i) {
        CHECK_CUDA(cudaEventCreateWithFlags(&event_pool[i],
                                            cudaEventDisableTiming));
    }
    struct PendingFeed {
        int target;
        cudaEvent_t ev;
    };
    std::vector<PendingFeed> pending_ring(event_ring_cap);
    int pending_head = 0;
    int pending_count = 0;
    int ev_alloc_cursor = 0;
    int admit_min_batch = 32;
    if (const char* env = std::getenv("QUIVER_ADMIT_MIN_BATCH"))
        admit_min_batch = std::max(1, std::atoi(env));

    while (last_done < num_queries) {
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

        int current_done = __atomic_load_n(
            (int32_t*)&h_comm->queries_completed, __ATOMIC_ACQUIRE);
        if (current_done > last_done) {
            double now = shared::elapsed();
            for (int i = last_done; i < current_done; i++) {
                query_latencies_buf[latency_count++] =
                    (now - query_start_times[i]) * 1000.0;
            }
            last_done = current_done;
        }

        int allowed = last_done + max_inflight_queries;
        int new_feed_target = std::min(num_queries, allowed);
        int gap = new_feed_target - next_to_feed;
        bool must_submit = (gap > 0) && (
            gap >= admit_min_batch ||
            new_feed_target == num_queries ||
            pending_count == 0);
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
            next_to_feed = new_feed_target;
            int tail = (pending_head + pending_count) % event_ring_cap;
            pending_ring[tail] = {next_to_feed, ev};
            ++pending_count;
        }
    }
    while (pending_count > 0) {
        CHECK_CUDA(cudaEventSynchronize(pending_ring[pending_head].ev));
        pending_head = (pending_head + 1) % event_ring_cap;
        --pending_count;
    }

    __atomic_store_n((int32_t*)&h_comm->terminate, 1, __ATOMIC_RELEASE);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();
    double t_after_kernel = shared::elapsed();

    nvtxRangePush("D2H");
    memcpy(nns, d_out_nns, sizeof(int) * topk * num_queries);
    memcpy(distances, d_out_dist, sizeof(float) * topk * num_queries);
    memcpy(found_cnt, d_out_fc, sizeof(int) * num_queries);
    nvtxRangePop();

    double t1 = shared::elapsed();

    poll_done.store(true, std::memory_order_release);
    for (auto& p : pollers) p.join();

    double search_time = t1 - t0;
    auto io_stats = loader->snapshot_stats();

    std::sort(query_latencies_buf.begin(),
              query_latencies_buf.begin() + latency_count);
    double ql_avg = 0, ql_p50 = 0, ql_p90 = 0, ql_p99 = 0, ql_p999 = 0, ql_max = 0;
    if (latency_count > 0) {
        ql_avg  = std::accumulate(query_latencies_buf.begin(),
                                  query_latencies_buf.begin() + latency_count,
                                  0.0) / latency_count;
        ql_p50  = query_latencies_buf[latency_count * 50 / 100];
        ql_p90  = query_latencies_buf[latency_count * 90 / 100];
        ql_p99  = query_latencies_buf[latency_count * 99 / 100];
        ql_p999 = query_latencies_buf[(size_t)(latency_count * 0.999)];
        ql_max  = query_latencies_buf[latency_count - 1];
    }

    std::ostringstream runtime_params;
    runtime_params << "num_blocks=" << num_blocks << " pipe_width=" << pipe_w
                   << " poll_threads=" << thread_cnt
                   << " poll_submit_batch_size=" << poll_submit_batch_size
                   << " prefeed=" << prefeed_depth
                   << " queries_per_block=" << Q;
    shared::RunReport report;
    report.program = program_name;
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

#ifdef QUIVER_LIGHT_BREAKDOWN
    if (std::strstr(program_name, "PQ-in-shared-memory") != nullptr) {
        std::vector<QuiverLightBreakdownTrace> traces(
            breakdown_sample_count);
        CHECK_CUDA(cudaMemcpy(
            traces.data(), d_breakdown_traces,
            sizeof(QuiverLightBreakdownTrace) * breakdown_sample_count,
            cudaMemcpyDeviceToHost));
        std::vector<double> compute, io_wait, resume;
        for (int i = 0; i < breakdown_sample_count; ++i) {
            const auto& tr = traces[i];
            if (!tr.valid || tr.query_done_ns <= tr.query_start_ns) continue;
            const int wait_n = std::min(
                tr.wait_interval_count,
                QUIVER_LIGHT_BREAKDOWN_MAX_WAIT_INTERVALS);
            const int compute_n = std::min(
                tr.compute_interval_count,
                QUIVER_LIGHT_BREAKDOWN_MAX_COMPUTE_INTERVALS);
            std::vector<std::pair<int64_t, int64_t>> waits;
            for (int h = 0; h < wait_n; ++h)
                waits.push_back({tr.wait_start_ns[h], tr.wait_end_ns[h]});
            shared::ae::append_hop_intervals(
                "plusk_pq_in_smem", i, tr.wait_start_ns, tr.wait_end_ns,
                wait_n, tr.compute_start_ns, tr.compute_end_ns, compute_n);
            const auto scaled = shared::breakdown::scale_components_to_total(
                {tr.useful_compute_ns / 1000.0,
                 shared::breakdown::merged_interval_sum_us(std::move(waits)),
                 tr.resumption_delay_ns / 1000.0},
                (tr.query_done_ns - tr.query_start_ns) / 1000.0);
            compute.push_back(scaled[0]);
            io_wait.push_back(scaled[1]);
            resume.push_back(scaled[2]);
        }
        shared::breakdown::print_closed_metric_samples(
            "plusk_pq_in_smem", compute, io_wait, resume, {}, {}, {}, {},
            {}, {}, {}, {}, {});
        shared::ae::append_closed_breakdown(
            "plusk_pq_in_smem", compute, io_wait, resume, {});
    }
#endif

    for (auto ev : event_pool) {
        CHECK_CUDA(cudaEventDestroy(ev));
    }
    CHECK_CUDA(cudaStreamDestroy(h2d_stream));
    CHECK_CUDA(cudaStreamDestroy(kernel_stream));
    CHECK_CUDA(cudaFree(d_qdata));
    CHECK_CUDA(cudaFree(d_nid));
    CHECK_CUDA(cudaFree(d_ndist));
    CHECK_CUDA(cudaFree(d_ctx));
    CHECK_CUDA(cudaFree(d_nns_dev));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_fc));
    CHECK_CUDA(cudaFreeHost(d_out_nns));
    CHECK_CUDA(cudaFreeHost(d_out_dist));
    CHECK_CUDA(cudaFreeHost(d_out_fc));
    CHECK_CUDA(cudaFree(d_next_query));
    CHECK_CUDA(cudaFree(d_queries_done));
#ifdef QUIVER_LIGHT_BREAKDOWN
    CHECK_CUDA(cudaFree(d_breakdown_traces));
    CHECK_CUDA(cudaFree(d_breakdown_cta_windows));
#endif
    CHECK_CUDA(cudaFreeHost(h_comm));
    CHECK_CUDA(cudaHostUnregister(h_buffers));
    loader->destroy_buffer(h_buffers);
    CHECK_CUDA(cudaFreeHost(h_controls));
    CHECK_CUDA(cudaHostUnregister((void*)qdata));
    CHECK_CUDA(cudaHostUnregister(nns));
    CHECK_CUDA(cudaHostUnregister(distances));
    CHECK_CUDA(cudaHostUnregister(found_cnt));
}

}  // namespace

void run_barrier_search(
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
    int repeat)
{
    if (data_type == shared::UINT8) {
        run_strawman_launcher(
            "Strawman barrier (per-hop block barrier)",
            strawmen::persistent_search_kernel_barrier<StreamingDispatch, uint8_t>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat);
    } else {
        run_strawman_launcher(
            "Strawman barrier (per-hop block barrier)",
            strawmen::persistent_search_kernel_barrier<StreamingDispatch, float>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat);
    }
}

void run_pq_in_smem_search(
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
    int repeat)
{
    // The PQ-in-shared-memory strawman ablates ONLY the memory class of pq_dists.  The PQ LUT is moved
    // from per-block global memory into per-block dynamic shared memory.
    // Per-slot LUT size = num_pivots(256) × num_chunks × sizeof(float)
    // SIFT-100M (num_chunks=64) → 64 KB/slot.
    //
    // NOTE: must read num_chunks from host_data (already populated by
    // PQSearch::read_data()), NOT from device_data — device_data is only
    // populated inside run_strawman_launcher when it calls pq->init_device().
    const int num_chunks_host = pq->get_data().num_chunks;
    const size_t pq_lut_bytes_per_slot =
        (size_t)shared::PQSearchData::num_pivots
        * (size_t)num_chunks_host
        * sizeof(float);
    if (data_type == shared::UINT8) {
        run_strawman_launcher(
            "Strawman PQ-in-shared-memory (PQ LUT in shared memory)",
            strawmen::persistent_search_kernel_pq_in_smem<StreamingDispatch, uint8_t>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            pq_lut_bytes_per_slot);
    } else if (data_type == shared::INT8) {
        run_strawman_launcher(
            "Strawman PQ-in-shared-memory (PQ LUT in shared memory)",
            strawmen::persistent_search_kernel_pq_in_smem<StreamingDispatch, int8_t>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            pq_lut_bytes_per_slot);
    } else {
        run_strawman_launcher(
            "Strawman PQ-in-shared-memory (PQ LUT in shared memory)",
            strawmen::persistent_search_kernel_pq_in_smem<StreamingDispatch, float>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            pq_lut_bytes_per_slot);
    }
}

void run_pq_in_gmem_search(
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
    int repeat)
{
    const int max_m = (int)layout.max_m0;
    const int node_size = (int)layout.node_size;
    const int aligned_ef = (max_m + ef_search + 31) / 32 * 32;
    const int num_chunks_host = pq->get_data().num_chunks;
    const size_t workspace_bytes = strawmen::pq_in_gmem_workspace_bytes(
        ef_search, max_m, node_size, aligned_ef, topk, num_chunks_host);

    if (data_type == shared::UINT8) {
        run_strawman_launcher(
            "Strawman PQ-in-global-memory (explicit context spill-load)",
            strawmen::persistent_search_kernel_pq_in_gmem<StreamingDispatch, uint8_t>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            /*extra_smem_per_slot=*/0,
            /*threads_per_block_override=*/128,
            workspace_bytes);
    } else if (data_type == shared::INT8) {
        run_strawman_launcher(
            "Strawman PQ-in-global-memory (explicit context spill-load)",
            strawmen::persistent_search_kernel_pq_in_gmem<StreamingDispatch, int8_t>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            /*extra_smem_per_slot=*/0,
            /*threads_per_block_override=*/128,
            workspace_bytes);
    } else {
        run_strawman_launcher(
            "Strawman PQ-in-global-memory (explicit context spill-load)",
            strawmen::persistent_search_kernel_pq_in_gmem<StreamingDispatch, float>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            /*extra_smem_per_slot=*/0,
            /*threads_per_block_override=*/128,
            workspace_bytes);
    }
}

void run_naive_block_multi_query_search(
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
    int repeat)
{
    // Naive block multi-query duplicates the original 128-thread query
    // executor Q times inside one block. Q=1 is the single-query resource
    // baseline; Q>1 adds independent per-slot resources and 128*Q threads.
    const int Q = queries_per_block > 0 ? queries_per_block : 1;
    if (Q < 1 || Q > strawmen::NAIVE_BLOCK_MULTI_QUERY_MAX_Q) {
        ERROR("[strawman S2] Q={} out of supported range [1,{}]",
              Q, strawmen::NAIVE_BLOCK_MULTI_QUERY_MAX_Q);
        return;
    }
    if (pipe_w < 1 || pipe_w > strawmen::NAIVE_BLOCK_MULTI_QUERY_MAX_W) {
        ERROR("[strawman S2] pipe_width={} out of supported range [1,{}]",
              pipe_w, strawmen::NAIVE_BLOCK_MULTI_QUERY_MAX_W);
        return;
    }
    if ((int)layout.max_m0 != kNaiveBlockMultiQueryThreadsPerQuery) {
        ERROR("[strawman S2] naive block multi-query supports only max_m={}, got {}",
              kNaiveBlockMultiQueryThreadsPerQuery, layout.max_m0);
        return;
    }
    const int naive_bmq_threads_per_block =
        Q * kNaiveBlockMultiQueryThreadsPerQuery;
    if (data_type == shared::UINT8) {
        run_strawman_launcher(
            "Strawman naive block multi-query",
            strawmen::persistent_search_kernel_naive_block_multi_query<StreamingDispatch, uint8_t>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            /*extra_smem_per_slot=*/0,
            /*threads_per_block_override=*/naive_bmq_threads_per_block);
    } else if (data_type == shared::INT8) {
        run_strawman_launcher(
            "Strawman naive block multi-query",
            strawmen::persistent_search_kernel_naive_block_multi_query<StreamingDispatch, int8_t>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            /*extra_smem_per_slot=*/0,
            /*threads_per_block_override=*/naive_bmq_threads_per_block);
    } else {
        run_strawman_launcher(
            "Strawman naive block multi-query",
            strawmen::persistent_search_kernel_naive_block_multi_query<StreamingDispatch, float>,
            layout, pipe_w, thread_cnt, mini_batch,
            loader, pq, nav, enter_point, starter,
            qdata, num_queries, topk, ef_search,
            nns, distances, found_cnt, queries_per_block, data_type, repeat,
            /*extra_smem_per_slot=*/0,
            /*threads_per_block_override=*/naive_bmq_threads_per_block);
    }
}

}  // namespace quiver

#endif  // QUIVER_BUILD_STRAWMEN
