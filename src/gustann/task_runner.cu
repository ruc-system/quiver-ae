#include "task_runner.cuh"

#include <algorithm>
#include <cstdlib>
#include <random>

#include "shared/common/gpu_primitives.cuh"
#include "shared/common/logging.hpp"
#include "shared/common/runtime.hpp"
#ifdef QUIVER_LIGHT_BREAKDOWN
#include "../shared/common/light_breakdown.hpp"
#endif
#include "shared/index/nav_kernel.cuh"
#include "shared/index/pq_device.cuh"

#include "kernels/merge_data.cuh"
#include "kernels/pipe_search.cuh"
#include "kernels/visit_select.cuh"

namespace gustann {

using shared::PAGE_SIZE;
using shared::elapsed;

#ifdef GUSTANN_FIXED_PIPE_WIDTH
#define GUSTANN_PIPE_W GUSTANN_FIXED_PIPE_WIDTH
#else
#define GUSTANN_PIPE_W pipe_w_
#endif

static __global__ void init_search(float *qdata, shared::PQSearchData *pq_data,
                                   int stream_offset, int dim) {
  pq_data->init_query(qdata + blockIdx.x * dim, stream_offset);
}

#ifdef GUSTANN_LATENCY_PROBE
static __global__ void record_probe_kernel_start(int64_t *out) {
  if (threadIdx.x == 0) {
    *out = gustann_globaltimer_ns();
  }
}
#endif

namespace {

#ifdef QUIVER_LIGHT_BREAKDOWN
double breakdown_cta_window_seconds_from_env() {
  const char *env = std::getenv("GUSTANN_BREAKDOWN_CTA_WINDOW_US");
  if (env == nullptr || env[0] == '\0') {
    env = std::getenv("QUIVER_BREAKDOWN_CTA_WINDOW_US");
  }
  if (env == nullptr || env[0] == '\0') {
    return 1000.0e-6;
  }
  char *end = nullptr;
  long value_us = std::strtol(env, &end, 10);
  if (end == env || value_us <= 0) {
    return 1000.0e-6;
  }
  return static_cast<double>(value_us) * 1e-6;
}
#endif

#ifdef GUSTANN_LATENCY_PROBE
int batch_tail_cases_per_runner_from_env() {
  const char *env = std::getenv("GUSTANN_BATCH_TAIL_CASES_PER_RUNNER");
  if (env == nullptr) {
    return 2;
  }
  return std::max(0, std::atoi(env));
}
#endif

void destroy_stream_noexcept(cudaStream_t &stream) noexcept {
  if (stream != nullptr) {
    cudaStreamDestroy(stream);
    stream = nullptr;
  }
}

#ifdef QUIVER_LIGHT_BREAKDOWN
void destroy_event_noexcept(cudaEvent_t &event) noexcept {
  if (event != nullptr) {
    cudaEventDestroy(event);
    event = nullptr;
  }
}
#endif

template <class T>
void free_device_noexcept(T *&ptr) noexcept {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

void free_host_pinned_noexcept(void *&ptr) noexcept {
  if (ptr != nullptr) {
    cudaFreeHost(ptr);
    ptr = nullptr;
  }
}

}  // namespace

TaskRunner::TaskRunner(int _tid, int _cid_base, int _mini_batch, int _num_dims,
                       int _topk, int64_t _num_data, int _max_m0,
                       int _ef_search, int _enter_point, uint8_t *_starter,
                       PQSearch *_pq, int nodes_per_page, int node_size,
                       int data_size, DataType data_type_,
                       std::shared_ptr<IndexLoader> _loader, NavGraph *_nav,
                       int _pipe_w) try {
  tid = _tid;
  mini_batch = _mini_batch;
#ifdef GUSTANN_FIXED_PIPE_WIDTH
  pipe_w_ = GUSTANN_FIXED_PIPE_WIDTH;
#else
  pipe_w_ = _pipe_w;
#endif

  num_dims = _num_dims;
  topk = _topk;
  num_data = _num_data;
  max_m0 = _max_m0;
  ef_search = _ef_search;
  aligned_ef = (ef_search + max_m0 + 31) / 32 * 32;
  starter = _starter;
  enter_point = _enter_point;

  nodes_per_page_ = nodes_per_page;
  node_size_ = node_size;
  data_size_ = data_size;
  data_type = data_type_;
  loader = _loader;

  nav_graph = _nav;
  pq = _pq;
  qcnt_ = 0;

  tcnt = (max_m0 + 31) / 32 * 32;

  num_reads = 0;
  time_gpu = 0;
  time_ssd = 0;
  latency = 0;
  cnt_query = 0;
  batch_start_time = 0;
  batch_finished_ = true;
  time_init_issue = 0;
  time_gpu_issue = 0;
  time_ssd_issue = 0;
  time_fin_issue = 0;
#ifdef QUIVER_LIGHT_BREAKDOWN
  time_h2d = 0;
  time_d2h = 0;
  time_resumption = 0;
  time_kernel_launch = 0;
  time_useful_compute = 0;
#endif

  stream_offset_ = _cid_base * mini_batch;
  CHECK_CUDA(cudaStreamCreate(&stream_));
#ifdef QUIVER_LIGHT_BREAKDOWN
  CHECK_CUDA(cudaEventCreate(&h2d_start_event_));
  CHECK_CUDA(cudaEventCreate(&h2d_done_event_));
  CHECK_CUDA(cudaEventCreate(&d2h_start_event_));
  CHECK_CUDA(cudaEventCreate(&d2h_done_event_));
  CHECK_CUDA(cudaEventCreate(&hop_compute_start_event_));
  CHECK_CUDA(cudaEventCreate(&hop_compute_done_event_));
#endif

  CHECK_CUDA(cudaMalloc(&d_qdata_, sizeof(float) * mini_batch * num_dims));
  CHECK_CUDA(cudaMalloc(&d_nns_, sizeof(int) * mini_batch * topk));
  CHECK_CUDA(cudaMalloc(&d_distances_, sizeof(float) * mini_batch * topk));
  CHECK_CUDA(cudaMalloc(&d_found_cnt_, sizeof(int) * mini_batch));
  CHECK_CUDA(
      cudaMalloc(&d_neighbors_id_, sizeof(uint32_t) * aligned_ef * mini_batch));
  CHECK_CUDA(
      cudaMalloc(&d_neighbors_dist_, sizeof(float) * aligned_ef * mini_batch));
  CHECK_CUDA(cudaMalloc(&d_ctx_, sizeof(Data) * mini_batch));
#ifdef GUSTANN_LATENCY_PROBE
  CHECK_CUDA(cudaMalloc(&d_probe_traces_,
                        sizeof(GustannProbeQueryTrace) * mini_batch));
  CHECK_CUDA(cudaMallocHost(&h_probe_traces_,
                            sizeof(GustannProbeQueryTrace) * mini_batch));
  CHECK_CUDA(
      cudaMalloc(&d_probe_active_samples_,
                 sizeof(int) * mini_batch * GUSTANN_PIPE_W));
  CHECK_CUDA(cudaMalloc(&d_probe_kernel_start_, sizeof(int64_t)));
#endif

  sub_lanes_.resize(GUSTANN_PIPE_W);
#ifdef GUSTANN_LATENCY_PROBE
  batch_tail_cases_per_runner_ = batch_tail_cases_per_runner_from_env();
  pending_batch_io_traces_.resize(GUSTANN_PIPE_W);
#endif
  for (int w = 0; w < GUSTANN_PIPE_W; w++) {
    auto &sl = sub_lanes_[w];
    sl.cid = _cid_base * GUSTANN_PIPE_W + w;
    sl.buffer = loader->create_buffer((int64_t)PAGE_SIZE * mini_batch);
    CHECK_CUDA(cudaHostRegister(
        sl.buffer, sizeof(uint8_t) * PAGE_SIZE * mini_batch,
        cudaHostRegisterDefault));
    sl.buffer_registered = true;
    CHECK_CUDA(cudaMallocHost(&sl.request, sizeof(int32_t) * mini_batch));
    sl.pending = false;
    sl.wait_start_time = 0.0;
  }

  state_ = Q_FIN;
  nns_ = nullptr;
  distances_ = nullptr;
  found_cnt_ = nullptr;
  start_pt_ = nullptr;
} catch (...) {
  cleanup();
  throw;
}

TaskRunner::~TaskRunner() { cleanup(); }

void TaskRunner::cleanup() noexcept {
  destroy_stream_noexcept(stream_);
#ifdef QUIVER_LIGHT_BREAKDOWN
  destroy_event_noexcept(h2d_start_event_);
  destroy_event_noexcept(h2d_done_event_);
  destroy_event_noexcept(d2h_start_event_);
  destroy_event_noexcept(d2h_done_event_);
  destroy_event_noexcept(hop_compute_start_event_);
  destroy_event_noexcept(hop_compute_done_event_);
#endif
  free_device_noexcept(d_qdata_);
  free_device_noexcept(d_nns_);
  free_device_noexcept(d_distances_);
  free_device_noexcept(d_found_cnt_);
  free_device_noexcept(d_neighbors_id_);
  free_device_noexcept(d_neighbors_dist_);
  free_device_noexcept(d_ctx_);
#ifdef GUSTANN_LATENCY_PROBE
  free_device_noexcept(d_probe_traces_);
  free_device_noexcept(d_probe_active_samples_);
  free_device_noexcept(d_probe_kernel_start_);
  if (h_probe_traces_ != nullptr) {
    void *probe_ptr = h_probe_traces_;
    free_host_pinned_noexcept(probe_ptr);
    h_probe_traces_ = nullptr;
  }
#endif

  for (auto &sl : sub_lanes_) {
    if (sl.buffer_registered && sl.buffer != nullptr) {
      cudaHostUnregister(sl.buffer);
      sl.buffer_registered = false;
    }
    if (sl.request != nullptr) {
      void *request_ptr = sl.request;
      free_host_pinned_noexcept(request_ptr);
      sl.request = nullptr;
    }
    if (sl.buffer != nullptr && loader) {
      loader->destroy_buffer(sl.buffer);
      sl.buffer = nullptr;
    }
    sl.buffer = nullptr;
    sl.pending = false;
    sl.wait_start_time = 0.0;
  }
}

void TaskRunner::init_query(const float *qdata, int _qcnt, int *_nns,
                            float *_dis, int *_found_cnt, int *_start_pt,
                            int _query_base) {
  batch_finished_ = false;
  time_init_issue -= elapsed();
  latency -= elapsed();
  cnt_query++;
  qcnt_ = _qcnt;
  nns_ = _nns;
  distances_ = _dis;
  found_cnt_ = _found_cnt;
  start_pt_ = _start_pt;
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
  query_base_ = _query_base;
#else
  (void)_query_base;
#endif
#ifdef GUSTANN_LATENCY_PROBE
  current_batch_has_kept_io_trace_ = false;
#endif

  for (int w = 0; w < GUSTANN_PIPE_W; w++) {
    sub_lanes_[w].pending = false;
    sub_lanes_[w].wait_start_time = 0.0;
#ifdef GUSTANN_LATENCY_PROBE
    pending_batch_io_traces_[w] = {};
#endif
  }
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
  current_hop_index_ = 0;
  pending_hop_index_ = 0;
  pending_hop_wait_us_ = 0.0;
#endif
  pending_hop_compute_start_ = 0.0;

  // Synchronize previous batch's D2H before starting timing for this batch.
  // batch_start_time must be recorded after this sync so that the previous
  // batch's completion time is not charged to this batch's latency.
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  batch_start_time = elapsed();
#ifdef QUIVER_LIGHT_BREAKDOWN
  breakdown_batch_start_time_ = batch_start_time;
  breakdown_batch_h2d_time_ = 0.0;
  breakdown_batch_d2h_time_ = 0.0;
  breakdown_batch_ssd_time_ = 0.0;
  breakdown_batch_resumption_time_ = 0.0;
  breakdown_batch_compute_time_ = 0.0;
  breakdown_batch_kernel_launch_time_ = 0.0;
  start_breakdown_cta_window();
#endif
  time_gpu -= elapsed();

  CHECK_CUDA(cudaMemsetAsync(d_found_cnt_, 0, sizeof(int) * qcnt_, stream_));
  CHECK_CUDA(cudaMemsetAsync(d_ctx_, 0, sizeof(Data) * qcnt_, stream_));
#ifdef GUSTANN_LATENCY_PROBE
  CHECK_CUDA(cudaMemsetAsync(d_probe_traces_, 0,
                             sizeof(GustannProbeQueryTrace) * qcnt_,
                             stream_));
  CHECK_CUDA(cudaMemsetAsync(d_probe_active_samples_, 0xff,
                             sizeof(int) * mini_batch * GUSTANN_PIPE_W,
                             stream_));
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
  CHECK_CUDA(cudaEventRecord(h2d_start_event_, stream_));
#endif
  CHECK_CUDA(cudaMemcpyAsync(d_qdata_, qdata, sizeof(float) * qcnt_ * num_dims,
                             cudaMemcpyHostToDevice, stream_));
#ifdef QUIVER_LIGHT_BREAKDOWN
  CHECK_CUDA(cudaEventRecord(h2d_done_event_, stream_));
  h2d_event_pending_ = true;
#endif

#ifdef QUIVER_LIGHT_BREAKDOWN
  double launch_start = elapsed();
#endif
  init_search<<<qcnt_, 64, 0, stream_>>>(d_qdata_, pq->get_device_ptr(),
                                         stream_offset_, num_dims);
#ifdef QUIVER_LIGHT_BREAKDOWN
  double launch_elapsed = elapsed() - launch_start;
  time_kernel_launch += launch_elapsed;
  breakdown_batch_kernel_launch_time_ += launch_elapsed;
#endif

  if (nav_graph) {
    int init_ef = std::min(ef_search, 5);
    int dim = nav_graph->data_len;
#ifdef QUIVER_LIGHT_BREAKDOWN
    launch_start = elapsed();
#endif
    shared::get_entry_kernel(data_type)<<<(qcnt_ + 1) / 2, 64, 0, stream_>>>(
        d_qdata_, nav_graph->data_dev, nav_graph->graph_dev, qcnt_,
        nav_graph->num_node, dim, nav_graph->max_m, init_ef, nav_graph->start,
        sub_lanes_[0].request, d_neighbors_id_, d_neighbors_dist_);
#ifdef QUIVER_LIGHT_BREAKDOWN
    launch_elapsed = elapsed() - launch_start;
    time_kernel_launch += launch_elapsed;
    breakdown_batch_kernel_launch_time_ += launch_elapsed;
#endif
  } else {
    for (int j = 0; j < qcnt_; j++) {
      memcpy(sub_lanes_[0].buffer + PAGE_SIZE * j, starter, PAGE_SIZE);
      sub_lanes_[0].request[j] = enter_point;
    }
  }

  state_ = Q_INIT;
  time_init_issue += elapsed();
}

#ifdef QUIVER_LIGHT_BREAKDOWN
void TaskRunner::set_breakdown_total_batches(int total_batches) {
  breakdown_total_batches_ = total_batches;
}

void TaskRunner::start_breakdown_cta_window() {
  if (pipe_w_ != 1 || breakdown_cta_window_active_) {
    return;
  }
  const double now = elapsed();
  breakdown_cta_window_active_ = true;
  breakdown_cta_waiting_ = false;
  breakdown_cta_window_start_time_ = now;
  breakdown_cta_window_deadline_time_ =
      now + breakdown_cta_window_seconds_from_env();
  breakdown_cta_pure_wait_time_ = 0.0;
  breakdown_cta_wait_start_time_ = 0.0;
}

void TaskRunner::finish_breakdown_cta_window(double now) {
  if (!breakdown_cta_window_active_) {
    return;
  }
  double end_time = now;
  if (breakdown_cta_window_deadline_time_ > 0.0 &&
      end_time > breakdown_cta_window_deadline_time_) {
    end_time = breakdown_cta_window_deadline_time_;
  }
  if (breakdown_cta_waiting_) {
    if (end_time > breakdown_cta_wait_start_time_) {
      breakdown_cta_pure_wait_time_ +=
          end_time - breakdown_cta_wait_start_time_;
    }
    breakdown_cta_waiting_ = false;
    breakdown_cta_wait_start_time_ = 0.0;
  }
  if (end_time > breakdown_cta_window_start_time_) {
    cta_window_samples.push_back(
        {(end_time - breakdown_cta_window_start_time_) * 1e6,
         breakdown_cta_pure_wait_time_ * 1e6});
  }
  breakdown_cta_window_active_ = false;
  breakdown_cta_window_start_time_ = 0.0;
  breakdown_cta_window_deadline_time_ = 0.0;
  breakdown_cta_pure_wait_time_ = 0.0;
  breakdown_cta_wait_start_time_ = 0.0;
  breakdown_cta_waiting_ = false;
}

void TaskRunner::tick_breakdown_cta_window() {
  if (!breakdown_cta_window_active_) {
    return;
  }
  const double now = elapsed();
  if (breakdown_cta_window_deadline_time_ > 0.0 &&
      now >= breakdown_cta_window_deadline_time_) {
    finish_breakdown_cta_window(now);
  }
}

void TaskRunner::start_breakdown_cta_wait() {
  if (!breakdown_cta_window_active_) {
    return;
  }
  const double now = elapsed();
  if (breakdown_cta_window_deadline_time_ > 0.0 &&
      now >= breakdown_cta_window_deadline_time_) {
    finish_breakdown_cta_window(now);
    return;
  }
  if (!breakdown_cta_waiting_) {
    breakdown_cta_wait_start_time_ = now;
    breakdown_cta_waiting_ = true;
  }
}

void TaskRunner::end_breakdown_cta_wait(double now) {
  if (!breakdown_cta_window_active_ || !breakdown_cta_waiting_) {
    return;
  }
  double wait_end = now;
  if (breakdown_cta_window_deadline_time_ > 0.0 &&
      wait_end > breakdown_cta_window_deadline_time_) {
    wait_end = breakdown_cta_window_deadline_time_;
  }
  if (wait_end > breakdown_cta_wait_start_time_) {
    breakdown_cta_pure_wait_time_ += wait_end - breakdown_cta_wait_start_time_;
  }
  breakdown_cta_waiting_ = false;
  breakdown_cta_wait_start_time_ = 0.0;
}

void TaskRunner::record_breakdown_batch_sample(double done_time) {
  if (breakdown_total_batches_ <= 0 || qcnt_ <= 0) {
    return;
  }
  const int sample_count = std::min(
      breakdown_total_batches_,
      static_cast<int>(
          shared::breakdown::QUIVER_LIGHT_BREAKDOWN_SAMPLE_BATCHES));
  if (sample_count <= 0) {
    return;
  }

  const int batch_id = query_base_ / mini_batch;
  if (shared::breakdown::sample_index(batch_id, breakdown_total_batches_,
                                      sample_count) < 0) {
    return;
  }
  const double latency_us =
      done_time > breakdown_batch_start_time_
          ? (done_time - breakdown_batch_start_time_) * 1e6
          : 0.0;
  breakdown_batch_samples.push_back(
      {batch_id, breakdown_batch_compute_time_ * 1e6,
       breakdown_batch_ssd_time_ * 1e6,
       breakdown_batch_resumption_time_ * 1e6,
       breakdown_batch_kernel_launch_time_ * 1e6,
       breakdown_batch_h2d_time_ * 1e6, breakdown_batch_d2h_time_ * 1e6,
       latency_us});
}
#endif

void TaskRunner::submit_ssd_sublane(int w) {
  auto &sl = sub_lanes_[w];
  std::vector<std::pair<int, void *>> pages;
#ifdef GUSTANN_LATENCY_PROBE
  std::vector<shared::TracedIoRequest> traced_pages;
  GustannBatchIoTrace trace;
  if (batch_tail_cases_per_runner_ > 0) {
    trace.query_base = query_base_;
    trace.qcnt = qcnt_;
    trace.hop_index = current_hop_index_;
    trace.sublane = w;
    trace.enqueue_ns.assign(qcnt_, 0);
    trace.submit_ns.assign(qcnt_, 0);
    trace.complete_ns.assign(qcnt_, 0);
  }
#endif
  for (int j = 0; j < qcnt_; j++) {
    if (sl.request[j] != -1) {
      if (!(sl.request[j] >= 0 && sl.request[j] < num_data)) {
        fprintf(stderr, "pipe_search: invalid request %d\n", sl.request[j]);
        std::abort();
      }
      int blockid = sl.request[j] / nodes_per_page_;
      pages.emplace_back(blockid, sl.buffer + PAGE_SIZE * j);
#ifdef GUSTANN_LATENCY_PROBE
      if (batch_tail_cases_per_runner_ > 0) {
        traced_pages.push_back({blockid, sl.buffer + PAGE_SIZE * j, j,
                                trace.enqueue_ns.data(), trace.submit_ns.data(),
                                trace.complete_ns.data()});
        trace.active_ios++;
      }
#endif
      num_reads++;
    }
  }
  if (!pages.empty()) {
#ifdef GUSTANN_LATENCY_PROBE
    if (batch_tail_cases_per_runner_ > 0) {
      pending_batch_io_traces_[w] = std::move(trace);
      loader->submit_traced_task(traced_pages, tid, sl.cid);
    } else
#endif
    {
    loader->submit_task(pages, tid, sl.cid);
    }
    sl.pending = true;
    sl.wait_start_time = elapsed();
  } else {
#ifdef GUSTANN_LATENCY_PROBE
    pending_batch_io_traces_[w] = {};
#endif
    sl.wait_start_time = 0.0;
  }
}

#ifdef GUSTANN_LATENCY_PROBE
void TaskRunner::maybe_keep_batch_io_tail_case(int w) {
  if (batch_tail_cases_per_runner_ <= 0) {
    return;
  }
  auto &trace = pending_batch_io_traces_[w];
  if (trace.active_ios <= 0) {
    return;
  }

  bool has_complete = false;
  for (int64_t complete_ns : trace.complete_ns) {
    if (complete_ns > 0) {
      has_complete = true;
      break;
    }
  }
  if (!has_complete) {
    trace = {};
    return;
  }

  batch_io_trace_seen_++;
  if (static_cast<int>(batch_io_tail_cases.size()) <
      batch_tail_cases_per_runner_) {
    batch_io_tail_cases.push_back(std::move(trace));
    current_batch_has_kept_io_trace_ = true;
  } else {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist(0, batch_io_trace_seen_ - 1);
    const uint64_t slot = dist(rng);
    if (slot < static_cast<uint64_t>(batch_tail_cases_per_runner_)) {
      batch_io_tail_cases[slot] = std::move(trace);
      current_batch_has_kept_io_trace_ = true;
    }
  }
  trace = {};
}
#endif

template <class T>
void TaskRunner::launch_merge_data(int w) {
  auto &sl = sub_lanes_[w];
#ifdef GUSTANN_LATENCY_PROBE
  record_probe_kernel_start<<<1, 1, 0, stream_>>>(d_probe_kernel_start_);
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
  const double launch_start = elapsed();
#endif
  merge_data_kernel<<<qcnt_, tcnt,
                      ((sizeof(int) * 3 + sizeof(float) * 2) *
                       (ef_search + max_m0)),
                      stream_>>>(sl.buffer, sl.request, pq->device_data.num_chunks,
                                 pq->device_data.pq_dists,
                                 pq->device_data.compressed_data,
                                 nodes_per_page_, node_size_, data_size_,
                                 stream_offset_, max_m0, ef_search,
                                 d_neighbors_id_, d_neighbors_dist_, d_ctx_
#ifdef GUSTANN_LATENCY_PROBE
                                 ,
                                 d_probe_traces_, d_probe_active_samples_,
                                 mini_batch, w, pending_hop_index_, query_base_,
                                 d_probe_kernel_start_
#endif
                                 );
#ifdef QUIVER_LIGHT_BREAKDOWN
  const double launch_elapsed = elapsed() - launch_start;
  time_kernel_launch += launch_elapsed;
  breakdown_batch_kernel_launch_time_ += launch_elapsed;
#endif
}

template <class T>
void TaskRunner::launch_visit_select(int w) {
  auto &sl = sub_lanes_[w];
#ifdef QUIVER_LIGHT_BREAKDOWN
  const double launch_start = elapsed();
#endif
  pipe_visit_select_kernel<T><<<(qcnt_ + 1) / 2, 64, 0, stream_>>>(
      d_qdata_, sl.buffer, sl.request, num_dims, topk, d_nns_, d_distances_,
      d_found_cnt_, nodes_per_page_, node_size_, max_m0, ef_search,
      d_neighbors_id_, d_neighbors_dist_, d_ctx_, qcnt_
#ifdef GUSTANN_LATENCY_PROBE
      ,
      d_probe_traces_, d_probe_active_samples_, mini_batch, w
#endif
      );
#ifdef QUIVER_LIGHT_BREAKDOWN
  const double launch_elapsed = elapsed() - launch_start;
  time_kernel_launch += launch_elapsed;
  breakdown_batch_kernel_launch_time_ += launch_elapsed;
#endif
}

void TaskRunner::process_sublane_gpu(int w) {
  auto &sl = sub_lanes_[w];
  if (data_type == shared::UINT8) {
    launch_merge_data<uint8_t>(w);
#ifdef NOFUSE_ABLATION
#ifdef QUIVER_LIGHT_BREAKDOWN
    const double launch_start = elapsed();
#endif
    visit_exact_dist_kernel<uint8_t><<<(qcnt_ + 1) / 2, 64, 0, stream_>>>(
        d_qdata_, sl.buffer, sl.request, num_dims, topk, d_nns_, d_distances_,
        d_found_cnt_, nodes_per_page_, node_size_, data_size_, qcnt_);
#ifdef QUIVER_LIGHT_BREAKDOWN
    const double launch_elapsed = elapsed() - launch_start;
    time_kernel_launch += launch_elapsed;
    breakdown_batch_kernel_launch_time_ += launch_elapsed;
#endif
#else
    launch_visit_select<uint8_t>(w);
#endif
  } else if (data_type == shared::INT8) {
    launch_merge_data<int8_t>(w);
#ifdef NOFUSE_ABLATION
#ifdef QUIVER_LIGHT_BREAKDOWN
    const double launch_start = elapsed();
#endif
    visit_exact_dist_kernel<int8_t><<<(qcnt_ + 1) / 2, 64, 0, stream_>>>(
        d_qdata_, sl.buffer, sl.request, num_dims, topk, d_nns_, d_distances_,
        d_found_cnt_, nodes_per_page_, node_size_, data_size_, qcnt_);
#ifdef QUIVER_LIGHT_BREAKDOWN
    const double launch_elapsed = elapsed() - launch_start;
    time_kernel_launch += launch_elapsed;
    breakdown_batch_kernel_launch_time_ += launch_elapsed;
#endif
#else
    launch_visit_select<int8_t>(w);
#endif
  } else if (data_type == shared::FLOAT) {
    launch_merge_data<float>(w);
#ifdef NOFUSE_ABLATION
#ifdef QUIVER_LIGHT_BREAKDOWN
    const double launch_start = elapsed();
#endif
    visit_exact_dist_kernel<float><<<(qcnt_ + 1) / 2, 64, 0, stream_>>>(
        d_qdata_, sl.buffer, sl.request, num_dims, topk, d_nns_, d_distances_,
        d_found_cnt_, nodes_per_page_, node_size_, data_size_, qcnt_);
#ifdef QUIVER_LIGHT_BREAKDOWN
    const double launch_elapsed = elapsed() - launch_start;
    time_kernel_launch += launch_elapsed;
    breakdown_batch_kernel_launch_time_ += launch_elapsed;
#endif
#else
    launch_visit_select<float>(w);
#endif
  } else {
    ERROR("Invalid Data Type!");
    exit(-1);
  }
}

void TaskRunner::select_next_sublane(int w) {
  auto &sl = sub_lanes_[w];
#ifdef QUIVER_LIGHT_BREAKDOWN
  const double launch_start = elapsed();
#endif
  pipe_select_next_kernel<<<(qcnt_ + 1) / 2, 64, 0, stream_>>>(
      max_m0, ef_search, d_neighbors_id_, d_neighbors_dist_, d_ctx_,
      sl.request, qcnt_);
#ifdef QUIVER_LIGHT_BREAKDOWN
  const double launch_elapsed = elapsed() - launch_start;
  time_kernel_launch += launch_elapsed;
  breakdown_batch_kernel_launch_time_ += launch_elapsed;
#endif
}

void TaskRunner::finish_query() {
  time_fin_issue -= elapsed();
#ifdef QUIVER_LIGHT_BREAKDOWN
  CHECK_CUDA(cudaEventRecord(d2h_start_event_, stream_));
#endif
  CHECK_CUDA(cudaMemcpyAsync(nns_, d_nns_, qcnt_ * topk * sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpyAsync(distances_, d_distances_,
                             qcnt_ * topk * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpyAsync(found_cnt_, d_found_cnt_, qcnt_ * sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
#ifdef QUIVER_LIGHT_BREAKDOWN
  CHECK_CUDA(cudaEventRecord(d2h_done_event_, stream_));
  d2h_event_pending_ = true;
#endif
#ifdef GUSTANN_LATENCY_PROBE
  CHECK_CUDA(cudaMemcpyAsync(h_probe_traces_, d_probe_traces_,
                             qcnt_ * sizeof(GustannProbeQueryTrace),
                             cudaMemcpyDeviceToHost, stream_));
#endif
  state_ = Q_RES;
  time_fin_issue += elapsed();
}

bool TaskRunner::update_state() {
  if (batch_finished_) {
    return true;
  }
#ifdef QUIVER_LIGHT_BREAKDOWN
  tick_breakdown_cta_window();
#endif
  try {
    switch (state_) {
    case Q_INIT: {
      auto err = cudaStreamQuery(stream_);
      if (err == cudaSuccess) {
#ifdef QUIVER_LIGHT_BREAKDOWN
        if (h2d_event_pending_) {
          float h2d_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, h2d_start_event_,
                                          h2d_done_event_));
          const double h2d_s = static_cast<double>(h2d_ms) * 1e-3;
          time_h2d += h2d_s;
          breakdown_batch_h2d_time_ += h2d_s;
          h2d_event_pending_ = false;
        }
#endif
        time_gpu += elapsed();
        if (nav_graph) {
          nav_graph->translate(sub_lanes_[0].request, qcnt_);
        }
        time_ssd_issue -= elapsed();
        submit_ssd_sublane(0);
        time_ssd_issue += elapsed();
        state_ = Q_BOOT;
      } else if (err != cudaErrorNotReady) {
        CHECK_CUDA(err);
      }
      break;
    }

    case Q_BOOT: {
      if (!sub_lanes_[0].pending) {
        break;
      }
      bool ready = loader->poll_task(sub_lanes_[0].cid);
      if (ready) {
        double now = elapsed();
#ifdef QUIVER_LIGHT_BREAKDOWN
        end_breakdown_cta_wait(now);
#endif
        double wait_s = now - sub_lanes_[0].wait_start_time;
        time_ssd += wait_s;
#ifdef QUIVER_LIGHT_BREAKDOWN
        breakdown_batch_ssd_time_ += wait_s;
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
        pending_hop_index_ = current_hop_index_;
        pending_hop_wait_us_ = wait_s * 1e6;
#endif
#ifdef GUSTANN_LATENCY_PROBE
        maybe_keep_batch_io_tail_case(0);
#endif
        sub_lanes_[0].pending = false;
        sub_lanes_[0].wait_start_time = 0.0;

        time_gpu_issue -= elapsed();
        pending_hop_compute_start_ = elapsed();
#ifdef QUIVER_LIGHT_BREAKDOWN
        time_resumption += pending_hop_compute_start_ - now;
        breakdown_batch_resumption_time_ +=
            pending_hop_compute_start_ - now;
#endif
        time_gpu -= pending_hop_compute_start_;

#ifdef QUIVER_LIGHT_BREAKDOWN
        CHECK_CUDA(cudaEventRecord(hop_compute_start_event_, stream_));
#endif
        process_sublane_gpu(0);
#ifdef NOFUSE_ABLATION
        for (int w = 0; w < GUSTANN_PIPE_W; w++) {
#else
        for (int w = 1; w < GUSTANN_PIPE_W; w++) {
#endif
          select_next_sublane(w);
        }
#ifdef QUIVER_LIGHT_BREAKDOWN
        CHECK_CUDA(cudaEventRecord(hop_compute_done_event_, stream_));
        hop_compute_event_pending_ = true;
#endif

        state_ = Q_BOOT_GPU;
#ifdef QUIVER_LIGHT_BREAKDOWN
      } else {
        start_breakdown_cta_wait();
#endif
      }
      break;
    }

    case Q_BOOT_GPU: {
      auto err = cudaStreamQuery(stream_);
      if (err == cudaSuccess) {
        double now = elapsed();
        double hop_compute_s = now - pending_hop_compute_start_;
#ifdef QUIVER_LIGHT_BREAKDOWN
        if (hop_compute_event_pending_) {
          float hop_compute_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(
              &hop_compute_ms, hop_compute_start_event_,
              hop_compute_done_event_));
          hop_compute_s = static_cast<double>(hop_compute_ms) * 1e-3;
          hop_compute_event_pending_ = false;
        }
#endif
        time_gpu += now;
        time_gpu_issue += elapsed();
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
        hop_latencies.push_back(
            {pending_hop_index_, pending_hop_wait_us_,
             hop_compute_s * 1e6});
        current_hop_index_ = pending_hop_index_ + 1;
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
        breakdown_batch_compute_time_ += hop_compute_s;
        time_useful_compute += hop_compute_s;
#endif

        time_ssd_issue -= elapsed();
        for (int w = 0; w < GUSTANN_PIPE_W; w++) {
          submit_ssd_sublane(w);
        }
        time_ssd_issue += elapsed();

        state_ = Q_PIPE;
      } else if (err != cudaErrorNotReady) {
        CHECK_CUDA(err);
      }
      break;
    }

    case Q_PIPE: {
      std::vector<int> completed;
#ifdef QUIVER_LIGHT_BREAKDOWN
      double ready_time_sum = 0.0;
      int ready_time_count = 0;
#endif
      for (int w = 0; w < GUSTANN_PIPE_W; w++) {
        if (sub_lanes_[w].pending) {
          bool ready = loader->poll_task(sub_lanes_[w].cid);
          if (ready) {
            double now = elapsed();
            double wait_s = now - sub_lanes_[w].wait_start_time;
            time_ssd += wait_s;
#ifdef QUIVER_LIGHT_BREAKDOWN
            ready_time_sum += now;
            ready_time_count++;
            breakdown_batch_ssd_time_ += wait_s;
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
            pending_hop_wait_us_ += wait_s * 1e6;
#endif
#ifdef GUSTANN_LATENCY_PROBE
            maybe_keep_batch_io_tail_case(w);
#endif
            sub_lanes_[w].pending = false;
            sub_lanes_[w].wait_start_time = 0.0;
            completed.push_back(w);
          }
        }
      }
      if (completed.empty()) {
        bool any_pending = false;
        for (int w = 0; w < GUSTANN_PIPE_W; w++) {
          if (sub_lanes_[w].pending) {
            any_pending = true;
            break;
          }
        }
        if (!any_pending) {
          finish_query();
#ifdef QUIVER_LIGHT_BREAKDOWN
        } else {
          start_breakdown_cta_wait();
#endif
        }
        break;
      }
#ifdef QUIVER_LIGHT_BREAKDOWN
      end_breakdown_cta_wait(elapsed());
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
      pending_hop_index_ = current_hop_index_;
      pending_hop_wait_us_ /= static_cast<double>(completed.size());
#endif

      time_gpu_issue -= elapsed();
      pending_hop_compute_start_ = elapsed();
#ifdef QUIVER_LIGHT_BREAKDOWN
      if (ready_time_count > 0) {
        const double resume_time =
            pending_hop_compute_start_ * ready_time_count - ready_time_sum;
        time_resumption += resume_time;
        breakdown_batch_resumption_time_ += resume_time;
      }
#endif
      time_gpu -= pending_hop_compute_start_;

#ifdef QUIVER_LIGHT_BREAKDOWN
      CHECK_CUDA(cudaEventRecord(hop_compute_start_event_, stream_));
#endif
      for (int w : completed) {
        process_sublane_gpu(w);
#ifdef NOFUSE_ABLATION
        select_next_sublane(w);
#endif
      }

      for (int w = 0; w < GUSTANN_PIPE_W; w++) {
        if (!sub_lanes_[w].pending) {
          bool in_completed = false;
          for (int c : completed) {
            if (c == w) {
              in_completed = true;
              break;
            }
          }
          if (!in_completed) {
            select_next_sublane(w);
          }
        }
      }
#ifdef QUIVER_LIGHT_BREAKDOWN
      CHECK_CUDA(cudaEventRecord(hop_compute_done_event_, stream_));
      hop_compute_event_pending_ = true;
#endif

      state_ = Q_PIPE_GPU;
      break;
    }

    case Q_PIPE_GPU: {
      auto err = cudaStreamQuery(stream_);
      if (err == cudaSuccess) {
        double now = elapsed();
        double hop_compute_s = now - pending_hop_compute_start_;
#ifdef QUIVER_LIGHT_BREAKDOWN
        if (hop_compute_event_pending_) {
          float hop_compute_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(
              &hop_compute_ms, hop_compute_start_event_,
              hop_compute_done_event_));
          hop_compute_s = static_cast<double>(hop_compute_ms) * 1e-3;
          hop_compute_event_pending_ = false;
        }
#endif
        time_gpu += now;
        time_gpu_issue += elapsed();
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
        hop_latencies.push_back(
            {pending_hop_index_, pending_hop_wait_us_,
             hop_compute_s * 1e6});
        current_hop_index_ = pending_hop_index_ + 1;
        pending_hop_wait_us_ = 0.0;
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
        breakdown_batch_compute_time_ += hop_compute_s;
        time_useful_compute += hop_compute_s;
#endif

        time_ssd_issue -= elapsed();
        for (int w = 0; w < GUSTANN_PIPE_W; w++) {
          if (!sub_lanes_[w].pending) {
            bool has_work = false;
            for (int j = 0; j < qcnt_; j++) {
              if (sub_lanes_[w].request[j] != -1) {
                has_work = true;
                break;
              }
            }
            if (has_work) {
              submit_ssd_sublane(w);
            }
          }
        }
        time_ssd_issue += elapsed();

        bool all_idle = true;
        for (int w = 0; w < GUSTANN_PIPE_W; w++) {
          if (sub_lanes_[w].pending) {
            all_idle = false;
            break;
          }
        }
        if (all_idle) {
          finish_query();
        } else {
          state_ = Q_PIPE;
        }
      } else if (err != cudaErrorNotReady) {
        CHECK_CUDA(err);
      }
      break;
    }

    case Q_RES: {
      auto err = cudaStreamQuery(stream_);
      if (err == cudaSuccess) {
#ifdef QUIVER_LIGHT_BREAKDOWN
        if (d2h_event_pending_) {
          float d2h_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, d2h_start_event_,
                                          d2h_done_event_));
          const double d2h_s = static_cast<double>(d2h_ms) * 1e-3;
          time_d2h += d2h_s;
          breakdown_batch_d2h_time_ += d2h_s;
          d2h_event_pending_ = false;
        }
#endif
        state_ = Q_FIN;
        double t_done = elapsed();
        latency += t_done;
        batch_latencies.push_back((t_done - batch_start_time) * 1000.0);
#ifdef GUSTANN_LATENCY_PROBE
        if (current_batch_has_kept_io_trace_) {
          probe_traces.insert(probe_traces.end(), h_probe_traces_,
                              h_probe_traces_ + qcnt_);
        }
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
        record_breakdown_batch_sample(t_done);
        finish_breakdown_cta_window(t_done);
#endif
        batch_finished_ = true;
      } else if (err != cudaErrorNotReady) {
        CHECK_CUDA(err);
      }
      break;
    }

    case Q_FIN:
      break;
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "[TaskRunner tid=%d] Exception in state %d: %s\n", tid,
            (int)state_, e.what());
    std::abort();
  }

  return state_ == Q_FIN;
}

}  // namespace gustann
