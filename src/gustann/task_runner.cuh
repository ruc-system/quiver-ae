#pragma once

#include <cstdint>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#include "shared/common/cuda_utils.cuh"
#include "shared/common/data_type.hpp"
#include "shared/common/search_state.cuh"
#include "shared/index/nav_graph.hpp"
#include "shared/index/pq_search.hpp"
#include "shared/io/loader.hpp"
#include "probe.cuh"

namespace gustann {

using shared::Data;
using shared::DataType;
using shared::IndexLoader;
using shared::NavGraph;
using shared::PQSearch;

struct SubLane {
  uint8_t *buffer = nullptr;
  int32_t *request = nullptr;
  int cid = 0;
  bool pending = false;
  bool buffer_registered = false;
  double wait_start_time = 0.0;
};

#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
struct HopLatencySample {
  int hop_index = 0;
  double wait_us = 0.0;
  double compute_us = 0.0;
};
#endif

#ifdef GUSTANN_LATENCY_PROBE
struct GustannBatchIoTrace {
  int query_base = 0;
  int qcnt = 0;
  int hop_index = 0;
  int sublane = 0;
  int active_ios = 0;
  std::vector<int64_t> enqueue_ns;
  std::vector<int64_t> submit_ns;
  std::vector<int64_t> complete_ns;
};
#endif

#ifdef QUIVER_LIGHT_BREAKDOWN
struct BreakdownCtaWindowSample {
  double window_us = 0.0;
  double pure_io_wait_us = 0.0;
};

struct BreakdownBatchSample {
  int batch_id = -1;
  double compute_us = 0.0;
  double io_wait_us = 0.0;
  double resume_us = 0.0;
  double kernel_launch_us = 0.0;
  double h2d_us = 0.0;
  double d2h_us = 0.0;
  double latency_us = 0.0;
};
#endif

class TaskRunner {
 public:
  ~TaskRunner();
  TaskRunner(const TaskRunner &) = delete;
  TaskRunner &operator=(const TaskRunner &) = delete;
  TaskRunner(TaskRunner &&) = delete;
  TaskRunner &operator=(TaskRunner &&) = delete;

  void init_query(const float *qdata, int _qcnt, int *_nns, float *_dis,
                  int *_found_cnt, int *_start_pt, int _query_base);
  bool update_state();

  TaskRunner(int _tid, int _cid_base, int _mini_batch, int _num_dims,
             int _topk, int64_t _num_data, int _max_m0, int _ef_search,
             int _enter_point, uint8_t *_starter, PQSearch *_pq,
             int nodes_per_page, int node_size, int data_size,
             DataType data_type, std::shared_ptr<IndexLoader> _loader,
             NavGraph *_nav, int _pipe_w = 1);

 public:
  double time_gpu;
  double time_ssd;
  double latency;
  int cnt_query;
  int num_reads = 0;
  double time_init_issue;
  double time_gpu_issue;
  double time_ssd_issue;
  double time_fin_issue;
#ifdef QUIVER_LIGHT_BREAKDOWN
  double time_h2d;
  double time_d2h;
  double time_resumption;
  double time_kernel_launch;
  double time_useful_compute;
#endif

  // Per-batch latency tracing.
  std::vector<double> batch_latencies;
  double batch_start_time;
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
  std::vector<HopLatencySample> hop_latencies;
#endif
#ifdef GUSTANN_LATENCY_PROBE
  std::vector<GustannProbeQueryTrace> probe_traces;
  std::vector<GustannBatchIoTrace> batch_io_tail_cases;
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
  std::vector<BreakdownCtaWindowSample> cta_window_samples;
  std::vector<BreakdownBatchSample> breakdown_batch_samples;
  void set_breakdown_total_batches(int total_batches);
#endif

 private:
  void cleanup() noexcept;
  void submit_ssd_sublane(int w);
#ifdef GUSTANN_LATENCY_PROBE
  void maybe_keep_batch_io_tail_case(int w);
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
  void record_breakdown_batch_sample(double done_time);
  void start_breakdown_cta_window();
  void finish_breakdown_cta_window(double now);
  void tick_breakdown_cta_window();
  void start_breakdown_cta_wait();
  void end_breakdown_cta_wait(double now);
#endif
  void process_sublane_gpu(int w);
  void select_next_sublane(int w);
  void finish_query();

  template <class T>
  void launch_merge_data(int w);

  template <class T>
  void launch_visit_select(int w);

  DataType data_type;
  int mini_batch, num_dims, topk, max_m0, ef_search, aligned_ef;
  int enter_point;
  int nodes_per_page_, node_size_, data_size_;
  int tid;
  int tcnt;
  int64_t num_data;
  PQSearch *pq;
  std::shared_ptr<IndexLoader> loader;
  uint8_t *starter;
  NavGraph *nav_graph;
  int pipe_w_;
  std::vector<SubLane> sub_lanes_;
  cudaStream_t stream_ = nullptr;
#ifdef QUIVER_LIGHT_BREAKDOWN
  cudaEvent_t h2d_start_event_ = nullptr;
  cudaEvent_t h2d_done_event_ = nullptr;
  cudaEvent_t d2h_start_event_ = nullptr;
  cudaEvent_t d2h_done_event_ = nullptr;
  cudaEvent_t hop_compute_start_event_ = nullptr;
  cudaEvent_t hop_compute_done_event_ = nullptr;
  bool h2d_event_pending_ = false;
  bool d2h_event_pending_ = false;
  bool hop_compute_event_pending_ = false;
#endif
  int stream_offset_ = 0;
  float *d_qdata_ = nullptr;
  int *d_nns_ = nullptr;
  float *d_distances_ = nullptr;
  int *d_found_cnt_ = nullptr;
  uint32_t *d_neighbors_id_ = nullptr;
  float *d_neighbors_dist_ = nullptr;
  Data *d_ctx_ = nullptr;
#ifdef GUSTANN_LATENCY_PROBE
  GustannProbeQueryTrace *d_probe_traces_ = nullptr;
  GustannProbeQueryTrace *h_probe_traces_ = nullptr;
  int *d_probe_active_samples_ = nullptr;
  int64_t *d_probe_kernel_start_ = nullptr;
  std::vector<GustannBatchIoTrace> pending_batch_io_traces_;
  int batch_tail_cases_per_runner_ = 0;
  uint64_t batch_io_trace_seen_ = 0;
  bool current_batch_has_kept_io_trace_ = false;
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
  int query_base_ = 0;
#endif
#ifdef QUIVER_LIGHT_BREAKDOWN
  int breakdown_total_batches_ = 0;
  double breakdown_batch_start_time_ = 0.0;
  double breakdown_batch_h2d_time_ = 0.0;
  double breakdown_batch_d2h_time_ = 0.0;
  double breakdown_batch_ssd_time_ = 0.0;
  double breakdown_batch_resumption_time_ = 0.0;
  double breakdown_batch_compute_time_ = 0.0;
  double breakdown_batch_kernel_launch_time_ = 0.0;
  bool breakdown_cta_window_active_ = false;
  bool breakdown_cta_waiting_ = false;
  double breakdown_cta_window_start_time_ = 0.0;
  double breakdown_cta_window_deadline_time_ = 0.0;
  double breakdown_cta_pure_wait_time_ = 0.0;
  double breakdown_cta_wait_start_time_ = 0.0;
#endif

  enum State { Q_INIT, Q_BOOT, Q_BOOT_GPU, Q_PIPE, Q_PIPE_GPU, Q_RES, Q_FIN }
      state_;
  int qcnt_ = 0;
  bool batch_finished_ = true;
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
  int current_hop_index_ = 0;
  int pending_hop_index_ = 0;
  double pending_hop_wait_us_ = 0.0;
#endif
  double pending_hop_compute_start_ = 0.0;

  int *nns_ = nullptr;
  float *distances_ = nullptr;
  int *found_cnt_ = nullptr;
  int *start_pt_ = nullptr;
};

}  // namespace gustann
