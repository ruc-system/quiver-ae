#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>

#include <sys/time.h>

#include "executor.hpp"

#include "shared/common/cuda_utils.cuh"
#include "shared/common/logging.hpp"
#include "shared/common/runtime.hpp"
#include "shared/common/run_report.hpp"
#include "shared/io/loader.hpp"
#include "shared/io/runtime_stats.hpp"
#include "../shared/common/ae_csv.hpp"
#ifdef QUIVER_LIGHT_BREAKDOWN
#include "../shared/common/light_breakdown.hpp"
#endif

#include "task_runner.cuh"

#include <nvtx3/nvToolsExt.h>

namespace gustann {

namespace {

double percentile(std::vector<double> &values, double pct) {
  if (values.empty()) {
    return 0.0;
  }
  size_t idx = static_cast<size_t>(values.size() * pct / 100.0);
  if (idx >= values.size()) {
    idx = values.size() - 1;
  }
  return values[idx];
}

void print_metric_summary(const char *name, std::vector<double> &values) {
  if (values.empty()) {
    std::printf("  %-18s %10d %12.1f %12.1f %12.1f %12.1f %12.1f\n",
                name, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
    return;
  }

  std::sort(values.begin(), values.end());
  double avg =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  std::printf("  %-18s %10zu %12.1f %12.1f %12.1f %12.1f %12.1f\n",
              name, values.size(), avg, percentile(values, 50),
              percentile(values, 90), percentile(values, 99), values.back());
}

#ifdef GUSTANN_LATENCY_PROBE
void print_hop_latency_breakdown(
    const std::vector<HopLatencySample> &hop_latencies) {
  if (hop_latencies.empty()) {
    std::printf("\n=== GustANN Batch Hop Latency: no samples ===\n");
    return;
  }

  std::vector<double> waits;
  std::vector<double> computes;
  waits.reserve(hop_latencies.size());
  computes.reserve(hop_latencies.size());
  for (const auto &sample : hop_latencies) {
    waits.push_back(sample.wait_us);
    computes.push_back(sample.compute_us);
  }

  std::printf("\n=== GustANN Batch Hop Latency (%zu hop samples) ===\n",
              hop_latencies.size());
  std::printf("  BatchHopWait: SSD wait per batch hop; BatchHopCompute: GPU compute per batch hop\n");
  std::printf("  %-18s %10s %12s %12s %12s %12s %12s\n", "Metric", "count",
              "avg_us", "p50_us", "p90_us", "p99_us", "max_us");
  std::printf("  %-18s %10s %12s %12s %12s %12s %12s\n",
              "------------------", "----------", "------------",
              "------------", "------------", "------------", "------------");
  print_metric_summary("BatchHopWait", waits);
  print_metric_summary("BatchHopCompute", computes);
  std::printf("=== End GustANN Batch Hop Latency ===\n\n");
}

size_t get_cdf_point_limit() {
  const char *env = std::getenv("GUSTANN_PROBE_CDF_POINTS");
  if (env == nullptr || env[0] == '\0') {
    return 200;
  }
  char *end = nullptr;
  unsigned long value = std::strtoul(env, &end, 10);
  if (end == env || value == 0) {
    return 200;
  }
  return static_cast<size_t>(value);
}

void write_cdf_points(std::ofstream &out, const char *metric,
                      std::vector<double> &values, size_t max_points) {
  if (values.empty()) {
    return;
  }

  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  const size_t points = std::min(n, max_points);
  size_t last_idx = static_cast<size_t>(-1);

  for (size_t p = 0; p < points; ++p) {
    size_t idx = p * n / points;
    if (idx == last_idx) {
      continue;
    }
    last_idx = idx;
    double cdf = static_cast<double>(idx + 1) / static_cast<double>(n);
    out << metric << "," << values[idx] << "," << cdf << "," << n << "\n";
  }
}

std::vector<double> collect_batch_end_to_end_io_us(
    const std::vector<GustannBatchIoTrace> &cases) {
  std::vector<double> values;
  for (const auto &trace : cases) {
    const size_t n = std::min({trace.enqueue_ns.size(), trace.submit_ns.size(),
                               trace.complete_ns.size()});
    values.reserve(values.size() + n);
    for (size_t i = 0; i < n; ++i) {
      const int64_t enqueue_ns = trace.enqueue_ns[i];
      const int64_t submit_ns = trace.submit_ns[i];
      const int64_t complete_ns = trace.complete_ns[i];
      if (enqueue_ns <= 0 || submit_ns < enqueue_ns || complete_ns < submit_ns) {
        continue;
      }
      values.push_back(
          static_cast<double>(complete_ns - enqueue_ns) / 1000.0);
    }
  }
  return values;
}

uint64_t query_hop_key(int query_id, int hop_index, int sublane) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(query_id)) << 24) |
         (static_cast<uint64_t>(static_cast<uint16_t>(hop_index)) << 8) |
         static_cast<uint64_t>(static_cast<uint8_t>(sublane));
}

std::unordered_set<uint64_t> build_sampled_io_query_hop_keys(
    const std::vector<GustannBatchIoTrace> &cases) {
  std::unordered_set<uint64_t> keys;
  for (const auto &trace : cases) {
    const size_t n = std::min({trace.enqueue_ns.size(), trace.submit_ns.size(),
                               trace.complete_ns.size()});
    keys.reserve(keys.size() + n);
    for (size_t i = 0; i < n; ++i) {
      const int64_t enqueue_ns = trace.enqueue_ns[i];
      const int64_t submit_ns = trace.submit_ns[i];
      const int64_t complete_ns = trace.complete_ns[i];
      if (enqueue_ns <= 0 || submit_ns < enqueue_ns || complete_ns < submit_ns) {
        continue;
      }
      keys.insert(query_hop_key(trace.query_base + static_cast<int>(i),
                                trace.hop_index, trace.sublane));
    }
  }
  return keys;
}

void collect_query_compute_metrics(
    const std::vector<GustannProbeQueryTrace> &probe_traces,
    const std::vector<GustannBatchIoTrace> &sampled_io_cases,
    std::vector<double> &query_end_to_end_computes,
    std::vector<double> &query_active_computes) {
  auto sampled_io_query_hops = build_sampled_io_query_hop_keys(sampled_io_cases);
  if (sampled_io_query_hops.empty()) {
    return;
  }
  for (const auto &trace : probe_traces) {
    int sample_count =
        std::min(trace.num_samples, GUSTANN_PROBE_MAX_SAMPLES);
    for (int i = 0; i < sample_count; ++i) {
      const auto &sample = trace.samples[i];
      if (sample.t_compute_done <= 0 || sample.hop_index < 0) {
        continue;
      }
      if (sampled_io_query_hops.find(query_hop_key(trace.query_id,
                                                   sample.hop_index,
                                                   sample.sublane)) ==
              sampled_io_query_hops.end()) {
        continue;
      }
      if (sample.t_kernel_start > 0 &&
          sample.t_compute_done >= sample.t_kernel_start) {
        query_end_to_end_computes.push_back(
            static_cast<double>(sample.t_compute_done - sample.t_kernel_start) /
            1000.0);
      }
      if (sample.t_merge_start > 0 &&
          sample.t_compute_done >= sample.t_merge_start) {
        query_active_computes.push_back(
            static_cast<double>(sample.t_compute_done - sample.t_merge_start) /
            1000.0);
      }
    }
  }
}

void write_probe_cdf_csv(const char *path,
                         const std::vector<HopLatencySample> &hop_latencies,
                         const std::vector<GustannProbeQueryTrace>
                             &probe_traces,
                         const std::vector<GustannBatchIoTrace>
                             &batch_io_cases) {
  if (path == nullptr || path[0] == '\0') {
    return;
  }

  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open GustANN probe CDF CSV: %s\n", path);
    return;
  }

  std::vector<double> batch_waits;
  std::vector<double> batch_computes;
  batch_waits.reserve(hop_latencies.size());
  batch_computes.reserve(hop_latencies.size());
  for (const auto &sample : hop_latencies) {
    batch_waits.push_back(sample.wait_us);
    batch_computes.push_back(sample.compute_us);
  }

  std::vector<double> query_end_to_end_computes;
  std::vector<double> query_active_computes;
  collect_query_compute_metrics(probe_traces, batch_io_cases,
                                query_end_to_end_computes,
                                query_active_computes);

  const size_t max_points = get_cdf_point_limit();
  out << "metric,value_us,cdf,total_count\n";
  write_cdf_points(out, "BatchHopWait", batch_waits, max_points);
  write_cdf_points(out, "BatchHopCompute", batch_computes, max_points);
  write_cdf_points(out, "QueryEndToEndCompute", query_end_to_end_computes,
                   max_points);
  auto end_to_end_ios = collect_batch_end_to_end_io_us(batch_io_cases);
  write_cdf_points(out, "EndToEndIO", end_to_end_ios, max_points);

  (void)max_points;
}

void print_query_probe_breakdown(
    const std::vector<GustannProbeQueryTrace> &probe_traces,
    const std::vector<GustannBatchIoTrace> &sampled_io_cases) {
  std::vector<double> query_end_to_end_computes;
  std::vector<double> query_active_computes;
  collect_query_compute_metrics(probe_traces, sampled_io_cases,
                                query_end_to_end_computes,
                                query_active_computes);
  const size_t valid_samples =
      std::max(query_end_to_end_computes.size(), query_active_computes.size());

  if (valid_samples == 0) {
    std::printf("\n=== GustANN GPU Per-Query Probe: no valid samples ===\n");
    return;
  }

  std::printf("\n=== GustANN GPU Query Compute Latency (%zu sampled query-hop IOs) ===\n",
              valid_samples);
  std::printf("  QueryEndToEndCompute: kernel launch marker -> query compute done\n");
  std::printf("  QueryActiveCompute: query merge start -> query compute done\n");
  std::printf("  Samples match the sampled query-hop IO set used by EndToEndIO/DeviceService\n");
  std::printf("  %-18s %10s %12s %12s %12s %12s %12s\n", "Metric", "count",
              "avg_us", "p50_us", "p90_us", "p99_us", "max_us");
  std::printf("  %-18s %10s %12s %12s %12s %12s %12s\n",
              "------------------", "----------", "------------",
              "------------", "------------", "------------", "------------");
  print_metric_summary("QueryEndToEndCompute", query_end_to_end_computes);
  print_metric_summary("QueryActiveCompute", query_active_computes);
  std::printf("=== End GustANN GPU Query Compute Latency ===\n\n");
}

int get_batch_tail_case_limit() {
  const char *env = std::getenv("GUSTANN_BATCH_TAIL_SAMPLE_CASES");
  if (env == nullptr) {
    return 256;
  }
  return std::max(0, std::atoi(env));
}

std::vector<GustannBatchIoTrace> select_batch_io_tail_cases(
    std::vector<GustannBatchIoTrace> cases) {
  const int limit = get_batch_tail_case_limit();
  if (limit <= 0 || cases.empty()) {
    return {};
  }
  cases.erase(std::remove_if(cases.begin(), cases.end(),
                             [](const GustannBatchIoTrace &trace) {
                               return collect_batch_end_to_end_io_us({trace})
                                   .empty();
                             }),
              cases.end());
  static std::mt19937_64 rng{std::random_device{}()};
  std::shuffle(cases.begin(), cases.end(), rng);
  if (static_cast<int>(cases.size()) > limit) {
    cases.resize(limit);
  }
  return cases;
}

struct BatchIoDetail {
  double device_service_us = 0.0;
  double end_to_end_us = 0.0;
};

std::vector<BatchIoDetail> build_batch_io_details(
    const GustannBatchIoTrace &trace) {
  std::vector<BatchIoDetail> details;
  const size_t n = std::min({trace.enqueue_ns.size(), trace.submit_ns.size(),
                             trace.complete_ns.size()});
  for (size_t i = 0; i < n; ++i) {
    const int64_t enqueue_ns = trace.enqueue_ns[i];
    const int64_t submit_ns = trace.submit_ns[i];
    const int64_t complete_ns = trace.complete_ns[i];
    if (enqueue_ns <= 0 || submit_ns < enqueue_ns || complete_ns < submit_ns) {
      continue;
    }
    details.push_back(
        {static_cast<double>(complete_ns - submit_ns) / 1000.0,
         static_cast<double>(complete_ns - enqueue_ns) / 1000.0});
  }
  return details;
}

void write_batch_tail_detail_csv(
    const char *path, const std::vector<GustannBatchIoTrace> &cases) {
  if (path == nullptr || path[0] == '\0' || cases.empty()) {
    return;
  }

  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open GustANN batch tail CSV: %s\n", path);
    return;
  }

  out << "case_id,query_base,hop_index,sublane,active_ios,query_slot,"
         "enqueue_ns,submit_ns,complete_ns,device_service_us,end_to_end_us\n";
  for (size_t case_id = 0; case_id < cases.size(); ++case_id) {
    const auto &trace = cases[case_id];
    const size_t n = std::min({trace.enqueue_ns.size(), trace.submit_ns.size(),
                               trace.complete_ns.size()});
    for (size_t i = 0; i < n; ++i) {
      const int64_t enqueue_ns = trace.enqueue_ns[i];
      const int64_t submit_ns = trace.submit_ns[i];
      const int64_t complete_ns = trace.complete_ns[i];
      if (enqueue_ns <= 0 || submit_ns < enqueue_ns || complete_ns < submit_ns) {
        continue;
      }
      out << case_id << "," << trace.query_base << "," << trace.hop_index
          << "," << trace.sublane << "," << trace.active_ios << "," << i
          << "," << enqueue_ns << "," << submit_ns << "," << complete_ns
          << "," << static_cast<double>(complete_ns - submit_ns) / 1000.0
          << "," << static_cast<double>(complete_ns - enqueue_ns) / 1000.0
          << "\n";
    }
  }

  std::printf("GustANN batch-internal IO case details written to %s\n",
              path);
}

void print_batch_internal_io_tail_cases(
    const std::vector<GustannBatchIoTrace> &cases) {
  if (cases.empty()) {
    std::printf("\n=== GustANN Batch IO Latency: no samples ===\n");
    return;
  }

  std::vector<BatchIoDetail> all_details;
  for (const auto &trace : cases) {
    auto details = build_batch_io_details(trace);
    all_details.insert(all_details.end(), details.begin(), details.end());
  }

  if (all_details.empty()) {
    std::printf("\n=== GustANN Batch IO Latency: no valid IO samples ===\n");
    return;
  }

  std::printf("\n=== GustANN Batch IO Latency (%zu sampled IOs from %zu submit_sublane calls) ===\n",
              all_details.size(), cases.size());
  std::printf("  EndToEndIO: enqueue -> complete; DeviceService: submit -> complete\n");

  std::vector<double> device_services;
  std::vector<double> end_to_end_ios;
  device_services.reserve(all_details.size());
  end_to_end_ios.reserve(all_details.size());
  for (const auto &detail : all_details) {
    device_services.push_back(detail.device_service_us);
    end_to_end_ios.push_back(detail.end_to_end_us);
  }

  std::printf("  %-18s %10s %12s %12s %12s %12s %12s\n", "Metric", "count",
              "avg_us", "p50_us", "p90_us", "p99_us", "max_us");
  std::printf("  %-18s %10s %12s %12s %12s %12s %12s\n",
              "------------------", "----------", "------------",
              "------------", "------------", "------------", "------------");
  print_metric_summary("EndToEndIO", end_to_end_ios);
  print_metric_summary("DeviceService", device_services);
  std::printf("=== End GustANN Batch IO Latency ===\n\n");

  write_batch_tail_detail_csv(std::getenv("GUSTANN_BATCH_TAIL_DETAIL_CSV"),
                              cases);
}
#endif

void print_runtime_summary(int mini_batch, int pipe_w, int worker_threads,
                           int runner_contexts, int num_queries,
                           shared::DataType data_type, int topk,
                           int ef_search, int repeat,
                           double search_time, double gpu_active_time,
                           double ssd_wait_time, double init_time,
                           double fin_time, int pages_read_total,
                           const std::vector<double> &batch_latencies,
#ifdef QUIVER_LIGHT_BREAKDOWN
                           const std::vector<BreakdownCtaWindowSample>
                               &cta_window_samples,
                           const std::vector<BreakdownBatchSample>
                               &breakdown_batch_samples,
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
                           const std::vector<HopLatencySample> &hop_latencies,
#endif
#ifdef GUSTANN_LATENCY_PROBE
                           const std::vector<GustannProbeQueryTrace>
                               &probe_traces,
                           const std::vector<GustannBatchIoTrace>
                               &batch_io_tail_cases,
#endif
                           const shared::LoaderStatsSnapshot &io_stats) {
  // All time accumulators are summed across all threads×contexts.
  // Normalize by parallelism to get wall-clock equivalent.
  int parallelism = worker_threads * runner_contexts;
  (void)gpu_active_time;
  (void)ssd_wait_time;
  (void)init_time;
  (void)fin_time;
  (void)pages_read_total;

  // Per-batch latency stats
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
  runtime_params << "mini_batch=" << mini_batch << " pipe_width=" << pipe_w
                 << " worker_threads=" << worker_threads
                 << " runner_contexts=" << runner_contexts
                 << " parallelism=" << parallelism;
  shared::RunReport report;
  report.program = "GustANN";
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
  report.mini_batch = mini_batch;
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
    std::vector<double> cta_window;
    std::vector<double> cta_active;
    std::vector<double> cta_pure_io_wait;
    std::vector<double> cta_active_ratio;
    compute_samples.reserve(breakdown_batch_samples.size());
    io_wait_samples.reserve(breakdown_batch_samples.size());
    resume_samples.reserve(breakdown_batch_samples.size());
    scheduling_gap_samples.reserve(breakdown_batch_samples.size());
    other_samples.reserve(breakdown_batch_samples.size());
    kernel_launch_samples.reserve(breakdown_batch_samples.size());
    h2d_samples.reserve(breakdown_batch_samples.size());
    d2h_samples.reserve(breakdown_batch_samples.size());
    cta_window.reserve(cta_window_samples.size());
    cta_active.reserve(cta_window_samples.size());
    cta_pure_io_wait.reserve(cta_window_samples.size());
    cta_active_ratio.reserve(cta_window_samples.size());
    for (const auto &sample : breakdown_batch_samples) {
      const double compute_us = sample.compute_us;
      const double io_wait_us = sample.io_wait_us;
      const double resume_us = sample.resume_us;
      const double other_us =
          std::max(0.0, sample.latency_us - compute_us - io_wait_us -
                            resume_us - sample.kernel_launch_us -
                            sample.h2d_us - sample.d2h_us);
      compute_samples.push_back(compute_us);
      io_wait_samples.push_back(io_wait_us);
      resume_samples.push_back(resume_us);
      scheduling_gap_samples.push_back(0.0);
      other_samples.push_back(other_us);
      kernel_launch_samples.push_back(sample.kernel_launch_us);
      h2d_samples.push_back(sample.h2d_us);
      d2h_samples.push_back(sample.d2h_us);
    }
    if (pipe_w == 1) {
      for (const auto &sample : cta_window_samples) {
        if (sample.window_us <= 0.0) {
          continue;
        }
        const double active_us =
            sample.window_us > sample.pure_io_wait_us
                ? sample.window_us - sample.pure_io_wait_us
                : 0.0;
        const auto cta_activity =
            shared::breakdown::make_cta_activity_sample(sample.window_us,
                                                        active_us);
        cta_window.push_back(sample.window_us);
        cta_active.push_back(cta_activity.active_us);
        cta_pure_io_wait.push_back(cta_activity.idle_us);
        cta_active_ratio.push_back(cta_activity.active_ratio_pct);
      }
    }
    if (!compute_samples.empty()) {
      shared::breakdown::print_closed_metric_samples(
          "GustANN", compute_samples, io_wait_samples, resume_samples,
          scheduling_gap_samples, other_samples, kernel_launch_samples,
          h2d_samples, d2h_samples, cta_window, cta_active,
          cta_pure_io_wait, cta_active_ratio);
      shared::ae::append_closed_breakdown(
          "GustANN", compute_samples, io_wait_samples, resume_samples,
          cta_active_ratio, kernel_launch_samples, h2d_samples, d2h_samples);
    }
    for (size_t i = 0; i < hop_latencies.size(); ++i) {
      shared::ae::append_hop_sample(
          "GustANN", static_cast<int>(i), hop_latencies[i].hop_index,
          hop_latencies[i].wait_us, hop_latencies[i].compute_us);
    }
  }
#endif
#ifdef GUSTANN_LATENCY_PROBE
  print_hop_latency_breakdown(hop_latencies);
  auto sampled_batch_io_cases = select_batch_io_tail_cases(batch_io_tail_cases);
  print_query_probe_breakdown(probe_traces, sampled_batch_io_cases);
  write_probe_cdf_csv(std::getenv("GUSTANN_PROBE_CSV"), hop_latencies,
                      probe_traces, sampled_batch_io_cases);
  print_batch_internal_io_tail_cases(sampled_batch_io_cases);
#endif
}

}  // namespace

HybridExecutor::HybridExecutor(const shared::Layout &layout,
                               const shared::DataType &data_type,
                               const std::string &fpath,
                               const HybridExecutorConfig &config)
    : layout_(layout), data_type_(data_type) {
  mini_batch_ = config.mini_batch;
  thread_cnt_ = config.thread_cnt;
  ctx_per_thread_ = config.ctx_per_thread;
#ifdef GUSTANN_FIXED_PIPE_WIDTH
  pipe_w_ = GUSTANN_FIXED_PIPE_WIDTH;
#else
  pipe_w_ = config.pipe_w;
#endif

  const std::string& memory_fpath =
      config.memory_index_file.empty() ? fpath : config.memory_index_file;
  const std::string& starter_fpath =
      (config.use_backend == HybridExecutorConfig::MEMORY) ? memory_fpath : fpath;

  FILE *input = fopen(starter_fpath.c_str(), "rb");
  if (!input) {
    ERROR("Failed to open index file: {}", starter_fpath);
    exit(-1);
  }

  starter_ = std::make_unique<uint8_t[]>(shared::PAGE_SIZE);
  fseek(input,
        (long)shared::PAGE_SIZE *
            (layout_.enter_point / layout_.nodes_per_page + 1),
        SEEK_SET);
  fread((char *)starter_.get(), sizeof(char), shared::PAGE_SIZE, input);

  fclose(input);

  if (config.use_backend == HybridExecutorConfig::SPDK) {
#if QUIVER_ENABLE_SPDK
    const auto &ssds = config.ssd_lists;
    if (ssds.empty()) {
      ERROR("NO SSD IN USE!");
      exit(-1);
    }
    int spdk_ctx_cnt = thread_cnt_ * ctx_per_thread_ * pipe_w_;
    loader_ = shared::create_spdk_loader(
        ssds, mini_batch_ * ctx_per_thread_ * pipe_w_, thread_cnt_,
        spdk_ctx_cnt);
#else
    ERROR("This binary was built without SPDK support");
    exit(-1);
#endif
  } else if (config.use_backend == HybridExecutorConfig::MEMORY) {
    loader_ = shared::create_mem_loader_sync(
        memory_fpath.c_str(), layout_.num_pages, config.memory_backend);
  } else {
    ERROR("Wrong IO Backend setting!");
    exit(-1);
  }
}

void HybridExecutor::search(const float *qdata, int num_queries, int topk,
                            int ef_search, int *nns, float *distances,
                            int *found_cnt, shared::PQSearch *pq_,
                            shared::NavGraph *nav_, int repeat) {
  struct HostRegistrationGuard {
    const void *ptr = nullptr;
    ~HostRegistrationGuard() {
      if (ptr) {
        cudaHostUnregister(const_cast<void *>(ptr));
      }
    }
  } qdata_guard;

  int batch_cnt = mini_batch_ * thread_cnt_ * ctx_per_thread_;
  CHECK_CUDA(cudaHostRegister((void *)qdata,
                              sizeof(float) * num_queries * layout_.num_dims,
                              cudaHostRegisterDefault));
  qdata_guard.ptr = qdata;

  if (pq_) {
    pq_->init_device(layout_.num_dims, layout_.num_data, batch_cnt, ef_search);
  } else {
    ERROR("PQ is not inited!");
    throw;
  }

  if (pipe_w_ > 1) {
    INFO("Pipe Search: W={}, B={}, C={}, T={}", pipe_w_, mini_batch_,
         ctx_per_thread_, thread_cnt_);
  }

  std::atomic<int> tot_reads(0);
  std::mutex summary_mu;
  double total_gpu = 0;
  double total_ssd = 0;
  double total_init = 0;
  double total_fin  = 0;
  std::vector<double> all_batch_latencies;
#ifdef QUIVER_LIGHT_BREAKDOWN
  std::vector<BreakdownCtaWindowSample> all_cta_window_samples;
  std::vector<BreakdownBatchSample> all_breakdown_batch_samples;
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
  std::vector<HopLatencySample> all_hop_latencies;
#endif
#ifdef GUSTANN_LATENCY_PROBE
  std::vector<GustannProbeQueryTrace> all_probe_traces;
  std::vector<GustannBatchIoTrace> all_batch_io_tail_cases;
#endif

  std::unique_ptr<int[]> start_pts = std::make_unique<int[]>(num_queries);
  memset(start_pts.get(), -1, sizeof(int) * num_queries);

  std::atomic<int> cur(0);
  auto worker = [&](int threadid) {
    shared::bind_core(threadid * 2 + 21);

    bool finished = false;
    int tot_task = 0;
    std::vector<std::unique_ptr<TaskRunner>> tasks;
    tasks.reserve(ctx_per_thread_);
    for (int i = 0; i < ctx_per_thread_; i++) {
      tasks.emplace_back(std::make_unique<TaskRunner>(
          threadid, threadid * ctx_per_thread_ + i, mini_batch_,
          (int)layout_.num_dims, topk, layout_.num_data, (int)layout_.max_m0,
          ef_search, (int)layout_.enter_point, starter_.get(), pq_,
          (int)layout_.nodes_per_page, (int)layout_.node_size,
          (int)layout_.data_size, data_type_, loader_, nav_, pipe_w_));
#ifdef QUIVER_LIGHT_BREAKDOWN
      const int total_batches =
          (num_queries + mini_batch_ - 1) / mini_batch_;
      tasks.back()->set_breakdown_total_batches(total_batches);
#endif
    }
    while (!finished) {
      finished = true;
      for (auto &task : tasks) {
        if (task->update_state()) {
          if (cur.load() < num_queries) {
            int qstart = cur.fetch_add(mini_batch_);
            int qend = std::min(qstart + mini_batch_, num_queries);
            int qcnt = qend - qstart;
            if (qstart < num_queries) {
#ifdef CACHE_START
              for (int i = qstart; i < qend; i++) {
                if (start_pts[i % 10000] != -1) {
                  start_pts[i] = start_pts[i % 10000];
                }
              }
#endif
              task->init_query(
                  qdata + (int64_t)qstart * layout_.num_dims, qcnt,
                  nns + qstart * topk, distances + qstart * topk,
                  found_cnt + qstart, start_pts.get() + qstart, qstart);
              finished = false;
            }
            tot_task += qcnt;
          }
        } else {
          finished = false;
        }
      }
    }

    double tot_gpu = 0;
    double tot_ssd = 0;
    double tot_init = 0;
    double tot_fin  = 0;

    for (auto &task : tasks) {
      tot_reads.fetch_add(task->num_reads);
      tot_gpu  += task->time_gpu;
      tot_ssd  += task->time_ssd;
      tot_init += task->time_init_issue;
      tot_fin  += task->time_fin_issue;
    }
    std::vector<double> local_batch_latencies;
#ifdef QUIVER_LIGHT_BREAKDOWN
    std::vector<BreakdownCtaWindowSample> local_cta_window_samples;
    std::vector<BreakdownBatchSample> local_breakdown_batch_samples;
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
    std::vector<HopLatencySample> local_hop_latencies;
#endif
#ifdef GUSTANN_LATENCY_PROBE
    std::vector<GustannProbeQueryTrace> local_probe_traces;
    std::vector<GustannBatchIoTrace> local_batch_io_tail_cases;
#endif
    for (auto &task : tasks) {
      local_batch_latencies.insert(local_batch_latencies.end(),
                                   task->batch_latencies.begin(),
                                   task->batch_latencies.end());
#ifdef QUIVER_LIGHT_BREAKDOWN
      local_cta_window_samples.insert(local_cta_window_samples.end(),
                                      task->cta_window_samples.begin(),
                                      task->cta_window_samples.end());
      local_breakdown_batch_samples.insert(
          local_breakdown_batch_samples.end(),
          task->breakdown_batch_samples.begin(),
          task->breakdown_batch_samples.end());
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
      local_hop_latencies.insert(local_hop_latencies.end(),
                                 task->hop_latencies.begin(),
                                 task->hop_latencies.end());
#endif
#ifdef GUSTANN_LATENCY_PROBE
      local_probe_traces.insert(local_probe_traces.end(),
                                task->probe_traces.begin(),
                                task->probe_traces.end());
      local_batch_io_tail_cases.insert(local_batch_io_tail_cases.end(),
                                       task->batch_io_tail_cases.begin(),
                                       task->batch_io_tail_cases.end());
#endif
    }

    std::lock_guard<std::mutex> lock(summary_mu);
    total_gpu += tot_gpu;
    total_ssd += tot_ssd;
    total_init += tot_init;
    total_fin  += tot_fin;
    all_batch_latencies.insert(all_batch_latencies.end(),
                               local_batch_latencies.begin(),
                               local_batch_latencies.end());
#ifdef QUIVER_LIGHT_BREAKDOWN
    all_cta_window_samples.insert(all_cta_window_samples.end(),
                                  local_cta_window_samples.begin(),
                                  local_cta_window_samples.end());
    all_breakdown_batch_samples.insert(
        all_breakdown_batch_samples.end(),
        local_breakdown_batch_samples.begin(),
        local_breakdown_batch_samples.end());
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
    all_hop_latencies.insert(all_hop_latencies.end(),
                             local_hop_latencies.begin(),
                             local_hop_latencies.end());
#endif
#ifdef GUSTANN_LATENCY_PROBE
    all_probe_traces.insert(all_probe_traces.end(), local_probe_traces.begin(),
                            local_probe_traces.end());
    all_batch_io_tail_cases.insert(all_batch_io_tail_cases.end(),
                                   local_batch_io_tail_cases.begin(),
                                   local_batch_io_tail_cases.end());
#endif
  };

  std::vector<std::thread> th;
  CHECK_CUDA(cudaDeviceSynchronize());
  nvtxRangePush("SearchKernel");
  double start = shared::elapsed();
  for (int i = 0; i < thread_cnt_; i++) {
    th.emplace_back(worker, i);
  }

  for (int i = 0; i < thread_cnt_; i++) {
    th[i].join();
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  double end = shared::elapsed();
  nvtxRangePop(); // SearchKernel
  DEBUG("End Search");
  auto io_stats = loader_->snapshot_stats();
  print_runtime_summary(mini_batch_, pipe_w_, thread_cnt_, ctx_per_thread_,
                        num_queries, data_type_, topk, ef_search, repeat,
                        end - start, total_gpu, total_ssd, total_init, total_fin,
                        tot_reads.load(), all_batch_latencies,
#ifdef QUIVER_LIGHT_BREAKDOWN
                        all_cta_window_samples, all_breakdown_batch_samples,
#endif
#if defined(GUSTANN_LATENCY_PROBE) || defined(QUIVER_LIGHT_BREAKDOWN)
                        all_hop_latencies,
#endif
#ifdef GUSTANN_LATENCY_PROBE
                        all_probe_traces, all_batch_io_tail_cases,
#endif
                        io_stats);
  CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace gustann
