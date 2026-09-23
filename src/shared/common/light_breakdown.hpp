#pragma once

#ifdef QUIVER_LIGHT_BREAKDOWN

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

namespace shared {
namespace breakdown {

static constexpr int QUIVER_LIGHT_BREAKDOWN_SAMPLE_QUERIES = 200;
static constexpr int QUIVER_LIGHT_BREAKDOWN_SAMPLE_BATCHES = 20;

__host__ __device__ __forceinline__ int sample_query_id(
    int sample_idx, int total_queries, int sample_count) {
  if (sample_idx < 0 || sample_idx >= sample_count || total_queries <= 0) {
    return -1;
  }
  if (total_queries <= sample_count) return sample_idx;
  if (sample_count <= 1) return 0;
  return static_cast<int>(
      (static_cast<long long>(sample_idx) *
       static_cast<long long>(total_queries - 1)) /
      static_cast<long long>(sample_count - 1));
}

__host__ __device__ __forceinline__ int sample_index(
    int qid, int total_queries, int sample_count) {
  if (qid < 0 || total_queries <= 0 || sample_count <= 0) return -1;
  if (total_queries <= sample_count) {
    return qid < total_queries ? qid : -1;
  }
  if (sample_count == 1) return qid == 0 ? 0 : -1;

  const long long denom = static_cast<long long>(total_queries - 1);
  int idx = static_cast<int>(
      (static_cast<long long>(qid) * static_cast<long long>(sample_count - 1)) /
      denom);
  for (int delta = 0; delta <= 1; ++delta) {
    int candidate = idx + delta;
    if (candidate >= 0 && candidate < sample_count &&
        sample_query_id(candidate, total_queries, sample_count) == qid) {
      return candidate;
    }
  }
  return -1;
}

__host__ __device__ __forceinline__ int sample_batch_id(
    int sample_idx, int total_batches, int sample_count) {
  return sample_query_id(sample_idx, total_batches, sample_count);
}

struct Breakdown {
  int query_count = 0;
  double useful_compute_us = 0.0;
  double resumption_delay_us = 0.0;
  double ssd_wait_us = 0.0;
  double kernel_launch_us = 0.0;
  double h2d_us = 0.0;
  double d2h_us = 0.0;
};

struct CtaActivitySample {
  double active_us = 0.0;
  double idle_us = 0.0;
  double active_ratio_pct = 0.0;
};

struct DistributionStats {
  size_t count = 0;
  double avg = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
};

inline double percentile_sorted(const std::vector<double>& sorted, double pct) {
  if (sorted.empty()) return 0.0;
  size_t idx = static_cast<size_t>(sorted.size() * pct / 100.0);
  if (idx >= sorted.size()) idx = sorted.size() - 1;
  return sorted[idx];
}

inline double average(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

inline std::vector<double> scale_components_to_total(
    const std::vector<double>& components, double target_total) {
  std::vector<double> scaled;
  scaled.reserve(components.size());
  const double total = std::accumulate(components.begin(), components.end(), 0.0);
  if (total <= 0.0 || target_total <= 0.0) {
    scaled.assign(components.size(), 0.0);
    return scaled;
  }
  for (double component : components) {
    scaled.push_back(target_total * component / total);
  }
  return scaled;
}

inline std::vector<std::pair<int64_t, int64_t>> merge_intervals(
    std::vector<std::pair<int64_t, int64_t>> intervals) {
  std::vector<std::pair<int64_t, int64_t>> merged;
  std::sort(intervals.begin(), intervals.end());
  for (auto interval : intervals) {
    if (interval.second <= interval.first) continue;
    if (merged.empty() || interval.first > merged.back().second) {
      merged.push_back(interval);
    } else if (interval.second > merged.back().second) {
      merged.back().second = interval.second;
    }
  }
  return merged;
}

inline double merged_interval_sum_us(
    std::vector<std::pair<int64_t, int64_t>> intervals) {
  auto merged = merge_intervals(std::move(intervals));
  int64_t total_ns = 0;
  for (const auto& interval : merged) {
    total_ns += interval.second - interval.first;
  }
  return static_cast<double>(total_ns) / 1000.0;
}

inline CtaActivitySample make_cta_activity_sample(double window_us,
                                                  double active_us) {
  CtaActivitySample sample;
  sample.active_us = active_us;
  sample.idle_us = window_us > active_us ? window_us - active_us : 0.0;
  if (window_us > 0.0) {
    sample.active_ratio_pct = 100.0 * active_us / window_us;
    if (sample.active_ratio_pct > 100.0) sample.active_ratio_pct = 100.0;
  }
  return sample;
}

inline DistributionStats make_distribution_stats(
    const std::vector<double>& values) {
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  DistributionStats stats;
  stats.count = sorted.size();
  stats.avg = average(sorted);
  stats.p50 = percentile_sorted(sorted, 50);
  stats.p90 = percentile_sorted(sorted, 90);
  stats.p99 = percentile_sorted(sorted, 99);
  stats.max = sorted.empty() ? 0.0 : sorted.back();
  return stats;
}

inline DistributionStats sum_distribution_stats(
    const DistributionStats& compute,
    const DistributionStats& io_wait,
    const DistributionStats& resume,
    const DistributionStats& scheduling_gap,
    const DistributionStats& other,
    const DistributionStats& kernel_launch,
    const DistributionStats& h2d,
    const DistributionStats& d2h) {
  DistributionStats stats;
  stats.count = std::max({compute.count, io_wait.count, resume.count,
                          scheduling_gap.count, other.count,
                          kernel_launch.count, h2d.count, d2h.count});
  stats.avg = compute.avg + io_wait.avg + resume.avg + scheduling_gap.avg +
              other.avg + kernel_launch.avg + h2d.avg + d2h.avg;
  stats.p50 = compute.p50 + io_wait.p50 + resume.p50 + scheduling_gap.p50 +
              other.p50 + kernel_launch.p50 + h2d.p50 + d2h.p50;
  stats.p90 = compute.p90 + io_wait.p90 + resume.p90 + scheduling_gap.p90 +
              other.p90 + kernel_launch.p90 + h2d.p90 + d2h.p90;
  stats.p99 = compute.p99 + io_wait.p99 + resume.p99 + scheduling_gap.p99 +
              other.p99 + kernel_launch.p99 + h2d.p99 + d2h.p99;
  stats.max = compute.max + io_wait.max + resume.max + scheduling_gap.max +
              other.max + kernel_launch.max + h2d.max + d2h.max;
  return stats;
}

inline void print_distribution_stats(const char* name,
                                     const DistributionStats& stats) {
  std::printf("  %-22s %10zu %12.2f %12.2f %12.2f %12.2f %12.2f\n",
              name, stats.count, stats.avg, stats.p50, stats.p90, stats.p99,
              stats.max);
}

inline void print_distribution(const char* name,
                               const std::vector<double>& values) {
  print_distribution_stats(name, make_distribution_stats(values));
}

inline void print_breakdown(const char* system, const Breakdown& b) {
  std::printf("\n========== Light Breakdown Breakdown: %s ==========\n", system);
  std::printf("Queries: %d\n", b.query_count);
  std::printf("Metric                         total_us     avg_us/query\n");
  std::printf("--------------------------- ------------ ----------------\n");
  const double n = b.query_count > 0 ? static_cast<double>(b.query_count) : 0.0;
  auto print_row = [&](const char* name, double total_us) {
    const double avg_us = n > 0.0 ? total_us / n : 0.0;
    std::printf("%-27s %12.2f %16.4f\n", name, total_us, avg_us);
  };
  print_row("Compute", b.useful_compute_us + b.resumption_delay_us);
  print_row("Resume", b.ssd_wait_us);
  print_row("kernel launch", b.kernel_launch_us);
  print_row("H2D", b.h2d_us);
  print_row("D2H", b.d2h_us);
  std::printf("===============================================\n\n");
}

inline void print_metric_samples(const char* system,
                                 const std::vector<double>& compute,
                                 const std::vector<double>& resume) {
  std::printf("\n========== Light Breakdown Samples: %s ==========\n", system);
  std::printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
              "Metric", "count", "avg_us", "p50_us", "p90_us", "p99_us",
              "max_us");
  std::printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
              "----------------------", "----------", "------------",
              "------------", "------------", "------------", "------------");
  print_distribution("Compute", compute);
  print_distribution("Resume", resume);
  std::printf("=============================================\n\n");
}

inline void print_detailed_metric_samples(
    const char* system,
    const std::vector<double>& compute,
    const std::vector<double>& resume,
    const std::vector<double>& kernel_launch,
    const std::vector<double>& h2d,
    const std::vector<double>& d2h,
    const std::vector<double>& latency,
    const std::vector<double>& cta_window,
    const std::vector<double>& cta_active,
    const std::vector<double>& cta_pure_io_wait,
    const std::vector<double>& cta_active_ratio) {
  std::printf("\n========== Light Breakdown Samples: %s ==========\n", system);
  std::printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
              "Metric", "count", "avg_us", "p50_us", "p90_us", "p99_us",
              "max_us");
  std::printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
              "----------------------", "----------", "------------",
              "------------", "------------", "------------", "------------");
  print_distribution("Compute", compute);
  print_distribution("Resume", resume);
  print_distribution("kernel launch", kernel_launch);
  print_distribution("H2D", h2d);
  print_distribution("D2H", d2h);
  print_distribution("Breakdown total", latency);
  print_distribution("CTA window", cta_window);
  print_distribution("CTA active", cta_active);
  print_distribution("CTA pure IO wait", cta_pure_io_wait);
  print_distribution("CTA active %", cta_active_ratio);
  std::printf("=============================================\n\n");
}

inline void print_closed_metric_samples(
    const char* system,
    const std::vector<double>& compute,
    const std::vector<double>& io_wait,
    const std::vector<double>& resume,
    const std::vector<double>& scheduling_gap,
    const std::vector<double>& other,
    const std::vector<double>& kernel_launch,
    const std::vector<double>& h2d,
    const std::vector<double>& d2h,
    const std::vector<double>& cta_window,
    const std::vector<double>& cta_active,
    const std::vector<double>& cta_pure_io_wait,
    const std::vector<double>& cta_active_ratio) {
  const DistributionStats compute_stats = make_distribution_stats(compute);
  const DistributionStats io_wait_stats = make_distribution_stats(io_wait);
  const DistributionStats resume_stats = make_distribution_stats(resume);
  const DistributionStats scheduling_gap_stats =
      make_distribution_stats(scheduling_gap);
  const DistributionStats other_stats = make_distribution_stats(other);
  const DistributionStats kernel_launch_stats =
      make_distribution_stats(kernel_launch);
  const DistributionStats h2d_stats = make_distribution_stats(h2d);
  const DistributionStats d2h_stats = make_distribution_stats(d2h);
  const DistributionStats breakdown_total_stats = sum_distribution_stats(
      compute_stats, io_wait_stats, resume_stats, scheduling_gap_stats,
      other_stats, kernel_launch_stats, h2d_stats, d2h_stats);

  std::printf("\n========== Light Breakdown Samples: %s ==========\n", system);
  std::printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
              "Metric", "count", "avg_us", "p50_us", "p90_us", "p99_us",
              "max_us");
  std::printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
              "----------------------", "----------", "------------",
              "------------", "------------", "------------", "------------");
  print_distribution_stats("Compute", compute_stats);
  print_distribution_stats("IO wait", io_wait_stats);
  print_distribution_stats("Resume", resume_stats);
  print_distribution_stats("Scheduling gap", scheduling_gap_stats);
  print_distribution_stats("Other", other_stats);
  print_distribution_stats("kernel launch", kernel_launch_stats);
  print_distribution_stats("H2D", h2d_stats);
  print_distribution_stats("D2H", d2h_stats);
  print_distribution_stats("Breakdown total", breakdown_total_stats);
  print_distribution("CTA window", cta_window);
  print_distribution("CTA active", cta_active);
  print_distribution("CTA pure IO wait", cta_pure_io_wait);
  print_distribution("CTA active %", cta_active_ratio);
  std::printf("=============================================\n\n");
}

inline void print_quiver_metric_samples(
    const std::vector<double>& compute,
    const std::vector<double>& io_wait,
    const std::vector<double>& resume,
    const std::vector<double>& scheduling_gap,
    const std::vector<double>& other,
    const std::vector<double>& kernel_launch,
    const std::vector<double>& h2d,
    const std::vector<double>& d2h,
    const std::vector<double>& latency,
    const std::vector<double>& cta_window,
    const std::vector<double>& cta_active,
    const std::vector<double>& cta_pure_io_wait,
    const std::vector<double>& cta_active_ratio) {
  (void)latency;
  print_closed_metric_samples("Quiver", compute, io_wait, resume,
                              scheduling_gap, other, kernel_launch, h2d, d2h,
                              cta_window, cta_active, cta_pure_io_wait,
                              cta_active_ratio);
}

inline void print_cta_window_samples(
    const char* system,
    const std::vector<double>& cta_window,
    const std::vector<double>& cta_active,
    const std::vector<double>& cta_pure_io_wait,
    const std::vector<double>& cta_active_ratio) {
  std::printf("\n========== Light Breakdown CTA Windows: %s ==========\n", system);
  std::printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
              "Metric", "count", "avg_us", "p50_us", "p90_us", "p99_us",
              "max_us");
  std::printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
              "----------------------", "----------", "------------",
              "------------", "------------", "------------", "------------");
  print_distribution("CTA window", cta_window);
  print_distribution("CTA active", cta_active);
  print_distribution("CTA pure IO wait", cta_pure_io_wait);
  print_distribution("CTA active %", cta_active_ratio);
  std::printf("=============================================\n\n");
}

}  // namespace breakdown
}  // namespace shared

#endif  // QUIVER_LIGHT_BREAKDOWN
