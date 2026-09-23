#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace shared {
namespace ae {

inline std::mutex& csv_mu() {
  static std::mutex mu;
  return mu;
}

inline const char* env(const char* key) {
  const char* value = std::getenv(key);
  return value != nullptr && value[0] != '\0' ? value : nullptr;
}

inline std::string env_or(const char* key, const char* fallback = "") {
  const char* value = env(key);
  return value != nullptr ? std::string(value) : std::string(fallback);
}

inline void append_line(const char* path, const char* header,
                        const std::string& line) {
  if (path == nullptr) return;
  std::lock_guard<std::mutex> lock(csv_mu());
  std::ofstream out(path, std::ios::app);
  if (!out) return;
  if (out.tellp() == 0) out << header << '\n';
  out << line << '\n';
}

struct MetricsRow {
  const char* system = "";
  int ef = 0;
  int num_blocks = 0;
  int mini_batch = 0;
  int queries_per_block = 0;
  int pipe_width = 0;
  double qps = 0.0;
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p99_ms = 0.0;
  double p999_ms = 0.0;
  double max_ms = 0.0;
  const char* latency_kind = "";
};

inline void append_metrics(const MetricsRow& row) {
  const char* path = env("QUIVER_AE_METRICS_CSV");
  if (path == nullptr) return;
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(6);
  os << env_or("QUIVER_AE_STEM") << ',' << env_or("QUIVER_AE_VARIANT")
     << ',' << row.system << ',' << env_or("QUIVER_AE_DATASET") << ','
     << row.ef << ',' << env_or("QUIVER_AE_TARGET_RECALL") << ','
     << row.num_blocks << ',' << row.mini_batch << ','
     << row.queries_per_block << ',' << row.pipe_width << ',' << row.qps
     << ',' << row.avg_ms << ',' << row.p50_ms << ',' << row.p90_ms << ','
     << row.p99_ms << ',' << row.p999_ms << ',' << row.max_ms << ','
     << row.latency_kind;
  append_line(path,
              "stem,variant,system,dataset,ef,target_recall,num_blocks,"
              "mini_batch,queries_per_block,pipe_width,qps,avg_ms,p50_ms,"
              "p90_ms,p99_ms,p999_ms,max_ms,latency_kind",
              os.str());
}

inline double percentile_sorted(const std::vector<double>& sorted,
                                double pct) {
  if (sorted.empty()) return 0.0;
  size_t index = static_cast<size_t>(sorted.size() * pct / 100.0);
  if (index >= sorted.size()) index = sorted.size() - 1;
  return sorted[index];
}

inline void append_breakdown_metric(const char* system, const char* metric,
                                    const std::vector<double>& values) {
  const char* path = env("QUIVER_AE_BREAKDOWN_CSV");
  if (path == nullptr || values.empty()) return;
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const double avg =
      std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(6);
  os << env_or("QUIVER_AE_STEM") << ',' << env_or("QUIVER_AE_VARIANT")
     << ',' << system << ',' << env_or("QUIVER_AE_DATASET") << ','
     << env_or("QUIVER_AE_TARGET_RECALL") << ',' << metric << ','
     << sorted.size() << ',' << avg << ',' << percentile_sorted(sorted, 50)
     << ',' << percentile_sorted(sorted, 90) << ','
     << percentile_sorted(sorted, 99) << ',' << sorted.back();
  append_line(path,
              "stem,variant,system,dataset,target_recall,metric,count,"
              "avg_us,p50_us,p90_us,p99_us,max_us",
              os.str());
}

inline void append_cta_sample(const char* system, int sample_i,
                              double active_pct) {
  const char* path = env("QUIVER_AE_CTA_SAMPLES_CSV");
  if (path == nullptr) return;
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(6);
  os << env_or("QUIVER_AE_STEM") << ',' << env_or("QUIVER_AE_VARIANT")
     << ',' << system << ',' << env_or("QUIVER_AE_DATASET") << ','
     << sample_i << ',' << active_pct;
  append_line(path,
              "stem,variant,system,dataset,sample_i,cta_active_pct",
              os.str());
}

inline void append_hop_sample(const char* system, int sample_id, int hop,
                              double wait_us, double compute_us) {
  const char* path = env("QUIVER_AE_HOP_SAMPLES_CSV");
  if (path == nullptr) return;
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(6);
  os << env_or("QUIVER_AE_STEM") << ',' << env_or("QUIVER_AE_VARIANT")
     << ',' << system << ',' << env_or("QUIVER_AE_DATASET") << ','
     << sample_id << ',' << hop << ',' << wait_us << ',' << compute_us;
  append_line(path,
              "stem,variant,system,dataset,sample_id,hop,wait_us,compute_us",
              os.str());
}

inline void append_hop_intervals(const char* system, int sample_id,
                                 const int64_t* wait_start,
                                 const int64_t* wait_end, int wait_n,
                                 const int64_t* compute_start,
                                 const int64_t* compute_end, int compute_n) {
  const int hops = wait_n > compute_n ? wait_n : compute_n;
  for (int h = 0; h < hops; ++h) {
    double wait_us = 0.0;
    double compute_us = 0.0;
    if (h < wait_n && wait_end[h] > wait_start[h]) {
      wait_us = static_cast<double>(wait_end[h] - wait_start[h]) / 1000.0;
    }
    if (h < compute_n && compute_end[h] > compute_start[h]) {
      compute_us =
          static_cast<double>(compute_end[h] - compute_start[h]) / 1000.0;
    }
    append_hop_sample(system, sample_id, h, wait_us, compute_us);
  }
}

inline void append_closed_breakdown(const char* system,
                                    const std::vector<double>& compute,
                                    const std::vector<double>& io_wait,
                                    const std::vector<double>& resume,
                                    const std::vector<double>& cta_active_pct,
                                    const std::vector<double>& kernel_launch = {},
                                    const std::vector<double>& h2d = {},
                                    const std::vector<double>& d2h = {}) {
  append_breakdown_metric(system, "Compute", compute);
  append_breakdown_metric(system, "IO wait", io_wait);
  append_breakdown_metric(system, "Resume", resume);
  append_breakdown_metric(system, "kernel launch", kernel_launch);
  append_breakdown_metric(system, "H2D", h2d);
  append_breakdown_metric(system, "D2H", d2h);
  append_breakdown_metric(system, "CTA active %", cta_active_pct);
  for (size_t i = 0; i < cta_active_pct.size(); ++i) {
    append_cta_sample(system, static_cast<int>(i), cta_active_pct[i]);
  }
}

}  // namespace ae
}  // namespace shared
