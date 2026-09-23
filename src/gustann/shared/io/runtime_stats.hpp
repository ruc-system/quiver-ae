#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace shared {

struct LoaderStatsSnapshot {
  uint64_t pages_enqueued = 0;
  uint64_t pages_submitted = 0;
  uint64_t pages_completed = 0;
  uint64_t submit_backpressure_count = 0;
  uint64_t nvme_error_count = 0;
  uint64_t hot_max_submitted_pages_per_sec = 0;
  uint64_t hot_sample_count = 0;
  std::vector<uint64_t> per_ssd_submitted;
  std::vector<uint64_t> per_ssd_completed;

  uint64_t pages_in_flight() const {
    if (pages_submitted < pages_completed) {
      return 0;
    }
    return pages_submitted - pages_completed;
  }

  uint64_t total_per_ssd_submitted() const {
    return std::accumulate(per_ssd_submitted.begin(), per_ssd_submitted.end(),
                           uint64_t{0});
  }

  uint64_t total_per_ssd_completed() const {
    return std::accumulate(per_ssd_completed.begin(), per_ssd_completed.end(),
                           uint64_t{0});
  }

  bool counts_closed() const {
    return pages_enqueued == pages_submitted &&
           pages_submitted == pages_completed &&
           total_per_ssd_submitted() == pages_submitted &&
           total_per_ssd_completed() == pages_completed;
  }

  bool has_backpressure() const { return submit_backpressure_count != 0; }

  bool has_errors() const { return nvme_error_count != 0; }
};

struct BatchLatencySummary {
  size_t count = 0;
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p99_ms = 0.0;
  double max_ms = 0.0;
};

inline BatchLatencySummary summarize_batch_latencies(
    const std::vector<double> &latencies_ms) {
  BatchLatencySummary summary;
  summary.count = latencies_ms.size();
  if (latencies_ms.empty()) {
    return summary;
  }

  std::vector<double> sorted = latencies_ms;
  std::sort(sorted.begin(), sorted.end());
  summary.avg_ms =
      std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
  summary.p50_ms = sorted[sorted.size() * 50 / 100];
  summary.p99_ms = sorted[sorted.size() * 99 / 100];
  summary.max_ms = sorted.back();
  return summary;
}

class LoaderStatsAccumulator {
 public:
  explicit LoaderStatsAccumulator(size_t num_ssds)
      : per_ssd_submitted_(num_ssds), per_ssd_completed_(num_ssds) {}

  void record_enqueued(uint64_t count = 1) {
    pages_enqueued_.fetch_add(count, std::memory_order_relaxed);
  }

  void record_submitted(size_t ssd_id, uint64_t count = 1) {
    pages_submitted_.fetch_add(count, std::memory_order_relaxed);
    if (ssd_id < per_ssd_submitted_.size()) {
      per_ssd_submitted_[ssd_id].fetch_add(count, std::memory_order_relaxed);
    }
  }

  void record_completed(size_t ssd_id, uint64_t count = 1) {
    pages_completed_.fetch_add(count, std::memory_order_relaxed);
    if (ssd_id < per_ssd_completed_.size()) {
      per_ssd_completed_[ssd_id].fetch_add(count, std::memory_order_relaxed);
    }
  }

  void record_submit_backpressure(uint64_t count = 1) {
    submit_backpressure_count_.fetch_add(count, std::memory_order_relaxed);
  }

  void record_nvme_error(uint64_t count = 1) {
    nvme_error_count_.fetch_add(count, std::memory_order_relaxed);
  }

  void record_hot_sample(uint64_t submitted_pages, double sample_seconds) {
    if (sample_seconds <= 0.0) {
      return;
    }
    const auto pages_per_sec = static_cast<uint64_t>(
        static_cast<double>(submitted_pages) / sample_seconds + 0.5);
    hot_sample_count_.fetch_add(1, std::memory_order_relaxed);
    uint64_t current =
        hot_max_submitted_pages_per_sec_.load(std::memory_order_relaxed);
    while (current < pages_per_sec &&
           !hot_max_submitted_pages_per_sec_.compare_exchange_weak(
               current, pages_per_sec, std::memory_order_relaxed)) {
    }
  }

  LoaderStatsSnapshot snapshot() const {
    LoaderStatsSnapshot stats;
    stats.pages_enqueued = pages_enqueued_.load(std::memory_order_relaxed);
    stats.pages_submitted = pages_submitted_.load(std::memory_order_relaxed);
    stats.pages_completed = pages_completed_.load(std::memory_order_relaxed);
    stats.submit_backpressure_count =
        submit_backpressure_count_.load(std::memory_order_relaxed);
    stats.nvme_error_count = nvme_error_count_.load(std::memory_order_relaxed);
    stats.hot_max_submitted_pages_per_sec =
        hot_max_submitted_pages_per_sec_.load(std::memory_order_relaxed);
    stats.hot_sample_count = hot_sample_count_.load(std::memory_order_relaxed);
    stats.per_ssd_submitted.resize(per_ssd_submitted_.size());
    stats.per_ssd_completed.resize(per_ssd_completed_.size());
    for (size_t i = 0; i < per_ssd_submitted_.size(); ++i) {
      stats.per_ssd_submitted[i] =
          per_ssd_submitted_[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < per_ssd_completed_.size(); ++i) {
      stats.per_ssd_completed[i] =
          per_ssd_completed_[i].load(std::memory_order_relaxed);
    }
    return stats;
  }

  void clear() {
    pages_enqueued_.store(0, std::memory_order_relaxed);
    pages_submitted_.store(0, std::memory_order_relaxed);
    pages_completed_.store(0, std::memory_order_relaxed);
    submit_backpressure_count_.store(0, std::memory_order_relaxed);
    nvme_error_count_.store(0, std::memory_order_relaxed);
    hot_max_submitted_pages_per_sec_.store(0, std::memory_order_relaxed);
    hot_sample_count_.store(0, std::memory_order_relaxed);
    for (auto &value : per_ssd_submitted_) {
      value.store(0, std::memory_order_relaxed);
    }
    for (auto &value : per_ssd_completed_) {
      value.store(0, std::memory_order_relaxed);
    }
  }

 private:
  std::atomic<uint64_t> pages_enqueued_{0};
  std::atomic<uint64_t> pages_submitted_{0};
  std::atomic<uint64_t> pages_completed_{0};
  std::atomic<uint64_t> submit_backpressure_count_{0};
  std::atomic<uint64_t> nvme_error_count_{0};
  std::atomic<uint64_t> hot_max_submitted_pages_per_sec_{0};
  std::atomic<uint64_t> hot_sample_count_{0};
  std::vector<std::atomic<uint64_t>> per_ssd_submitted_;
  std::vector<std::atomic<uint64_t>> per_ssd_completed_;
};

}  // namespace shared
