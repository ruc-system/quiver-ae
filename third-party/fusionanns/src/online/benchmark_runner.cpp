#include "benchmark_runner.h"

#include "common/nvtx_utils.h"
#include "index/centroid_navigator_sptag.h"
#ifdef FUSIONANNS_USE_CUVS_CAGRA
#include "index/centroid_navigator_cagra.h"
#endif
#include "online/cluster_scheduler.h"
#include "online/int8_gemm_scorer.h"
#include "online/pq_lut_scorer.h"
#include "online/residual_sq8_scorer.h"

#include <faiss/Index.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <atomic>
#include <boost/unordered/unordered_flat_set.hpp>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include <omp.h>

#ifndef FUSIONANNS_RECORD_TIMING_EVENT
#define FUSIONANNS_RECORD_TIMING_EVENT(enabled, evt, stream)                   \
  do {                                                                         \
    if (enabled) {                                                             \
      CUDA_CHECK(cudaEventRecord((evt), (stream)));                            \
    }                                                                          \
  } while (0)
#endif

namespace fusionann::online {

namespace {

std::string to_lower_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

class ReusableThreadPool {
public:
  ReusableThreadPool() = default;
  ~ReusableThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      shutdown_ = true;
      cv_.notify_all();
    }
    for (auto &worker : workers_) {
      if (worker.thread.joinable()) {
        worker.thread.join();
      }
    }
  }

  void ensure_workers(int count) {
    if (count <= 0)
      return;
    std::unique_lock<std::mutex> lk(mutex_);
    int current = static_cast<int>(workers_.size());
    if (count <= current) {
      ensure_generation_capacity_locked(current);
      return;
    }
    int target = count;
    ensure_generation_capacity_locked(target);
    workers_.reserve(static_cast<size_t>(target));
    for (int idx = current; idx < target; ++idx) {
      workers_.push_back(Worker{});
    }
    for (int idx = current; idx < target; ++idx) {
      workers_[idx].thread =
          std::thread(&ReusableThreadPool::worker_loop, this, idx);
    }
  }

  void run(int worker_count, const std::function<void(int)> &fn) {
    if (worker_count <= 0)
      return;
    ensure_workers(worker_count);
    std::unique_lock<std::mutex> lk(mutex_);
    current_fn_ = fn;
    active_count_ = worker_count;
    completed_ = 0;
    ++current_generation_;
    const int generation = current_generation_;
    for (int i = 0; i < worker_count; ++i) {
      request_generation_[i] = generation;
    }
    cv_.notify_all();
    done_cv_.wait(lk, [&] { return completed_ == active_count_; });
    active_count_ = 0;
    current_fn_ = nullptr;
  }

private:
  struct Worker {
    std::thread thread;
  };

  void worker_loop(int index) {
    {
      std::string thread_name = "fusionann_worker_" + std::to_string(index);
      fusionann::common::nvtx_name_current_thread(thread_name.c_str());
    }
    std::unique_lock<std::mutex> lk(mutex_);
    while (true) {
      cv_.wait(lk, [&] {
        return shutdown_ || request_generation_[index] > ack_generation_[index];
      });
      if (shutdown_)
        break;
      auto fn = current_fn_;
      int expected_gen = request_generation_[index];
      lk.unlock();
      fn(index);
      lk.lock();
      ack_generation_[index] = expected_gen;
      ++completed_;
      if (completed_ == active_count_) {
        done_cv_.notify_one();
      }
    }
  }

  void ensure_generation_capacity_locked(int count) {
    if (static_cast<int>(request_generation_.size()) < count) {
      request_generation_.resize(static_cast<size_t>(count), 0);
    }
    if (static_cast<int>(ack_generation_.size()) < count) {
      ack_generation_.resize(static_cast<size_t>(count), 0);
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  bool shutdown_ = false;
  std::vector<Worker> workers_;
  std::vector<int> request_generation_;
  std::vector<int> ack_generation_;
  std::function<void(int)> current_fn_;
  int active_count_ = 0;
  int completed_ = 0;
  int current_generation_ = 0;
};

ReusableThreadPool &global_thread_pool() {
  static ReusableThreadPool pool;
  return pool;
}

class PostingHotspotScope {
public:
  explicit PostingHotspotScope(PostingListAccessor &metadata)
      : metadata_(metadata) {
    const char *flag = std::getenv("FUSIONANN_POSTING_HOTSPOTS");
    if (!flag || flag[0] == '\0' || flag[0] == '0') {
      return;
    }
    enabled_ = metadata_.enable_hotspot_tracking();
    topk_ = parse_topk(std::getenv("FUSIONANN_POSTING_HOTSPOTS_TOPK"));
    if (enabled_) {
      std::cout << "  > Posting list hotspot tracking enabled (top " << topk_
                << ")." << std::endl;
    }
  }

  PostingHotspotScope(const PostingHotspotScope &) = delete;
  PostingHotspotScope &operator=(const PostingHotspotScope &) = delete;

  ~PostingHotspotScope() {
    if (enabled_) {
      metadata_.disable_hotspot_tracking();
    }
  }

  bool enabled() const { return enabled_; }

  void report() const {
    if (!enabled_) {
      return;
    }
    uint64_t total = metadata_.hotspot_total_hits();
    auto top = metadata_.top_hotspot_lists(topk_);
    std::cout << "  > Posting list hotspots (samples=" << total << ")\n";
    if (top.empty()) {
      std::cout << "    (no posting list hits recorded)\n";
      return;
    }
    for (size_t i = 0; i < top.size(); ++i) {
      const auto &entry = top[i];
      double pct = total > 0 ? (static_cast<double>(entry.hits) * 100.0 /
                                static_cast<double>(total))
                             : 0.0;
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << pct;
      std::cout << "    #" << (i + 1) << " list=" << entry.list_id
                << " hits=" << entry.hits << " (" << oss.str() << "%)\n";
    }
  }

private:
  static size_t parse_topk(const char *env_value) {
    if (!env_value || env_value[0] == '\0') {
      return 10;
    }
    char *end = nullptr;
    long parsed = std::strtol(env_value, &end, 10);
    if (end == env_value || parsed <= 0) {
      return 10;
    }
    return static_cast<size_t>(parsed);
  }

  PostingListAccessor &metadata_;
  bool enabled_ = false;
  size_t topk_ = 10;
};

} // namespace

CsvLogger::CsvLogger(const std::string &path) : ofs_(path, std::ios::app) {
  if (!ofs_) {
    throw std::runtime_error("Failed to open CSV log at " + path);
  }
  if (ofs_.tellp() == 0) {
    ofs_ << "qid,thread,total_ms,graph_ms,gather_ms,dt_ms,overlap_ms,overlap_"
            "ratio,";
    ofs_ << "dedup_ms,pq_ms,sort_ms,h2d_ids_ms,d2h_topn_ms,rerank_ms,rerank_io_"
            "ms,rerank_l2_ms,";
    ofs_ << "io_hits,io_misses,io_calls,io_evictions,pages_uniq,cand_in,";
    ofs_ << "cand_unique,batches,topk,rerank_size,gpu_cand_in,gpu_cand_unique,";
    ofs_ << "gpu_tmp_bytes,gpu_wait_ms,rerank_examined,mini_batch_size,"
            "mini_batch_pages"
         << '\n';
  }
}

void CsvLogger::append(const QueryStats &s) {
  std::lock_guard<std::mutex> lk(mutex_);
  write_row(s);
}

void CsvLogger::append_no_lock(const QueryStats &s) { write_row(s); }

void CsvLogger::write_row(const QueryStats &s) {
  auto old_flags = ofs_.flags();
  auto old_precision = ofs_.precision();
  ofs_ << std::fixed << std::setprecision(3);
  ofs_ << s.qid << ',' << s.thread_id << ',' << s.t_total_ms << ','
       << s.t_graph_ms << ',' << s.t_gather_ms << ',' << s.t_dt_ms << ','
       << s.overlap_dt_graph_ms << ',' << s.overlap_ratio << ',' << s.t_dedup_ms
       << ',' << s.t_pq_ms << ',' << s.t_sort_ms << ',' << s.t_h2d_ids_ms << ','
       << s.t_d2h_topn_ms << ',' << s.t_rerank_ms << ',' << s.t_rerank_io_ms
       << ',' << s.t_rerank_l2_ms << ',' << s.io_hits << ',' << s.io_misses
       << ',' << s.io_calls << ',' << s.io_evictions << ','
       << s.pages_uniq_in_batch << ',' << s.cand_in << ',' << s.cand_unique
       << ',' << s.batches << ',' << s.topk << ',' << s.rerank_size << ','
       << s.gpu_candidates_in << ',' << s.gpu_candidates_unique << ','
       << s.gpu_tmp_storage_bytes << ',' << s.gpu_wait_ms << ','
       << s.rerank_examined << ',' << s.mini_batch_size << ','
       << s.mini_batch_pages << '\n';
  ofs_.flags(old_flags);
  ofs_.precision(old_precision);
}

template <typename T>
static double percentile_copy(std::vector<T> values, double p) {
  if (values.empty())
    return 0.0;
  if (p <= 0.0)
    return static_cast<double>(values.front());
  if (p >= 100.0)
    return static_cast<double>(values.back());
  std::sort(values.begin(), values.end());
  double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(idx));
  size_t hi = static_cast<size_t>(std::ceil(idx));
  double weight = idx - static_cast<double>(lo);
  double lo_val = static_cast<double>(values[lo]);
  double hi_val = static_cast<double>(values[hi]);
  return (1.0 - weight) * lo_val + weight * hi_val;
}

RunSummary summarize_run(const std::vector<QueryStats> &stats,
                         const std::vector<double> &recalls, double wall_ms,
                         const ServerConfig &cfg, int threads_used) {
  RunSummary summary;
  summary.wall_ms = wall_ms;
  summary.total_queries = static_cast<uint64_t>(stats.size());
  summary.threads = threads_used;
  summary.nprobe = cfg.nprobe;
  summary.rerank_size = cfg.rerank_size;
  summary.batch_size = cfg.batch_size;
  summary.top_k = cfg.top_k;
  summary.recall_k = cfg.recall_k;

  if (summary.wall_ms > 0.0 && summary.total_queries > 0) {
    summary.qps_wall =
        static_cast<double>(summary.total_queries) / (summary.wall_ms / 1000.0);
  }

  if (stats.empty())
    return summary;

  std::vector<double> totals;
  totals.reserve(stats.size());
  std::vector<double> graph_lat;
  graph_lat.reserve(stats.size());
  std::vector<double> gather_lat;
  gather_lat.reserve(stats.size());
  std::vector<double> dt_lat;
  dt_lat.reserve(stats.size());
  std::vector<double> rerank_lat;
  rerank_lat.reserve(stats.size());
  std::vector<double> rerank_io_lat;
  rerank_io_lat.reserve(stats.size());
  std::vector<double> rerank_l2_lat;
  rerank_l2_lat.reserve(stats.size());
  std::vector<double> gpu_waits;
  gpu_waits.reserve(stats.size());
  std::vector<double> miss_rates;
  miss_rates.reserve(stats.size());
  double total_overlap_ratio = 0.0;
  double total_rerank_examined = 0.0;
  double total_batch_pages = 0.0;
  double total_batch_size = 0.0;
  uint64_t total_batches = 0;

  for (const auto &s : stats) {
    totals.push_back(s.t_total_ms);
    graph_lat.push_back(s.t_graph_ms);
    gather_lat.push_back(s.t_gather_ms);
    dt_lat.push_back(static_cast<double>(s.t_dt_ms));
    rerank_lat.push_back(s.t_rerank_ms);
    rerank_io_lat.push_back(s.t_rerank_io_ms);
    rerank_l2_lat.push_back(s.t_rerank_l2_ms);
    gpu_waits.push_back(s.gpu_wait_ms);
    summary.io_hits += s.io_hits;
    summary.io_misses += s.io_misses;
    summary.io_calls += s.io_calls;
    summary.io_evictions += s.io_evictions;
    double denom = static_cast<double>(s.io_hits + s.io_misses);
    double miss_rate =
        denom > 0.0 ? static_cast<double>(s.io_misses) / denom : 0.0;
    miss_rates.push_back(miss_rate);
    total_overlap_ratio += s.overlap_ratio;
    total_rerank_examined += static_cast<double>(s.rerank_examined);
    total_batch_pages += static_cast<double>(s.pages_uniq_in_batch);
    total_batch_size += s.mini_batch_size;
    total_batches += static_cast<uint64_t>(s.batches);
  }

  double sum_total_ms = std::accumulate(totals.begin(), totals.end(), 0.0);
  summary.avg_ms =
      totals.empty() ? 0.0 : sum_total_ms / static_cast<double>(totals.size());
  summary.qps_effective = summary.avg_ms > 0.0 ? 1000.0 / summary.avg_ms : 0.0;
  summary.p50_ms = percentile_copy(totals, 50.0);
  summary.p90_ms = percentile_copy(totals, 90.0);
  summary.p99_ms = percentile_copy(totals, 99.0);

  summary.avg_graph_ms =
      std::accumulate(graph_lat.begin(), graph_lat.end(), 0.0) /
      static_cast<double>(graph_lat.size());
  summary.avg_gather_ms =
      std::accumulate(gather_lat.begin(), gather_lat.end(), 0.0) /
      static_cast<double>(gather_lat.size());
  summary.avg_dt_ms = std::accumulate(dt_lat.begin(), dt_lat.end(), 0.0) /
                      static_cast<double>(dt_lat.size());
  summary.avg_rerank_ms =
      std::accumulate(rerank_lat.begin(), rerank_lat.end(), 0.0) /
      static_cast<double>(rerank_lat.size());
  summary.avg_rerank_io_ms =
      std::accumulate(rerank_io_lat.begin(), rerank_io_lat.end(), 0.0) /
      static_cast<double>(rerank_io_lat.size());
  summary.avg_rerank_l2_ms =
      std::accumulate(rerank_l2_lat.begin(), rerank_l2_lat.end(), 0.0) /
      static_cast<double>(rerank_l2_lat.size());
  summary.avg_overlap_ratio =
      total_overlap_ratio / static_cast<double>(stats.size());

  summary.avg_gpu_wait_ms =
      std::accumulate(gpu_waits.begin(), gpu_waits.end(), 0.0) /
      static_cast<double>(gpu_waits.size());
  summary.avg_rerank_examined =
      total_rerank_examined / static_cast<double>(stats.size());
  summary.avg_inflight =
      summary.wall_ms > 0.0 ? sum_total_ms / summary.wall_ms : 0.0;

  if (total_batches > 0) {
    summary.avg_batch_pages =
        total_batch_pages / static_cast<double>(total_batches);
    summary.avg_batch_size =
        total_batch_size / static_cast<double>(total_batches);
  }

  if (!recalls.empty()) {
    std::vector<double> recall_copy = recalls;
    summary.avg_recall =
        std::accumulate(recall_copy.begin(), recall_copy.end(), 0.0) /
        static_cast<double>(recall_copy.size());
    summary.recall_p50 = percentile_copy(recall_copy, 50.0);
    summary.recall_p90 = percentile_copy(recall_copy, 90.0);
    summary.recall_p99 = percentile_copy(recall_copy, 99.0);
  }

  if (!miss_rates.empty()) {
    summary.miss_rate_avg =
        std::accumulate(miss_rates.begin(), miss_rates.end(), 0.0) /
        static_cast<double>(miss_rates.size());
    summary.miss_rate_p50 = percentile_copy(miss_rates, 50.0);
    summary.miss_rate_p90 = percentile_copy(miss_rates, 90.0);
    summary.miss_rate_p99 = percentile_copy(miss_rates, 99.0);
  }

  return summary;
}

void append_summary_csv(const std::string &path, const RunSummary &summary) {
  std::ofstream ofs(path, std::ios::app);
  if (!ofs) {
    throw std::runtime_error("Failed to open summary CSV at " + path);
  }
  if (ofs.tellp() == 0) {
    ofs << "threads,nprobe,rerank,batch,topk,recall_k,queries,wall_ms,"
           "qps_effective,qps_wall,avg_ms,p50_ms,p90_ms,p99_ms,avg_recall,"
           "recall_p50,recall_p90,recall_p99,avg_graph_ms,avg_gather_ms,avg_dt_"
           "ms,"
           "avg_rerank_ms,avg_rerank_io_ms,avg_rerank_l2_ms,avg_overlap,avg_"
           "gpu_wait_ms,avg_rerank_examined,"
           "avg_batch_pages,avg_batch_size,avg_inflight,miss_rate_avg,miss_"
           "rate_p50,"
           "miss_rate_p90,miss_rate_p99,io_hits,io_misses,io_calls,"
           "io_evictions\n";
  }
  ofs << summary.threads << ',' << summary.nprobe << ',' << summary.rerank_size
      << ',' << summary.batch_size << ',' << summary.top_k << ','
      << summary.recall_k << ',' << summary.total_queries << ','
      << summary.wall_ms << ',' << summary.qps_effective << ','
      << summary.qps_wall << ',' << summary.avg_ms << ',' << summary.p50_ms
      << ',' << summary.p90_ms << ',' << summary.p99_ms << ','
      << summary.avg_recall << ',' << summary.recall_p50 << ','
      << summary.recall_p90 << ',' << summary.recall_p99 << ','
      << summary.avg_graph_ms << ',' << summary.avg_gather_ms << ','
      << summary.avg_dt_ms << ',' << summary.avg_rerank_ms << ','
      << summary.avg_rerank_io_ms << ',' << summary.avg_rerank_l2_ms << ','
      << summary.avg_overlap_ratio << ',' << summary.avg_gpu_wait_ms << ','
      << summary.avg_rerank_examined << ',' << summary.avg_batch_pages << ','
      << summary.avg_batch_size << ',' << summary.avg_inflight << ','
      << summary.miss_rate_avg << ',' << summary.miss_rate_p50 << ','
      << summary.miss_rate_p90 << ',' << summary.miss_rate_p99 << ','
      << summary.io_hits << ',' << summary.io_misses << ',' << summary.io_calls
      << ',' << summary.io_evictions << '\n';
}

std::vector<int> parse_int_list(const std::string &spec) {
  std::vector<int> values;
  if (spec.empty())
    return values;
  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty())
      continue;
    try {
      int v = std::stoi(token);
      values.push_back(v);
    } catch (const std::exception &) {
      throw std::invalid_argument("Failed to parse integer in list: " + token);
    }
  }
  return values;
}

double compute_recall(const std::vector<long> &results, const int *ground_truth,
                      int k) {
  std::unordered_set<int> gt_set(ground_truth, ground_truth + k);
  int hits = 0;
  const int limit = std::min(k, static_cast<int>(results.size()));
  for (int i = 0; i < limit; ++i) {
    if (gt_set.count(results[i])) {
      hits++;
    }
  }
  return k > 0 ? static_cast<double>(hits) / k : 0.0;
}

std::vector<long>
heuristic_rerank(const float *query_vector,
                 const std::vector<ResultPair> &approximate_results,
                 IOManager &io_manager, int dim, int top_k, int rerank_size,
                 int batch_size, float stability_delta, int stability_iters,
                 QueryStats *stats, bool enable_detail) {
  if (rerank_size <= 0)
    return {};
  if (batch_size <= 0)
    batch_size = 1;
  const float epsilon_delta =
      std::max(1e-5f, std::isfinite(stability_delta) ? stability_delta : 0.05f);
  const int beta_stability = std::max(1, stability_iters);

  std::priority_queue<std::pair<float, long>> max_heap;
  int stability_counter = 0;
  std::unordered_set<long> last_top_k_ids;
  const int effective_rerank_size =
      std::min(static_cast<int>(approximate_results.size()), rerank_size);
  int processed = 0;

  double rerank_io_ms = 0.0;
  double rerank_l2_ms = 0.0;
  const bool capture_detail = enable_detail && stats;
  auto timed_accumulate = [&](double &bucket, const char *nvtx_label,
                              std::uint32_t color, auto &&fn) {
    FUSIONANNS_NVTX_RANGE_COLOR(nvtx_label, color);
    if (!capture_detail) {
      fn();
      return;
    }
    auto begin = std::chrono::steady_clock::now();
    fn();
    auto end = std::chrono::steady_clock::now();
    bucket += std::chrono::duration<double, std::milli>(end - begin).count();
  };

  int current_index = 0;
  while (current_index < effective_rerank_size) {
    int current_batch_size =
        std::min(batch_size, effective_rerank_size - current_index);
    std::vector<long> batch_ids(current_batch_size);
    for (int j = 0; j < current_batch_size; ++j) {
      batch_ids[j] = approximate_results[current_index + j].id;
    }

    if (stats) {
      stats->batches += 1;
    }
    std::vector<float> raw_vectors_batch;
    uint32_t unique_pages = 0;
    timed_accumulate(rerank_io_ms, "Rerank.IOBatchFetch", 0xFFFFF176u, [&] {
      io_manager.get_vectors(batch_ids, raw_vectors_batch,
                             capture_detail ? &unique_pages : nullptr);
    });
    if (stats) {
      stats->pages_uniq_in_batch += unique_pages;
      stats->mini_batch_size += static_cast<double>(current_batch_size);
      stats->mini_batch_pages += static_cast<double>(unique_pages);
      stats->rerank_examined += current_batch_size;
    }

    timed_accumulate(rerank_l2_ms, "Rerank.L2Batch", 0xFFFFB74Du, [&] {
      for (int j = 0; j < current_batch_size; ++j) {
        long vec_id = batch_ids[j];
        const float *raw_vector = raw_vectors_batch.data() + j * dim;
        float exact_dist = faiss::fvec_L2sqr(query_vector, raw_vector, dim);

        if (max_heap.size() < top_k || exact_dist < max_heap.top().first) {
          if (max_heap.size() >= top_k) {
            max_heap.pop();
          }
          max_heap.push({exact_dist, vec_id});
        }
      }
    });

    processed += current_batch_size;
    current_index += current_batch_size;

    if (max_heap.size() < top_k)
      continue;
    // 第一次 heap 满时，记录初始快照
    if (last_top_k_ids.empty()) {
      auto bootstrap_heap = max_heap;
      while (!bootstrap_heap.empty()) {
        last_top_k_ids.insert(bootstrap_heap.top().second);
        bootstrap_heap.pop();
      }
      continue;
    }
    std::unordered_set<long> current_top_k_ids;
    auto temp_heap = max_heap;
    while (!temp_heap.empty()) {
      current_top_k_ids.insert(temp_heap.top().second);
      temp_heap.pop();
    }

    int next_index = current_index;

    int intersection_size = 0;
    for (long id : current_top_k_ids) {
      if (last_top_k_ids.count(id)) {
        intersection_size++;
      }
    }
    double delta = static_cast<double>(top_k - intersection_size) / top_k;
    if (delta < epsilon_delta) {
      stability_counter++;
    } else {
      stability_counter = 0;
    }
    bool request_break = stability_counter >= beta_stability;
    if (request_break) {
      break;
    }
    last_top_k_ids = std::move(current_top_k_ids);
  }

  std::vector<std::pair<float, long>> final_results;
  while (!max_heap.empty()) {
    final_results.push_back(max_heap.top());
    max_heap.pop();
  }
  std::sort(final_results.begin(), final_results.end());
  std::vector<long> final_ids;
  for (const auto &p : final_results) {
    final_ids.push_back(p.second);
  }
  if (stats) {
    stats->t_rerank_io_ms = rerank_io_ms;
    stats->t_rerank_l2_ms = rerank_l2_ms;
  }
  return final_ids;
}

void load_ivecs(const std::string &filename, std::vector<int> &data, long &num,
                int &dim) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  ifs.read(reinterpret_cast<char *>(&dim), sizeof(int));
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  num = file_size / ((dim * sizeof(int)) + sizeof(int));
  data.resize(num * dim);
  ifs.seekg(0, std::ios::beg);
  std::vector<int> temp_vec(dim);
  int temp_dim = dim;
  for (long i = 0; i < num; ++i) {
    ifs.read(reinterpret_cast<char *>(&temp_dim), sizeof(int));
    if (temp_dim != dim) {
      throw std::runtime_error("Inconsistent dimension in " + filename);
    }
    ifs.read(reinterpret_cast<char *>(temp_vec.data()), dim * sizeof(int));
    std::copy(temp_vec.begin(), temp_vec.end(), data.begin() + i * dim);
  }
}

template <typename ScorerT>
RunOnceResult
run_once_impl(ScorerT &scorer, const ServerConfig &cfg, IOManager &io_manager,
              PostingListAccessor &metadata, ICentroidNavigator &centroid_nav,
              const std::vector<float> &query_data, int query_dim,
              const std::vector<int> &gt_data, int gt_dim,
              const std::string &stats_csv_path) {
  FUSIONANNS_NVTX_RANGE("RunOnceTotal");
  const int available_queries =
      query_dim > 0 ? static_cast<int>(query_data.size() / query_dim) : 0;
  if (available_queries <= 0) {
    throw std::runtime_error("No query vectors available for benchmarking");
  }
  int queries_to_run = cfg.queries_to_run > 0
                           ? std::min(cfg.queries_to_run, available_queries)
                           : available_queries;
  if (queries_to_run <= 0) {
    throw std::runtime_error("queries_to_run resolved to zero");
  }
  int warmup_queries = std::clamp(cfg.warmup, 0, queries_to_run);
  int repeats = std::max(cfg.repeat, 1);
  int measurement_queries = queries_to_run * repeats;
  if (measurement_queries <= 0) {
    measurement_queries = queries_to_run;
  }
  const bool has_groundtruth =
      !gt_data.empty() && gt_dim >= cfg.recall_k && gt_dim >= cfg.top_k;

  if (query_dim != scorer.query_dim()) {
    throw std::invalid_argument("Query dimension mismatch for scorer");
  }

  const int candidate_capacity = scorer.candidate_capacity();
  if (candidate_capacity <= 0) {
    throw std::runtime_error("Invalid candidate capacity from scorer");
  }

  std::cout << "  > Initializing GPU block pool with C_max="
            << candidate_capacity << "..." << std::endl;
  GpuBlockPool block_pool;
  {
    FUSIONANNS_NVTX_RANGE("InitBlockPool");
    if (!block_pool.init(scorer.bytes_per_block(), cfg.reserve_ratio)) {
      throw std::runtime_error("Failed to initialize GPU block pool");
    }
  }
  std::cout << "  > GPU block pool initialized successfully." << std::endl;

  int thread_count = std::min(cfg.threads, block_pool.capacity());
  thread_count = std::max(thread_count, 1);
  std::cout << "  > Using " << thread_count
            << " worker threads for benchmarking." << std::endl;
  ReusableThreadPool &worker_pool = global_thread_pool();

  SPTAGCentroidNavigator *sptag_nav =
      dynamic_cast<SPTAGCentroidNavigator *>(&centroid_nav);
  if (sptag_nav) {
    sptag_nav->tune_for_nprobe(cfg.nprobe);
  }

#ifdef FUSIONANNS_USE_CUVS_CAGRA
  // CAGRA: 初始化 resource pool + batch collector
  // batch_size = thread_count，让所有并发线程的查询可以合并为一次 GPU kernel
  // slot 数量 = 2（一个在执行 kernel，另一个在准备下一批数据，形成流水线）
  CAGRACentroidNavigator *cagra_nav =
      dynamic_cast<CAGRACentroidNavigator *>(&centroid_nav);
  std::unique_ptr<CagraBatchCollector> cagra_collector;
  if (cagra_nav) {
    int cagra_batch_size = thread_count;  // 最大收集 thread_count 个查询
    int cagra_slots = 2;  // 双缓冲：一个执行 kernel，一个准备数据
    cagra_nav->init_resource_pool(cagra_slots, cfg.nprobe, cagra_batch_size);
    cagra_nav->enable_profiling(true);
    cagra_nav->reset_profile_stats();
    // 创建 batch collector（超时 200us：如果凑不满 batch 就提前 flush）
    cagra_collector = cagra_nav->create_batch_collector(
        cagra_batch_size, cfg.nprobe, 200);
  }
#else
  void *cagra_collector = nullptr;  // placeholder
#endif

  RunOnceResult result;
  result.stats.reserve(static_cast<size_t>(measurement_queries));
  result.recalls.reserve(static_cast<size_t>(measurement_queries));
  std::mutex stats_mutex;
  std::atomic<bool> worker_error{false};
  std::exception_ptr worker_eptr = nullptr;
  const int stats_sample_period = std::max(1, cfg.stats_sample_rate);
  auto should_emit_csv = [&](size_t sequence) -> bool {
    if (!cfg.record_query_stats || stats_csv_path.empty()) {
      return false;
    }
    return (sequence % static_cast<size_t>(stats_sample_period)) == 0;
  };

  struct Task {
    size_t sequence = 0;
    int query_index = 0;
    bool record = false;
    bool emit_to_csv = false;
  };

  struct StreamContext {
    cudaStream_t xfer = nullptr;
    cudaStream_t comp = nullptr;
    StreamContext() = default;
    StreamContext(const StreamContext &) = delete;
    StreamContext &operator=(const StreamContext &) = delete;
    StreamContext(StreamContext &&other) noexcept
        : xfer(other.xfer), comp(other.comp) {
      other.xfer = nullptr;
      other.comp = nullptr;
    }
    StreamContext &operator=(StreamContext &&other) noexcept {
      if (this != &other) {
        destroy();
        xfer = other.xfer;
        comp = other.comp;
        other.xfer = nullptr;
        other.comp = nullptr;
      }
      return *this;
    }
    ~StreamContext() { destroy(); }
    void destroy() {
      if (xfer) {
        cudaStreamDestroy(xfer);
        xfer = nullptr;
      }
      if (comp) {
        cudaStreamDestroy(comp);
        comp = nullptr;
      }
    }
  };

  auto make_stream_context = [&]() {
    StreamContext ctx;
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.xfer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.comp, cudaStreamNonBlocking));
    return ctx;
  };

  std::vector<StreamContext> stream_pool;
  stream_pool.reserve(static_cast<size_t>(thread_count));
  for (int i = 0; i < thread_count; ++i) {
    stream_pool.push_back(make_stream_context());
  }
  std::vector<GpuBlock *> thread_blocks;
  thread_blocks.reserve(static_cast<size_t>(thread_count));
  struct BlockLeaseGuard {
    GpuBlockPool *pool = nullptr;
    std::vector<GpuBlock *> *blocks = nullptr;
    ~BlockLeaseGuard() {
      if (!pool || !blocks)
        return;
      for (GpuBlock *&block : *blocks) {
        if (block) {
          pool->release_block(block);
          block = nullptr;
        }
      }
    }
  } block_guard{&block_pool, &thread_blocks};
  for (int i = 0; i < thread_count; ++i) {
    thread_blocks.push_back(block_pool.acquire_block());
  }

  // ========== Pre-allocate per-thread pinned memory ==========
  // Pinned memory 应在主线程串行分配，避免 CUDA 运行时锁竞争
  struct ThreadPinnedBuffers {
    float *h_query = nullptr;
    float *d_query_mapped = nullptr;
    long long *h_candidate = nullptr;
    long long *d_candidate_mapped = nullptr;
  };
  std::vector<ThreadPinnedBuffers> pinned_pool;
  pinned_pool.resize(static_cast<size_t>(thread_count));
  {
    unsigned int host_flags = cudaHostAllocPortable | cudaHostAllocMapped;
    for (int i = 0; i < thread_count; ++i) {
      ThreadPinnedBuffers &buf = pinned_pool[i];
      CUDA_CHECK(cudaHostAlloc(&buf.h_query,
                               sizeof(float) * static_cast<size_t>(query_dim),
                               host_flags));
      CUDA_CHECK(cudaHostGetDevicePointer(
          reinterpret_cast<void **>(&buf.d_query_mapped), buf.h_query, 0));
      CUDA_CHECK(cudaHostAlloc(&buf.h_candidate,
                               sizeof(long long) *
                                   static_cast<size_t>(candidate_capacity),
                               host_flags));
      CUDA_CHECK(cudaHostGetDevicePointer(
          reinterpret_cast<void **>(&buf.d_candidate_mapped), buf.h_candidate,
          0));
    }
  }
  struct PinnedPoolGuard {
    std::vector<ThreadPinnedBuffers> *pool = nullptr;
    ~PinnedPoolGuard() {
      if (!pool)
        return;
      for (auto &buf : *pool) {
        if (buf.h_query) {
          cudaFreeHost(buf.h_query);
          buf.h_query = nullptr;
        }
        if (buf.h_candidate) {
          cudaFreeHost(buf.h_candidate);
          buf.h_candidate = nullptr;
        }
      }
    }
  } pinned_guard{&pinned_pool};

  auto create_events = [&](ThreadGpuEvents &events) {
    auto make_event = [&](cudaEvent_t &evt, unsigned int flags) {
      if (flags == 0) {
        CUDA_CHECK(cudaEventCreate(&evt));
      } else {
        CUDA_CHECK(cudaEventCreateWithFlags(&evt, flags));
      }
    };
    make_event(events.query_ready, cudaEventDisableTiming);
    make_event(events.dt_start, 0);
    make_event(events.dt_done, 0);
    make_event(events.h2d_start, 0);
    make_event(events.h2d_stop, 0);
    make_event(events.dedup_start, 0);
    make_event(events.dedup_stop, 0);
    make_event(events.pq_start, 0);
    make_event(events.pq_stop, 0);
    make_event(events.d2h_start, 0);
    make_event(events.d2h_stop, 0);
    int desired_host_capacity =
        std::max(32, std::max(candidate_capacity, cfg.rerank_size));
    auto ensure_host_buffer = [&](ThreadGpuEvents &evt) {
      if (evt.host_capacity >= desired_host_capacity && evt.h_dists &&
          evt.h_ids) {
        if (!evt.d_dists_mapped) {
          CUDA_CHECK(cudaHostGetDevicePointer(
              reinterpret_cast<void **>(&evt.d_dists_mapped), evt.h_dists, 0));
        }
        if (!evt.d_ids_mapped) {
          CUDA_CHECK(cudaHostGetDevicePointer(
              reinterpret_cast<void **>(&evt.d_ids_mapped), evt.h_ids, 0));
        }
        return;
      }
      if (evt.h_dists) {
        cudaFreeHost(evt.h_dists);
        evt.h_dists = nullptr;
      }
      if (evt.h_ids) {
        cudaFreeHost(evt.h_ids);
        evt.h_ids = nullptr;
      }
      unsigned int host_flags = cudaHostAllocPortable | cudaHostAllocMapped;
      CUDA_CHECK(cudaHostAlloc(&evt.h_dists,
                               sizeof(float) *
                                   static_cast<size_t>(desired_host_capacity),
                               host_flags));
      CUDA_CHECK(cudaHostAlloc(&evt.h_ids,
                               sizeof(long long) *
                                   static_cast<size_t>(desired_host_capacity),
                               host_flags));
      evt.host_capacity = desired_host_capacity;
      CUDA_CHECK(cudaHostGetDevicePointer(
          reinterpret_cast<void **>(&evt.d_dists_mapped), evt.h_dists, 0));
      CUDA_CHECK(cudaHostGetDevicePointer(
          reinterpret_cast<void **>(&evt.d_ids_mapped), evt.h_ids, 0));
    };
    ensure_host_buffer(events);

    if (!events.h_unique_count) {
      CUDA_CHECK(cudaHostAlloc(&events.h_unique_count, sizeof(int),
                               cudaHostAllocPortable));
    }
    if (events.h_unique_count) {
      *events.h_unique_count = 0;
    }
    events.pending_copy_count = 0;
    events.recorded_h2d = false;
    events.recorded_dedup = false;
    events.recorded_pq = false;
    events.recorded_d2h = false;
    events.host_results_current = false;
  };

  auto destroy_events = [&](ThreadGpuEvents &events) {
    auto destroy = [](cudaEvent_t &evt) {
      if (evt) {
        cudaEventDestroy(evt);
        evt = nullptr;
      }
    };
    destroy(events.query_ready);
    destroy(events.dt_start);
    destroy(events.dt_done);
    destroy(events.h2d_start);
    destroy(events.h2d_stop);
    destroy(events.dedup_start);
    destroy(events.dedup_stop);
    destroy(events.pq_start);
    destroy(events.pq_stop);
    destroy(events.d2h_start);
    destroy(events.d2h_stop);
    if (events.h_dists) {
      cudaFreeHost(events.h_dists);
      events.h_dists = nullptr;
    }
    if (events.h_ids) {
      cudaFreeHost(events.h_ids);
      events.h_ids = nullptr;
    }
    events.d_dists_mapped = nullptr;
    events.d_ids_mapped = nullptr;
    if (events.h_unique_count) {
      cudaFreeHost(events.h_unique_count);
      events.h_unique_count = nullptr;
    }
    events.host_capacity = 0;
    events.pending_copy_count = 0;
    events.host_results_current = false;
  };

  // ========== Pre-allocate per-thread GPU events ==========
  // Events 应在主线程串行创建，避免 cudaEventCreate 的锁竞争
  std::vector<ThreadGpuEvents> events_pool;
  events_pool.resize(static_cast<size_t>(thread_count));
  for (int i = 0; i < thread_count; ++i) {
    create_events(events_pool[i]);
  }
  struct EventsPoolGuard {
    std::vector<ThreadGpuEvents> *pool = nullptr;
    std::function<void(ThreadGpuEvents &)> destroyer;
    ~EventsPoolGuard() {
      if (!pool || !destroyer)
        return;
      for (auto &evt : *pool) {
        destroyer(evt);
      }
    }
  } events_guard{&events_pool, destroy_events};

  auto process_task = [&](const Task &task, int thread_id,
                          float *h_query_pinned, const float *d_query_pinned,
                          long long *h_candidate_pinned,
                          const long long *d_candidate_pinned,
                          QueryStats &stats_out, bool record_stats,
                          bool emit_to_csv, cudaStream_t stream_xfer,
                          cudaStream_t stream_comp, GpuBlock *block,
                          ThreadGpuEvents &events,
                          std::vector<QueryStats> *thread_stats,
                          std::vector<double> *thread_recalls) {
    FUSIONANNS_NVTX_RANGE_COLOR(
        "ProcessQuery", 0xFF1E88E5u); // 单次查询处理 / per-query processing
    const bool timing_enabled = cfg.record_query_stats && record_stats;
    auto t_total_begin = std::chrono::steady_clock::now();
    const float *query_vector =
        query_data.data() + static_cast<size_t>(task.query_index) * query_dim;
    std::memcpy(h_query_pinned, query_vector,
                static_cast<size_t>(query_dim) * sizeof(float));

    if (!block) {
      throw std::runtime_error("Null GPU block assigned to worker");
    }
    auto workspace = scorer.prepare_workspace(*block);
    stats_out.gpu_wait_ms = 0.0;
    std::chrono::steady_clock::time_point t_dt_begin_host;
    t_dt_begin_host = std::chrono::steady_clock::now();
    scorer.prepare_query(h_query_pinned, query_dim, workspace, stream_xfer,
                         stream_comp, events, timing_enabled, d_query_pinned);

    auto t_graph_begin = std::chrono::steady_clock::now();
    ProbeResult probe;
    {
      FUSIONANNS_NVTX_RANGE_COLOR("GraphSearch",
                                  0xFFE91E63u); // 图遍历 / graph traversal
#ifdef FUSIONANNS_USE_CUVS_CAGRA
      if (cagra_collector) {
        // CAGRA batch mode: 提交查询到 batch collector，等待批量 GPU kernel 返回
        auto future = cagra_collector->submit(query_vector);
        probe = future.get();
      } else {
        probe = centroid_nav.search_one(query_vector, cfg.nprobe);
      }
#else
      probe = centroid_nav.search_one(query_vector, cfg.nprobe);
#endif
    }
    auto t_graph_end = std::chrono::steady_clock::now();
    stats_out.t_graph_ms =
        std::chrono::duration<double, std::milli>(t_graph_end - t_graph_begin)
            .count();

    stats_out.cand_in = 0;
    stats_out.cand_unique = 0;
    auto t_gather_begin = std::chrono::steady_clock::now();
    double dedup_ms = 0.0;

    // Threshold: if max_vector_id > 16M the bitset would exceed 2MB,
    // fall back to hash set approach.
    static constexpr size_t kBitsetMaxId = 16u << 20; // 16M
    const size_t max_vid = metadata.max_vector_id();

    if (max_vid <= kBitsetMaxId) {
      // ── Fast path: fused gather + dedup via thread_local bitset ──
      FUSIONANNS_NVTX_RANGE_COLOR("GatherDedup",
                                  0xFF5C6BC0u); // fused gather+dedup
      const size_t num_words = (max_vid / 64) + 1;
      static thread_local std::vector<uint64_t> seen_bits;
      if (seen_bits.size() < num_words)
        seen_bits.resize(num_words, 0);

      size_t min_word = num_words;
      size_t max_word = 0;

      for (long centroid_id : probe.ids) {
        if (centroid_id < 0)
          continue;
        PostingListAccessor::View view =
            metadata.view(static_cast<int>(centroid_id));
        for (size_t j = 0; j < view.size; ++j) {
          int32_t raw_id = view.data[j];
          stats_out.cand_in++;
          if (raw_id < 0)
            continue;
          size_t id = static_cast<size_t>(raw_id);
          size_t word = id >> 6;       // id / 64
          uint64_t bit = uint64_t(1) << (id & 63);
          if (!(seen_bits[word] & bit)) {
            seen_bits[word] |= bit;
            if (word < min_word) min_word = word;
            if (word > max_word) max_word = word;
            if (stats_out.cand_unique < candidate_capacity) {
              h_candidate_pinned[stats_out.cand_unique++] =
                  static_cast<long long>(raw_id);
            }
          }
        }
      }

      // Reset only the touched portion of the bitset.
      if (min_word <= max_word) {
        std::memset(&seen_bits[min_word], 0,
                    (max_word - min_word + 1) * sizeof(uint64_t));
      }
    } else {
      // ── Slow path: hash set fallback for very large ID spaces ──
      FUSIONANNS_NVTX_RANGE_COLOR("GatherDedup",
                                  0xFF5C6BC0u); // fused gather+dedup (fallback)
      boost::unordered_flat_set<long long> seen;
      seen.reserve(static_cast<size_t>(candidate_capacity) * 2);
      seen.max_load_factor(0.6f);

      for (long centroid_id : probe.ids) {
        if (centroid_id < 0)
          continue;
        PostingListAccessor::View view =
            metadata.view(static_cast<int>(centroid_id));
        for (size_t j = 0; j < view.size; ++j) {
          int32_t raw_id = view.data[j];
          stats_out.cand_in++;
          if (raw_id < 0)
            continue;
          long long id = static_cast<long long>(raw_id);
          auto inserted = seen.insert(id);
          if (inserted.second) {
            if (stats_out.cand_unique < candidate_capacity) {
              h_candidate_pinned[stats_out.cand_unique++] = id;
            }
          }
        }
      }
    }

    auto t_gather_end = std::chrono::steady_clock::now();
    stats_out.t_gather_ms =
        std::chrono::duration<double, std::milli>(t_gather_end - t_gather_begin)
            .count();
    dedup_ms = stats_out.t_gather_ms; // dedup is fused into gather now

    int num_candidates =
        std::min(stats_out.cand_unique, candidate_capacity);
    const int initial_num_candidates = stats_out.cand_in;

    static thread_local std::vector<ResultPair> approximate_results;
    approximate_results.clear(); // 只重置 size，保留 capacity

    GpuTimings gpu_timings_total{};
    if (record_stats) {
      gpu_timings_total.tmp_storage_bytes = scorer.tmp_storage_bytes();
      gpu_timings_total.candidates_in = initial_num_candidates;
      gpu_timings_total.candidates_unique = stats_out.cand_unique;
      gpu_timings_total.topk_requested = cfg.rerank_size;
      gpu_timings_total.dedup_ms = static_cast<float>(dedup_ms);
    }
    if (timing_enabled) {
      stats_out.t_dedup_ms = static_cast<float>(dedup_ms);
    }

    events.recorded_dedup = false;

    num_candidates = stats_out.cand_unique;
    if (events.h_unique_count) {
      *events.h_unique_count = num_candidates;
    }
    scorer.score_candidates(num_candidates, h_candidate_pinned, workspace,
                            stream_xfer, stream_comp, events, timing_enabled,
                            d_candidate_pinned, cfg.use_gpu_topk);
    if (record_stats) {
      gpu_timings_total.candidates_unique = num_candidates;
    }

    double cpu_topk_ms = 0.0;
    if (cfg.rerank_size > 0 && num_candidates > 0) {
      GpuTimings iteration_timings{};
      GpuTimings *timings_ptr = timing_enabled ? &iteration_timings : nullptr;
      {
        FUSIONANNS_NVTX_RANGE_COLOR(
            "FetchApproxResults",
            0xFFCDDC39u); // PQ结果收集 / fetch GPU results
        scorer.fetch_results(num_candidates, workspace, stream_comp, events,
                             timings_ptr, approximate_results,
                             h_candidate_pinned, &stats_out.gpu_wait_ms,
                             cfg.rerank_size, cfg.use_gpu_topk,
                             d_candidate_pinned);
      }

      if (!cfg.use_gpu_topk && !approximate_results.empty()) {
        FUSIONANNS_NVTX_RANGE_COLOR(
            "CPUSelectTopR",
            0xFFAED581u); // CPU选择稳定TopR / CPU selection
        auto t_topk_begin = std::chrono::steady_clock::now();
        int limit = std::max(0, cfg.rerank_size);
        if (limit > 0 && limit < static_cast<int>(approximate_results.size())) {
          auto nth =
              approximate_results.begin() +
              static_cast<std::vector<ResultPair>::difference_type>(limit);
          std::nth_element(approximate_results.begin(), nth,
                           approximate_results.end(),
                           [](const ResultPair &a, const ResultPair &b) {
                             return a.dist < b.dist;
                           });
          approximate_results.resize(static_cast<size_t>(limit));
        }
        std::sort(approximate_results.begin(), approximate_results.end(),
                  [](const ResultPair &a, const ResultPair &b) {
                    return a.dist < b.dist;
                  });
        auto t_topk_end = std::chrono::steady_clock::now();
        cpu_topk_ms =
            std::chrono::duration<double, std::milli>(t_topk_end - t_topk_begin)
                .count();
      }

      if (timing_enabled) {
        if (events.recorded_pq) {
          CUDA_CHECK(cudaEventElapsedTime(&iteration_timings.pq_ms,
                                          events.pq_start, events.pq_stop));
        }
        if (events.recorded_d2h) {
          CUDA_CHECK(cudaEventElapsedTime(&iteration_timings.d2h_topn_ms,
                                          events.d2h_start, events.d2h_stop));
        }
      }
      iteration_timings.candidates_unique = static_cast<int>(
          std::min<std::size_t>(approximate_results.size(),
                                static_cast<std::size_t>(num_candidates)));
      events.pending_copy_count = 0;
      events.recorded_d2h = false;
      events.recorded_pq = false;

      if (record_stats) {
        gpu_timings_total.pq_ms = iteration_timings.pq_ms;
        gpu_timings_total.d2h_topn_ms = iteration_timings.d2h_topn_ms;
        gpu_timings_total.sort_ms = 0.0f;
      }
    } else {
      approximate_results.clear();
      events.pending_copy_count = 0;
      events.recorded_d2h = false;
      events.recorded_pq = false;
    }

    if (timing_enabled && events.recorded_h2d) {
      cudaError_t q = cudaEventQuery(events.h2d_stop);
      if (q == cudaSuccess) {
        CUDA_CHECK(cudaEventElapsedTime(&gpu_timings_total.h2d_ids_ms,
                                        events.h2d_start, events.h2d_stop));
      } else if (q != cudaErrorNotReady) {
        CUDA_CHECK(q);
      }
      events.recorded_h2d = false;
    }
    if (timing_enabled) {
      cudaError_t q = cudaEventQuery(events.dt_done);
      if (q == cudaSuccess) {
        CUDA_CHECK(cudaEventElapsedTime(&stats_out.t_dt_ms, events.dt_start,
                                        events.dt_done));
      } else if (q != cudaErrorNotReady) {
        CUDA_CHECK(q);
      }
    } else {
      stats_out.t_dt_ms = 0.0f;
    }

    if (record_stats) {
      if (timing_enabled) {
        stats_out.t_h2d_ids_ms = gpu_timings_total.h2d_ids_ms;
        stats_out.t_dedup_ms = gpu_timings_total.dedup_ms;
        stats_out.t_pq_ms = gpu_timings_total.pq_ms;
        stats_out.t_sort_ms = static_cast<float>(cpu_topk_ms);
        stats_out.t_d2h_topn_ms = gpu_timings_total.d2h_topn_ms;
      } else {
        stats_out.t_h2d_ids_ms = 0.0f;
        stats_out.t_dedup_ms = 0.0f;
        stats_out.t_pq_ms = 0.0f;
        stats_out.t_sort_ms = 0.0f;
        stats_out.t_d2h_topn_ms = 0.0f;
      }
      stats_out.gpu_candidates_in = gpu_timings_total.candidates_in;
      stats_out.gpu_candidates_unique = gpu_timings_total.candidates_unique;
      stats_out.gpu_tmp_storage_bytes = gpu_timings_total.tmp_storage_bytes;
      stats_out.cand_unique = gpu_timings_total.candidates_unique;
    } else {
      stats_out.t_h2d_ids_ms = 0.0;
      stats_out.t_dedup_ms = 0.0;
      stats_out.t_pq_ms = 0.0;
      stats_out.t_sort_ms = 0.0;
      stats_out.t_d2h_topn_ms = 0.0;
      stats_out.gpu_candidates_in = initial_num_candidates;
      stats_out.gpu_candidates_unique = num_candidates;
      stats_out.gpu_tmp_storage_bytes = scorer.tmp_storage_bytes();
      stats_out.cand_unique = num_candidates;
    }

    auto io_before = io_manager.snapshot();
    auto t_rerank_begin = std::chrono::steady_clock::now();
    std::vector<long> final_results;
    {
      FUSIONANNS_NVTX_RANGE_COLOR(
          "HeuristicRerank",
          0xFFFF9800u); // SSD重排阶段 / SSD rerank stage
      final_results = heuristic_rerank(
          query_vector, approximate_results, io_manager, query_dim, cfg.top_k,
          cfg.rerank_size, cfg.batch_size, cfg.rerank_delta,
          cfg.rerank_stable_iters, record_stats ? &stats_out : nullptr,
          timing_enabled);
    }
    auto t_rerank_end = std::chrono::steady_clock::now();
    auto io_after = io_manager.snapshot();
    stats_out.t_rerank_ms =
        std::chrono::duration<double, std::milli>(t_rerank_end - t_rerank_begin)
            .count();
    stats_out.io_hits = io_after.hits - io_before.hits;
    stats_out.io_misses = io_after.misses - io_before.misses;
    stats_out.io_calls = io_after.io_count - io_before.io_count;
    stats_out.io_evictions = io_after.evictions - io_before.evictions;

    auto t_total_end = std::chrono::steady_clock::now();
    stats_out.t_total_ms =
        std::chrono::duration<double, std::milli>(t_total_end - t_total_begin)
            .count();

    if (timing_enabled) {
      auto dt_span_steady =
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double, std::milli>(stats_out.t_dt_ms));
      auto t_dt_end_estimated = t_dt_begin_host + dt_span_steady;

      double dt_begin_ms = std::chrono::duration<double, std::milli>(
                               t_dt_begin_host - t_total_begin)
                               .count();
      double dt_end_ms = std::chrono::duration<double, std::milli>(
                             t_dt_end_estimated - t_total_begin)
                             .count();

      double graph_begin_ms = std::chrono::duration<double, std::milli>(
                                  t_graph_begin - t_total_begin)
                                  .count();
      double graph_end_ms =
          std::chrono::duration<double, std::milli>(t_graph_end - t_total_begin)
              .count();
      double overlap_ms =
          std::max(0.0, std::min(dt_end_ms, graph_end_ms) -
                            std::max(dt_begin_ms, graph_begin_ms));
      double union_ms = (dt_end_ms - dt_begin_ms) +
                        (graph_end_ms - graph_begin_ms) - overlap_ms;
      stats_out.overlap_dt_graph_ms = overlap_ms;
      double ratio = union_ms > 0.0 ? (overlap_ms / union_ms) : 0.0;
      if (ratio < 0.0)
        ratio = 0.0;
      if (ratio > 1.0)
        ratio = 1.0;
      stats_out.overlap_ratio = ratio;
    } else {
      stats_out.overlap_dt_graph_ms = 0.0;
      stats_out.overlap_ratio = 0.0;
    }

    if (record_stats && has_groundtruth && thread_recalls) {
      double recall = compute_recall(final_results,
                                     gt_data.data() +
                                         static_cast<size_t>(task.query_index) *
                                             static_cast<size_t>(gt_dim),
                                     cfg.recall_k);
      thread_recalls->push_back(recall);
    } else if (record_stats && !has_groundtruth) {
      if (thread_recalls) {
        thread_recalls->push_back(0.0);
      }
    }

    stats_out.emit_to_csv = emit_to_csv;
    if (record_stats && thread_stats) {
      thread_stats->push_back(stats_out);
    }
  };

  const int total_sequences_min = measurement_queries;
  std::chrono::steady_clock::time_point run_start;

  auto make_worker = [&](auto &&get_task) {
    return [&](int thread_id) {
      if (sptag_nav) {
        sptag_nav->apply_thread_local_settings();
      }
      StreamContext &streams = stream_pool[thread_id];
      GpuBlock *block = thread_blocks[thread_id];
      if (!block) {
        throw std::runtime_error("Worker missing GPU block");
      }
      // 使用预分配的 pinned memory 和 events，避免 worker 线程内 CUDA API
      // 锁竞争
      ThreadPinnedBuffers &pinned = pinned_pool[thread_id];
      ThreadGpuEvents &events = events_pool[thread_id];
      // 重置 events 状态（复用同一批 events 对象）
      events.pending_copy_count = 0;
      events.recorded_h2d = false;
      events.recorded_dedup = false;
      events.recorded_pq = false;
      events.recorded_d2h = false;
      events.host_results_current = false;
      if (events.h_unique_count) {
        *events.h_unique_count = 0;
      }

      std::vector<QueryStats> local_stats;
      std::vector<double> local_recalls;
      Task task;
      while (get_task(task)) {
        bool record_stats = task.record;
        bool emit_to_csv = task.emit_to_csv;
        QueryStats stats;
        stats.qid = task.query_index;
        stats.thread_id = thread_id;
        stats.topk = cfg.top_k;
        stats.rerank_size = cfg.rerank_size;
        auto t_total_begin = std::chrono::steady_clock::now();
        stats.t_total_ms = 0.0;
        process_task(task, thread_id, pinned.h_query, pinned.d_query_mapped,
                     pinned.h_candidate, pinned.d_candidate_mapped, stats,
                     record_stats, emit_to_csv, streams.xfer, streams.comp,
                     block, events, record_stats ? &local_stats : nullptr,
                     record_stats ? &local_recalls : nullptr);
      }
      if (!local_stats.empty()) {
        std::lock_guard<std::mutex> lk(stats_mutex);
        result.stats.insert(result.stats.end(), local_stats.begin(),
                            local_stats.end());
        result.recalls.insert(result.recalls.end(), local_recalls.begin(),
                              local_recalls.end());
      }
    };
  };

  auto run_parallel_warmup = [&](int count) {
    if (count <= 0)
      return;
    std::atomic<int> next_seq{0};
    auto get_task = [&](Task &task) -> bool {
      int seq = next_seq.fetch_add(1, std::memory_order_acq_rel);
      if (seq >= count)
        return false;
      task.sequence = static_cast<size_t>(seq);
      task.query_index = seq % queries_to_run;
      task.record = false;
      task.emit_to_csv = false;
      return true;
    };
    auto worker_fn = make_worker(get_task);
    std::atomic<bool> warmup_error{false};
    std::exception_ptr warmup_eptr = nullptr;
    auto guarded = [&](int thread_id) {
      try {
        worker_fn(thread_id);
      } catch (...) {
        if (!warmup_error.exchange(true)) {
          warmup_eptr = std::current_exception();
        }
      }
    };
    worker_pool.run(thread_count, guarded);
    if (warmup_error.load()) {
      if (warmup_eptr) {
        std::rethrow_exception(warmup_eptr);
      }
      throw std::runtime_error("Unknown warmup worker failure");
    }
    for (int t = 0; t < thread_count; ++t) {
      CUDA_CHECK(cudaStreamSynchronize(stream_pool[t].xfer));
      CUDA_CHECK(cudaStreamSynchronize(stream_pool[t].comp));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  std::cout << "  > Running warmup with " << warmup_queries << " queries..."
            << std::endl;
  {
    FUSIONANNS_NVTX_RANGE("WarmupQueries");
    run_parallel_warmup(warmup_queries);
  }
  std::cout << "  > Warmup completed." << std::endl;

#ifdef FUSIONANNS_USE_CUVS_CAGRA
  // 输出 warmup 阶段的 CAGRA 延迟 breakdown
  if (cagra_nav && cagra_nav->profiling_enabled()) {
    const auto &warmup_profile = cagra_nav->get_profile_stats();
    std::cout << "\n" << warmup_profile.to_string() << std::endl;
    // reset 后继续收集 measurement 阶段数据
    cagra_nav->reset_profile_stats();
  }
#endif

  cudaProfilerStart();
  if (!cfg.open_loop) {
    struct ClosedLoopState {
      int base_queries = 0;
      int warmup = 0;
      int min_sequences = 0;
      std::atomic<int> next_seq;
      std::atomic<bool> stop_flag;
      bool time_gate = false;
      ClosedLoopState(int base, int warm, int min_seq, bool gate)
          : base_queries(base), warmup(warm), min_sequences(min_seq),
            next_seq(0), stop_flag(false), time_gate(gate) {}
    };
    ClosedLoopState state(queries_to_run, 0, total_sequences_min,
                          cfg.min_duration_s > 0);

    FUSIONANNS_NVTX_RANGE("ClosedLoopRun");

    run_start = std::chrono::steady_clock::now();
    std::thread timer_thread;
    if (state.time_gate) {
      timer_thread = std::thread([&state, duration = cfg.min_duration_s]() {
        std::this_thread::sleep_for(std::chrono::seconds(duration));
        state.stop_flag.store(true, std::memory_order_release);
      });
    }

    auto get_task = [&](Task &task) -> bool {
      while (true) {
        int seq = state.next_seq.fetch_add(1, std::memory_order_acq_rel);
        if (!state.time_gate) {
          if (seq >= state.min_sequences)
            return false;
        } else {
          if (seq >= state.min_sequences &&
              state.stop_flag.load(std::memory_order_acquire)) {
            return false;
          }
        }
        task.sequence = static_cast<size_t>(seq);
        task.query_index = seq % state.base_queries;
        task.record = true;
        task.emit_to_csv = should_emit_csv(task.sequence);
        return true;
      }
    };

    auto worker_fn = make_worker(get_task);
    auto guarded = [&](int thread_id) {
      try {
        worker_fn(thread_id);
      } catch (...) {
        if (!worker_error.exchange(true)) {
          worker_eptr = std::current_exception();
        }
      }
    };
    worker_pool.run(thread_count, guarded);
    if (timer_thread.joinable()) {
      timer_thread.join();
    }
  } else {
    if (cfg.open_loop_rps <= 0.0) {
      throw std::invalid_argument(
          "open_loop requires positive --open_loop_rps");
    }

    FUSIONANNS_NVTX_RANGE("OpenLoopRun");

    size_t target_sequences = static_cast<size_t>(measurement_queries);
    if (cfg.min_duration_s > 0) {
      size_t min_required = static_cast<size_t>(std::ceil(
          cfg.open_loop_rps * static_cast<double>(cfg.min_duration_s)));
      target_sequences = std::max(target_sequences, min_required);
    }

    std::deque<Task> queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> generator_done{false};

    run_start = std::chrono::steady_clock::now();

    std::thread producer([&]() {
      auto interval =
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(1.0 / cfg.open_loop_rps));
      auto next_release = std::chrono::steady_clock::now();
      for (size_t seq = 0; seq < target_sequences; ++seq) {
        Task task;
        task.sequence = seq;
        task.query_index =
            static_cast<int>(seq % static_cast<size_t>(queries_to_run));
        task.record = true;
        task.emit_to_csv = should_emit_csv(task.sequence);
        {
          std::lock_guard<std::mutex> lk(queue_mutex);
          queue.push_back(task);
        }
        queue_cv.notify_one();
        next_release += interval;
        std::this_thread::sleep_until(next_release);
      }
      generator_done.store(true, std::memory_order_release);
      queue_cv.notify_all();
    });

    auto get_task = [&](Task &task) -> bool {
      std::unique_lock<std::mutex> lk(queue_mutex);
      queue_cv.wait(lk, [&] {
        return !queue.empty() || generator_done.load(std::memory_order_acquire);
      });
      if (queue.empty())
        return false;
      task = queue.front();
      queue.pop_front();
      return true;
    };

    auto worker_fn = make_worker(get_task);
    auto guarded = [&](int thread_id) {
      try {
        worker_fn(thread_id);
      } catch (...) {
        if (!worker_error.exchange(true)) {
          worker_eptr = std::current_exception();
        }
      }
    };
    worker_pool.run(thread_count, guarded);
    producer.join();
  }

  if (worker_error.load()) {
    if (worker_eptr) {
      std::rethrow_exception(worker_eptr);
    }
    throw std::runtime_error("Unknown worker failure");
  }

  auto run_end = std::chrono::steady_clock::now();
  double wall_ms =
      std::chrono::duration<double, std::milli>(run_end - run_start).count();

  result.summary =
      summarize_run(result.stats, result.recalls, wall_ms, cfg, thread_count);

#ifdef FUSIONANNS_USE_CUVS_CAGRA
  // 输出 measurement 阶段的 CAGRA 延迟 breakdown
  if (cagra_nav && cagra_nav->profiling_enabled()) {
    const auto &meas_profile = cagra_nav->get_profile_stats();
    std::cout << "\n[Measurement Phase]\n" << meas_profile.to_string() << std::endl;
    cagra_nav->enable_profiling(false);
  }
  // 关停 batch collector
  if (cagra_collector) {
    cagra_collector->shutdown();
    cagra_collector.reset();
  }
#endif

  if (cfg.record_query_stats && !stats_csv_path.empty() &&
      !result.stats.empty()) {
    std::cout << "  > Flushing stats to CSV..." << std::endl;
    CsvLogger deferred_logger(stats_csv_path);
    for (const auto &stat : result.stats) {
      if (!stat.emit_to_csv)
        continue;
      deferred_logger.append_no_lock(stat);
    }
  }

  // Surface asynchronous kernel failures before returning measurements.  A
  // failed launch must never be reported as a successful low-recall run.
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaProfilerStop();

  return result;
}

RunOnceResult run_once(const ServerConfig &cfg, IOManager &io_manager,
                       PostingListAccessor &metadata,
                       ICentroidNavigator &centroid_nav,
                       const faiss::ProductQuantizer &pq, uint8_t *pq_codes_gpu,
                       float *pq_codebooks_gpu,
                       const std::vector<float> &query_data, int query_dim,
                       const std::vector<int> &gt_data, int gt_dim,
                       const std::string &stats_csv_path) {
  PostingHotspotScope hotspot_scope(metadata);
  double avg_posting = metadata.nlist() == 0
                           ? 0.0
                           : static_cast<double>(metadata.total_postings()) /
                                 static_cast<double>(metadata.nlist());
  int C_max = cfg.c_max;
  if (C_max <= 0) {
    double estimated = static_cast<double>(cfg.nprobe) * avg_posting * 1.5;
    C_max = static_cast<int>(std::ceil(std::max(estimated, 1024.0)));
    C_max = std::max(C_max, cfg.rerank_size * 2);
  }

  std::string scorer_name =
      cfg.scorer.empty() ? std::string("pq_lut") : to_lower_copy(cfg.scorer);
  RunOnceResult run_result;
  if (scorer_name == "int8_gemm") {
    auto corpus =
        acquire_int8_corpus(cfg.index_path + "#d" + std::to_string(query_dim),
                            io_manager, query_dim, cfg.threads);
    Int8GemmScorer scorer(corpus, C_max);
    run_result =
        run_once_impl(scorer, cfg, io_manager, metadata, centroid_nav,
                      query_data, query_dim, gt_data, gt_dim, stats_csv_path);
  } else {
    PqLutScorer scorer(pq, pq_codes_gpu, pq_codebooks_gpu, query_dim, C_max);
    run_result =
        run_once_impl(scorer, cfg, io_manager, metadata, centroid_nav,
                      query_data, query_dim, gt_data, gt_dim, stats_csv_path);
  }
  hotspot_scope.report();
  return run_result;
}

// ============================================================================
// RSQ8 IVF Scan Mode
// ============================================================================

RunOnceResult run_once_rsq8(const ServerConfig &cfg,
                            ICentroidNavigator &centroid_nav,
                            const std::string &rsq8_index_path,
                            const std::vector<float> &query_data, int query_dim,
                            const std::vector<int> &gt_data, int gt_dim,
                            const std::string &stats_csv_path) {
  FUSIONANNS_NVTX_RANGE("RunOnceRSQ8");

  const int available_queries =
      query_dim > 0 ? static_cast<int>(query_data.size() / query_dim) : 0;
  if (available_queries <= 0) {
    throw std::runtime_error("No queries available for RSQ8 scoring");
  }

  const int queries_to_run =
      cfg.queries_to_run > 0 ? std::min(cfg.queries_to_run, available_queries)
                             : available_queries;

  std::cout << "[RSQ8] Loading GPU index from " << rsq8_index_path << std::endl;

  // 加载 RSQ8 索引到 GPU
  std::unique_ptr<fusionann::RSQ8GpuIndex> rsq8_gpu_index;
  {
    FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.LoadIndex", 0xFFFF5722u);
    cudaStream_t load_stream;
    CUDA_CHECK(cudaStreamCreate(&load_stream));
    rsq8_gpu_index =
        fusionann::RSQ8GpuIndex::load(rsq8_index_path, load_stream);
    CUDA_CHECK(cudaStreamSynchronize(load_stream));
    CUDA_CHECK(cudaStreamDestroy(load_stream));
  }

  // 创建 Scorer
  std::shared_ptr<fusionann::RSQ8Scorer> scorer;
  {
    FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.CreateScorer", 0xFFFF9800u);
    scorer = std::make_shared<fusionann::RSQ8Scorer>(
        std::shared_ptr<fusionann::RSQ8GpuIndex>(
            rsq8_gpu_index.release(),
            [](fusionann::RSQ8GpuIndex *p) { delete p; }));
  }

  // 创建 ClusterScheduler
  fusionann::ClusterScheduler scheduler(query_dim, cfg.threads);

  // 获取 centroids 用于计算残差
  const int nlist = static_cast<int>(scorer->index()->nlist);
  std::vector<float> centroids(nlist * query_dim);
  {
    FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.CopyCentroids", 0xFFFFC107u);
    CUDA_CHECK(cudaMemcpy(centroids.data(), scorer->index()->d_centroids,
                          centroids.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // 计算最大聚类大小（用于分配 workspace）
  uint32_t max_cluster_size = 0;
  {
    FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.ComputeMaxClusterSize", 0xFF8BC34Au);
    for (uint32_t i = 0; i < scorer->index()->nlist; ++i) {
      max_cluster_size =
          std::max(max_cluster_size, scorer->index()->cluster_sizes[i]);
    }
  }

  std::cout << "[RSQ8] Index: " << nlist << " clusters, "
            << scorer->index()->total_vectors << " vectors, "
            << "max_cluster_size=" << max_cluster_size << std::endl;
  std::cout << "[RSQ8] Running " << queries_to_run
            << " queries with nprobe=" << cfg.nprobe << ", top_k=" << cfg.top_k
            << ", rsq8_batch_size=" << cfg.rsq8_batch_size << std::endl;

  // 结果收集
  RunOnceResult result;
  result.stats.resize(queries_to_run);
  result.recalls.resize(queries_to_run);
  // 多线程并行资源
  const int num_workers = std::max(1, cfg.threads);
  const int rsq8_batch = std::max(1, cfg.rsq8_batch_size);

  // 使用 GpuBlockPool 预分配 GPU 内存（零 cudaMalloc per batch）
  // max_queries = batch_size * nprobe (每个 query 会被调度到 nprobe 个聚类)
  // max_problems = batch_size * nprobe (最坏情况每个聚类都不同)
  const int workspace_max_queries = rsq8_batch * cfg.nprobe;
  const int workspace_max_problems = rsq8_batch * cfg.nprobe;

  fusionann::RSQ8WorkspacePlan rsq8_plan = fusionann::RSQ8WorkspacePlan::make(
      workspace_max_queries, static_cast<int>(max_cluster_size),
      workspace_max_problems, cfg.top_k, query_dim);

  std::cout << "[RSQ8] Workspace plan: gpu_bytes="
            << rsq8_plan.gpu_bytes_per_block
            << " pinned_bytes=" << rsq8_plan.pinned_bytes_total << std::endl;

  // 初始化 GpuBlockPool
  GpuBlockPool rsq8_block_pool;
  std::vector<GpuBlock *> thread_blocks(num_workers);
  std::vector<std::unique_ptr<fusionann::RSQ8PinnedBuffers>> pinned_buffers(
      num_workers);
  std::vector<cudaStream_t> streams(num_workers);

  {
    FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.AllocateWorkspaces", 0xFF4CAF50u);
    if (!rsq8_block_pool.init(rsq8_plan.gpu_bytes_per_block,
                              cfg.reserve_ratio)) {
      throw std::runtime_error("Failed to initialize RSQ8 GPU block pool");
    }
    for (int t = 0; t < num_workers; ++t) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[t], cudaStreamNonBlocking));
      thread_blocks[t] = rsq8_block_pool.acquire_block();
      if (!thread_blocks[t]) {
        throw std::runtime_error("Failed to acquire GpuBlock for RSQ8");
      }
      pinned_buffers[t] = fusionann::RSQ8PinnedBuffers::allocate(rsq8_plan);
    }
  }

  // RAII guard for releasing blocks
  struct BlockReleaseGuard {
    GpuBlockPool *pool;
    std::vector<GpuBlock *> *blocks;
    ~BlockReleaseGuard() {
      if (!pool || !blocks)
        return;
      for (auto *b : *blocks) {
        if (b)
          pool->release_block(b);
      }
    }
  } block_guard{&rsq8_block_pool, &thread_blocks};

  const int warmup_queries = std::min(std::max(cfg.warmup, 0), queries_to_run);

  auto run_batches = [&](int query_limit, bool collect_results) {
    const int total_batches = (query_limit + rsq8_batch - 1) / rsq8_batch;

#pragma omp parallel num_threads(num_workers)
    {
      const int tid = omp_get_thread_num();
      cudaStream_t my_stream = streams[tid];
      GpuBlock *my_block = thread_blocks[tid];
      fusionann::RSQ8PinnedBuffers &my_pinned = *pinned_buffers[tid];
      std::vector<int32_t> probe_ids;
      std::vector<float> probe_dists;

#pragma omp for schedule(dynamic)
      for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
        FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.ProcessBatch", 0xFF1E88E5u);

        const int batch_start = batch_idx * rsq8_batch;
        if (batch_start >= query_limit)
          continue;
        const int batch_end = std::min(batch_start + rsq8_batch, query_limit);
        const int batch_count = batch_end - batch_start;

        auto batch_begin = std::chrono::steady_clock::now();

        // Step 1: Coarse search - 批量找到每个查询的 nprobe 个最近聚类
        const float *batch_queries =
            query_data.data() + static_cast<size_t>(batch_start) * query_dim;
        {
          FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.CoarseSearch", 0xFFE91E63u);
          centroid_nav.search_batch_flat(batch_queries, batch_count, cfg.nprobe,
                                         probe_ids, probe_dists);
        }

        // Step 2: 准备 workspace view (为调度提供 packed residuals buffer)
        fusionann::RSQ8WorkspaceView my_view =
            fusionann::rsq8_prepare_workspace(*my_block, my_pinned, rsq8_plan);

        // Step 3: 调度 - 将 Query->Clusters 反转为 Cluster->Queries
        std::vector<fusionann::ClusterTask> tasks;
        {
          FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.Schedule", 0xFF3F51B5u);
          tasks = scheduler.schedule(
              batch_queries, batch_count, probe_ids.data(), cfg.nprobe,
              centroids.data(), nlist, my_view.h_residuals_staging);
        }

        // Step 4: 批量评分
        std::vector<std::vector<fusionann::RSQ8Result>> all_results;
        {
          FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.GPUScoring", 0xFF2196F3u);
          scorer->score_clusters_batch(tasks, my_view, all_results, my_stream);
        }

        // Step 5: 同步
        {
          FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.Sync", 0xFF00BCD4u);
          CUDA_CHECK(cudaStreamSynchronize(my_stream));
        }

        if (!collect_results) {
          continue;
        }

        // 收集结果 (从 pinned memory 读取)
        {
          FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.CollectResults", 0xFF009688u);
          for (size_t p = 0; p < tasks.size(); ++p) {
            const auto &task = tasks[p];
            const int M = task.num_queries;
            const int query_offset = task.query_offset;
            for (int i = 0; i < M; ++i) {
              int q_local = task.query_indices[i];
              if (q_local >= static_cast<int>(all_results.size())) {
                all_results.resize(q_local + 1);
              }
              size_t row_base =
                  static_cast<size_t>(query_offset + i) * my_view.topk;
              for (int k = 0; k < my_view.topk; ++k) {
                fusionann::RSQ8Result res;
                res.global_id = my_view.h_all_topk_indices_pinned[row_base + k];
                res.distance =
                    my_view.h_all_topk_distances_pinned[row_base + k];
                all_results[q_local].push_back(res);
              }
            }
          }
        }

        // Step 6: 对每个查询合并结果取 Top-K
        {
          FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.MergeTopK", 0xFF673AB7u);
          // 确保 all_results 覆盖所有查询
          if (static_cast<int>(all_results.size()) < batch_count) {
            all_results.resize(batch_count);
          }
          for (int i = 0; i < batch_count; ++i) {
            auto &results = all_results[i];
            // 按距离排序
            std::sort(results.begin(), results.end(),
                      [](const fusionann::RSQ8Result &a,
                         const fusionann::RSQ8Result &b) {
                        return a.distance < b.distance;
                      });
            // 保留 Top-K
            if (static_cast<int>(results.size()) > cfg.top_k) {
              results.resize(cfg.top_k);
            }

            // 计算 Recall
            int global_qid = batch_start + i;
            std::vector<long> final_ids;
            for (const auto &r : results) {
              final_ids.push_back(r.global_id);
            }

            // Ground truth
            const int *gt_ptr =
                (gt_dim > 0 && !gt_data.empty())
                    ? gt_data.data() + static_cast<size_t>(global_qid) * gt_dim
                    : nullptr;
            double recall = gt_ptr
                                ? compute_recall(final_ids, gt_ptr,
                                                 std::min(cfg.recall_k, gt_dim))
                                : 0.0;

            auto batch_end_time = std::chrono::steady_clock::now();
            double batch_ms = std::chrono::duration<double, std::milli>(
                                  batch_end_time - batch_begin)
                                  .count();

            // 写入结果
            result.recalls[global_qid] = recall;
            result.stats[global_qid].qid = global_qid;
            result.stats[global_qid].topk = cfg.top_k;
            result.stats[global_qid].t_total_ms = batch_ms / batch_count;
          }
        }
      }
    }
  };

  if (warmup_queries > 0) {
    run_batches(warmup_queries, false);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaProfilerStart();
  auto wall_start = std::chrono::steady_clock::now();
  run_batches(queries_to_run, true);
  auto wall_end = std::chrono::steady_clock::now();
  cudaProfilerStop();
  double wall_ms =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  // 清理资源 (GpuBlock 由 block_guard RAII 自动释放)
  {
    FUSIONANNS_NVTX_RANGE_COLOR("RSQ8.Cleanup", 0xFF607D8Bu);
    for (int t = 0; t < num_workers; ++t) {
      CUDA_CHECK(cudaStreamDestroy(streams[t]));
    }
    pinned_buffers.clear();
  }

  // 生成 Summary
  result.summary =
      summarize_run(result.stats, result.recalls, wall_ms, cfg, cfg.threads);

  std::cout << "[RSQ8] Completed " << queries_to_run << " queries in "
            << wall_ms << " ms"
            << " (QPS=" << (queries_to_run / wall_ms * 1000.0) << ")"
            << " Recall@" << cfg.recall_k << "=" << result.summary.avg_recall
            << std::endl;

  return result;
}

} // namespace fusionann::online
