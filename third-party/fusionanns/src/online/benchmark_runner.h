#pragma once

#include "index/centroid_navigator.h"
#include "online/gpu_kernels.cuh"
#include "online/gpu_pool.h"
#include "online/io_manager.h"
#include "posting_list_accessor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      throw std::runtime_error(cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)
#endif

namespace fusionann::online {

struct QueryStats {
  int qid = -1;
  int thread_id = -1;
  bool emit_to_csv = true;
  double t_total_ms = 0.0;
  double t_graph_ms = 0.0;
  double t_gather_ms = 0.0;
  double t_rerank_ms = 0.0;
  double t_rerank_io_ms = 0.0;
  double t_rerank_l2_ms = 0.0;
  float t_dt_ms = 0.0f;
  float t_h2d_ids_ms = 0.0f;
  float t_dedup_ms = 0.0f;
  float t_pq_ms = 0.0f;
  float t_sort_ms = 0.0f;
  float t_d2h_topn_ms = 0.0f;
  double overlap_dt_graph_ms = 0.0;
  double overlap_ratio = 0.0;
  uint64_t io_hits = 0;
  uint64_t io_misses = 0;
  uint64_t io_calls = 0;
  uint64_t io_evictions = 0;
  uint32_t pages_uniq_in_batch = 0;
  int cand_in = 0;
  int cand_unique = 0;
  int batches = 0;
  int topk = 0;
  int rerank_size = 0;
  int gpu_candidates_in = 0;
  int gpu_candidates_unique = 0;
  size_t gpu_tmp_storage_bytes = 0;
  int rerank_examined = 0;
  double gpu_wait_ms = 0.0;
  double mini_batch_pages = 0.0;
  double mini_batch_size = 0.0;
};

struct RunSummary {
  double wall_ms = 0.0;
  double qps_effective = 0.0;
  double qps_wall = 0.0;
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p99_ms = 0.0;
  double avg_graph_ms = 0.0;
  double avg_dt_ms = 0.0;
  double avg_gather_ms = 0.0;
  double avg_rerank_ms = 0.0;
  double avg_rerank_io_ms = 0.0;
  double avg_rerank_l2_ms = 0.0;
  double avg_overlap_ratio = 0.0;
  double avg_recall = 0.0;
  double recall_p50 = 0.0;
  double recall_p90 = 0.0;
  double recall_p99 = 0.0;
  double avg_gpu_wait_ms = 0.0;
  double avg_rerank_examined = 0.0;
  double avg_batch_pages = 0.0;
  double avg_batch_size = 0.0;
  double avg_inflight = 0.0;
  double miss_rate_avg = 0.0;
  double miss_rate_p50 = 0.0;
  double miss_rate_p90 = 0.0;
  double miss_rate_p99 = 0.0;
  uint64_t io_hits = 0;
  uint64_t io_misses = 0;
  uint64_t io_calls = 0;
  uint64_t io_evictions = 0;
  uint64_t total_queries = 0;
  int threads = 0;
  int nprobe = 0;
  int rerank_size = 0;
  int batch_size = 0;
  int top_k = 0;
  int recall_k = 0;
};

struct ServerConfig {
  int threads = 4;
  int top_k = 10;
  int nprobe = 10;
  int rerank_size = 400;
  int batch_size = 20;
  float rerank_delta = 0.03f;
  int rerank_stable_iters = 5;
  std::string io_backend = "io_uring";
  float reserve_ratio = 0.8f;
  int c_max = 0;
  int queries_to_run = 0;
  int cache_mb = 4096;
  int warmup = 10;
  int repeat = 1;
  int recall_k = 10;
  double target_recall = -1.0;
  std::string index_prefix = "sift";
  std::string query_path = "data/sift/sift_query.fvecs";
  std::string query_format = "fvecs";
  std::string groundtruth_path = "data/sift/sift_groundtruth.ivecs";
  std::string groundtruth_format = "ivecs";
  std::string sweep_threads;
  std::string sweep_nprobe;
  std::string sweep_rerank;
  std::string scorer = "pq_lut";
  int rsq8_batch_size = 32;
  bool open_loop = false;
  double open_loop_rps = 0.0;
  int min_duration_s = 0;
  std::string summary_csv = "indices/run_summary.csv";
  bool record_query_stats = true;
  std::string index_path = "indices";
  int stats_sample_rate = 1;
#ifdef FUSIONANNS_USE_CUVS_CAGRA
  bool use_cagra = true;      // cuVS 已链接：默认启用 CAGRA 质心图搜索
#else
  bool use_cagra = false;     // cuVS 未链接：默认关闭 CAGRA
#endif
#ifdef FUSIONANNS_USE_CUVS_SELECT_K
  bool use_gpu_topk = true;   // cuVS 已链接：默认启用 GPU select_k TopK
#else
  bool use_gpu_topk = false;  // cuVS 未链接：默认关闭 GPU TopK
#endif
};

struct RunOnceResult {
  RunSummary summary;
  std::vector<QueryStats> stats;
  std::vector<double> recalls;
};

class CsvLogger {
public:
  explicit CsvLogger(const std::string &path);
  void append(const QueryStats &s);
  void append_no_lock(const QueryStats &s);

private:
  void write_row(const QueryStats &s);
  std::mutex mutex_;
  std::ofstream ofs_;
};

RunSummary summarize_run(const std::vector<QueryStats> &stats,
                         const std::vector<double> &recalls, double wall_ms,
                         const ServerConfig &cfg, int threads_used);

void append_summary_csv(const std::string &path, const RunSummary &summary);

std::vector<int> parse_int_list(const std::string &spec);

RunOnceResult run_once(const ServerConfig &cfg, IOManager &io_manager,
                       PostingListAccessor &metadata,
                       ICentroidNavigator &centroid_nav,
                       const faiss::ProductQuantizer &pq, uint8_t *pq_codes_gpu,
                       float *pq_codebooks_gpu,
                       const std::vector<float> &query_data, int query_dim,
                       const std::vector<int> &gt_data, int gt_dim,
                       const std::string &stats_csv_path);

/// RSQ8 专用运行函数 (IVF 扫描模式)
/// @param rsq8_index_path RSQ8 索引文件路径 (residual_sq8_index.bin)
RunOnceResult run_once_rsq8(const ServerConfig &cfg,
                            ICentroidNavigator &centroid_nav,
                            const std::string &rsq8_index_path,
                            const std::vector<float> &query_data, int query_dim,
                            const std::vector<int> &gt_data, int gt_dim,
                            const std::string &stats_csv_path);

void load_ivecs(const std::string &filename, std::vector<int> &data, long &num,
                int &dim);

double compute_recall(const std::vector<long> &results, const int *ground_truth,
                      int k);

std::vector<long>
heuristic_rerank(const float *query_vector,
                 const std::vector<ResultPair> &approximate_results,
                 IOManager &io_manager, int dim, int top_k, int rerank_size,
                 int batch_size, float stability_delta, int stability_iters,
                 QueryStats *stats = nullptr, bool enable_detail = false);

} // namespace fusionann::online
