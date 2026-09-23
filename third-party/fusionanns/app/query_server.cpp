#include "common/datasource.h"
#include "common/io.h"
#include "common/nvtx_utils.h"
#include "common/types.h"

#include "index/centroid_navigator.h"
#include "index/centroid_navigator_sptag.h"
#ifdef FUSIONANNS_USE_CUVS_CAGRA
#include "index/centroid_navigator_cagra.h"
#endif
#include "online/benchmark_runner.h"
#include "online/io_manager.h"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>
// #include <gperftools/profiler.h>

using ::IOManager;
using fusionann::online::append_summary_csv;
using fusionann::online::load_ivecs;
using fusionann::online::parse_int_list;
using fusionann::online::PostingListAccessor;
using fusionann::online::run_once;
using fusionann::online::run_once_rsq8;
using fusionann::online::RunOnceResult;
using fusionann::online::RunSummary;
using fusionann::online::ServerConfig;

namespace fs = std::filesystem;

namespace {

std::string to_lower_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

BaseFormat parse_base_format(const std::string &value) {
  std::string fmt = to_lower_copy(value);
  if (fmt == "fvecs") {
    return BaseFormat::FVECs;
  }
  if (fmt == "u8bin" || fmt == "bin") {
    return BaseFormat::U8BIN;
  }
  if (fmt == "bvecs") {
    return BaseFormat::BVECS;
  }
  if (fmt == "fbin") {
    return BaseFormat::FBIN;
  }
  throw std::runtime_error("invalid dataset format: " + value);
}

enum class GroundTruthFormat { Ivecs, Bin };

GroundTruthFormat parse_groundtruth_format(const std::string &value) {
  std::string fmt = to_lower_copy(value);
  if (fmt == "ivecs") {
    return GroundTruthFormat::Ivecs;
  }
  if (fmt == "bin") {
    return GroundTruthFormat::Bin;
  }
  throw std::runtime_error("invalid ground truth format: " + value);
}

std::string index_component_path(const std::string &base_dir,
                                 const std::string &prefix,
                                 const std::string &suffix) {
  return (fs::path(base_dir) / (prefix + suffix)).string();
}

std::string resolve_index_component(const std::string &label,
                                    std::initializer_list<std::string> paths) {
  for (const auto &path : paths) {
    if (!path.empty() && fs::exists(path)) {
      return path;
    }
  }
  std::ostringstream oss;
  oss << "Missing index component for " << label << ". Checked:";
  for (const auto &path : paths) {
    oss << ' ' << path;
  }
  throw std::runtime_error(oss.str());
}

void load_groundtruth_bin(const std::string &path, std::vector<int> &data,
                          long &num, int &dim) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Cannot open ground truth file: " + path);
  }
  uint32_t n = 0;
  uint32_t k = 0;
  f.read(reinterpret_cast<char *>(&n), sizeof(uint32_t));
  f.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));
  if (!f) {
    throw std::runtime_error("Failed to read ground truth header: " + path);
  }
  if (n == 0 || k == 0) {
    throw std::runtime_error("Ground truth file reports empty data: " + path);
  }
  std::vector<uint32_t> raw(static_cast<size_t>(n) * static_cast<size_t>(k));
  f.read(reinterpret_cast<char *>(raw.data()),
         static_cast<std::streamsize>(raw.size() * sizeof(uint32_t)));
  if (f.gcount() !=
      static_cast<std::streamsize>(raw.size() * sizeof(uint32_t))) {
    throw std::runtime_error("Ground truth payload truncated: " + path);
  }
  data.resize(raw.size());
  std::transform(raw.begin(), raw.end(), data.begin(),
                 [](uint32_t v) { return static_cast<int>(v); });
  num = static_cast<long>(n);
  dim = static_cast<int>(k);
}

ServerConfig parse_args(int argc, char **argv) {
  ServerConfig cfg;
  CLI::App app{"FusionANNS Query Server"};
  app.add_option("--threads", cfg.threads,
                 "Number of worker threads for rerank stage");
  app.add_option("--topk", cfg.top_k, "Number of results to return");
  app.add_option("--nprobe", cfg.nprobe,
                 "Coarse clusters to probe during graph search");
  app.add_option("--rerank_size", cfg.rerank_size,
                 "Candidate count passed to SSD reranker");
  app.add_option("--batch_size", cfg.batch_size,
                 "Mini-batch size when fetching raw vectors from SSD");
  app.add_option("--rerank_delta", cfg.rerank_delta,
                 "Stability delta threshold for heuristic rerank");
  app.add_option("--rerank_stable_iters", cfg.rerank_stable_iters,
                 "Consecutive stable mini-batches before stopping");
  app.add_option("--io_backend", cfg.io_backend,
                 "IO backend (pread|mmap|io_uring)");
  app.add_option("--reserve_ratio", cfg.reserve_ratio,
                 "GPU block pool reserve ratio (0, 0.95]");
  app.add_option("--cmax", cfg.c_max,
                 "Override candidate buffer size (C_max); 0 auto tunes");
  app.add_option("--queries", cfg.queries_to_run,
                 "Limit number of queries to execute (0 = all)");
  app.add_option("--cache_mb", cfg.cache_mb,
                 "Approximate DRAM cache budget for SSD pages (MB)");
  app.add_option("--warmup", cfg.warmup,
                 "Warmup queries before measurement (per run)");
  app.add_option("--repeat", cfg.repeat,
                 "Repeat the query set multiple times (>=1)");
  app.add_option("--recall_k", cfg.recall_k, "Recall@K metric");
  app.add_option("--target_recall", cfg.target_recall,
                 "Target recall threshold for auto tuning (<=0 disabled)");
  app.add_option("--sweep_threads", cfg.sweep_threads,
                 "Comma separated thread counts to benchmark");
  app.add_option("--sweep_nprobe", cfg.sweep_nprobe,
                 "Comma separated nprobe values to benchmark");
  app.add_option("--sweep_rerank", cfg.sweep_rerank,
                 "Comma separated rerank sizes to benchmark");
  app.add_flag("--open_loop", cfg.open_loop,
               "Drive workload in open-loop mode");
  app.add_option("--open_loop_rps", cfg.open_loop_rps,
                 "Target requests per second for open-loop mode");
  app.add_option("--min_duration_s", cfg.min_duration_s,
                 "Minimum duration for a benchmark run (seconds)");
  app.add_option("--summary_csv", cfg.summary_csv,
                 "Aggregate summary CSV output path");
  app.add_option("--index-prefix", cfg.index_prefix,
                 "Index file prefix under indices/ (default: sift)");
  app.add_option("--query", cfg.query_path,
                 "Path to the query dataset "
                 "(default: data/sift/sift_query.fvecs)");
  app.add_option("--query-format", cfg.query_format,
                 "Format of the query dataset (fvecs|u8bin|bvecs|fbin)");
  app.add_option("--groundtruth", cfg.groundtruth_path,
                 "Path to ground truth dataset "
                 "(default: data/sift/sift_groundtruth.ivecs)");
  app.add_option("--groundtruth-format", cfg.groundtruth_format,
                 "Format of ground truth dataset (ivecs|bin)");
  app.add_option("--index-path", cfg.index_path,
                 "Path to the index directory (default: indices)");
  app.add_option("--scorer", cfg.scorer,
                 "Coarse scorer backend (pq_lut|int8_gemm|rsq8)");
  app.add_option("--rsq8_batch_size", cfg.rsq8_batch_size,
                 "Query batch size for RSQ8 scoring path");
  app.add_flag("--use-cagra,!--no-use-cagra", cfg.use_cagra,
               "Use CAGRA for centroid graph search when index exists");
  app.add_flag("--gpu-topk,!--no-gpu-topk", cfg.use_gpu_topk,
               "Use GPU select_k for TopK in PQ scoring");

  bool disable_query_stats = false;
  app.add_flag("--disable-query-stats", disable_query_stats,
               "Disable per-query stats CSV output");
  app.add_option("--stats_sample_rate", cfg.stats_sample_rate,
                 "Record one out of N queries when emitting per-query stats")
      ->check(CLI::PositiveNumber);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e));
  }

  if (cfg.threads <= 0)
    cfg.threads = 1;
  if (cfg.rerank_size <= 0)
    cfg.rerank_size = 1;
  if (cfg.batch_size <= 0)
    cfg.batch_size = 1;
  if (!std::isfinite(cfg.rerank_delta) || cfg.rerank_delta <= 0.0f)
    cfg.rerank_delta = 0.03f;
  if (cfg.rerank_stable_iters <= 0)
    cfg.rerank_stable_iters = 5;
  if (cfg.cache_mb < 0)
    cfg.cache_mb = 0;
  if (cfg.warmup < 0)
    cfg.warmup = 0;
  if (cfg.repeat <= 0)
    cfg.repeat = 1;
  if (cfg.recall_k <= 0)
    cfg.recall_k = cfg.top_k;
  if (cfg.min_duration_s < 0)
    cfg.min_duration_s = 0;
  if (cfg.open_loop_rps < 0.0)
    cfg.open_loop_rps = 0.0;
  std::transform(
      cfg.io_backend.begin(), cfg.io_backend.end(), cfg.io_backend.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  cfg.query_format = to_lower_copy(cfg.query_format);
  cfg.groundtruth_format = to_lower_copy(cfg.groundtruth_format);
  if (cfg.reserve_ratio <= 0.0f || cfg.reserve_ratio > 0.95f)
    cfg.reserve_ratio = 0.8f;
  cfg.scorer = to_lower_copy(cfg.scorer);
  cfg.record_query_stats = !disable_query_stats;
  cfg.stats_sample_rate = std::max(1, cfg.stats_sample_rate);
  if (cfg.rsq8_batch_size <= 0)
    cfg.rsq8_batch_size = 32;
  return cfg;
}

} // namespace

int main(int argc, char **argv) {
  uint8_t *pq_codes_gpu = nullptr;
  float *pq_codebooks_gpu = nullptr;
  try {
    ServerConfig config = parse_args(argc, argv);

    std::cout << "--- FusionANNS Online Query Server ---" << std::endl;
#ifdef FUSIONANNS_USE_CUVS_SELECT_K
    std::cout << "  [cuVS] GPU TopK: "
              << (config.use_gpu_topk ? "enabled (select_k)" : "disabled (CPU)")
              << std::endl;
#else
    if (config.use_gpu_topk) {
      std::cerr << "  [WARN] --gpu-topk requested but libcuvs is not linked. "
                   "Auto-disabling GPU TopK (falling back to CPU)." << std::endl;
      config.use_gpu_topk = false;
    }
    std::cout << "  [cuVS] GPU TopK: disabled (libcuvs not linked)" << std::endl;
#endif
#ifdef FUSIONANNS_USE_CUVS_CAGRA
    std::cout << "  [cuVS] CAGRA centroid graph: "
              << (config.use_cagra ? "enabled when available" : "disabled")
              << std::endl;
#else
    if (config.use_cagra) {
      std::cerr << "  [WARN] --use-cagra requested but libcuvs is not linked. "
                   "Auto-disabling CAGRA (falling back to SPTAG)." << std::endl;
      config.use_cagra = false;
    }
    std::cout << "  [cuVS] CAGRA centroid graph: disabled (libcuvs not linked)"
              << std::endl;
#endif

    std::cout << "[1/4] Loading index components..." << std::endl;
    IOManager::BackendKind backend_kind = IOManager::BackendKind::Pread;
    if (config.io_backend == "mmap") {
      backend_kind = IOManager::BackendKind::MMap;
    } else if (config.io_backend == "pread" || config.io_backend == "cached") {
      backend_kind = IOManager::BackendKind::Pread;
    } else if (config.io_backend == "io_uring" ||
               config.io_backend == "iouring") {
      backend_kind = IOManager::BackendKind::IoUring;
    } else {
      throw std::invalid_argument("Unknown IO backend: " + config.io_backend);
    }
    size_t cache_pages = 0;
    if (config.cache_mb > 0) {
      uint64_t cache_bytes =
          static_cast<uint64_t>(config.cache_mb) * 1024ULL * 1024ULL;
      cache_pages = static_cast<size_t>(cache_bytes / SSD_PAGE_SIZE);
      if (cache_pages == 0) {
        cache_pages = 1;
      }
    }

    std::string pq_codec_path;
    std::string pq_codes_path;
    std::string metadata_path;
    std::unique_ptr<faiss::ProductQuantizer> pq_holder;
    std::unique_ptr<IOManager> io_manager;
    std::unique_ptr<PostingListAccessor> metadata;
    std::unique_ptr<ICentroidNavigator> centroid_nav;
    int dim = 0;

    // 判断是否为 RSQ8 模式
    const bool is_rsq8_mode = (config.scorer == "rsq8");

    {
      FUSIONANNS_NVTX_RANGE("LoadIndexComponents");

      // 加载 centroid navigator (所有模式都需要)
#ifdef FUSIONANNS_USE_CUVS_CAGRA
      fs::path cagra_path =
          fs::path(config.index_path) / "centroid_graph_cagra.bin";
      if (config.use_cagra && fs::exists(cagra_path)) {
        centroid_nav = std::make_unique<CAGRACentroidNavigator>();
        centroid_nav->load(cagra_path.string());
        std::cout << "  > Centroid navigator loaded from " << cagra_path
                  << " (CAGRA GPU)" << std::endl;
      } else if (config.use_cagra && !fs::exists(cagra_path)) {
        std::cerr << "  [WARN] --use-cagra requested but " << cagra_path
                  << " not found; falling back to SPTAG. Rebuild index with "
                     "libcuvs to generate CAGRA centroid graph."
                  << std::endl;
      }
      if (!centroid_nav)
#endif
      {
        centroid_nav = std::make_unique<SPTAGCentroidNavigator>();
        fs::path nav_dir =
            fs::path(config.index_path) / "proximity_graph_index";
        if (!fs::exists(nav_dir)) {
          nav_dir = fs::path(config.index_path) / "centroid_graph_sptag";
        }
        centroid_nav->load(nav_dir.string());
        if (auto *sptag_nav =
                dynamic_cast<SPTAGCentroidNavigator *>(centroid_nav.get())) {
          sptag_nav->tune_for_nprobe(config.nprobe);
        }
        std::cout << "  > Centroid navigator loaded from " << nav_dir
                  << std::endl;
      }

      if (is_rsq8_mode) {
        // RSQ8 模式: 从 RSQ8 索引读取维度
        std::string rsq8_path =
            (fs::path(config.index_path) / "residual_sq8_index.bin").string();
        if (!fs::exists(rsq8_path)) {
          throw std::runtime_error("RSQ8 index not found: " + rsq8_path);
        }
        // 读取 RSQ8IndexHeader 获取 dim 和 nlist
        // Header layout: magic(4) | version(4) | nlist(4) | dim(4) | ...
        std::ifstream ifs(rsq8_path, std::ios::binary);
        uint32_t magic, version, header_nlist, header_dim;
        ifs.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));
        ifs.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
        ifs.read(reinterpret_cast<char *>(&header_nlist), sizeof(uint32_t));
        ifs.read(reinterpret_cast<char *>(&header_dim), sizeof(uint32_t));

        if (magic != 0x52535138) { // "RSQ8"
          throw std::runtime_error("Invalid RSQ8 index magic: " + rsq8_path);
        }
        dim = static_cast<int>(header_dim);
        std::cout << "  > RSQ8 index found: " << rsq8_path << " (dim=" << dim
                  << ", nlist=" << header_nlist << ", version=" << version
                  << ")" << std::endl;
      } else {
        // PQ 模式: 加载 PQ codec 和相关文件
        pq_codec_path = resolve_index_component(
            "PQ codec",
            {(fs::path(config.index_path) / "pq_codec").string(),
             index_component_path(config.index_path, config.index_prefix,
                                  ".pq_codec")});
        pq_codes_path = resolve_index_component(
            "PQ codes",
            {(fs::path(config.index_path) / "pq_codes").string(),
             index_component_path(config.index_path, config.index_prefix,
                                  ".pq_codes")});
        metadata_path = resolve_index_component(
            "posting list metadata",
            {(fs::path(config.index_path) / "posting_lists_metadata").string(),
             index_component_path(config.index_path, config.index_prefix,
                                  ".metadata")});

        pq_holder.reset(faiss::read_ProductQuantizer(pq_codec_path.c_str()));
        const faiss::ProductQuantizer &pq_ref = *pq_holder;
        dim = static_cast<int>(pq_ref.dsub * pq_ref.M);
        if (dim <= 0) {
          throw std::runtime_error("Invalid dimension inferred from PQ codec");
        }

        io_manager = std::make_unique<IOManager>(
            (fs::path(config.index_path) / "vector_location.map").string(),
            (fs::path(config.index_path) / "packed_raw_vectors.bin").string(),
            dim, backend_kind, cache_pages);
        metadata = std::make_unique<PostingListAccessor>(metadata_path);

        std::cout << "  > PQ codec: " << pq_codec_path << " (d=" << dim << ')'
                  << std::endl;
        std::cout << "  > PQ codes: " << pq_codes_path << std::endl;
        std::cout << "  > Posting lists: " << metadata->nlist()
                  << " lists covering " << metadata->total_postings()
                  << " postings (" << metadata_path << ')' << std::endl;
      }

      std::cout << "  > All index components loaded." << std::endl;
    }

    std::cout << "\n[2/4] Initializing GPU..." << std::endl;
    std::vector<uint8_t> pq_codes_cpu;

    if (!is_rsq8_mode) {
      // PQ 模式: 加载 PQ 到 GPU
      const faiss::ProductQuantizer &pq = *pq_holder;
      IOManager &io_manager_ref = *io_manager;
      PostingListAccessor &metadata_ref = *metadata;
      (void)io_manager_ref;
      (void)metadata_ref;

      FUSIONANNS_NVTX_RANGE("InitializeGPU");
      load_binary<uint8_t>(pq_codes_path, pq_codes_cpu);

      size_t codes_bytes = pq_codes_cpu.size() * sizeof(uint8_t);
      size_t free_b = 0, total_b = 0;
      auto st = cudaMemGetInfo(&free_b, &total_b);
      if (st != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(st)
                  << '\n';
      }
      std::cout << "  > PQ codes bytes: "
                << (codes_bytes / (1024.0 * 1024 * 1024)) << " GiB\n"
                << "  > GPU free/total: " << (free_b / (1024.0 * 1024 * 1024))
                << '/' << (total_b / (1024.0 * 1024 * 1024)) << " GiB\n";

      CUDA_CHECK(
          cudaMalloc(&pq_codes_gpu, pq_codes_cpu.size() * sizeof(uint8_t)));
      CUDA_CHECK(cudaMemcpy(pq_codes_gpu, pq_codes_cpu.data(),
                            pq_codes_cpu.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));
      std::cout << "  > Copied PQ codes to GPU." << std::endl;

      upload_pq_codebooks_to_gpu(pq, &pq_codebooks_gpu, static_cast<int>(pq.M),
                                 static_cast<int>(pq.ksub),
                                 static_cast<int>(pq.dsub));
    } else {
      std::cout << "  > RSQ8 mode: GPU index will be loaded on demand."
                << std::endl;
    }

    std::cout << "\n[3/4] Loading query data and ground truth..." << std::endl;
    std::vector<float> query_data;
    long query_num = 0;
    int query_dim = 0;
    std::vector<int> gt_data;
    long gt_num = 0;
    int gt_dim = 0;
    {
      FUSIONANNS_NVTX_RANGE("LoadQueriesGroundTruth");
      BaseFormat query_fmt = parse_base_format(config.query_format);
      auto query_source = make_source(query_fmt, config.query_path, 0, dim);
      if (query_source->dim() != dim) {
        throw std::runtime_error("Query dimension mismatch: expected " +
                                 std::to_string(dim) + ", file has " +
                                 std::to_string(query_source->dim()));
      }
      size_t query_total = query_source->size();
      if (query_total == 0) {
        throw std::runtime_error("No query vectors available in " +
                                 config.query_path);
      }
      query_data.resize(query_total * static_cast<size_t>(dim));
      const size_t block =
          std::max<size_t>(1, std::min<size_t>(query_total, 1024));
      for (size_t offset = 0; offset < query_total; offset += block) {
        size_t take = std::min(block, query_total - offset);
        query_source->read_block(offset, take,
                                 query_data.data() +
                                     offset * static_cast<size_t>(dim));
      }
      query_num = static_cast<long>(query_total);
      query_dim = dim;
      GroundTruthFormat gt_fmt =
          parse_groundtruth_format(config.groundtruth_format);
      if (gt_fmt == GroundTruthFormat::Ivecs) {
        load_ivecs(config.groundtruth_path, gt_data, gt_num, gt_dim);
      } else {
        load_groundtruth_bin(config.groundtruth_path, gt_data, gt_num, gt_dim);
      }
    }
    std::cout << "  > Loaded " << query_num << " queries (d=" << query_dim
              << ") and ground truth (" << gt_dim << " per query)."
              << std::endl;

    auto dedup_values = [](std::vector<int> &vals) {
      vals.erase(std::remove_if(vals.begin(), vals.end(),
                                [](int v) { return v <= 0; }),
                 vals.end());
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    };

    std::vector<int> thread_candidates = parse_int_list(config.sweep_threads);
    if (thread_candidates.empty()) {
      thread_candidates.push_back(config.threads);
    }
    dedup_values(thread_candidates);
    if (thread_candidates.empty()) {
      thread_candidates.push_back(std::max(1, config.threads));
    }

    std::vector<int> nprobe_candidates = parse_int_list(config.sweep_nprobe);
    bool auto_nprobe = false;
    if (nprobe_candidates.empty()) {
      nprobe_candidates.push_back(std::max(1, config.nprobe));
      auto_nprobe = config.target_recall > 0.0;
    }
    dedup_values(nprobe_candidates);
    if (nprobe_candidates.empty()) {
      nprobe_candidates.push_back(1);
    }

    std::vector<int> rerank_candidates = parse_int_list(config.sweep_rerank);
    bool auto_rerank = false;
    if (rerank_candidates.empty()) {
      rerank_candidates.push_back(std::max(1, config.rerank_size));
      auto_rerank = config.target_recall > 0.0;
    }
    dedup_values(rerank_candidates);
    if (rerank_candidates.empty()) {
      rerank_candidates.push_back(1);
    }

    const std::string stats_csv_path =
        config.record_query_stats
            ? (fs::path(config.index_path) / "query_stats.csv").string()
            : "";
    if (config.record_query_stats) {
      std::ofstream reset(stats_csv_path, std::ios::trunc);
    }

    const ServerConfig base_cfg = config;
    RunSummary best_summary{};
    ServerConfig best_cfg_snapshot = config;
    bool have_best = false;

    std::cout << "\n[4/4] Performing search with re-ranking..." << std::endl;

    constexpr int kMaxNprobeAuto = 4096;
    constexpr int kMaxRerankAuto = 131072;

    {
      FUSIONANNS_NVTX_RANGE("BenchmarkSweep");
      size_t np_idx = 0;
      while (np_idx < nprobe_candidates.size()) {
        size_t rr_idx = 0;
        while (rr_idx < rerank_candidates.size()) {
          bool recall_met_combo = base_cfg.target_recall <= 0.0;
          for (int threads_val : thread_candidates) {
            ServerConfig sweep_cfg = base_cfg;
            sweep_cfg.nprobe = nprobe_candidates[np_idx];
            sweep_cfg.rerank_size = rerank_candidates[rr_idx];
            sweep_cfg.threads = threads_val;

            std::cout << "\n-- Running benchmark: threads=" << sweep_cfg.threads
                      << " nprobe=" << sweep_cfg.nprobe
                      << " rerank_size=" << sweep_cfg.rerank_size << " --"
                      << std::endl;

            RunOnceResult run_result;
            int prof = 0;
            {
              FUSIONANNS_NVTX_RANGE_COLOR(
                  "RunOnce",
                  0xFF4CAF50u); // 绿色强调运行 / highlight run in green
              // if (!prof) {
              //   ProfilerStart("./run_once.prof");
              // }
              if (sweep_cfg.scorer == "rsq8") {
                std::string rsq8_path =
                    (fs::path(sweep_cfg.index_path) / "residual_sq8_index.bin")
                        .string();
                run_result = run_once_rsq8(sweep_cfg, *centroid_nav, rsq8_path,
                                           query_data, query_dim, gt_data,
                                           gt_dim, stats_csv_path);
              } else {
                run_result = run_once(sweep_cfg, *io_manager, *metadata,
                                      *centroid_nav, *pq_holder, pq_codes_gpu,
                                      pq_codebooks_gpu, query_data, query_dim,
                                      gt_data, gt_dim, stats_csv_path);
              }
              // if (prof < 1000) {
              //   prof++;
              // } else {
              //   ProfilerStop();
              // }
            }
            const RunSummary &summary = run_result.summary;
            append_summary_csv(base_cfg.summary_csv, summary);

            bool meets_target = base_cfg.target_recall <= 0.0 ||
                                summary.avg_recall >= base_cfg.target_recall;
            if (meets_target) {
              recall_met_combo = true;
            }

            if (!have_best ||
                summary.qps_effective > best_summary.qps_effective) {
              best_summary = summary;
              best_cfg_snapshot = sweep_cfg;
              have_best = true;
            }

            std::cout << "  threads=" << sweep_cfg.threads
                      << " nprobe=" << sweep_cfg.nprobe
                      << " rerank=" << sweep_cfg.rerank_size
                      << " -> qps_eff=" << std::fixed << std::setprecision(2)
                      << summary.qps_effective
                      << ", qps_wall=" << summary.qps_wall
                      << ", avg=" << summary.avg_ms
                      << "ms, p50=" << summary.p50_ms
                      << "ms, p90=" << summary.p90_ms
                      << "ms, p99=" << summary.p99_ms << "ms, inflight≈"
                      << std::setprecision(2) << summary.avg_inflight
                      << ", miss_rate(avg/p90)=" << std::setprecision(3)
                      << summary.miss_rate_avg << '/' << summary.miss_rate_p90
                      << ", Recall@" << sweep_cfg.recall_k << '='
                      << std::defaultfloat << std::setprecision(6)
                      << summary.avg_recall << std::endl;
          }

          if (base_cfg.target_recall > 0.0 && !recall_met_combo) {
            bool rerank_extended = false;
            if (auto_rerank && rr_idx == rerank_candidates.size() - 1) {
              int current = rerank_candidates.back();
              if (current < kMaxRerankAuto) {
                int next = std::min(current * 2, kMaxRerankAuto);
                if (next != current) {
                  std::cout << "  > Increasing rerank_size to " << next
                            << " to chase target recall" << std::endl;
                  rerank_candidates.push_back(next);
                  rerank_extended = true;
                  rr_idx = rerank_candidates.size() - 1;
                  recall_met_combo = base_cfg.target_recall <= 0.0;
                  continue;
                }
              }
            }
            if (!rerank_extended && auto_nprobe &&
                np_idx == nprobe_candidates.size() - 1) {
              int current = nprobe_candidates.back();
              if (current < kMaxNprobeAuto) {
                int next = std::min(current * 2, kMaxNprobeAuto);
                if (next != current) {
                  std::cout << "  > Increasing nprobe to " << next
                            << " to chase target recall" << std::endl;
                  nprobe_candidates.push_back(next);
                  break;
                }
              }
            }
            if (!rerank_extended) {
              std::cout << "  ! Target recall " << base_cfg.target_recall
                        << " not met for nprobe=" << nprobe_candidates[np_idx]
                        << " rerank=" << rerank_candidates[rr_idx] << std::endl;
            }
          }

          ++rr_idx;
        }
        ++np_idx;
      }
    }

    if (have_best) {
      std::cout << "\nBest qps_eff configuration: threads="
                << best_cfg_snapshot.threads
                << " nprobe=" << best_cfg_snapshot.nprobe
                << " rerank=" << best_cfg_snapshot.rerank_size
                << " => qps_eff=" << std::fixed << std::setprecision(2)
                << best_summary.qps_effective
                << ", qps_wall=" << best_summary.qps_wall
                << ", avg=" << best_summary.avg_ms << "ms, inflight≈"
                << std::setprecision(2) << best_summary.avg_inflight
                << ", miss_rate(avg/p90)=" << std::setprecision(3)
                << best_summary.miss_rate_avg << '/'
                << best_summary.miss_rate_p90 << ", Recall@"
                << best_cfg_snapshot.recall_k << '=' << std::defaultfloat
                << std::setprecision(6) << best_summary.avg_recall << std::endl;
    }

    cudaFree(pq_codebooks_gpu);
    pq_codebooks_gpu = nullptr;
    cudaFree(pq_codes_gpu);
    pq_codes_gpu = nullptr;
  } catch (const std::exception &ex) {
    if (pq_codebooks_gpu) {
      cudaFree(pq_codebooks_gpu);
      pq_codebooks_gpu = nullptr;
    }
    if (pq_codes_gpu) {
      cudaFree(pq_codes_gpu);
      pq_codes_gpu = nullptr;
    }
    std::cerr << "Fatal error: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}
