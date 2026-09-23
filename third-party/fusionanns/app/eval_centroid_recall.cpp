/**
 * eval_centroid_recall.cpp
 *
 * Evaluate centroid graph search recall at various nprobe values.
 * Loads SPTAG graph, queries, and ground truth, then sweeps nprobe
 * and reports recall@K and latency statistics.
 *
 * Directly configures SPTAG search parameters (MaxCheck, CEF) for
 * large nprobe values (up to 20000+) that exceed the defaults in
 * SPTAGCentroidNavigator::tune_for_nprobe().
 *
 * Usage:
 *   ./bin/eval_centroid_recall \
 *     --graph indices_simulation/proximity_graph_index \
 *     --queries data/simulation/queries.fvecs \
 *     --groundtruth data/simulation/groundtruth_centroids.ivecs \
 *     --nprobe 1000,2000,5000,8000,10000,12000,15000,20000 \
 *     --threads 48
 */

#include "common/io.h"
#include "online/benchmark_runner.h"

#include <CLI/CLI.hpp>

#include <inc/Core/Common.h>
#include <inc/Core/VectorIndex.h>
#include <inc/Helper/StringConvert.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using fusionann::online::load_ivecs;

namespace {

void check(SPTAG::ErrorCode code, const char *ctx) {
  if (code != SPTAG::ErrorCode::Success) {
    throw std::runtime_error(std::string(ctx) + " failed: " +
                             SPTAG::Helper::Convert::ConvertToString(code));
  }
}

std::vector<int> parse_int_list(const std::string &spec) {
  std::vector<int> result;
  if (spec.empty())
    return result;
  std::istringstream iss(spec);
  std::string token;
  while (std::getline(iss, token, ',')) {
    if (!token.empty()) {
      result.push_back(std::stoi(token));
    }
  }
  return result;
}

double compute_recall_set(const std::vector<long> &results,
                          const int *ground_truth, int k) {
  std::unordered_set<long> gt_set;
  for (int i = 0; i < k; ++i) {
    gt_set.insert(static_cast<long>(ground_truth[i]));
  }
  int hits = 0;
  int check_size = std::min(static_cast<int>(results.size()), k);
  for (int i = 0; i < check_size; ++i) {
    if (gt_set.count(results[i]) > 0) {
      ++hits;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

struct SearchResult {
  std::vector<long> ids;
  std::vector<float> dists;
};

SearchResult search_one_sptag(
    const std::shared_ptr<SPTAG::VectorIndex> &index, const float *query,
    int nprobe) {
  std::vector<SPTAG::BasicResult> results(
      static_cast<size_t>(std::max(nprobe, 1)));
  SPTAG::QueryResult qres(const_cast<char *>(reinterpret_cast<const char *>(query)),
                           nprobe, /*withMeta=*/false, results.data());
  auto code = index->SearchIndex(qres);
  if (code != SPTAG::ErrorCode::Success) {
    throw std::runtime_error("SPTAG SearchIndex failed");
  }

  SearchResult out;
  out.ids.reserve(results.size());
  out.dists.reserve(results.size());
  for (const auto &res : results) {
    if (res.VID < 0)
      continue;
    out.ids.push_back(static_cast<long>(res.VID));
    out.dists.push_back(res.Dist);
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  std::string graph_path = "indices_simulation/proximity_graph_index";
  std::string queries_path = "data/simulation/queries.fvecs";
  std::string gt_path = "data/simulation/groundtruth_centroids.ivecs";
  std::string nprobe_spec = "1000,2000,5000,8000,10000,12000,15000,20000";
  int threads = 1;
  int recall_k = 10000;
  int maxcheck_factor = 10;

  CLI::App app{"Evaluate centroid graph search recall"};
  app.add_option("--graph", graph_path, "Path to SPTAG graph index directory");
  app.add_option("--queries", queries_path, "Path to query vectors (fvecs)");
  app.add_option("--groundtruth", gt_path,
                 "Path to ground truth centroids (ivecs)");
  app.add_option("--nprobe", nprobe_spec,
                 "Comma-separated nprobe values to sweep");
  app.add_option("--threads", threads, "Number of search threads");
  app.add_option("--recall_k", recall_k, "Recall@K metric (default: 10000)");
  app.add_option("--maxcheck-factor", maxcheck_factor,
                 "MaxCheck = nprobe * factor (default: 10)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  try {
    std::vector<int> nprobe_values = parse_int_list(nprobe_spec);
    if (nprobe_values.empty()) {
      std::cerr << "No nprobe values specified" << std::endl;
      return 1;
    }

    // Load SPTAG graph directly
    std::cout << "[1/3] Loading SPTAG graph from " << graph_path << "..."
              << std::endl;
    SPTAG::SetLogger(std::make_shared<SPTAG::Helper::SimpleLogger>(
        SPTAG::Helper::LogLevel::LL_Warning));

    std::shared_ptr<SPTAG::VectorIndex> index;
    check(SPTAG::VectorIndex::LoadIndex(graph_path, index), "LoadIndex");
    if (!index) {
      throw std::runtime_error("SPTAG LoadIndex returned null");
    }
    int dim = index->GetFeatureDim();
    std::cout << "  > Graph loaded (dim=" << dim << ")" << std::endl;

    // Load queries
    std::cout << "[2/3] Loading queries and ground truth..." << std::endl;
    std::vector<float> query_data;
    long query_num = 0;
    int query_dim = 0;
    load_fvecs(queries_path, query_data, query_num, query_dim);
    std::cout << "  > Loaded " << query_num << " queries (dim=" << query_dim
              << ")" << std::endl;

    std::vector<int> gt_data;
    long gt_num = 0;
    int gt_dim = 0;
    load_ivecs(gt_path, gt_data, gt_num, gt_dim);
    std::cout << "  > Loaded ground truth: " << gt_num << " queries x "
              << gt_dim << " neighbors" << std::endl;

    if (query_num != gt_num) {
      std::cerr << "Query count mismatch: " << query_num << " vs " << gt_num
                << std::endl;
      return 1;
    }
    if (recall_k > gt_dim) {
      std::cerr << "Warning: recall_k=" << recall_k << " > gt_dim=" << gt_dim
                << ", clamping to gt_dim" << std::endl;
      recall_k = gt_dim;
    }

    // Set search threads
    check(index->SetParameter("NumberOfThreads", "1"),
          "SetParameter NumberOfThreads");
    check(index->UpdateIndex(), "UpdateIndex");

    // Sweep nprobe values
    std::cout << "\n[3/3] Evaluating recall (maxcheck_factor="
              << maxcheck_factor << ")...\n" << std::endl;
    std::cout << std::setw(10) << "nprobe" << std::setw(10) << "maxcheck"
              << std::setw(10) << "cef" << std::setw(14) << "avg_recall"
              << std::setw(14) << "min_recall" << std::setw(16)
              << "avg_search_ms" << std::setw(16) << "p99_search_ms"
              << std::setw(16) << "p50_search_ms" << std::endl;
    std::cout << std::string(106, '-') << std::endl;

    for (int nprobe : nprobe_values) {
      // Configure search parameters — no cap on MaxCheck
      int maxcheck = nprobe * maxcheck_factor;
      int cef = std::max(nprobe, 512);

      check(index->SetParameter("MaxCheck", std::to_string(maxcheck)),
            "SetParameter MaxCheck");
      check(index->SetParameter("CEF", std::to_string(cef)),
            "SetParameter CEF");

      std::vector<double> recalls(query_num);
      std::vector<double> latencies(query_num);

      // Run searches
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (long q = 0; q < query_num; ++q) {
        const float *query = query_data.data() + q * query_dim;

        auto t0 = std::chrono::steady_clock::now();
        auto result = search_one_sptag(index, query, nprobe);
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies[q] = ms;

        const int *gt_ptr = gt_data.data() + q * gt_dim;
        recalls[q] = compute_recall_set(result.ids, gt_ptr, recall_k);
      }

      // Statistics
      double avg_recall =
          std::accumulate(recalls.begin(), recalls.end(), 0.0) / query_num;
      double min_recall = *std::min_element(recalls.begin(), recalls.end());

      std::sort(latencies.begin(), latencies.end());
      double avg_ms =
          std::accumulate(latencies.begin(), latencies.end(), 0.0) / query_num;
      double p50_ms = latencies[static_cast<size_t>(query_num * 0.50)];
      double p99_ms = latencies[static_cast<size_t>(query_num * 0.99)];

      std::cout << std::setw(10) << nprobe << std::setw(10) << maxcheck
                << std::setw(10) << cef << std::setw(14) << std::fixed
                << std::setprecision(6) << avg_recall << std::setw(14)
                << min_recall << std::setw(16) << std::setprecision(3)
                << avg_ms << std::setw(16) << p99_ms << std::setw(16)
                << p50_ms << std::endl;
    }

    std::cout << "\nDone!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
