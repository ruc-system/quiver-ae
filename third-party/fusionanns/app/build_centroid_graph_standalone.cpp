/**
 * build_centroid_graph_standalone.cpp
 *
 * Standalone tool to build a SPTAG centroid graph from centroids.fvecs.
 * Directly configures SPTAG with parameters suitable for large-scale
 * (1M+ vectors) graph construction.
 *
 * Usage:
 *   ./bin/build_centroid_graph_standalone \
 *     --centroids data/simulation/centroids.fvecs \
 *     --output indices_simulation/proximity_graph_index \
 *     --threads 48
 */

#include "common/io.h"

#include <CLI/CLI.hpp>

#include <inc/Core/Common.h>
#include <inc/Core/VectorIndex.h>
#include <inc/Helper/StringConvert.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void check(SPTAG::ErrorCode code, const char *ctx) {
  if (code != SPTAG::ErrorCode::Success) {
    throw std::runtime_error(std::string(ctx) + " failed: " +
                             SPTAG::Helper::Convert::ConvertToString(code));
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string centroids_path = "data/simulation/centroids.fvecs";
  std::string output_path = "indices_simulation/proximity_graph_index";
  int threads = 0;
  int neighborhood_size = 32;
  int refine_iters = 2;

  CLI::App app{"Build SPTAG centroid graph from centroids.fvecs"};
  app.add_option("--centroids", centroids_path,
                 "Path to centroids file (fvecs format)");
  app.add_option("--output", output_path,
                 "Output directory for SPTAG graph index");
  app.add_option("--threads", threads,
                 "Number of build threads (0 = auto)");
  app.add_option("--neighborhood-size", neighborhood_size,
                 "Graph neighborhood size (default: 32)");
  app.add_option("--refine-iters", refine_iters,
                 "Graph refine iterations (default: 2)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  try {
    // Load centroids
    std::cout << "[1/3] Loading centroids from " << centroids_path << "..."
              << std::endl;
    std::vector<float> centroids;
    long num = 0;
    int dim = 0;
    load_fvecs(centroids_path, centroids, num, dim);
    std::cout << "  > Loaded " << num << " centroids (dim=" << dim << ")"
              << std::endl;

    // Build SPTAG graph directly
    std::cout << "[2/3] Building SPTAG BKT graph..." << std::endl;
    std::cout << "  > NeighborhoodSize=" << neighborhood_size
              << " RefineIterations=" << refine_iters
              << " Threads=" << threads << std::endl;

    SPTAG::SetLogger(std::make_shared<SPTAG::Helper::SimpleLogger>(
        SPTAG::Helper::LogLevel::LL_Info));

    auto index = SPTAG::VectorIndex::CreateInstance(
        SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
    if (!index) {
      throw std::runtime_error("SPTAG VectorIndex creation failed");
    }

    // Configure — use conservative defaults for large scale
    check(index->SetParameter("DistCalcMethod", "L2"), "DistCalcMethod");
    check(index->SetParameter("BKTKmeansK", "16"), "BKTKmeansK");
    check(index->SetParameter("BKTLeafSize", "8"), "BKTLeafSize");
    check(index->SetParameter("NeighborhoodSize",
                              std::to_string(neighborhood_size)),
          "NeighborhoodSize");
    check(index->SetParameter("GraphNeighborhoodScale", "2"),
          "GraphNeighborhoodScale");
    check(index->SetParameter("RefineIterations",
                              std::to_string(refine_iters)),
          "RefineIterations");
    if (threads > 0) {
      check(index->SetParameter("NumberOfThreads", std::to_string(threads)),
            "NumberOfThreads");
    }
    check(index->SetParameter("HashTableExponent", "2"), "HashTableExponent");
    check(index->SetParameter("MaxCheck", "2048"), "MaxCheck");
    check(index->SetParameter("CEF", "256"), "CEF");
    check(index->SetParameter("TPTNumber", "16"), "TPTNumber");
    check(index->SetParameter("TPTLeafSize", "2000"), "TPTLeafSize");
    int sample_cap = std::max(32, std::min(static_cast<int>(num), 2048));
    check(index->SetParameter("Samples", std::to_string(sample_cap)),
          "Samples");

    auto t0 = std::chrono::steady_clock::now();
    check(index->BuildIndex(centroids.data(),
                            static_cast<SPTAG::SizeType>(num),
                            static_cast<SPTAG::DimensionType>(dim)),
          "BuildIndex");
    auto t1 = std::chrono::steady_clock::now();

    double build_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  > Build completed in " << build_sec << " seconds"
              << std::endl;

    // Save
    std::cout << "[3/3] Saving graph to " << output_path << "..." << std::endl;
    fs::create_directories(output_path);
    check(index->SaveIndex(output_path), "SaveIndex");
    std::cout << "  > Done!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
