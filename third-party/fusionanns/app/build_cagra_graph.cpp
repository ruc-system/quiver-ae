/// build_cagra_graph: 从已有 SPTAG 索引中提取 centroid 向量，构建 CAGRA 质心图
///
/// 用法:
///   build_cagra_graph --index-path ./indices_sift1m
///
/// 输入: <index-path>/proximity_graph_index/vectors.bin  (SPTAG 格式)
/// 输出: <index-path>/centroid_graph_cagra.bin

#ifdef FUSIONANNS_USE_CUVS_CAGRA
#include "index/centroid_navigator_cagra.h"
#endif

#include <CLI/CLI.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/// 读取 SPTAG vectors.bin 格式: [nrows(int32)][dim(int32)][nrows*dim*float32]
static std::vector<float> load_sptag_vectors(const std::string &path,
                                             int &nrows, int &dim) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open vectors file: " + path);
  }

  int32_t n = 0, d = 0;
  ifs.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
  ifs.read(reinterpret_cast<char *>(&d), sizeof(int32_t));

  if (n <= 0 || d <= 0) {
    throw std::runtime_error("Invalid vectors.bin header: nrows=" +
                             std::to_string(n) + " dim=" + std::to_string(d));
  }

  nrows = static_cast<int>(n);
  dim = static_cast<int>(d);

  std::vector<float> data(static_cast<size_t>(n) * static_cast<size_t>(d));
  ifs.read(reinterpret_cast<char *>(data.data()),
           static_cast<std::streamsize>(data.size() * sizeof(float)));

  if (!ifs) {
    throw std::runtime_error("Failed to read all vector data from " + path);
  }

  return data;
}

int main(int argc, char **argv) {
#ifndef FUSIONANNS_USE_CUVS_CAGRA
  std::cerr
      << "ERROR: This binary was compiled without CAGRA support.\n"
      << "Rebuild with --libcuvs_root pointing to a valid libcuvs installation."
      << std::endl;
  return 1;
#else
  std::string index_path = "./indices_sift1m";

  CLI::App app{"Build CAGRA centroid graph from existing SPTAG index"};
  app.add_option("--index-path", index_path,
                 "Path to the index directory (default: ./indices_sift1m)");
  CLI11_PARSE(app, argc, argv);

  try {
    // 1. 定位 SPTAG vectors.bin
    fs::path vectors_path =
        fs::path(index_path) / "proximity_graph_index" / "vectors.bin";
    if (!fs::exists(vectors_path)) {
      throw std::runtime_error("SPTAG vectors file not found: " +
                               vectors_path.string());
    }

    // 2. 读取 centroid 向量
    int nrows = 0, dim = 0;
    std::cout << "Loading centroids from " << vectors_path << " ..."
              << std::endl;
    auto centroids = load_sptag_vectors(vectors_path.string(), nrows, dim);
    std::cout << "  Loaded " << nrows << " centroids (dim=" << dim << ")"
              << std::endl;

    // 3. 构建 CAGRA 图
    fs::path output_path =
        fs::path(index_path) / "centroid_graph_cagra.bin";
    std::cout << "Building CAGRA centroid graph ..." << std::endl;

    CAGRACentroidNavigator cagra_nav;
    cagra_nav.build(centroids.data(), nrows, dim, 0);

    // 4. 保存
    cagra_nav.save(output_path.string());
    std::cout << "  CAGRA centroid graph saved to " << output_path
              << std::endl;
    std::cout << "Done!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
#endif
}
