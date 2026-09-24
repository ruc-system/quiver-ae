#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "shared/common/logging.hpp"

namespace bin_common {

struct QueryData {
  std::unique_ptr<float[]> values;
  size_t dims = 0;
  size_t count = 0;
};

struct GroundTruthData {
  std::unique_ptr<int[]> values;
  size_t width = 0;
  size_t count = 0;
};

template <class T>
inline std::unique_ptr<T[]> read_typed_bin(const std::string& fname,
                                           size_t& d_out,
                                           size_t& n_out) {
  std::ifstream input(fname, std::ios::binary);
  if (!input.is_open()) {
    ERROR("Failed to open file {}", fname);
    std::exit(1);
  }

  int32_t npts = 0;
  int32_t ndims = 0;
  input.read(reinterpret_cast<char*>(&npts), sizeof(int32_t));
  input.read(reinterpret_cast<char*>(&ndims), sizeof(int32_t));

  d_out = static_cast<size_t>(ndims);
  n_out = static_cast<size_t>(npts);

  auto result = std::make_unique<T[]>(d_out * n_out);
  input.read(reinterpret_cast<char*>(result.get()),
             sizeof(T) * d_out * n_out);
  return result;
}

inline std::vector<std::string> read_ssd_list(const std::string& ssd_list_file) {
  std::vector<std::string> ssd_lists;
  if (ssd_list_file.empty()) {
    return ssd_lists;
  }

  std::ifstream input(ssd_list_file);
  if (!input.is_open()) {
    ERROR("Failed to open SSD list file {}", ssd_list_file);
    std::exit(1);
  }

  std::string line;
  while (input >> line) {
    INFO("USE SSD: \"{}\"", line);
    ssd_lists.push_back(line);
  }
  return ssd_lists;
}

inline QueryData load_queries_as_float(const std::string& query_file,
                                       int repeat = 1) {
  QueryData queries;

  if (query_file.find(".u8bin") != std::string::npos) {
    auto raw = read_typed_bin<uint8_t>(query_file, queries.dims, queries.count);
    queries.values = std::make_unique<float[]>(queries.dims * queries.count);
    std::copy(raw.get(), raw.get() + queries.dims * queries.count,
              queries.values.get());
  } else if (query_file.find(".i8bin") != std::string::npos) {
    auto raw = read_typed_bin<int8_t>(query_file, queries.dims, queries.count);
    queries.values = std::make_unique<float[]>(queries.dims * queries.count);
    std::copy(raw.get(), raw.get() + queries.dims * queries.count,
              queries.values.get());
  } else {
    queries.values = read_typed_bin<float>(query_file, queries.dims,
                                           queries.count);
  }

  INFO("Read {} queries", queries.count);

  if (repeat > 1) {
    const size_t original_count = queries.count;
    auto repeated =
        std::make_unique<float[]>(queries.dims * original_count * repeat);
    for (int i = 0; i < repeat; ++i) {
      std::copy(queries.values.get(),
                queries.values.get() + queries.dims * original_count,
                repeated.get() + i * queries.dims * original_count);
    }
    queries.values = std::move(repeated);
    queries.count = original_count * static_cast<size_t>(repeat);
  }

  return queries;
}

inline GroundTruthData load_ground_truth(const std::string& gt_file) {
  GroundTruthData gt;
  gt.values = read_typed_bin<int>(gt_file, gt.width, gt.count);
  return gt;
}

template <class T>
inline void write_typed_bin(const std::string& path, const T* values,
                            size_t count, size_t dims) {
  if (count > static_cast<size_t>(INT32_MAX) ||
      dims > static_cast<size_t>(INT32_MAX)) {
    throw std::runtime_error("Binary result dimensions exceed int32 range");
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open binary result file " + path);
  }

  const int32_t count_header = static_cast<int32_t>(count);
  const int32_t dims_header = static_cast<int32_t>(dims);
  output.write(reinterpret_cast<const char*>(&count_header),
               sizeof(count_header));
  output.write(reinterpret_cast<const char*>(&dims_header),
               sizeof(dims_header));
  output.write(reinterpret_cast<const char*>(values),
               sizeof(T) * count * dims);
  if (!output) {
    throw std::runtime_error("Failed to write binary result file " + path);
  }
}

}  // namespace bin_common
