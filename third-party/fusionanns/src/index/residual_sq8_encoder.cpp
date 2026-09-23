#include "index/residual_sq8_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fusionann {

namespace {

/// FP32 -> FP16 转换 (使用 IEEE 754 规范)
inline uint16_t fp32_to_fp16(float value) {
  if (!std::isfinite(value)) {
    return value > 0 ? 0x7C00 : 0xFC00; // +inf / -inf
  }

  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(float));

  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = bits & 0x7FFFFF;

  if (exponent < -24) {
    return static_cast<uint16_t>(sign << 15); // 下溢为 0
  }
  if (exponent > 15) {
    return static_cast<uint16_t>((sign << 15) | 0x7C00); // 上溢为 inf
  }
  if (exponent < -14) {
    // 非规格化数
    mantissa |= 0x800000;
    mantissa >>= (-14 - exponent);
    return static_cast<uint16_t>((sign << 15) | (mantissa >> 13));
  }

  return static_cast<uint16_t>((sign << 15) |
                               (static_cast<uint32_t>(exponent + 15) << 10) |
                               (mantissa >> 13));
}

/// 计算 L2 范数的平方
inline float compute_l2_norm_sq(const float *vec, int dim) {
  float acc = 0.0f;
  for (int i = 0; i < dim; ++i) {
    acc += vec[i] * vec[i];
  }
  return acc;
}

/// 计算残差: residual = vec - centroid
inline void compute_residual(const float *vec, const float *centroid,
                             float *residual, int dim) {
  for (int i = 0; i < dim; ++i) {
    residual[i] = vec[i] - centroid[i];
  }
}

} // namespace

// ========== RSQ8EncodedIndex Implementation ==========

size_t RSQ8EncodedIndex::compute_file_size() const {
  size_t size = sizeof(RSQ8IndexHeader);
  size += nlist * sizeof(RSQ8ClusterHeader);
  size += centroids.size() * sizeof(float);

  for (const auto &cluster : clusters) {
    size += cluster.codes.size() * sizeof(int8_t);
    size += cluster.norms.size() * sizeof(float);
    size += cluster.global_ids.size() * sizeof(int32_t);
  }

  return size;
}

void RSQ8EncodedIndex::save(const std::string &path) const {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }

  // 计算偏移量
  uint64_t current_offset = sizeof(RSQ8IndexHeader);
  uint64_t headers_offset = current_offset;
  current_offset += nlist * sizeof(RSQ8ClusterHeader);
  uint64_t centroids_offset = current_offset;
  current_offset += centroids.size() * sizeof(float);

  // 准备 ClusterHeaders
  std::vector<RSQ8ClusterHeader> headers(nlist);
  for (uint32_t i = 0; i < nlist; ++i) {
    const auto &cluster = clusters[i];
    headers[i].data_offset = current_offset;
    current_offset += cluster.codes.size() * sizeof(int8_t);
    // 对齐到 128 字节 (HBM transaction)
    current_offset = (current_offset + RSQ8_CACHE_LINE - 1) / RSQ8_CACHE_LINE *
                     RSQ8_CACHE_LINE;

    headers[i].norm_offset = current_offset;
    current_offset += cluster.norms.size() * sizeof(float);
    current_offset = (current_offset + RSQ8_CACHE_LINE - 1) / RSQ8_CACHE_LINE *
                     RSQ8_CACHE_LINE;

    headers[i].id_offset = current_offset;
    current_offset += cluster.global_ids.size() * sizeof(int32_t);
    current_offset = (current_offset + RSQ8_CACHE_LINE - 1) / RSQ8_CACHE_LINE *
                     RSQ8_CACHE_LINE;

    headers[i].padded_count = cluster.padded_count;
    headers[i].original_count = cluster.original_count;
    headers[i].scale = cluster.scale;
    std::memset(headers[i].reserved, 0, sizeof(headers[i].reserved));
  }

  // 写入文件头
  RSQ8IndexHeader file_header{};
  file_header.magic = RSQ8_MAGIC;
  file_header.version = RSQ8_VERSION;
  file_header.nlist = nlist;
  file_header.dim = dim;
  file_header.total_vectors = total_vectors;
  file_header.headers_offset = headers_offset;
  file_header.centroids_offset = centroids_offset;
  std::memset(file_header.reserved, 0, sizeof(file_header.reserved));

  ofs.write(reinterpret_cast<const char *>(&file_header), sizeof(file_header));
  ofs.write(reinterpret_cast<const char *>(headers.data()),
            headers.size() * sizeof(RSQ8ClusterHeader));
  ofs.write(reinterpret_cast<const char *>(centroids.data()),
            centroids.size() * sizeof(float));

  // 写入每个聚类的数据
  std::vector<char> padding_buffer(RSQ8_CACHE_LINE, 0);
  for (uint32_t i = 0; i < nlist; ++i) {
    const auto &cluster = clusters[i];

    ofs.write(reinterpret_cast<const char *>(cluster.codes.data()),
              cluster.codes.size() * sizeof(int8_t));
    // Padding
    size_t pos = ofs.tellp();
    size_t aligned_pos =
        (pos + RSQ8_CACHE_LINE - 1) / RSQ8_CACHE_LINE * RSQ8_CACHE_LINE;
    if (aligned_pos > pos) {
      ofs.write(padding_buffer.data(), aligned_pos - pos);
    }

    ofs.write(reinterpret_cast<const char *>(cluster.norms.data()),
              cluster.norms.size() * sizeof(float));
    pos = ofs.tellp();
    aligned_pos =
        (pos + RSQ8_CACHE_LINE - 1) / RSQ8_CACHE_LINE * RSQ8_CACHE_LINE;
    if (aligned_pos > pos) {
      ofs.write(padding_buffer.data(), aligned_pos - pos);
    }

    ofs.write(reinterpret_cast<const char *>(cluster.global_ids.data()),
              cluster.global_ids.size() * sizeof(int32_t));
    pos = ofs.tellp();
    aligned_pos =
        (pos + RSQ8_CACHE_LINE - 1) / RSQ8_CACHE_LINE * RSQ8_CACHE_LINE;
    if (aligned_pos > pos) {
      ofs.write(padding_buffer.data(), aligned_pos - pos);
    }
  }

  ofs.flush();
  if (!ofs) {
    throw std::runtime_error("Failed to write RSQ8 index file: " + path);
  }

  std::cout << "  > RSQ8 index saved: " << path << " ("
            << (current_offset / 1024.0 / 1024.0) << " MiB)" << std::endl;
}

RSQ8EncodedIndex RSQ8EncodedIndex::load(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open RSQ8 index file: " + path);
  }

  RSQ8IndexHeader file_header{};
  ifs.read(reinterpret_cast<char *>(&file_header), sizeof(file_header));

  if (file_header.magic != RSQ8_MAGIC) {
    throw std::runtime_error("Invalid RSQ8 index magic number");
  }
  if (file_header.version != RSQ8_VERSION) {
    throw std::runtime_error("Unsupported RSQ8 index version: " +
                             std::to_string(file_header.version));
  }

  RSQ8EncodedIndex index;
  index.nlist = file_header.nlist;
  index.dim = file_header.dim;
  index.total_vectors = file_header.total_vectors;

  // 读取 ClusterHeaders
  std::vector<RSQ8ClusterHeader> headers(index.nlist);
  ifs.seekg(static_cast<std::streamoff>(file_header.headers_offset));
  ifs.read(reinterpret_cast<char *>(headers.data()),
           headers.size() * sizeof(RSQ8ClusterHeader));

  // 读取 centroids
  index.centroids.resize(static_cast<size_t>(index.nlist) * index.dim);
  ifs.seekg(static_cast<std::streamoff>(file_header.centroids_offset));
  ifs.read(reinterpret_cast<char *>(index.centroids.data()),
           index.centroids.size() * sizeof(float));

  // 读取每个聚类的数据
  index.clusters.resize(index.nlist);
  for (uint32_t i = 0; i < index.nlist; ++i) {
    auto &cluster = index.clusters[i];
    const auto &header = headers[i];

    cluster.original_count = header.original_count;
    cluster.padded_count = header.padded_count;
    cluster.scale = header.scale;

    size_t vec_size = static_cast<size_t>(header.padded_count) * index.dim;
    cluster.codes.resize(vec_size);
    ifs.seekg(static_cast<std::streamoff>(header.data_offset));
    ifs.read(reinterpret_cast<char *>(cluster.codes.data()),
             vec_size * sizeof(int8_t));

    cluster.norms.resize(header.padded_count);
    ifs.seekg(static_cast<std::streamoff>(header.norm_offset));
    ifs.read(reinterpret_cast<char *>(cluster.norms.data()),
             header.padded_count * sizeof(float));

    cluster.global_ids.resize(header.padded_count);
    ifs.seekg(static_cast<std::streamoff>(header.id_offset));
    ifs.read(reinterpret_cast<char *>(cluster.global_ids.data()),
             header.padded_count * sizeof(int32_t));
  }

  return index;
}

// ========== RSQ8Encoder Implementation ==========

RSQ8Encoder::RSQ8Encoder(const RSQ8EncoderOptions &options)
    : options_(options) {
#ifdef _OPENMP
  if (options_.num_threads > 0) {
    omp_set_num_threads(options_.num_threads);
  }
#endif
}

float RSQ8Encoder::compute_cluster_scale(const float *residuals,
                                         size_t count) const {
  if (count == 0)
    return 1.0f;

  const int dim = options_.dim;

  if (options_.use_max_abs) {
    // 使用最大绝对值
    float max_abs = 0.0f;
    for (size_t i = 0; i < count * static_cast<size_t>(dim); ++i) {
      float val = std::fabs(residuals[i]);
      if (val > max_abs)
        max_abs = val;
    }
    return max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
  }

  // 使用百分位
  std::vector<float> abs_values;
  abs_values.reserve(count * static_cast<size_t>(dim));
  for (size_t i = 0; i < count * static_cast<size_t>(dim); ++i) {
    abs_values.push_back(std::fabs(residuals[i]));
  }

  size_t percentile_idx = static_cast<size_t>(
      static_cast<float>(abs_values.size() - 1) * options_.scale_percentile);
  std::nth_element(abs_values.begin(),
                   abs_values.begin() +
                       static_cast<std::ptrdiff_t>(percentile_idx),
                   abs_values.end());

  float percentile_val = abs_values[percentile_idx];
  return percentile_val > 0.0f ? percentile_val / 127.0f : 1.0f;
}

RSQ8ClusterData RSQ8Encoder::encode_cluster(const float *centroid,
                                            const std::vector<int32_t> &ids,
                                            GatherFn &gather) const {

  RSQ8ClusterData result;
  result.original_count = static_cast<uint32_t>(ids.size());
  result.padded_count = rsq8_padded_count(result.original_count);

  if (ids.empty()) {
    result.scale = 1.0f;
    return result;
  }

  const int dim = options_.dim;
  const size_t n = ids.size();
  const size_t n_padded = result.padded_count;

  // 获取原始向量
  std::vector<float> vectors(n * static_cast<size_t>(dim));
  gather(ids.data(), n, vectors.data());

  // 计算残差
  std::vector<float> residuals(n * static_cast<size_t>(dim));
  for (size_t i = 0; i < n; ++i) {
    compute_residual(vectors.data() + i * dim, centroid,
                     residuals.data() + i * dim, dim);
  }

  // 计算 scale
  result.scale = compute_cluster_scale(residuals.data(), n);
  const float inv_scale = 1.0f / result.scale;

  // 分配输出缓冲区
  result.codes.resize(n_padded * static_cast<size_t>(dim), 0);
  result.norms.resize(
      n_padded,
      std::numeric_limits<float>::infinity()); // Padding 位置设为 INFINITY
  result.global_ids.resize(n_padded, -1);      // Padding 位置设为 -1

  // 量化残差
  for (size_t i = 0; i < n; ++i) {
    const float *r = residuals.data() + i * dim;
    int8_t *c = result.codes.data() + i * dim;

    float norm_sq = 0.0f;
    for (int d = 0; d < dim; ++d) {
      float scaled = r[d] * inv_scale;
      scaled = std::max(-127.0f, std::min(127.0f, std::round(scaled)));
      c[d] = static_cast<int8_t>(scaled);

      // 使用量化后的值计算 norm (更准确地反映量化误差)
      float reconstructed = static_cast<float>(c[d]) * result.scale;
      norm_sq += reconstructed * reconstructed;
    }

    // 直接存储 FP32，无需转换，避免 FP16 溢出问题
    result.norms[i] = norm_sq;
    result.global_ids[i] = ids[i];
  }

  return result;
}

RSQ8EncodedIndex
RSQ8Encoder::encode(const float *centroids,
                    const std::vector<std::vector<int32_t>> &posting_lists,
                    GatherFn gather) {

  const size_t nlist = posting_lists.size();
  const int dim = options_.dim;

  RSQ8EncodedIndex result;
  result.nlist = static_cast<uint32_t>(nlist);
  result.dim = static_cast<uint32_t>(dim);
  result.total_vectors = 0;

  // 复制 centroids
  result.centroids.assign(centroids, centroids + nlist * dim);

  // 编码每个聚类
  result.clusters.resize(nlist);

  std::cout << "  > RSQ8 encoding " << nlist << " clusters..." << std::endl;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (size_t i = 0; i < nlist; ++i) {
    const float *centroid = centroids + i * dim;
    result.clusters[i] = encode_cluster(centroid, posting_lists[i], gather);

#ifdef _OPENMP
    if (i % 1000 == 0) {
#pragma omp critical
      {
        std::cout << "    [RSQ8] Encoded " << i << "/" << nlist << " clusters"
                  << std::endl;
      }
    }
#endif
  }

  // 统计总向量数
  for (size_t i = 0; i < nlist; ++i) {
    result.total_vectors += result.clusters[i].original_count;
  }

  std::cout << "  > RSQ8 encoding complete: " << result.total_vectors
            << " vectors in " << nlist << " clusters" << std::endl;

  return result;
}

} // namespace fusionann
