#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

// CUDA host/device 属性宏 (兼容非 CUDA 编译环境)
#ifdef __CUDACC__
#define RSQ8_HOST_DEVICE __host__ __device__
#else
#define RSQ8_HOST_DEVICE
#endif

namespace fusionann {

// Residual SQ8 索引格式定义
// 设计用于 RTX 4090 Tensor Core 高效访问

/// 单个聚类的头信息 (存储在索引文件开头)
struct alignas(64) RSQ8ClusterHeader {
  uint64_t data_offset;  ///< Int8 残差向量数据在文件中的字节偏移
  uint64_t norm_offset;  ///< FP16 预计算 ||r||² 数据的字节偏移
  uint64_t id_offset;    ///< Global ID 数组的字节偏移
  uint32_t padded_count; ///< Padding 后的向量数量 (对齐到 64)
  uint32_t original_count; ///< 原始向量数量 (不含 Padding)
  float scale;             ///< 该聚类的量化 scale: max(|residual|) / 127
  uint32_t reserved[3];    ///< 保留字段，保证 64 字节对齐
};

static_assert(sizeof(RSQ8ClusterHeader) == 64,
              "RSQ8ClusterHeader must be 64 bytes for cache alignment");

/// Residual SQ8 索引文件头
struct RSQ8IndexHeader {
  uint32_t magic;            ///< 魔数: 0x52535138 ("RSQ8")
  uint32_t version;          ///< 版本号
  uint32_t nlist;            ///< 聚类数量
  uint32_t dim;              ///< 向量维度
  uint64_t total_vectors;    ///< 总向量数量
  uint64_t headers_offset;   ///< ClusterHeader 数组的偏移
  uint64_t centroids_offset; ///< 聚类中心 (FP32) 的偏移
  uint64_t reserved[4];      ///< 保留字段
};

constexpr uint32_t RSQ8_MAGIC = 0x52535138; // "RSQ8" in little-endian
constexpr uint32_t RSQ8_VERSION = 1;

/// Padding 对齐常量
constexpr size_t RSQ8_ALIGNMENT = 64;   ///< Tensor Core tile 对齐
constexpr size_t RSQ8_CACHE_LINE = 128; ///< HBM transaction 对齐

/// 计算 Padding 后的数量 (对齐到 alignment)
inline uint32_t rsq8_padded_count(uint32_t original,
                                  size_t alignment = RSQ8_ALIGNMENT) {
  return static_cast<uint32_t>((static_cast<size_t>(original) + alignment - 1) /
                               alignment * alignment);
}

/// INFINITY 值用于 Padding 位置的 norm (FP16)
/// 确保 Padding 位置不会被选入 Top-K
inline uint16_t rsq8_infinity_fp16() {
  // IEEE 754 FP16 正无穷: 0x7C00
  return 0x7C00;
}

/// 运行时索引视图 (加载到 GPU 显存后的结构)
struct RSQ8IndexView {
  int8_t *d_vectors; ///< [sum(padded_count), dim] 所有聚类的 Int8 残差
  uint16_t *
      d_norms; ///< [sum(padded_count)] FP16 预计算 ||r||² (含 INFINITY padding)
  int32_t *d_global_ids;        ///< [sum(padded_count)] Global Vector IDs
  float *d_centroids;           ///< [nlist, dim] 聚类中心
  RSQ8ClusterHeader *d_headers; ///< [nlist] 聚类元信息

  uint32_t nlist;
  uint32_t dim;
  uint64_t total_vectors;

  /// 获取指定聚类的 Int8 数据指针
  RSQ8_HOST_DEVICE int8_t *cluster_vectors(uint32_t cluster_id) const {
    return d_vectors + d_headers[cluster_id].data_offset / sizeof(int8_t);
  }

  /// 获取指定聚类的 norm 数据指针
  RSQ8_HOST_DEVICE uint16_t *cluster_norms(uint32_t cluster_id) const {
    return d_norms + d_headers[cluster_id].norm_offset / sizeof(uint16_t);
  }

  /// 获取指定聚类的 Global ID 数据指针
  RSQ8_HOST_DEVICE int32_t *cluster_ids(uint32_t cluster_id) const {
    return d_global_ids + d_headers[cluster_id].id_offset / sizeof(int32_t);
  }

  /// 获取指定聚类的中心
  RSQ8_HOST_DEVICE float *cluster_centroid(uint32_t cluster_id) const {
    return d_centroids + static_cast<size_t>(cluster_id) * dim;
  }
};

} // namespace fusionann
