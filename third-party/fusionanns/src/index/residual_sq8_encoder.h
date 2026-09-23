#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common/residual_sq8_format.h"

namespace fusionann {

/// 残差 SQ8 编码器选项
struct RSQ8EncoderOptions {
  int dim = 128;                  ///< 向量维度
  float scale_percentile = 0.99f; ///< Scale 计算百分位 (0.99 = 99th percentile)
  bool use_max_abs = false; ///< true: 使用最大绝对值; false: 使用百分位
  int num_threads = 0;      ///< OpenMP 线程数 (0 = 自动)
};

/// 单个聚类的编码结果
struct RSQ8ClusterData {
  std::vector<int8_t> codes;       ///< [padded_count, dim] Int8 残差编码
  std::vector<float> norms;        ///< [padded_count] FP32 预计算 ||r||²
  std::vector<int32_t> global_ids; ///< [padded_count] Global Vector IDs
  float scale;                     ///< 该聚类的量化 scale
  uint32_t original_count;         ///< 原始向量数量
  uint32_t padded_count;           ///< Padding 后的向量数量
};

/// 完整的 RSQ8 编码结果
struct RSQ8EncodedIndex {
  std::vector<RSQ8ClusterData> clusters; ///< 每个聚类的编码数据
  std::vector<float> centroids;          ///< [nlist, dim] 聚类中心 (FP32)
  uint32_t nlist;                        ///< 聚类数量
  uint32_t dim;                          ///< 向量维度
  uint64_t total_vectors;                ///< 总向量数量

  /// 计算索引文件大小 (字节)
  size_t compute_file_size() const;

  /// 保存到文件
  void save(const std::string &path) const;

  /// 从文件加载
  static RSQ8EncodedIndex load(const std::string &path);
};

/// 残差 SQ8 编码器
///
/// 工作流程:
/// 1. 接收聚类结果 (centroids + assignments)
/// 2. 计算每个向量的残差 r = v - centroid
/// 3. 对每个聚类计算 scale
/// 4. 量化残差为 Int8
/// 5. 预计算 ||r||² (FP16)
class RSQ8Encoder {
public:
  using GatherFn =
      std::function<void(const int32_t *ids, size_t count, float *out)>;

  explicit RSQ8Encoder(const RSQ8EncoderOptions &options);

  /// 编码整个数据集
  /// @param centroids [nlist, dim] 聚类中心
  /// @param posting_lists 每个聚类包含的向量 ID 列表
  /// @param gather 按 ID 获取向量的回调函数
  /// @return 编码后的索引
  RSQ8EncodedIndex
  encode(const float *centroids,
         const std::vector<std::vector<int32_t>> &posting_lists,
         GatherFn gather);

private:
  /// 计算单个聚类的 scale
  float compute_cluster_scale(const float *residuals, size_t count) const;

  /// 编码单个聚类
  RSQ8ClusterData encode_cluster(const float *centroid,
                                 const std::vector<int32_t> &ids,
                                 GatherFn &gather) const;

  RSQ8EncoderOptions options_;
};

} // namespace fusionann
