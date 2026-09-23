#pragma once

#include "common/datasource.h"
#include "index/centroid_navigator.h"
#include "index/hbc_builder.h"
#include "index/replication_types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace faiss {
class ProductQuantizer;
}

struct BuilderOptions {
  BaseFormat base_fmt = BaseFormat::FVECs;
  std::string base_path;

  bool has_learn = false;
  BaseFormat learn_fmt = BaseFormat::FVECs;
  std::string learn_path;
  size_t learn_n_hint = 0;

  int nlist = 4096;
  int branch = 32;
  int beam = 256;
  int m = 16;
  int nbits = 8;
  size_t chunk = 1'000'000;
  uint32_t page = 4096;
  std::string output_dir = "indices";
  float epsilon = 0.6f;
  int max_replications = 8;
  float imbalance_delta = 0.05f;
  bool preload = true;
  size_t hbc_preload_gib = 16;
  size_t hbc_stream_block_mb = 32;
  size_t cache_size_mb = 0;
  bool use_mmap = false;

  // Residual SQ8 选项
  bool use_residual_sq8 = false;       ///< 启用残差 SQ8 编码
  float rsq8_scale_percentile = 0.99f; ///< Scale 计算百分位
  bool rsq8_use_max_abs = false;       ///< 使用最大绝对值计算 scale
};

class FusionAnnsBuilder {
public:
  explicit FusionAnnsBuilder(const BuilderOptions &opt);
  void build();

private:
  void ensure_dirs_() const;
  std::vector<float> sample_train_(size_t n_train);
  HbcResult run_hbc_();
  std::unique_ptr<ICentroidNavigator>
  build_centroid_navigator_(const std::vector<HbcNode *> &leaves);
  ReplicationResult run_replication_(const HbcResult &hbc,
                                     const ICentroidNavigator &navigator);
  void train_pq_and_encode_();
  void encode_pq_codes_parallel_(const faiss::ProductQuantizer &pq);
  void encode_residual_sq8_(); ///< RSQ8 编码替代 PQ
  void create_packed_layout_();
  void save_posting_list_metadata_() const;
  void preload_base_dataset_();
  void read_block_cached_(size_t start, size_t count, float *out) const;
  void read_gather_cached_(const int64_t *ids, size_t count, float *out) const;
  void read_gather_cached_(const int32_t *ids, size_t count, float *out) const;
  void build_packed_order_();
  std::filesystem::path output_dir_path_() const;
  std::filesystem::path output_path_(const std::string &relative) const;

private:
  BuilderOptions opt_;
  std::unique_ptr<IDataSource> base_;
  std::unique_ptr<IDataSource> learn_;
  size_t base_num_ = 0;
  int dim_ = 0;
  std::vector<float> base_cache_;
  std::vector<std::vector<int32_t>> primary_lists_;
  std::vector<std::vector<int32_t>> replication_lists_;
  std::vector<int32_t> vector_to_leaf_assignment_;
  std::vector<int64_t> packed_order_;
  bool has_preload_ = false;
  bool has_packed_order_ = false;
  int default_threads_ = 1;
  std::vector<float> leaf_centroids_; ///< [nlist, dim] 叶子聚类中心
};
