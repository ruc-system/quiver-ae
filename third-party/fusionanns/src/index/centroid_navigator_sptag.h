#pragma once

#include "index/centroid_navigator.h"

#include <memory>

namespace SPTAG {
class VectorIndex;
}

class SPTAGCentroidNavigator final : public ICentroidNavigator {
public:
  SPTAGCentroidNavigator();
  ~SPTAGCentroidNavigator() override;

  void build(const float *centroids, int nlist, int dim,
             int num_threads) override;
  void save(const std::string &path) const override;
  void load(const std::string &path) override;
  ProbeResult search_one(const float *query, int nprobe) const override;

  /// 批量搜索（使用 SPTAG 批量 API）
  std::vector<ProbeResult> search_batch(const float *queries, int num_queries,
                                        int nprobe) const override;

  void search_batch_flat(const float *queries, int num_queries, int nprobe,
                         std::vector<int32_t> &out_ids,
                         std::vector<float> &out_dists) const override;

  /// 获取向量维度
  int get_dim() const override { return dim_; }

  void tune_for_nprobe(int nprobe);
  void apply_thread_local_settings() const;

private:
  void ensure_ready() const;
  void configure_build_defaults(int nlist, int nthreads = 0);
  void configure_search_defaults(int nprobe) const;

  std::shared_ptr<SPTAG::VectorIndex> index_;
  int dim_ = 0;
};
