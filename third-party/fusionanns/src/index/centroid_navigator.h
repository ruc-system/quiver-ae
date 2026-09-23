#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

struct ProbeResult {
  std::vector<long> ids;
  std::vector<float> dists;
};

class ICentroidNavigator {
public:
  virtual ~ICentroidNavigator() = default;
  virtual void build(const float *centroids, int nlist, int dim,
                     int num_threads = 0) = 0;
  virtual void save(const std::string &path) const = 0;
  virtual void load(const std::string &path) = 0;
  virtual ProbeResult search_one(const float *query, int nprobe) const = 0;

  /// 批量搜索（默认实现：串行调用 search_one）
  virtual std::vector<ProbeResult>
  search_batch(const float *queries, int num_queries, int nprobe) const {
    std::vector<ProbeResult> results(num_queries);
    for (int i = 0; i < num_queries; ++i) {
      results[i] = search_one(queries + i * get_dim(), nprobe);
    }
    return results;
  }

  /// 批量搜索（flat 输出: [num_queries * nprobe]）
  virtual void search_batch_flat(const float *queries, int num_queries,
                                 int nprobe, std::vector<int32_t> &out_ids,
                                 std::vector<float> &out_dists) const {
    auto results = search_batch(queries, num_queries, nprobe);
    const size_t total =
        static_cast<size_t>(num_queries) * static_cast<size_t>(nprobe);
    out_ids.resize(total, -1);
    out_dists.resize(total, std::numeric_limits<float>::infinity());
    for (int q = 0; q < num_queries; ++q) {
      const auto &probe = results[q];
      const int count = std::min(static_cast<int>(probe.ids.size()), nprobe);
      const size_t base = static_cast<size_t>(q) * static_cast<size_t>(nprobe);
      for (int k = 0; k < count; ++k) {
        out_ids[base + static_cast<size_t>(k)] =
            static_cast<int32_t>(probe.ids[k]);
        out_dists[base + static_cast<size_t>(k)] = probe.dists[k];
      }
    }
  }

  /// 获取向量维度
  virtual int get_dim() const = 0;
};
