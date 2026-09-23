#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fusionann {

/// 粗排结果 (单个查询命中的聚类)
struct ClusterProbe {
  int32_t cluster_id; ///< 聚类 ID
  float distance;     ///< 到聚类中心的距离
};

/// 单个聚类的批处理任务
struct ClusterTask {
  int32_t cluster_id = -1; ///< 聚类 ID
  int32_t query_offset = 0; ///< packed residual/query offset (in queries)
  int32_t num_queries = 0;  ///< 查询数量
  const int32_t *query_indices = nullptr; ///< [num_queries] 查询索引
  const float *residual_queries = nullptr; ///< [num_queries, dim] 残差

  /// 返回该任务中的查询数量
  size_t num_queries_count() const { return static_cast<size_t>(num_queries); }
};

/// 聚类调度器 - 将 "Query 找 Cluster" 反转为 "Cluster 找 Queries"
///
/// 设计目标:
/// - 高效处理大 batch (10000 queries) + 多 cluster (16384 nlist)
/// - 使用简单 vector of vectors 避免 map 开销
/// - OpenMP 并行加速
class ClusterScheduler {
public:
  /// 构造函数
  /// @param dim 向量维度
  /// @param num_threads OpenMP 线程数 (0 = 自动)
  explicit ClusterScheduler(int dim, int num_threads = 0);

  /// 调度查询到聚类
  ///
  /// @param query_batch [num_queries, dim] 查询向量批次
  /// @param num_queries 查询数量
  /// @param probe_ids [num_queries, nprobe] 扁平粗排结果
  /// @param nprobe 每个查询的 probe 数
  /// @param centroids [nlist, dim] 聚类中心
  /// @param nlist 聚类数量
  /// @return 非空聚类的任务列表 (已计算残差)
  std::vector<ClusterTask>
  schedule(const float *query_batch, int num_queries,
           const int32_t *probe_ids, int nprobe,
           const float *centroids, int nlist,
           float *packed_residuals = nullptr);

private:
  int dim_;
  int num_threads_;
};

// ============================================================================
// 内联实现
// ============================================================================

inline ClusterScheduler::ClusterScheduler(int dim, int num_threads)
    : dim_(dim), num_threads_(num_threads) {
#ifdef _OPENMP
  if (num_threads_ > 0) {
    omp_set_num_threads(num_threads_);
  }
#else
  (void)num_threads_;
#endif
}

inline std::vector<ClusterTask> ClusterScheduler::schedule(
    const float *query_batch, int num_queries,
    const int32_t *probe_ids, int nprobe,
    const float *centroids, int nlist, float *packed_residuals) {

  // Step 1: 统计每个 cluster 的命中数量 (per-thread 计数，避免锁竞争)
#ifdef _OPENMP
  int max_threads = num_threads_ > 0 ? num_threads_ : omp_get_max_threads();
  if (omp_in_parallel()) {
    max_threads = 1;
  }
#else
  const int max_threads = 1;
#endif

  if (probe_ids == nullptr || nprobe <= 0) {
    return {};
  }

  struct SchedulerBuffers {
    int cached_nlist = 0;
    int cached_dim = 0;
    std::vector<int32_t> thread_counts;
    std::vector<int32_t> cluster_counts;
    std::vector<int32_t> cluster_offsets;
    std::vector<int32_t> flat_queries;
    std::vector<float> residuals;
    std::vector<int32_t> non_empty_clusters;
    std::vector<ClusterTask> tasks;
  };

  static thread_local SchedulerBuffers buffers;

  const size_t nlist_size = static_cast<size_t>(nlist);
  const size_t thread_counts_size =
      static_cast<size_t>(max_threads) * nlist_size;
  if (buffers.cached_nlist != nlist || buffers.cached_dim != dim_) {
    buffers.cached_nlist = nlist;
    buffers.cached_dim = dim_;
  }
  buffers.thread_counts.resize(thread_counts_size);
  std::fill(buffers.thread_counts.begin(), buffers.thread_counts.end(), 0);

#ifdef _OPENMP
#pragma omp parallel num_threads(max_threads)
  {
    const int tid = omp_get_thread_num();
    int32_t *counts = buffers.thread_counts.data() +
                      static_cast<size_t>(tid) * nlist_size;
#pragma omp for schedule(static)
    for (int q = 0; q < num_queries; ++q) {
      const size_t base =
          static_cast<size_t>(q) * static_cast<size_t>(nprobe);
      for (int k = 0; k < nprobe; ++k) {
        int32_t cid = probe_ids[base + static_cast<size_t>(k)];
        if (cid >= 0 && cid < nlist) {
          counts[cid] += 1;
        }
      }
    }
  }
#else
  int32_t *counts = buffers.thread_counts.data();
  for (int q = 0; q < num_queries; ++q) {
    const size_t base =
        static_cast<size_t>(q) * static_cast<size_t>(nprobe);
    for (int k = 0; k < nprobe; ++k) {
      int32_t cid = probe_ids[base + static_cast<size_t>(k)];
      if (cid >= 0 && cid < nlist) {
        counts[cid] += 1;
      }
    }
  }
#endif

  // Step 2: 汇总计数 + 前缀和
  buffers.cluster_counts.resize(nlist);
  for (int c = 0; c < nlist; ++c) {
    int32_t sum = 0;
    for (int t = 0; t < max_threads; ++t) {
      sum += buffers.thread_counts[static_cast<size_t>(t) * nlist_size + c];
    }
    buffers.cluster_counts[c] = sum;
  }

  buffers.cluster_offsets.resize(static_cast<size_t>(nlist) + 1);
  buffers.cluster_offsets[0] = 0;
  for (int c = 0; c < nlist; ++c) {
    buffers.cluster_offsets[static_cast<size_t>(c) + 1] =
        buffers.cluster_offsets[static_cast<size_t>(c)] +
        buffers.cluster_counts[c];
  }

  const int32_t total_hits =
      buffers.cluster_offsets[static_cast<size_t>(nlist)];
  buffers.flat_queries.resize(static_cast<size_t>(total_hits));

  // Step 3: 将 per-thread 计数转换为 per-thread 写入偏移
  for (int c = 0; c < nlist; ++c) {
    int32_t offset = buffers.cluster_offsets[static_cast<size_t>(c)];
    for (int t = 0; t < max_threads; ++t) {
      size_t idx = static_cast<size_t>(t) * nlist_size + c;
      int32_t count = buffers.thread_counts[idx];
      buffers.thread_counts[idx] = offset;
      offset += count;
    }
  }

  // Step 4: 填充 flat_queries
#ifdef _OPENMP
#pragma omp parallel num_threads(max_threads)
  {
    const int tid = omp_get_thread_num();
    int32_t *offsets = buffers.thread_counts.data() +
                       static_cast<size_t>(tid) * nlist_size;
#pragma omp for schedule(static)
    for (int q = 0; q < num_queries; ++q) {
      const size_t base =
          static_cast<size_t>(q) * static_cast<size_t>(nprobe);
      for (int k = 0; k < nprobe; ++k) {
        int32_t cid = probe_ids[base + static_cast<size_t>(k)];
        if (cid >= 0 && cid < nlist) {
          int32_t pos = offsets[cid]++;
          buffers.flat_queries[static_cast<size_t>(pos)] = q;
        }
      }
    }
  }
#else
  int32_t *offsets = buffers.thread_counts.data();
  for (int q = 0; q < num_queries; ++q) {
    const size_t base =
        static_cast<size_t>(q) * static_cast<size_t>(nprobe);
    for (int k = 0; k < nprobe; ++k) {
      int32_t cid = probe_ids[base + static_cast<size_t>(k)];
      if (cid >= 0 && cid < nlist) {
        int32_t pos = offsets[cid]++;
        buffers.flat_queries[static_cast<size_t>(pos)] = q;
      }
    }
  }
#endif

  float *residual_storage = packed_residuals;
  if (!residual_storage) {
    buffers.residuals.resize(static_cast<size_t>(total_hits) *
                             static_cast<size_t>(dim_));
    residual_storage = buffers.residuals.data();
  }

  // Step 5: 统计非空聚类
  buffers.non_empty_clusters.clear();
  buffers.non_empty_clusters.reserve(nlist);
  for (int c = 0; c < nlist; ++c) {
    if (buffers.cluster_counts[c] > 0) {
      buffers.non_empty_clusters.push_back(c);
    }
  }

  // Step 6: 并行构建任务并计算残差
  buffers.tasks.resize(buffers.non_empty_clusters.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
  for (size_t i = 0; i < buffers.non_empty_clusters.size(); ++i) {
    int32_t cluster_id = buffers.non_empty_clusters[i];
    const float *centroid = centroids + static_cast<size_t>(cluster_id) * dim_;

    ClusterTask &task = buffers.tasks[i];
    task.cluster_id = cluster_id;
    const int32_t count = buffers.cluster_counts[cluster_id];
    const int32_t start =
        buffers.cluster_offsets[static_cast<size_t>(cluster_id)];

    task.query_offset = start;
    task.num_queries = count;
    task.query_indices = buffers.flat_queries.data() + start;
    task.residual_queries = residual_storage +
                            static_cast<size_t>(start) * dim_;

    for (int32_t j = 0; j < count; ++j) {
      int32_t q_idx = task.query_indices[j];

      const float *query = query_batch + static_cast<size_t>(q_idx) * dim_;
      float *residual = residual_storage +
                        (static_cast<size_t>(start + j) * dim_);

      for (int d = 0; d < dim_; ++d) {
        residual[d] = query[d] - centroid[d];
      }
    }
  }

  return buffers.tasks;
}

} // namespace fusionann
