#pragma once

#include "common/residual_sq8_format.h"
#include "online/cluster_scheduler.h"
#include "online/gpu_pool.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace fusionann {

// ============================================================================
// RSQ8 Workspace (GpuBlock-based, zero cudaMalloc per batch)
// ============================================================================

/// RSQ8 内存布局计划
struct RSQ8WorkspacePlan {
  int max_queries = 0;      ///< 最大查询数 (batch_size * nprobe)
  int max_cluster_size = 0; ///< 最大聚类大小
  int max_problems = 0;     ///< 最大 GEMM 问题数 (≈ 非空聚类数)
  int topk = 0;
  int dim = 0;

  // 计算得出的各缓冲区大小
  size_t residuals_fp32_bytes = 0;
  size_t query_int8_bytes = 0;
  size_t query_norms_bytes = 0;
  size_t query_scales_bytes = 0;
  size_t gemm_output_bytes = 0;
  size_t topk_indices_bytes = 0;
  size_t topk_distances_bytes = 0;
  size_t all_topk_bytes = 0;
  size_t grouped_problem_bytes = 0;
  size_t grouped_ptr_bytes = 0;
  size_t grouped_ld_bytes = 0;

  size_t gpu_bytes_per_block = 0; ///< GpuBlock 总大小
  size_t pinned_bytes_total = 0;  ///< Pinned memory 总大小

  /// 根据参数计算内存布局
  static RSQ8WorkspacePlan make(int max_queries, int max_cluster_size,
                                int max_problems, int topk, int dim);
};

/// 批量 Top-K 的 problem 描述符
struct RSQ8TopkProblemDesc {
  uint64_t gemm_offset;      ///< GEMM output 中的偏移 (元素数)
  uint64_t norm_offset;      ///< index norms 中的偏移
  uint64_t id_offset;        ///< index global_ids 中的偏移
  uint64_t out_offset;       ///< 输出的偏移 (topk 元素数)
  int32_t num_queries;       ///< M (该 problem 的 query 数)
  int32_t num_candidates;    ///< N (该 problem 的 candidate 数)
  int32_t query_norm_offset; ///< query_norms 中该 problem 的起始位置
  int32_t query_prefix_sum; ///< 前缀和: 该 problem 之前所有 queries 的总数
                            ///< (用于二分查找)
  float scale; ///< cluster scale
};

/// RSQ8 工作空间视图 (从 GpuBlock 划分)
struct RSQ8WorkspaceView {
  // GPU 缓冲区 (从 GpuBlock 划分)
  float *d_residuals_fp32 = nullptr;
  int8_t *d_query_int8 = nullptr;
  float *d_query_norms = nullptr;
  float *d_query_scales = nullptr;
  int32_t *d_gemm_output = nullptr;
  float *d_distances = nullptr;
  int32_t *d_topk_indices = nullptr;
  float *d_topk_distances = nullptr;
  int32_t *d_all_topk_indices = nullptr;
  float *d_all_topk_distances = nullptr;

  // Grouped GEMM 参数缓冲区
  void *d_grouped_problem_sizes = nullptr;
  void *d_grouped_ptr_A = nullptr;
  void *d_grouped_ptr_B = nullptr;
  void *d_grouped_ptr_C = nullptr;
  void *d_grouped_ptr_D = nullptr;
  void *d_grouped_lda = nullptr;
  void *d_grouped_ldb = nullptr;
  void *d_grouped_ldc = nullptr;
  void *d_grouped_ldd = nullptr;

  // Pinned memory (外部管理)
  int32_t *h_topk_indices_pinned = nullptr;
  float *h_topk_distances_pinned = nullptr;
  int32_t *h_all_topk_indices_pinned = nullptr;
  float *h_all_topk_distances_pinned = nullptr;

  // Host-side vectors (外部管理)
  std::vector<int> *problem_sizes_m = nullptr;
  std::vector<int> *problem_sizes_n = nullptr;
  std::vector<int> *problem_sizes_k = nullptr;
  std::vector<int64_t> *lda_host = nullptr;
  std::vector<int64_t> *ldb_host = nullptr;
  std::vector<int64_t> *ldc_host = nullptr;
  std::vector<int64_t> *ldd_host = nullptr;

  // 批量操作 pinned 缓冲区 (外部管理)
  float *h_residuals_staging = nullptr;
  RSQ8TopkProblemDesc *h_topk_descs = nullptr;

  // 批量操作 GPU 缓冲区 (从 GpuBlock 划分)
  RSQ8TopkProblemDesc *d_topk_descs = nullptr;

  // 容量信息
  int max_queries = 0;
  int max_cluster_size = 0;
  int max_problems = 0;
  int topk = 0;
  int dim = 0;
};

/// RSQ8 Pinned Memory 持有者 (单独分配，生命周期独立于 GpuBlock)
struct RSQ8PinnedBuffers {
  int32_t *h_topk_indices = nullptr;
  float *h_topk_distances = nullptr;
  int32_t *h_all_topk_indices = nullptr;
  float *h_all_topk_distances = nullptr;
  cublasHandle_t cublas_handle = nullptr;

  // Host vectors for GEMM args
  std::vector<int> problem_sizes_m, problem_sizes_n, problem_sizes_k;
  std::vector<int64_t> lda_host, ldb_host, ldc_host, ldd_host;

  // 批量操作的 pinned 缓冲区
  float *h_residuals_staging = nullptr; ///< 合并后的 residuals staging
  RSQ8TopkProblemDesc *h_topk_descs = nullptr; ///< 批量 topk 描述符

  size_t topk_capacity = 0;
  size_t all_topk_capacity = 0;
  size_t residuals_staging_capacity =
      0;                          ///< h_residuals_staging 容量 (float 数)
  size_t topk_descs_capacity = 0; ///< h_topk_descs 容量 (个数)

  static std::unique_ptr<RSQ8PinnedBuffers>
  allocate(const RSQ8WorkspacePlan &plan);
  ~RSQ8PinnedBuffers();

  RSQ8PinnedBuffers() = default;
  RSQ8PinnedBuffers(const RSQ8PinnedBuffers &) = delete;
  RSQ8PinnedBuffers &operator=(const RSQ8PinnedBuffers &) = delete;
};

/// 从 GpuBlock 和 PinnedBuffers 构建 WorkspaceView
RSQ8WorkspaceView rsq8_prepare_workspace(GpuBlock &block,
                                         RSQ8PinnedBuffers &pinned,
                                         const RSQ8WorkspacePlan &plan);

/// ResidualSQ8 索引的 GPU 运行时表示
struct RSQ8GpuIndex {
  // GPU 数据指针
  int8_t *d_vectors = nullptr;     ///< [total_padded, dim] Int8 残差向量
  float *d_norms = nullptr;        ///< [total_padded] FP32 预计算 ||r||²
  int32_t *d_global_ids = nullptr; ///< [total_padded] Global Vector IDs
  float *d_centroids = nullptr;    ///< [nlist, dim] 聚类中心
  RSQ8ClusterHeader *d_headers = nullptr; ///< [nlist] 聚类头信息

  // 元信息
  uint32_t nlist = 0;
  uint32_t dim = 0;
  uint64_t total_vectors = 0;
  uint64_t total_padded = 0; ///< 所有聚类 padding 后的总向量数

  // 每个聚类的起始偏移 (主机端缓存)
  std::vector<uint64_t> cluster_offsets;
  std::vector<uint32_t> cluster_sizes;
  std::vector<float> cluster_scales;

  /// 从文件加载索引到 GPU
  static std::unique_ptr<RSQ8GpuIndex> load(const std::string &path,
                                            cudaStream_t stream = nullptr);

  /// 释放 GPU 资源
  ~RSQ8GpuIndex();

  // 禁止拷贝
  RSQ8GpuIndex() = default;
  RSQ8GpuIndex(const RSQ8GpuIndex &) = delete;
  RSQ8GpuIndex &operator=(const RSQ8GpuIndex &) = delete;
  RSQ8GpuIndex(RSQ8GpuIndex &&) = default;
  RSQ8GpuIndex &operator=(RSQ8GpuIndex &&) = default;
};

/// Epilogue 模式
enum class RSQ8EpilogueMode {
  TwoStage, ///< Phase 4.1: GEMM → 显存写 → finalize kernel
  Fused     ///< Phase 4.2: CUTLASS Epilogue 融合 (TODO)
};

/// ResidualSQ8 评分结果 (单个查询)
struct RSQ8Result {
  int32_t global_id;
  float distance;
};

/// ResidualSQ8 评分器工作空间
struct RSQ8ScorerWorkspace {
  // Query 相关
  int8_t *d_query_int8 = nullptr;  ///< [max_queries, dim] Int8 残差查询
  float *d_query_norms = nullptr;  ///< [max_queries] ||q_residual||²
  float *d_query_scales = nullptr; ///< [max_queries] query scale
  float *d_residuals_fp32 =
      nullptr; ///< [max_queries, dim] FP32 残差 (临时缓冲)

  // GEMM 输出 (Phase 4.1)
  int32_t *d_gemm_output = nullptr; ///< [max_queries, max_cluster_size]

  // 距离输出
  float *d_distances = nullptr; ///< [max_queries, max_cluster_size]

  // Top-K 相关
  int32_t *d_topk_indices = nullptr; ///< [max_queries, topk]
  float *d_topk_distances = nullptr; ///< [max_queries, topk]

  // Pinned memory for results
  int32_t *h_topk_indices_pinned = nullptr;
  float *h_topk_distances_pinned = nullptr;

  // Batch Top-K (packed by query)
  int32_t *d_all_topk_indices = nullptr;
  float *d_all_topk_distances = nullptr;
  int32_t *h_all_topk_indices_pinned = nullptr;
  float *h_all_topk_distances_pinned = nullptr;
  size_t all_topk_capacity_elems = 0; ///< total (query, k) elements

  // Per-workspace GEMM resources
  cublasHandle_t cublas_handle = nullptr;

  // 容量
  int max_queries = 0;
  int max_cluster_size = 0;
  int topk = 0;
  int dim = 0;

  // Batch processing 缓冲区容量 (for Grouped GEMM)
  size_t batch_residuals_fp32_capacity =
      0; ///< d_residuals_fp32 当前容量 (bytes)
  size_t batch_query_int8_capacity = 0;  ///< d_query_int8 当前容量 (bytes)
  size_t batch_query_norms_capacity = 0; ///< d_query_norms 当前容量 (bytes)
  size_t batch_query_scales_capacity = 0; ///< d_query_scales 当前容量 (bytes)
  size_t batch_gemm_output_capacity = 0; ///< d_gemm_output 当前容量 (bytes)
  size_t batch_distances_capacity = 0;   ///< d_distances 当前容量 (bytes)
  size_t batch_topk_indices_capacity = 0; ///< d_topk_indices 当前容量 (bytes)
  size_t batch_topk_distances_capacity =
      0; ///< d_topk_distances 当前容量 (bytes)
  size_t batch_topk_indices_pinned_capacity =
      0; ///< h_topk_indices_pinned 当前容量 (bytes)
  size_t batch_topk_distances_pinned_capacity =
      0; ///< h_topk_distances_pinned 当前容量 (bytes)

  // Grouped GEMM workspace
  void *d_grouped_gemm_workspace = nullptr; ///< Grouped GEMM workspace
  size_t grouped_gemm_workspace_size = 0;
  void *d_grouped_problem_sizes = nullptr; ///< [num_problems] GemmCoord
  void *d_grouped_ptr_A = nullptr;         ///< [num_problems] A pointers
  void *d_grouped_ptr_B = nullptr;         ///< [num_problems] B pointers
  void *d_grouped_ptr_C = nullptr;         ///< [num_problems] C pointers
  void *d_grouped_ptr_D = nullptr;         ///< [num_problems] D pointers
  void *d_grouped_lda = nullptr;           ///< [num_problems] lda
  void *d_grouped_ldb = nullptr;           ///< [num_problems] ldb
  void *d_grouped_ldc = nullptr;           ///< [num_problems] ldc
  void *d_grouped_ldd = nullptr;           ///< [num_problems] ldd
  int grouped_gemm_problem_capacity = 0;

  // Batch processing buffers (host-side, for Grouped GEMM arguments)
  // problem_sizes: [num_problems][3] = {M, N, K}
  std::vector<int> problem_sizes_m, problem_sizes_n, problem_sizes_k;
  std::vector<int64_t> lda_host, ldb_host, ldc_host, ldd_host;

  /// 分配工作空间
  static std::unique_ptr<RSQ8ScorerWorkspace>
  allocate(int max_queries, int max_cluster_size, int topk, int dim);

  /// 释放资源
  ~RSQ8ScorerWorkspace();

  RSQ8ScorerWorkspace() = default;
  RSQ8ScorerWorkspace(const RSQ8ScorerWorkspace &) = delete;
  RSQ8ScorerWorkspace &operator=(const RSQ8ScorerWorkspace &) = delete;
};

class RSQ8Scorer {
public:
  /// 构造函数
  /// @param index GPU 索引
  /// @param mode Epilogue 模式
  RSQ8Scorer(std::shared_ptr<RSQ8GpuIndex> index,
             RSQ8EpilogueMode mode = RSQ8EpilogueMode::TwoStage);

  /// 析构函数
  ~RSQ8Scorer();

  /// 对单个聚类中的所有查询进行评分
  ///
  /// @param task 聚类任务 (包含残差查询)
  /// @param workspace 工作空间
  /// @param stream CUDA 流
  void score_cluster(const ClusterTask &task, RSQ8ScorerWorkspace &workspace,
                     cudaStream_t stream);

  /// 批量对多个聚类进行评分 (CUTLASS Grouped GEMM)
  ///
  /// @param tasks 聚类任务列表
  /// @param workspace 工作空间
  /// @param all_results 输出结果 [batch_size][nprobe * topk]
  /// @param stream CUDA 流
  void score_clusters_batch(const std::vector<ClusterTask> &tasks,
                            RSQ8ScorerWorkspace &workspace,
                            std::vector<std::vector<RSQ8Result>> &all_results,
                            cudaStream_t stream);

  /// 批量评分 (GpuBlock 版本，零 cudaMalloc)
  void score_clusters_batch(const std::vector<ClusterTask> &tasks,
                            RSQ8WorkspaceView &view,
                            std::vector<std::vector<RSQ8Result>> &all_results,
                            cudaStream_t stream);

  /// 计算所需 GpuBlock 大小
  static size_t bytes_per_block(const RSQ8WorkspacePlan &plan) {
    return plan.gpu_bytes_per_block;
  }

  /// 获取 Top-K 结果
  ///
  /// @param num_queries 查询数量
  /// @param topk Top-K 大小
  /// @param workspace 工作空间
  /// @param results 输出结果 [num_queries][topk]
  /// @param stream CUDA 流
  void fetch_topk(int num_queries, int topk, RSQ8ScorerWorkspace &workspace,
                  std::vector<std::vector<RSQ8Result>> &results,
                  cudaStream_t stream);

  /// 获取索引指针
  const RSQ8GpuIndex *index() const { return index_.get(); }

private:
  std::shared_ptr<RSQ8GpuIndex> index_;
  RSQ8EpilogueMode mode_;
};

// ============================================================================
// CUDA Kernels (声明)
// ============================================================================

/// 将 FP32 残差量化为 Int8 并计算 ||residual||² 与 scale
/// @param residuals [num_queries, dim] FP32 输入
/// @param out_int8 [num_queries, dim] Int8 输出
/// @param out_norms [num_queries] ||residual||² 输出
/// @param out_scales [num_queries] query scale 输出
void rsq8_prepare_queries_int8_kernel(const float *residuals, int8_t *out_int8,
                                      float *out_norms, float *out_scales,
                                      int num_queries, int dim,
                                      cudaStream_t stream);

/// Phase 4.1: 计算最终距离
/// dist = ||q||² + ||r_reconstructed||² - 2 * dot(q, r_reconstructed)
///
/// @param gemm_output [M, N] GEMM 输出 (dot products as int32)
/// @param query_norms [M] ||q_residual||²
/// @param query_scales [M] query scale
/// @param precomputed_norms [N] precomputed ||r_reconstructed||² in FP32
/// @param scale cluster scale factor
/// @param distances [M, N] output distances
void rsq8_finalize_distances_kernel(const int32_t *gemm_output,
                                    const float *query_norms,
                                    const float *query_scales,
                                    const float *precomputed_norms, float scale,
                                    float *distances, int M, int N,
                                    cudaStream_t stream);

/// Top-K 选择
/// NOTE: distances 会被原地修改 (标记已选元素) 以减少额外内存开销
void rsq8_topk_kernel(float *distances, const int32_t *global_ids,
                      int num_queries, int num_candidates, int topk,
                      int32_t *out_indices, float *out_distances,
                      cudaStream_t stream);

// ============================================================================
// 批量 CUDA Kernels (优化: 减少 kernel launch 开销)
// ============================================================================

/// 批量 Top-K (单次 kernel launch 处理所有 problems 的所有 queries)
void rsq8_topk_fused_batched_kernel(
    const int32_t *gemm_output_base, const float *query_norms_base,
    const float *query_scales_base, const float *norms_base,
    const int32_t *global_ids_base, const RSQ8TopkProblemDesc *d_descs,
    int num_problems, int total_queries, int max_candidates, int topk,
    int32_t *out_indices, float *out_distances, cudaStream_t stream);

} // namespace fusionann
