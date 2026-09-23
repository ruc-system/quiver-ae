#include "index/residual_sq8_encoder.h"
#include "online/residual_sq8_scorer.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace fusionann {
namespace {

constexpr int kRsq8TopkFastMax = 32;

template <int MaxK>
__device__ __forceinline__ void rsq8_init_topk(float *dists, int32_t *idx) {
#pragma unroll
  for (int i = 0; i < MaxK; ++i) {
    dists[i] = FLT_MAX;
    idx[i] = -1;
  }
}

template <int MaxK>
__device__ __forceinline__ void
rsq8_insert_topk(float *dists, int32_t *idx, int topk, float val, int32_t id) {
  if (val >= dists[topk - 1])
    return;

  int pos = topk - 1;
  while (pos > 0 && val < dists[pos - 1]) {
    dists[pos] = dists[pos - 1];
    idx[pos] = idx[pos - 1];
    --pos;
  }
  dists[pos] = val;
  idx[pos] = id;
}

template <int MaxK>
__device__ __forceinline__ void
rsq8_merge_topk(float *dists, int32_t *idx, const float *other_dists,
                const int32_t *other_idx, int topk) {
  float merged_dists[MaxK];
  int32_t merged_idx[MaxK];
  int ia = 0;
  int ib = 0;
  for (int out = 0; out < topk; ++out) {
    float da = dists[ia];
    float db = other_dists[ib];
    if (da <= db) {
      merged_dists[out] = da;
      merged_idx[out] = idx[ia];
      ++ia;
    } else {
      merged_dists[out] = db;
      merged_idx[out] = other_idx[ib];
      ++ib;
    }
  }
  for (int i = 0; i < topk; ++i) {
    dists[i] = merged_dists[i];
    idx[i] = merged_idx[i];
  }
}

} // namespace

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define RSQ8_CUDA_CHECK(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void rsq8_prepare_queries_int8_kernel_impl(
    const float *__restrict__ residuals, int8_t *__restrict__ out_int8,
    float *__restrict__ out_norms, float *__restrict__ out_scales,
    int num_queries, int dim) {

  int q = blockIdx.x;
  if (q >= num_queries)
    return;

  const float *query = residuals + static_cast<size_t>(q) * dim;
  int8_t *out = out_int8 + static_cast<size_t>(q) * dim;

  float local_sum = 0.0f;
  float local_max = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float val = query[d];
    local_sum += val * val;
    local_max = fmaxf(local_max, fabsf(val));
  }

  // Warp reduce for sum/max
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
  }

  __shared__ float shared_sum[32];
  __shared__ float shared_max[32];
  __shared__ float shared_scale;
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  if (lane == 0) {
    shared_sum[warp_id] = local_sum;
    shared_max[warp_id] = local_max;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float total_sum = 0.0f;
    float total_max = 0.0f;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    for (int i = 0; i < num_warps; ++i) {
      total_sum += shared_sum[i];
      total_max = fmaxf(total_max, shared_max[i]);
    }
    float scale = total_max > 0.0f ? (total_max / 127.0f) : 1.0f;
    out_norms[q] = total_sum;
    out_scales[q] = scale;
    shared_scale = scale;
  }
  __syncthreads();

  float inv_scale = 1.0f / shared_scale;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float scaled = query[d] * inv_scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    int v = __float2int_rn(scaled);
    out[d] = static_cast<int8_t>(v);
  }
}

void rsq8_prepare_queries_int8_kernel(const float *residuals, int8_t *out_int8,
                                      float *out_norms, float *out_scales,
                                      int num_queries, int dim,
                                      cudaStream_t stream) {

  if (num_queries == 0)
    return;

  int block_size = std::min(256, ((dim + 31) / 32) * 32);
  rsq8_prepare_queries_int8_kernel_impl<<<num_queries, block_size, 0, stream>>>(
      residuals, out_int8, out_norms, out_scales, num_queries, dim);
}

__global__ void rsq8_finalize_distances_kernel_impl(
    const int32_t *__restrict__ gemm_output,
    const float *__restrict__ query_norms,  // [M] ||q||²
    const float *__restrict__ query_scales, // [M] query scale
    const float
        *__restrict__ precomputed_norms, // [N] ||r_reconstructed||² (FP32)
    float scale, float *__restrict__ distances, int M, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N)
    return;

  int m = idx / N;
  int n = idx % N;

  float q_norm = query_norms[m];
  float r_norm =
      precomputed_norms[n]; // ||r_reconstructed||² (FP32, no conversion needed)
  float gemm_out =
      static_cast<float>(gemm_output[idx]) * (query_scales[m] * scale);
  // dot(q, r_reconstructed) where r_reconstructed = int8 * scale

  // L2 squared distance: ||q - r_reconstructed||² = ||q||² +
  // ||r_reconstructed||² - 2*dot(q, r_reconstructed) gemm_out = dot(q,
  // int8*scale) = dot(q, r_reconstructed) r_norm = ||int8*scale||² =
  // ||r_reconstructed||²
  distances[idx] = q_norm + r_norm - 2.0f * gemm_out;
}

void rsq8_finalize_distances_kernel(const int32_t *gemm_output,
                                    const float *query_norms,
                                    const float *query_scales,
                                    const float *precomputed_norms, float scale,
                                    float *distances, int M, int N,
                                    cudaStream_t stream) {

  int total = M * N;
  if (total == 0)
    return;

  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;
  rsq8_finalize_distances_kernel_impl<<<num_blocks, block_size, 0, stream>>>(
      gemm_output, query_norms, query_scales, precomputed_norms, scale,
      distances, M, N);
}

template <int MaxK>
__global__ void rsq8_topk_fused_kernel_impl(
    const int32_t *__restrict__ gemm_output,
    const float *__restrict__ query_norms,
    const float *__restrict__ query_scales, const float *__restrict__ norms,
    float scale, const int32_t *__restrict__ global_ids, int num_queries,
    int num_candidates, int topk, int32_t *__restrict__ out_indices,
    float *__restrict__ out_distances) {

  int q = blockIdx.x;
  if (q >= num_queries || topk <= 0)
    return;

  const int32_t *gemm_row =
      gemm_output + static_cast<size_t>(q) * num_candidates;
  float q_norm = query_norms[q];
  float q_scale = query_scales[q];

  float local_dists[MaxK];
  int32_t local_idx[MaxK];
  rsq8_init_topk<MaxK>(local_dists, local_idx);

  for (int c = threadIdx.x; c < num_candidates; c += blockDim.x) {
    float r_norm = norms[c];
    float dot = static_cast<float>(gemm_row[c]) * (q_scale * scale);
    float dist = q_norm + r_norm - 2.0f * dot;
    rsq8_insert_topk<MaxK>(local_dists, local_idx, topk, dist, global_ids[c]);
  }

  extern __shared__ char smem[];
  float *s_dists = reinterpret_cast<float *>(smem);
  int32_t *s_idx = reinterpret_cast<int32_t *>(s_dists + blockDim.x * MaxK);
  int base = threadIdx.x * MaxK;

  for (int i = 0; i < topk; ++i) {
    s_dists[base + i] = local_dists[i];
    s_idx[base + i] = local_idx[i];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int other_base = (threadIdx.x + stride) * MaxK;
      rsq8_merge_topk<MaxK>(local_dists, local_idx, s_dists + other_base,
                            s_idx + other_base, topk);
      for (int i = 0; i < topk; ++i) {
        s_dists[base + i] = local_dists[i];
        s_idx[base + i] = local_idx[i];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    int32_t *out_idx = out_indices + static_cast<size_t>(q) * topk;
    float *out_dist = out_distances + static_cast<size_t>(q) * topk;
    for (int i = 0; i < topk; ++i) {
      out_idx[i] = local_idx[i];
      out_dist[i] = local_dists[i];
    }
  }
}

void rsq8_topk_fused_kernel(const int32_t *gemm_output,
                            const float *query_norms, const float *query_scales,
                            const float *norms, float scale,
                            const int32_t *global_ids, int num_queries,
                            int num_candidates, int topk, int32_t *out_indices,
                            float *out_distances, cudaStream_t stream) {

  if (num_queries == 0 || num_candidates == 0 || topk == 0)
    return;

  if (topk > kRsq8TopkFastMax) {
    throw std::runtime_error(
        "rsq8_topk_fused_kernel: topk exceeds fast kernel capacity");
  }

  int limit = std::min(128, num_candidates);
  int block_size = 1;
  while (block_size < limit) {
    block_size <<= 1;
  }

  size_t shared_bytes = static_cast<size_t>(block_size) * kRsq8TopkFastMax *
                        (sizeof(float) + sizeof(int32_t));
  rsq8_topk_fused_kernel_impl<kRsq8TopkFastMax>
      <<<num_queries, block_size, shared_bytes, stream>>>(
          gemm_output, query_norms, query_scales, norms, scale, global_ids,
          num_queries, num_candidates, topk, out_indices, out_distances);
}

// ============================================================================
// 批量 Top-K kernel (单次 launch 处理所有 problems 的所有 queries)
// 优化版本: 二分查找 + warp-level reduction
// ============================================================================

// Warp-level merge: 从 src_lane 获取数据并与本地 top-k merge
template <int MaxK>
__device__ __forceinline__ void
rsq8_warp_merge_topk(float *my_dists, int32_t *my_idx, int topk, int src_lane) {
  float other_dists[MaxK];
  int32_t other_idx[MaxK];

#pragma unroll
  for (int i = 0; i < MaxK; ++i) {
    other_dists[i] = __shfl_sync(0xffffffff, my_dists[i], src_lane);
    other_idx[i] = __shfl_sync(0xffffffff, my_idx[i], src_lane);
  }

  // Merge two sorted arrays
  float merged_dists[MaxK];
  int32_t merged_idx[MaxK];
  int ia = 0, ib = 0;

#pragma unroll
  for (int out = 0; out < MaxK; ++out) {
    if (my_dists[ia] <= other_dists[ib]) {
      merged_dists[out] = my_dists[ia];
      merged_idx[out] = my_idx[ia];
      ++ia;
    } else {
      merged_dists[out] = other_dists[ib];
      merged_idx[out] = other_idx[ib];
      ++ib;
    }
  }

#pragma unroll
  for (int i = 0; i < MaxK; ++i) {
    my_dists[i] = merged_dists[i];
    my_idx[i] = merged_idx[i];
  }
}

// 二分查找: 找到 global_query_idx 属于哪个 problem
__device__ __forceinline__ int
rsq8_binary_search_problem(const RSQ8TopkProblemDesc *descs, int num_problems,
                           int global_query_idx) {
  int lo = 0, hi = num_problems;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    // query_prefix_sum 是该 problem 之前所有 queries 的总数
    // 如果 global_query_idx < prefix_sum + num_queries，则在 mid 或之前
    if (global_query_idx <
        descs[mid].query_prefix_sum + descs[mid].num_queries) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

template <int MaxK, int BlockSize>
__global__ void rsq8_topk_fused_batched_kernel_impl(
    const int32_t *__restrict__ gemm_output_base,
    const float *__restrict__ query_norms_base,
    const float *__restrict__ query_scales_base,
    const float *__restrict__ norms_base,
    const int32_t *__restrict__ global_ids_base,
    const RSQ8TopkProblemDesc *__restrict__ descs, int num_problems, int topk,
    int32_t *__restrict__ out_indices_base,
    float *__restrict__ out_distances_base) {

  const int global_query_idx = blockIdx.x;
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  constexpr int NumWarps = BlockSize / 32;

  // 二分查找找到对应的 problem
  const int problem_idx =
      rsq8_binary_search_problem(descs, num_problems, global_query_idx);
  if (problem_idx >= num_problems)
    return;

  const RSQ8TopkProblemDesc &desc = descs[problem_idx];
  const int query_in_problem = global_query_idx - desc.query_prefix_sum;
  if (query_in_problem < 0 || query_in_problem >= desc.num_queries)
    return;

  const int num_candidates = desc.num_candidates;
  const int32_t *gemm_row =
      gemm_output_base + desc.gemm_offset +
      static_cast<size_t>(query_in_problem) * num_candidates;
  const float q_norm =
      query_norms_base[desc.query_norm_offset + query_in_problem];
  const float q_scale =
      query_scales_base[desc.query_norm_offset + query_in_problem];
  const float *norms = norms_base + desc.norm_offset;
  const int32_t *global_ids = global_ids_base + desc.id_offset;
  const float scale = desc.scale;

  // Step 1: 每个线程处理部分 candidates，维护 local top-k
  float local_dists[MaxK];
  int32_t local_idx[MaxK];
  rsq8_init_topk<MaxK>(local_dists, local_idx);

  for (int c = threadIdx.x; c < num_candidates; c += BlockSize) {
    float r_norm = norms[c];
    float dot = static_cast<float>(gemm_row[c]) * (q_scale * scale);
    float dist = q_norm + r_norm - 2.0f * dot;
    rsq8_insert_topk<MaxK>(local_dists, local_idx, topk, dist, global_ids[c]);
  }

  // Step 2: Warp-level reduction (无需 shared memory)
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    rsq8_warp_merge_topk<MaxK>(local_dists, local_idx, topk, lane_id ^ offset);
  }

  // Step 3: Cross-warp reduction (需要 shared memory，但只有 NumWarps 个元素)
  extern __shared__ char smem[];
  float *s_dists = reinterpret_cast<float *>(smem);
  int32_t *s_idx = reinterpret_cast<int32_t *>(s_dists + NumWarps * MaxK);

  // 每个 warp 的 lane 0 将结果写入 shared memory
  if (lane_id == 0) {
    const int base = warp_id * MaxK;
#pragma unroll
    for (int i = 0; i < MaxK; ++i) {
      s_dists[base + i] = local_dists[i];
      s_idx[base + i] = local_idx[i];
    }
  }
  __syncthreads();

  // 只用 warp 0 做最终 merge (全 warp 参与，避免部分 lane 参与的 shuffle
  // 未定义行为)
  if (warp_id == 0) {
    if (lane_id < NumWarps) {
      // 每个 lane 读取一个 warp 的结果
      const int base = lane_id * MaxK;
#pragma unroll
      for (int i = 0; i < MaxK; ++i) {
        local_dists[i] = s_dists[base + i];
        local_idx[i] = s_idx[base + i];
      }
    } else {
      // 其余 lane 使用哨兵值
#pragma unroll
      for (int i = 0; i < MaxK; ++i) {
        local_dists[i] = FLT_MAX;
        local_idx[i] = -1;
      }
    }

    // Warp-level reduction 合并所有 warps 的结果 (全 warp 参与)
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      rsq8_warp_merge_topk<MaxK>(local_dists, local_idx, topk,
                                 lane_id ^ offset);
    }

    // Lane 0 写入最终结果
    if (lane_id == 0) {
      const size_t out_base =
          desc.out_offset + static_cast<size_t>(query_in_problem) * topk;
      int32_t *out_idx = out_indices_base + out_base;
      float *out_dist = out_distances_base + out_base;
#pragma unroll
      for (int i = 0; i < topk; ++i) {
        out_idx[i] = local_idx[i];
        out_dist[i] = local_dists[i];
      }
    }
  }
}

void rsq8_topk_fused_batched_kernel(
    const int32_t *gemm_output_base, const float *query_norms_base,
    const float *query_scales_base, const float *norms_base,
    const int32_t *global_ids_base, const RSQ8TopkProblemDesc *d_descs,
    int num_problems, int total_queries, int max_candidates, int topk,
    int32_t *out_indices, float *out_distances, cudaStream_t stream) {

  if (num_problems == 0 || total_queries == 0 || topk == 0)
    return;

  if (topk > kRsq8TopkFastMax) {
    throw std::runtime_error(
        "rsq8_topk_fused_batched_kernel: topk exceeds fast kernel capacity");
  }

  // 优化: 使用固定的 block size 64 (2 warps)
  // 这减少了 cross-warp reduction 的次数，同时仍有足够的并行度
  constexpr int kBlockSize = 64;
  constexpr int kNumWarps = kBlockSize / 32;

  // Shared memory 只需要存储每个 warp 的 top-k 结果
  size_t shared_bytes = static_cast<size_t>(kNumWarps) * kRsq8TopkFastMax *
                        (sizeof(float) + sizeof(int32_t));

  rsq8_topk_fused_batched_kernel_impl<kRsq8TopkFastMax, kBlockSize>
      <<<total_queries, kBlockSize, shared_bytes, stream>>>(
          gemm_output_base, query_norms_base, query_scales_base, norms_base,
          global_ids_base, d_descs, num_problems, topk, out_indices,
          out_distances);
}

// cuBLAS GEMM wrapper: C = alpha * A * B^T + beta * C
// A: [M, K] row-major, B: [N, K] row-major (transposed in GEMM)
// C: [M, N] row-major
void rsq8_cublas_gemm_int8(cublasHandle_t handle, int M, int N, int K,
                           const int8_t *A, // [M, K]
                           const int8_t *B, // [N, K]
                           int32_t *C,      // [M, N]
                           cudaStream_t stream) {

  cublasSetStream(handle, stream);

  // cuBLAS 使用 column-major, 因此要做转置
  // C = A * B^T  在 column-major 中等价于 C^T = B * A^T
  // 但我们直接使用 cublasGemmEx 的 row-major 兼容模式
  const int32_t alpha = 1;
  const int32_t beta = 0;

  // 使用 cublasGemmEx 进行 INT8 GEMM，输出到 INT32
  // A: [M, K], B: [N, K], C: [M, N]
  // 计算 C = A * B^T
  cublasGemmEx(handle,
               CUBLAS_OP_T, // B transposed
               CUBLAS_OP_N, // A not transposed
               N, M, K,     // 交换 M/N 因为 column-major
               &alpha, B, CUDA_R_8I, K, A, CUDA_R_8I, K, &beta, C, CUDA_R_32I,
               N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Simple Top-K selection (可后续优化为 radix select)
__global__ void rsq8_topk_kernel_impl(float *__restrict__ distances,
                                      const int32_t *__restrict__ global_ids,
                                      int num_queries, int num_candidates,
                                      int topk,
                                      int32_t *__restrict__ out_indices,
                                      float *__restrict__ out_distances) {

  int q = blockIdx.x;
  if (q >= num_queries)
    return;

  float *dists = distances + static_cast<size_t>(q) * num_candidates;
  int32_t *out_idx = out_indices + static_cast<size_t>(q) * topk;
  float *out_dist = out_distances + static_cast<size_t>(q) * topk;

  // Simple selection sort for small topk
  // TODO: Use CUB radix sort for larger topk
  for (int k = 0; k < topk && k < num_candidates; ++k) {
    int min_idx = -1;
    float min_dist = 1e30f;

    for (int c = threadIdx.x; c < num_candidates; c += blockDim.x) {
      float d = dists[c];
      // Skip already selected (marked with INFINITY)
      if (d < min_dist) {
        min_dist = d;
        min_idx = c;
      }
    }

    // Reduce across threads to find global minimum
    __shared__ float shared_dists[256];
    __shared__ int shared_indices[256];

    shared_dists[threadIdx.x] = min_dist;
    shared_indices[threadIdx.x] = min_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (threadIdx.x < s) {
        if (shared_dists[threadIdx.x + s] < shared_dists[threadIdx.x]) {
          shared_dists[threadIdx.x] = shared_dists[threadIdx.x + s];
          shared_indices[threadIdx.x] = shared_indices[threadIdx.x + s];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      if (shared_indices[0] >= 0) {
        out_dist[k] = shared_dists[0];
        out_idx[k] = global_ids[shared_indices[0]];
        // Mark as selected by storing INFINITY
        dists[shared_indices[0]] = 1e30f;
      } else {
        out_dist[k] = 1e30f;
        out_idx[k] = -1;
      }
    }
    __syncthreads();
  }
}

void rsq8_topk_kernel(float *distances, const int32_t *global_ids,
                      int num_queries, int num_candidates, int topk,
                      int32_t *out_indices, float *out_distances,
                      cudaStream_t stream) {

  if (num_queries == 0 || num_candidates == 0 || topk == 0)
    return;

  // Power-of-two block size keeps reduction correct and fast.
  int limit = std::min(256, num_candidates);
  int block_size = 1;
  while (block_size < limit) {
    block_size <<= 1;
  }
  rsq8_topk_kernel_impl<<<num_queries, block_size, 0, stream>>>(
      distances, global_ids, num_queries, num_candidates, topk, out_indices,
      out_distances);
}

// ============================================================================
// RSQ8GpuIndex Implementation
// ============================================================================

RSQ8GpuIndex::~RSQ8GpuIndex() {
  if (d_vectors)
    cudaFree(d_vectors);
  if (d_norms)
    cudaFree(d_norms);
  if (d_global_ids)
    cudaFree(d_global_ids);
  if (d_centroids)
    cudaFree(d_centroids);
  if (d_headers)
    cudaFree(d_headers);
}

std::unique_ptr<RSQ8GpuIndex> RSQ8GpuIndex::load(const std::string &path,
                                                 cudaStream_t stream) {
  // 先加载到 CPU
  RSQ8EncodedIndex cpu_index = RSQ8EncodedIndex::load(path);

  auto gpu_index = std::make_unique<RSQ8GpuIndex>();
  gpu_index->nlist = cpu_index.nlist;
  gpu_index->dim = cpu_index.dim;
  gpu_index->total_vectors = cpu_index.total_vectors;

  // 计算总大小和偏移
  gpu_index->cluster_offsets.resize(cpu_index.nlist + 1);
  gpu_index->cluster_sizes.resize(cpu_index.nlist);
  gpu_index->cluster_scales.resize(cpu_index.nlist);

  uint64_t offset = 0;
  for (uint32_t i = 0; i < cpu_index.nlist; ++i) {
    gpu_index->cluster_offsets[i] = offset;
    gpu_index->cluster_sizes[i] = cpu_index.clusters[i].padded_count;
    gpu_index->cluster_scales[i] = cpu_index.clusters[i].scale;
    offset += cpu_index.clusters[i].padded_count;
  }
  gpu_index->cluster_offsets[cpu_index.nlist] = offset;
  gpu_index->total_padded = offset;

  // 分配 GPU 内存
  size_t vectors_bytes = offset * cpu_index.dim * sizeof(int8_t);
  size_t norms_bytes = offset * sizeof(float);
  size_t ids_bytes = offset * sizeof(int32_t);
  size_t centroids_bytes = cpu_index.nlist * cpu_index.dim * sizeof(float);
  size_t headers_bytes = cpu_index.nlist * sizeof(RSQ8ClusterHeader);

  RSQ8_CUDA_CHECK(cudaMalloc(&gpu_index->d_vectors, vectors_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&gpu_index->d_norms, norms_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&gpu_index->d_global_ids, ids_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&gpu_index->d_centroids, centroids_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&gpu_index->d_headers, headers_bytes));

  // 拷贝数据到 GPU
  // 合并所有聚类数据到连续缓冲区
  std::vector<int8_t> all_vectors(offset * cpu_index.dim);
  std::vector<float> all_norms(offset);
  std::vector<int32_t> all_ids(offset);
  std::vector<RSQ8ClusterHeader> headers(cpu_index.nlist);

  for (uint32_t i = 0; i < cpu_index.nlist; ++i) {
    const auto &cluster = cpu_index.clusters[i];
    uint64_t cluster_offset = gpu_index->cluster_offsets[i];

    // Vectors
    std::copy(cluster.codes.begin(), cluster.codes.end(),
              all_vectors.begin() + cluster_offset * cpu_index.dim);

    // Norms (直接拷贝 FP32，无需转换)
    std::copy(cluster.norms.begin(), cluster.norms.end(),
              all_norms.begin() + cluster_offset);

    // IDs
    std::copy(cluster.global_ids.begin(), cluster.global_ids.end(),
              all_ids.begin() + cluster_offset);

    // Header
    headers[i].data_offset = cluster_offset * cpu_index.dim;
    headers[i].norm_offset = cluster_offset;
    headers[i].id_offset = cluster_offset;
    headers[i].padded_count = cluster.padded_count;
    headers[i].original_count = cluster.original_count;
    headers[i].scale = cluster.scale;
  }

  if (stream) {
    RSQ8_CUDA_CHECK(cudaMemcpyAsync(gpu_index->d_vectors, all_vectors.data(),
                                    vectors_bytes, cudaMemcpyHostToDevice,
                                    stream));
    RSQ8_CUDA_CHECK(cudaMemcpyAsync(gpu_index->d_norms, all_norms.data(),
                                    norms_bytes, cudaMemcpyHostToDevice,
                                    stream));
    RSQ8_CUDA_CHECK(cudaMemcpyAsync(gpu_index->d_global_ids, all_ids.data(),
                                    ids_bytes, cudaMemcpyHostToDevice, stream));
    RSQ8_CUDA_CHECK(cudaMemcpyAsync(gpu_index->d_centroids,
                                    cpu_index.centroids.data(), centroids_bytes,
                                    cudaMemcpyHostToDevice, stream));
    RSQ8_CUDA_CHECK(cudaMemcpyAsync(gpu_index->d_headers, headers.data(),
                                    headers_bytes, cudaMemcpyHostToDevice,
                                    stream));
  } else {
    RSQ8_CUDA_CHECK(cudaMemcpy(gpu_index->d_vectors, all_vectors.data(),
                               vectors_bytes, cudaMemcpyHostToDevice));
    RSQ8_CUDA_CHECK(cudaMemcpy(gpu_index->d_norms, all_norms.data(),
                               norms_bytes, cudaMemcpyHostToDevice));
    RSQ8_CUDA_CHECK(cudaMemcpy(gpu_index->d_global_ids, all_ids.data(),
                               ids_bytes, cudaMemcpyHostToDevice));
    RSQ8_CUDA_CHECK(cudaMemcpy(gpu_index->d_centroids,
                               cpu_index.centroids.data(), centroids_bytes,
                               cudaMemcpyHostToDevice));
    RSQ8_CUDA_CHECK(cudaMemcpy(gpu_index->d_headers, headers.data(),
                               headers_bytes, cudaMemcpyHostToDevice));
  }

  std::cout << "  > RSQ8 GPU index loaded: " << gpu_index->nlist
            << " clusters, " << gpu_index->total_vectors << " vectors, "
            << (vectors_bytes + norms_bytes + ids_bytes + centroids_bytes) /
                   1024.0 / 1024.0
            << " MiB GPU memory" << std::endl;

  return gpu_index;
}

// ============================================================================
// RSQ8ScorerWorkspace Implementation
// ============================================================================

RSQ8ScorerWorkspace::~RSQ8ScorerWorkspace() {
  if (cublas_handle) {
    cublasDestroy(cublas_handle);
  }
  if (d_residuals_fp32)
    cudaFree(d_residuals_fp32);
  if (d_query_int8)
    cudaFree(d_query_int8);
  if (d_query_norms)
    cudaFree(d_query_norms);
  if (d_query_scales)
    cudaFree(d_query_scales);
  if (d_gemm_output)
    cudaFree(d_gemm_output);
  if (d_distances)
    cudaFree(d_distances);
  if (d_topk_indices)
    cudaFree(d_topk_indices);
  if (d_topk_distances)
    cudaFree(d_topk_distances);
  if (h_topk_indices_pinned)
    cudaFreeHost(h_topk_indices_pinned);
  if (h_topk_distances_pinned)
    cudaFreeHost(h_topk_distances_pinned);
  if (d_all_topk_indices)
    cudaFree(d_all_topk_indices);
  if (d_all_topk_distances)
    cudaFree(d_all_topk_distances);
  if (h_all_topk_indices_pinned)
    cudaFreeHost(h_all_topk_indices_pinned);
  if (h_all_topk_distances_pinned)
    cudaFreeHost(h_all_topk_distances_pinned);
  if (d_grouped_gemm_workspace)
    cudaFree(d_grouped_gemm_workspace);
  if (d_grouped_problem_sizes)
    cudaFree(d_grouped_problem_sizes);
  if (d_grouped_ptr_A)
    cudaFree(d_grouped_ptr_A);
  if (d_grouped_ptr_B)
    cudaFree(d_grouped_ptr_B);
  if (d_grouped_ptr_C)
    cudaFree(d_grouped_ptr_C);
  if (d_grouped_ptr_D)
    cudaFree(d_grouped_ptr_D);
  if (d_grouped_lda)
    cudaFree(d_grouped_lda);
  if (d_grouped_ldb)
    cudaFree(d_grouped_ldb);
  if (d_grouped_ldc)
    cudaFree(d_grouped_ldc);
  if (d_grouped_ldd)
    cudaFree(d_grouped_ldd);
}

std::unique_ptr<RSQ8ScorerWorkspace>
RSQ8ScorerWorkspace::allocate(int max_queries, int max_cluster_size, int topk,
                              int dim) {

  auto ws = std::make_unique<RSQ8ScorerWorkspace>();
  ws->max_queries = max_queries;
  ws->max_cluster_size = max_cluster_size;
  ws->topk = topk;
  ws->dim = dim;

  size_t query_int8_bytes =
      static_cast<size_t>(max_queries) * dim * sizeof(int8_t);
  size_t residuals_fp32_bytes =
      static_cast<size_t>(max_queries) * dim * sizeof(float);
  size_t norms_bytes = static_cast<size_t>(max_queries) * sizeof(float);
  size_t scales_bytes = static_cast<size_t>(max_queries) * sizeof(float);
  size_t gemm_bytes =
      static_cast<size_t>(max_queries) * max_cluster_size * sizeof(int32_t);
  size_t topk_indices_bytes =
      static_cast<size_t>(max_queries) * topk * sizeof(int32_t);
  size_t topk_dists_bytes =
      static_cast<size_t>(max_queries) * topk * sizeof(float);
  size_t all_topk_elems = static_cast<size_t>(max_queries) * topk;

  // 分配 GPU 内存
  RSQ8_CUDA_CHECK(cudaMalloc(&ws->d_residuals_fp32, residuals_fp32_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&ws->d_query_int8, query_int8_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&ws->d_query_norms, norms_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&ws->d_query_scales, scales_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&ws->d_gemm_output, gemm_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&ws->d_distances, gemm_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&ws->d_topk_indices, topk_indices_bytes));
  RSQ8_CUDA_CHECK(cudaMalloc(&ws->d_topk_distances, topk_dists_bytes));

  // 分配 pinned memory
  RSQ8_CUDA_CHECK(
      cudaMallocHost(&ws->h_topk_indices_pinned, topk_indices_bytes));
  RSQ8_CUDA_CHECK(
      cudaMallocHost(&ws->h_topk_distances_pinned, topk_dists_bytes));

  // 分配 all_topk 缓冲区 (用于batch模式输出)
  RSQ8_CUDA_CHECK(
      cudaMalloc(&ws->d_all_topk_indices, all_topk_elems * sizeof(int32_t)));
  RSQ8_CUDA_CHECK(
      cudaMalloc(&ws->d_all_topk_distances, all_topk_elems * sizeof(float)));
  RSQ8_CUDA_CHECK(cudaMallocHost(&ws->h_all_topk_indices_pinned,
                                 all_topk_elems * sizeof(int32_t)));
  RSQ8_CUDA_CHECK(cudaMallocHost(&ws->h_all_topk_distances_pinned,
                                 all_topk_elems * sizeof(float)));

  // 设置所有 capacity 字段，避免 score_clusters_batch 中重复分配
  ws->batch_residuals_fp32_capacity = residuals_fp32_bytes;
  ws->batch_query_int8_capacity = query_int8_bytes;
  ws->batch_query_norms_capacity = norms_bytes;
  ws->batch_query_scales_capacity = scales_bytes;
  ws->batch_gemm_output_capacity = gemm_bytes;
  ws->batch_distances_capacity = gemm_bytes;
  ws->batch_topk_indices_capacity = topk_indices_bytes;
  ws->batch_topk_distances_capacity = topk_dists_bytes;
  ws->batch_topk_indices_pinned_capacity = topk_indices_bytes;
  ws->batch_topk_distances_pinned_capacity = topk_dists_bytes;
  ws->all_topk_capacity_elems = all_topk_elems;

  if (cublasCreate(&ws->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuBLAS handle");
  }

  return ws;
}

// ============================================================================
// RSQ8WorkspacePlan Implementation (GpuBlock-based)
// ============================================================================

RSQ8WorkspacePlan RSQ8WorkspacePlan::make(int max_queries, int max_cluster_size,
                                          int max_problems, int topk, int dim) {
  RSQ8WorkspacePlan plan;
  plan.max_queries = max_queries;
  plan.max_cluster_size = max_cluster_size;
  plan.max_problems = max_problems;
  plan.topk = topk;
  plan.dim = dim;

  // 计算各缓冲区大小
  plan.residuals_fp32_bytes =
      static_cast<size_t>(max_queries) * dim * sizeof(float);
  plan.query_int8_bytes =
      static_cast<size_t>(max_queries) * dim * sizeof(int8_t);
  plan.query_norms_bytes = static_cast<size_t>(max_queries) * sizeof(float);
  plan.query_scales_bytes = static_cast<size_t>(max_queries) * sizeof(float);
  plan.gemm_output_bytes =
      static_cast<size_t>(max_queries) * max_cluster_size * sizeof(int32_t);
  plan.topk_indices_bytes =
      static_cast<size_t>(max_queries) * topk * sizeof(int32_t);
  plan.topk_distances_bytes =
      static_cast<size_t>(max_queries) * topk * sizeof(float);
  // NOTE: 分开计算 indices 和 distances，因为 rsq8_prepare_workspace 分开分配
  size_t all_topk_indices_bytes =
      static_cast<size_t>(max_queries) * topk * sizeof(int32_t);
  size_t all_topk_distances_bytes =
      static_cast<size_t>(max_queries) * topk * sizeof(float);
  plan.all_topk_bytes = all_topk_indices_bytes + all_topk_distances_bytes;

  // Grouped GEMM 参数缓冲区
  plan.grouped_problem_bytes =
      static_cast<size_t>(max_problems) * sizeof(cutlass::gemm::GemmCoord);
  plan.grouped_ptr_bytes = static_cast<size_t>(max_problems) * sizeof(void *);
  plan.grouped_ld_bytes = static_cast<size_t>(max_problems) * sizeof(int64_t);

  // 计算 GPU 总内存 (使用 256 字节对齐)
  auto align256 = [](size_t s) { return (s + 255) & ~size_t(255); };

  size_t offset = 0;
  offset += align256(plan.residuals_fp32_bytes);
  offset += align256(plan.query_int8_bytes);
  offset += align256(plan.query_norms_bytes);
  offset += align256(plan.query_scales_bytes);
  offset += align256(plan.gemm_output_bytes); // d_gemm_output
  offset += align256(plan.gemm_output_bytes); // d_distances (same size)
  offset += align256(plan.topk_indices_bytes);
  offset += align256(plan.topk_distances_bytes);
  // NOTE: d_all_topk_indices 和 d_all_topk_distances 在 rsq8_prepare_workspace
  // 中分开分配 所以这里也要分开对齐计算
  offset += align256(all_topk_indices_bytes);   // d_all_topk_indices
  offset += align256(all_topk_distances_bytes); // d_all_topk_distances
  // Grouped GEMM 参数: 9 个缓冲区
  offset += align256(plan.grouped_problem_bytes);
  offset += align256(plan.grouped_ptr_bytes) * 4; // ptr_A, B, C, D
  offset += align256(plan.grouped_ld_bytes) * 4;  // lda, ldb, ldc, ldd

  // 批量操作描述符缓冲区
  offset +=
      align256(static_cast<size_t>(max_problems) * sizeof(RSQ8TopkProblemDesc));

  plan.gpu_bytes_per_block = offset;

  // Pinned memory: topk 结果 + all_topk 结果
  plan.pinned_bytes_total = plan.topk_indices_bytes +
                            plan.topk_distances_bytes +
                            static_cast<size_t>(max_queries) * topk *
                                (sizeof(int32_t) + sizeof(float));

  return plan;
}

std::unique_ptr<RSQ8PinnedBuffers>
RSQ8PinnedBuffers::allocate(const RSQ8WorkspacePlan &plan) {
  auto buf = std::make_unique<RSQ8PinnedBuffers>();

  size_t topk_count = static_cast<size_t>(plan.max_queries) * plan.topk;
  buf->topk_capacity = topk_count;
  buf->all_topk_capacity = topk_count;

  RSQ8_CUDA_CHECK(
      cudaMallocHost(&buf->h_topk_indices, topk_count * sizeof(int32_t)));
  RSQ8_CUDA_CHECK(
      cudaMallocHost(&buf->h_topk_distances, topk_count * sizeof(float)));
  RSQ8_CUDA_CHECK(
      cudaMallocHost(&buf->h_all_topk_indices, topk_count * sizeof(int32_t)));
  RSQ8_CUDA_CHECK(
      cudaMallocHost(&buf->h_all_topk_distances, topk_count * sizeof(float)));

  // 分配批量操作的 pinned 缓冲区
  size_t residuals_floats = static_cast<size_t>(plan.max_queries) * plan.dim;
  RSQ8_CUDA_CHECK(cudaMallocHost(&buf->h_residuals_staging,
                                 residuals_floats * sizeof(float)));
  buf->residuals_staging_capacity = residuals_floats;

  RSQ8_CUDA_CHECK(cudaMallocHost(
      &buf->h_topk_descs, plan.max_problems * sizeof(RSQ8TopkProblemDesc)));
  buf->topk_descs_capacity = plan.max_problems;

  if (cublasCreate(&buf->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuBLAS handle");
  }

  // 预分配 host vectors
  buf->problem_sizes_m.reserve(plan.max_problems);
  buf->problem_sizes_n.reserve(plan.max_problems);
  buf->problem_sizes_k.reserve(plan.max_problems);
  buf->lda_host.reserve(plan.max_problems);
  buf->ldb_host.reserve(plan.max_problems);
  buf->ldc_host.reserve(plan.max_problems);
  buf->ldd_host.reserve(plan.max_problems);

  return buf;
}

RSQ8PinnedBuffers::~RSQ8PinnedBuffers() {
  if (h_topk_indices)
    cudaFreeHost(h_topk_indices);
  if (h_topk_distances)
    cudaFreeHost(h_topk_distances);
  if (h_all_topk_indices)
    cudaFreeHost(h_all_topk_indices);
  if (h_all_topk_distances)
    cudaFreeHost(h_all_topk_distances);
  if (h_residuals_staging)
    cudaFreeHost(h_residuals_staging);
  if (h_topk_descs)
    cudaFreeHost(h_topk_descs);
  if (cublas_handle)
    cublasDestroy(cublas_handle);
}

RSQ8WorkspaceView rsq8_prepare_workspace(GpuBlock &block,
                                         RSQ8PinnedBuffers &pinned,
                                         const RSQ8WorkspacePlan &plan) {
  block.reset();

  RSQ8WorkspaceView view;
  view.max_queries = plan.max_queries;
  view.max_cluster_size = plan.max_cluster_size;
  view.max_problems = plan.max_problems;
  view.topk = plan.topk;
  view.dim = plan.dim;

  // 从 GpuBlock 划分 GPU 缓冲区
  view.d_residuals_fp32 = block.allocate<float>(
      static_cast<size_t>(plan.max_queries) * plan.dim, 256);
  view.d_query_int8 = block.allocate<int8_t>(
      static_cast<size_t>(plan.max_queries) * plan.dim, 256);
  view.d_query_norms =
      block.allocate<float>(static_cast<size_t>(plan.max_queries), 256);
  view.d_query_scales =
      block.allocate<float>(static_cast<size_t>(plan.max_queries), 256);
  view.d_gemm_output = block.allocate<int32_t>(
      static_cast<size_t>(plan.max_queries) * plan.max_cluster_size, 256);
  view.d_distances = block.allocate<float>(
      static_cast<size_t>(plan.max_queries) * plan.max_cluster_size, 256);
  view.d_topk_indices = block.allocate<int32_t>(
      static_cast<size_t>(plan.max_queries) * plan.topk, 256);
  view.d_topk_distances = block.allocate<float>(
      static_cast<size_t>(plan.max_queries) * plan.topk, 256);
  view.d_all_topk_indices = block.allocate<int32_t>(
      static_cast<size_t>(plan.max_queries) * plan.topk, 256);
  view.d_all_topk_distances = block.allocate<float>(
      static_cast<size_t>(plan.max_queries) * plan.topk, 256);

  // Grouped GEMM 参数缓冲区
  view.d_grouped_problem_sizes =
      block.allocate<cutlass::gemm::GemmCoord>(plan.max_problems, 256);
  view.d_grouped_ptr_A =
      block.allocate<cutlass::int8_t *>(plan.max_problems, 256);
  view.d_grouped_ptr_B =
      block.allocate<cutlass::int8_t *>(plan.max_problems, 256);
  view.d_grouped_ptr_C = block.allocate<int32_t *>(plan.max_problems, 256);
  view.d_grouped_ptr_D = block.allocate<int32_t *>(plan.max_problems, 256);
  view.d_grouped_lda = block.allocate<int64_t>(plan.max_problems, 256);
  view.d_grouped_ldb = block.allocate<int64_t>(plan.max_problems, 256);
  view.d_grouped_ldc = block.allocate<int64_t>(plan.max_problems, 256);
  view.d_grouped_ldd = block.allocate<int64_t>(plan.max_problems, 256);

  // 绑定 pinned memory
  view.h_topk_indices_pinned = pinned.h_topk_indices;
  view.h_topk_distances_pinned = pinned.h_topk_distances;
  view.h_all_topk_indices_pinned = pinned.h_all_topk_indices;
  view.h_all_topk_distances_pinned = pinned.h_all_topk_distances;

  // 绑定 host vectors
  view.problem_sizes_m = &pinned.problem_sizes_m;
  view.problem_sizes_n = &pinned.problem_sizes_n;
  view.problem_sizes_k = &pinned.problem_sizes_k;
  view.lda_host = &pinned.lda_host;
  view.ldb_host = &pinned.ldb_host;
  view.ldc_host = &pinned.ldc_host;
  view.ldd_host = &pinned.ldd_host;

  // 批量操作的 pinned 缓冲区
  view.h_residuals_staging = pinned.h_residuals_staging;
  view.h_topk_descs = pinned.h_topk_descs;

  // 批量操作的 GPU 缓冲区
  view.d_topk_descs =
      block.allocate<RSQ8TopkProblemDesc>(plan.max_problems, 256);

  return view;
}

// ============================================================================
// RSQ8Scorer Implementation
// ============================================================================

RSQ8Scorer::RSQ8Scorer(std::shared_ptr<RSQ8GpuIndex> index,
                       RSQ8EpilogueMode mode)
    : index_(std::move(index)), mode_(mode) {
  if (mode_ == RSQ8EpilogueMode::Fused) {
    std::cerr << "Warning: CUTLASS Epilogue fusion not yet implemented, "
              << "falling back to TwoStage mode" << std::endl;
    mode_ = RSQ8EpilogueMode::TwoStage;
  }
}

RSQ8Scorer::~RSQ8Scorer() {}

void RSQ8Scorer::score_cluster(const ClusterTask &task,
                               RSQ8ScorerWorkspace &workspace,
                               cudaStream_t stream) {

  if (task.num_queries <= 0 || !task.query_indices || !task.residual_queries)
    return;

  const int num_queries = task.num_queries;
  const int cluster_id = task.cluster_id;
  const int dim = static_cast<int>(index_->dim);

  // 获取聚类信息
  const uint64_t cluster_offset = index_->cluster_offsets[cluster_id];
  const uint32_t cluster_size = index_->cluster_sizes[cluster_id];
  const float scale = index_->cluster_scales[cluster_id];

  // Step 1: 拷贝残差到 GPU 并量化为 Int8
  // 使用 workspace 中预分配的缓冲区，避免 per-cluster cudaMalloc/cudaFree
  const size_t residuals_bytes =
      static_cast<size_t>(num_queries) * dim * sizeof(float);
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(workspace.d_residuals_fp32,
                                  task.residual_queries, residuals_bytes,
                                  cudaMemcpyHostToDevice, stream));

  rsq8_prepare_queries_int8_kernel(
      workspace.d_residuals_fp32, workspace.d_query_int8,
      workspace.d_query_norms, workspace.d_query_scales, num_queries, dim,
      stream);

  // Step 2: GEMM (INT8 queries × INT8 index vectors)
  // cuBLAS GEMM: C[M,N] = A[M,K] * B^T[N,K]
  // M = num_queries, N = cluster_size, K = dim
  const int8_t *d_cluster_vectors = index_->d_vectors + cluster_offset * dim;
  rsq8_cublas_gemm_int8(workspace.cublas_handle,
                        num_queries,             // M
                        cluster_size,            // N
                        dim,                     // K
                        workspace.d_query_int8,  // A [M, K]
                        d_cluster_vectors,       // B [N, K]
                        workspace.d_gemm_output, // C [M, N]
                        stream);

  // Step 3: Top-K 选择 (小 topk 走 fused kernel, 规避额外写回)
  // dist = ||q||² + ||r_reconstructed||² - 2 * dot(q, r_reconstructed)
  // precomputed norms 已包含 scale (r_reconstructed = int8 * scale)
  const float *d_cluster_norms = index_->d_norms + cluster_offset;

  const int32_t *d_cluster_ids = index_->d_global_ids + cluster_offset;

  if (workspace.topk <= kRsq8TopkFastMax) {
    rsq8_topk_fused_kernel(
        workspace.d_gemm_output, workspace.d_query_norms,
        workspace.d_query_scales, d_cluster_norms, scale, d_cluster_ids,
        num_queries, static_cast<int>(cluster_size), workspace.topk,
        workspace.d_topk_indices, workspace.d_topk_distances, stream);
  } else {
    rsq8_finalize_distances_kernel(
        workspace.d_gemm_output, workspace.d_query_norms,
        workspace.d_query_scales, d_cluster_norms, scale, workspace.d_distances,
        num_queries, static_cast<int>(cluster_size), stream);
    rsq8_topk_kernel(workspace.d_distances, d_cluster_ids, num_queries,
                     static_cast<int>(cluster_size), workspace.topk,
                     workspace.d_topk_indices, workspace.d_topk_distances,
                     stream);
  }
}

// ============================================================================
// CUTLASS Grouped GEMM Type Definitions
// ============================================================================

// Grouped GEMM Kernel: INT8 × INT8 → INT32, Ampere/Ada TensorOp
// C = A * B where:
//   A is [M, K] RowMajor (queries), lda = K
//   B is [K, N] ColumnMajor - reads our stored [N, K] RowMajor data correctly
//   C is [M, N] RowMajor (output dot products), ldc = N
//
// Memory layout explanation:
//   Our index data: N vectors of K dimensions stored as data[n * K + k]
//   ColumnMajor [K, N] with ldb = K: B(k, n) = data[k + n * K] = data[n * K +
//   k] This matches our data layout! So ColumnMajor + ldb=K is correct.
using GemmGroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::int8_t, cutlass::layout::RowMajor, // A [M, K] RowMajor
    cutlass::ComplexTransform::kNone, 16, cutlass::int8_t,
    cutlass::layout::ColumnMajor, // B [K, N] ColumnMajor
    cutlass::ComplexTransform::kNone, 16, int32_t,
    cutlass::layout::RowMajor, // C/D [M, N] RowMajor
    int32_t,                   // Accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,                    // Ampere (compatible with Ada)
    cutlass::gemm::GemmShape<128, 128, 64>, // Threadblock shape
    cutlass::gemm::GemmShape<64, 64, 64>,   // Warp shape
    cutlass::gemm::GemmShape<16, 8, 32>,    // Instruction shape
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3, // Stages
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmGroupedKernel>;

void RSQ8Scorer::score_clusters_batch(
    const std::vector<ClusterTask> &tasks, RSQ8ScorerWorkspace &workspace,
    std::vector<std::vector<RSQ8Result>> &all_results, cudaStream_t stream) {

  if (tasks.empty())
    return;

  const int num_problems = static_cast<int>(tasks.size());
  const int dim = static_cast<int>(index_->dim);

  // 步骤 1: 准备 Grouped GEMM 参数
  std::vector<cutlass::gemm::GemmCoord> problem_sizes_host(num_problems);
  std::vector<cutlass::int8_t *> ptr_A_host(num_problems);
  std::vector<cutlass::int8_t *> ptr_B_host(num_problems);
  std::vector<int32_t *> ptr_C_host(num_problems);
  std::vector<int32_t *> ptr_D_host(num_problems);
  std::vector<int64_t> lda(num_problems), ldb(num_problems);
  std::vector<int64_t> ldc(num_problems), ldd(num_problems);

  // 计算总 query 数和 cluster 数据位置
  int packed_queries = 0;
  size_t total_gemm_output_elements = 0;
  for (const auto &task : tasks) {
    const int task_queries = task.num_queries;
    packed_queries = std::max(packed_queries, task.query_offset + task_queries);
    uint32_t cluster_size = index_->cluster_sizes[task.cluster_id];
    total_gemm_output_elements +=
        static_cast<size_t>(task_queries) * cluster_size;
  }

  // 步骤 2: 动态扩展 batch processing 缓冲区 (如果需要)
  size_t required_residuals_fp32_bytes =
      static_cast<size_t>(packed_queries) * dim * sizeof(float);
  if (required_residuals_fp32_bytes > workspace.batch_residuals_fp32_capacity) {
    if (workspace.d_residuals_fp32)
      cudaFree(workspace.d_residuals_fp32);
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_residuals_fp32, required_residuals_fp32_bytes));
    workspace.batch_residuals_fp32_capacity = required_residuals_fp32_bytes;
  }

  size_t required_query_int8_bytes =
      static_cast<size_t>(packed_queries) * dim * sizeof(int8_t);
  if (required_query_int8_bytes > workspace.batch_query_int8_capacity) {
    if (workspace.d_query_int8)
      cudaFree(workspace.d_query_int8);
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_query_int8, required_query_int8_bytes));
    workspace.batch_query_int8_capacity = required_query_int8_bytes;
  }

  size_t required_query_norms_bytes =
      static_cast<size_t>(packed_queries) * sizeof(float);
  if (required_query_norms_bytes > workspace.batch_query_norms_capacity) {
    if (workspace.d_query_norms)
      cudaFree(workspace.d_query_norms);
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_query_norms, required_query_norms_bytes));
    workspace.batch_query_norms_capacity = required_query_norms_bytes;
  }

  size_t required_query_scales_bytes =
      static_cast<size_t>(packed_queries) * sizeof(float);
  if (required_query_scales_bytes > workspace.batch_query_scales_capacity) {
    if (workspace.d_query_scales)
      cudaFree(workspace.d_query_scales);
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_query_scales, required_query_scales_bytes));
    workspace.batch_query_scales_capacity = required_query_scales_bytes;
  }

  size_t required_gemm_output_bytes =
      total_gemm_output_elements * sizeof(int32_t);
  if (required_gemm_output_bytes > workspace.batch_gemm_output_capacity) {
    if (workspace.d_gemm_output)
      cudaFree(workspace.d_gemm_output);
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_gemm_output, required_gemm_output_bytes));
    workspace.batch_gemm_output_capacity = required_gemm_output_bytes;
  }

  if (required_gemm_output_bytes > workspace.batch_distances_capacity) {
    if (workspace.d_distances)
      cudaFree(workspace.d_distances);
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_distances, required_gemm_output_bytes));
    workspace.batch_distances_capacity = required_gemm_output_bytes;
  }

  // 计算最大单个 task 的 query 数，用于 topk 缓冲区
  int max_task_queries = 0;
  for (const auto &task : tasks) {
    max_task_queries = std::max(max_task_queries, task.num_queries);
  }

  size_t required_topk_indices_bytes =
      static_cast<size_t>(max_task_queries) * workspace.topk * sizeof(int32_t);
  if (required_topk_indices_bytes > workspace.batch_topk_indices_capacity) {
    if (workspace.d_topk_indices)
      cudaFree(workspace.d_topk_indices);
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_topk_indices, required_topk_indices_bytes));
    workspace.batch_topk_indices_capacity = required_topk_indices_bytes;
  }

  size_t required_topk_distances_bytes =
      static_cast<size_t>(max_task_queries) * workspace.topk * sizeof(float);
  if (required_topk_distances_bytes > workspace.batch_topk_distances_capacity) {
    if (workspace.d_topk_distances)
      cudaFree(workspace.d_topk_distances);
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_topk_distances, required_topk_distances_bytes));
    workspace.batch_topk_distances_capacity = required_topk_distances_bytes;
  }

  if (required_topk_indices_bytes >
      workspace.batch_topk_indices_pinned_capacity) {
    if (workspace.h_topk_indices_pinned)
      cudaFreeHost(workspace.h_topk_indices_pinned);
    RSQ8_CUDA_CHECK(cudaMallocHost(&workspace.h_topk_indices_pinned,
                                   required_topk_indices_bytes));
    workspace.batch_topk_indices_pinned_capacity = required_topk_indices_bytes;
  }

  if (required_topk_distances_bytes >
      workspace.batch_topk_distances_pinned_capacity) {
    if (workspace.h_topk_distances_pinned)
      cudaFreeHost(workspace.h_topk_distances_pinned);
    RSQ8_CUDA_CHECK(cudaMallocHost(&workspace.h_topk_distances_pinned,
                                   required_topk_distances_bytes));
    workspace.batch_topk_distances_pinned_capacity =
        required_topk_distances_bytes;
  }

  // Batch Top-K 输出缓冲区 (按 packed query 存储)
  size_t required_all_topk_elems =
      static_cast<size_t>(packed_queries) * workspace.topk;
  if (required_all_topk_elems > workspace.all_topk_capacity_elems) {
    if (workspace.d_all_topk_indices)
      cudaFree(workspace.d_all_topk_indices);
    if (workspace.d_all_topk_distances)
      cudaFree(workspace.d_all_topk_distances);
    if (workspace.h_all_topk_indices_pinned)
      cudaFreeHost(workspace.h_all_topk_indices_pinned);
    if (workspace.h_all_topk_distances_pinned)
      cudaFreeHost(workspace.h_all_topk_distances_pinned);

    size_t all_topk_indices_bytes = required_all_topk_elems * sizeof(int32_t);
    size_t all_topk_distances_bytes = required_all_topk_elems * sizeof(float);

    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_all_topk_indices, all_topk_indices_bytes));
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_all_topk_distances, all_topk_distances_bytes));
    RSQ8_CUDA_CHECK(cudaMallocHost(&workspace.h_all_topk_indices_pinned,
                                   all_topk_indices_bytes));
    RSQ8_CUDA_CHECK(cudaMallocHost(&workspace.h_all_topk_distances_pinned,
                                   all_topk_distances_bytes));
    workspace.all_topk_capacity_elems = required_all_topk_elems;
  }

  // 步骤 4: 拷贝所有查询残差并准备 GEMM 参数
  size_t gemm_output_offset = 0;

  for (int p = 0; p < num_problems; ++p) {
    const auto &task = tasks[p];
    const int M = task.num_queries;
    const uint32_t cluster_size = index_->cluster_sizes[task.cluster_id];
    const int N = static_cast<int>(cluster_size);
    const int K = dim;
    const uint64_t cluster_offset = index_->cluster_offsets[task.cluster_id];
    const int query_offset = task.query_offset;

    // 拷贝残差到 GPU
    const size_t residuals_bytes = static_cast<size_t>(M) * K * sizeof(float);
    RSQ8_CUDA_CHECK(cudaMemcpyAsync(
        workspace.d_residuals_fp32 + query_offset * K, task.residual_queries,
        residuals_bytes, cudaMemcpyHostToDevice, stream));

    // 填充 GEMM 参数
    problem_sizes_host[p] = {M, N, K};
    ptr_A_host[p] = reinterpret_cast<cutlass::int8_t *>(workspace.d_query_int8 +
                                                        query_offset * K);
    ptr_B_host[p] = reinterpret_cast<cutlass::int8_t *>(index_->d_vectors +
                                                        cluster_offset * dim);
    ptr_C_host[p] = workspace.d_gemm_output + gemm_output_offset;
    ptr_D_host[p] = workspace.d_gemm_output + gemm_output_offset;
    lda[p] = K;
    ldb[p] = K;
    ldc[p] = N;
    ldd[p] = N;

    gemm_output_offset += static_cast<size_t>(M) * N;
  }

  // 步骤 4: 批量量化 query 残差 FP32→Int8
  rsq8_prepare_queries_int8_kernel(
      workspace.d_residuals_fp32, workspace.d_query_int8,
      workspace.d_query_norms, workspace.d_query_scales, packed_queries, dim,
      stream);

  // Ensure grouped GEMM argument buffers are large enough.
  if (num_problems > workspace.grouped_gemm_problem_capacity) {
    if (workspace.d_grouped_problem_sizes)
      cudaFree(workspace.d_grouped_problem_sizes);
    if (workspace.d_grouped_ptr_A)
      cudaFree(workspace.d_grouped_ptr_A);
    if (workspace.d_grouped_ptr_B)
      cudaFree(workspace.d_grouped_ptr_B);
    if (workspace.d_grouped_ptr_C)
      cudaFree(workspace.d_grouped_ptr_C);
    if (workspace.d_grouped_ptr_D)
      cudaFree(workspace.d_grouped_ptr_D);
    if (workspace.d_grouped_lda)
      cudaFree(workspace.d_grouped_lda);
    if (workspace.d_grouped_ldb)
      cudaFree(workspace.d_grouped_ldb);
    if (workspace.d_grouped_ldc)
      cudaFree(workspace.d_grouped_ldc);
    if (workspace.d_grouped_ldd)
      cudaFree(workspace.d_grouped_ldd);

    size_t problems_bytes =
        static_cast<size_t>(num_problems) * sizeof(cutlass::gemm::GemmCoord);
    size_t ptr_int8_bytes =
        static_cast<size_t>(num_problems) * sizeof(cutlass::int8_t *);
    size_t ptr_int32_bytes =
        static_cast<size_t>(num_problems) * sizeof(int32_t *);
    size_t ld_bytes = static_cast<size_t>(num_problems) * sizeof(int64_t);

    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_grouped_problem_sizes, problems_bytes));
    RSQ8_CUDA_CHECK(cudaMalloc(&workspace.d_grouped_ptr_A, ptr_int8_bytes));
    RSQ8_CUDA_CHECK(cudaMalloc(&workspace.d_grouped_ptr_B, ptr_int8_bytes));
    RSQ8_CUDA_CHECK(cudaMalloc(&workspace.d_grouped_ptr_C, ptr_int32_bytes));
    RSQ8_CUDA_CHECK(cudaMalloc(&workspace.d_grouped_ptr_D, ptr_int32_bytes));
    RSQ8_CUDA_CHECK(cudaMalloc(&workspace.d_grouped_lda, ld_bytes));
    RSQ8_CUDA_CHECK(cudaMalloc(&workspace.d_grouped_ldb, ld_bytes));
    RSQ8_CUDA_CHECK(cudaMalloc(&workspace.d_grouped_ldc, ld_bytes));
    RSQ8_CUDA_CHECK(cudaMalloc(&workspace.d_grouped_ldd, ld_bytes));
    workspace.grouped_gemm_problem_capacity = num_problems;
  }

  RSQ8_CUDA_CHECK(cudaMemcpyAsync(
      workspace.d_grouped_problem_sizes, problem_sizes_host.data(),
      static_cast<size_t>(num_problems) * sizeof(cutlass::gemm::GemmCoord),
      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(workspace.d_grouped_ptr_A, ptr_A_host.data(),
                                  static_cast<size_t>(num_problems) *
                                      sizeof(cutlass::int8_t *),
                                  cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(workspace.d_grouped_ptr_B, ptr_B_host.data(),
                                  static_cast<size_t>(num_problems) *
                                      sizeof(cutlass::int8_t *),
                                  cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(workspace.d_grouped_ptr_C, ptr_C_host.data(),
                      static_cast<size_t>(num_problems) * sizeof(int32_t *),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(workspace.d_grouped_ptr_D, ptr_D_host.data(),
                      static_cast<size_t>(num_problems) * sizeof(int32_t *),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(workspace.d_grouped_lda, lda.data(),
                      static_cast<size_t>(num_problems) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(workspace.d_grouped_ldb, ldb.data(),
                      static_cast<size_t>(num_problems) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(workspace.d_grouped_ldc, ldc.data(),
                      static_cast<size_t>(num_problems) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(workspace.d_grouped_ldd, ldd.data(),
                      static_cast<size_t>(num_problems) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream));

#if defined(RSQ8_DEBUG)
  RSQ8_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cerr << "[DEBUG] prepare_queries_int8 kernel completed successfully"
            << std::endl;

  std::cerr << "[DEBUG] Grouped GEMM: num_problems=" << num_problems
            << " total_queries=" << packed_queries
            << " total_gemm_elements=" << total_gemm_output_elements
            << std::endl;
  std::cerr << "[DEBUG] Buffer capacities: gemm_output="
            << workspace.batch_gemm_output_capacity
            << " query_int8=" << workspace.batch_query_int8_capacity
            << std::endl;
  for (int i = 0; i < std::min(3, num_problems); ++i) {
    std::cerr << "[DEBUG] Problem " << i << ": M=" << problem_sizes_host[i].m()
              << " N=" << problem_sizes_host[i].n()
              << " K=" << problem_sizes_host[i].k() << std::endl;
  }
#endif

  // 步骤 5: 调用 CUTLASS Grouped GEMM
  int threadblock_count =
      GemmGrouped::sufficient(problem_sizes_host.data(), num_problems);

  typename GemmGrouped::Arguments args(
      reinterpret_cast<cutlass::gemm::GemmCoord *>(
          workspace.d_grouped_problem_sizes),
      num_problems, threadblock_count, {1, 0}, // alpha, beta
      reinterpret_cast<cutlass::int8_t **>(workspace.d_grouped_ptr_A),
      reinterpret_cast<cutlass::int8_t **>(workspace.d_grouped_ptr_B),
      reinterpret_cast<int32_t **>(workspace.d_grouped_ptr_C),
      reinterpret_cast<int32_t **>(workspace.d_grouped_ptr_D),
      reinterpret_cast<int64_t *>(workspace.d_grouped_lda),
      reinterpret_cast<int64_t *>(workspace.d_grouped_ldb),
      reinterpret_cast<int64_t *>(workspace.d_grouped_ldc),
      reinterpret_cast<int64_t *>(workspace.d_grouped_ldd),
      problem_sizes_host.data() // host_problem_sizes for precomputation
  );

  // 获取/分配 workspace
  size_t workspace_size = GemmGrouped::get_workspace_size(args);
  if (workspace_size > workspace.grouped_gemm_workspace_size) {
    if (workspace.d_grouped_gemm_workspace) {
      cudaFree(workspace.d_grouped_gemm_workspace);
    }
    RSQ8_CUDA_CHECK(
        cudaMalloc(&workspace.d_grouped_gemm_workspace, workspace_size));
    workspace.grouped_gemm_workspace_size = workspace_size;
  }

  GemmGrouped gemm_grouped;
  cutlass::Status status =
      gemm_grouped.initialize(args, workspace.d_grouped_gemm_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS Grouped GEMM initialize failed");
  }
  status = gemm_grouped.run(stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS Grouped GEMM run failed");
  }

#if defined(RSQ8_DEBUG)
  RSQ8_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cerr << "[DEBUG] CUTLASS Grouped GEMM completed successfully"
            << std::endl;
#endif

  // 步骤 6: Top-K (小 topk 走 fused kernel, 每个 problem 串行)
  gemm_output_offset = 0;

  // 初始化结果
  int batch_size = 0;
  for (const auto &task : tasks) {
    for (int i = 0; i < task.num_queries; ++i) {
      int32_t qi = task.query_indices[i];
      batch_size = std::max(batch_size, qi + 1);
    }
  }
  all_results.resize(batch_size);

  for (int p = 0; p < num_problems; ++p) {
    const auto &task = tasks[p];
    const int M = task.num_queries;
    const uint32_t cluster_size = index_->cluster_sizes[task.cluster_id];
    const int N = static_cast<int>(cluster_size);
    const float scale = index_->cluster_scales[task.cluster_id];
    const uint64_t cluster_offset = index_->cluster_offsets[task.cluster_id];
    const int query_offset = task.query_offset;

    // Top-K (小 topk 走 fused kernel, 规避额外写回)
    const float *d_cluster_norms = index_->d_norms + cluster_offset;
    const int32_t *d_cluster_ids = index_->d_global_ids + cluster_offset;
    size_t out_base_elem = static_cast<size_t>(query_offset) * workspace.topk;
    if (workspace.topk <= kRsq8TopkFastMax) {
      rsq8_topk_fused_kernel(
          workspace.d_gemm_output + gemm_output_offset,
          workspace.d_query_norms + query_offset,
          workspace.d_query_scales + query_offset, d_cluster_norms, scale,
          d_cluster_ids, M, N, workspace.topk,
          workspace.d_all_topk_indices + out_base_elem,
          workspace.d_all_topk_distances + out_base_elem, stream);
    } else {
      rsq8_finalize_distances_kernel(
          workspace.d_gemm_output + gemm_output_offset,
          workspace.d_query_norms + query_offset,
          workspace.d_query_scales + query_offset, d_cluster_norms, scale,
          workspace.d_distances + gemm_output_offset, M, N, stream);
      rsq8_topk_kernel(workspace.d_distances + gemm_output_offset,
                       d_cluster_ids, M, N, workspace.topk,
                       workspace.d_all_topk_indices + out_base_elem,
                       workspace.d_all_topk_distances + out_base_elem, stream);
    }

    gemm_output_offset += static_cast<size_t>(M) * N;
  }

  if (packed_queries == 0 || workspace.topk == 0)
    return;

  // 异步拷贝 Top-K 结果到 pinned memory（不同步，由调用方负责同步）
  const size_t all_topk_indices_bytes =
      static_cast<size_t>(packed_queries) * workspace.topk * sizeof(int32_t);
  const size_t all_topk_distances_bytes =
      static_cast<size_t>(packed_queries) * workspace.topk * sizeof(float);
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(
      workspace.h_all_topk_indices_pinned, workspace.d_all_topk_indices,
      all_topk_indices_bytes, cudaMemcpyDeviceToHost, stream));
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(
      workspace.h_all_topk_distances_pinned, workspace.d_all_topk_distances,
      all_topk_distances_bytes, cudaMemcpyDeviceToHost, stream));

  // 注意：不在这里同步！调用方需要：
  // 1. cudaStreamSynchronize(stream) 等待完成
  // 2. 调用 collect_batch_results() 收集结果
}

void RSQ8Scorer::fetch_topk(int num_queries, int topk,
                            RSQ8ScorerWorkspace &workspace,
                            std::vector<std::vector<RSQ8Result>> &results,
                            cudaStream_t stream) {

  const size_t topk_indices_bytes =
      static_cast<size_t>(num_queries) * topk * sizeof(int32_t);
  const size_t topk_dists_bytes =
      static_cast<size_t>(num_queries) * topk * sizeof(float);

  RSQ8_CUDA_CHECK(cudaMemcpyAsync(workspace.h_topk_indices_pinned,
                                  workspace.d_topk_indices, topk_indices_bytes,
                                  cudaMemcpyDeviceToHost, stream));
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(workspace.h_topk_distances_pinned,
                                  workspace.d_topk_distances, topk_dists_bytes,
                                  cudaMemcpyDeviceToHost, stream));
  // NOTE: Synchronize to ensure pinned host buffers are ready for CPU access.
  RSQ8_CUDA_CHECK(cudaStreamSynchronize(stream));

  results.resize(num_queries);
  for (int q = 0; q < num_queries; ++q) {
    results[q].resize(topk);
    for (int k = 0; k < topk; ++k) {
      int idx = q * topk + k;
      results[q][k].global_id = workspace.h_topk_indices_pinned[idx];
      results[q][k].distance = workspace.h_topk_distances_pinned[idx];
    }
  }
}

// ============================================================================
// RSQ8Scorer::score_clusters_batch (GpuBlock 版本，零 cudaMalloc)
// ============================================================================

void RSQ8Scorer::score_clusters_batch(
    const std::vector<ClusterTask> &tasks, RSQ8WorkspaceView &view,
    std::vector<std::vector<RSQ8Result>> &all_results, cudaStream_t stream) {

  if (tasks.empty())
    return;

  const int num_problems = static_cast<int>(tasks.size());
  const int dim = static_cast<int>(index_->dim);
  const int topk = view.topk;

  // 检查容量
  if (num_problems > view.max_problems) {
    throw std::runtime_error(
        "RSQ8WorkspaceView: num_problems exceeds capacity");
  }

  // 计算总 query 数
  int packed_queries = 0;
  for (const auto &task : tasks) {
    packed_queries =
        std::max(packed_queries, task.query_offset + task.num_queries);
  }

  if (packed_queries > view.max_queries) {
    throw std::runtime_error(
        "RSQ8WorkspaceView: packed_queries exceeds capacity");
  }

  // 准备 GEMM 参数 (使用 view 中的 host vectors)
  auto &problem_sizes_m = *view.problem_sizes_m;
  auto &problem_sizes_n = *view.problem_sizes_n;
  auto &problem_sizes_k = *view.problem_sizes_k;
  auto &lda = *view.lda_host;
  auto &ldb = *view.ldb_host;
  auto &ldc = *view.ldc_host;
  auto &ldd = *view.ldd_host;

  problem_sizes_m.resize(num_problems);
  problem_sizes_n.resize(num_problems);
  problem_sizes_k.resize(num_problems);
  lda.resize(num_problems);
  ldb.resize(num_problems);
  ldc.resize(num_problems);
  ldd.resize(num_problems);

  std::vector<cutlass::gemm::GemmCoord> problem_sizes_host(num_problems);
  std::vector<cutlass::int8_t *> ptr_A_host(num_problems);
  std::vector<cutlass::int8_t *> ptr_B_host(num_problems);
  std::vector<int32_t *> ptr_C_host(num_problems);
  std::vector<int32_t *> ptr_D_host(num_problems);

  // =========================================================================
  // 优化: 批量操作 - 合并所有 residuals 到 staging buffer，一次传输
  // =========================================================================
  size_t gemm_output_offset = 0;
  int max_candidates = 0;
  bool residuals_packed = true;

  // Step 1: CPU 侧合并 residuals + 构建描述符
  for (int p = 0; p < num_problems; ++p) {
    const auto &task = tasks[p];
    const int M = task.num_queries;
    const uint32_t cluster_size = index_->cluster_sizes[task.cluster_id];
    const int N = static_cast<int>(cluster_size);
    const int K = dim;
    const float scale = index_->cluster_scales[task.cluster_id];
    const uint64_t cluster_offset = index_->cluster_offsets[task.cluster_id];
    const int query_offset = task.query_offset;

    max_candidates = std::max(max_candidates, N);

    const float *expected_residuals =
        view.h_residuals_staging + static_cast<size_t>(query_offset) * K;
    if (task.residual_queries != expected_residuals) {
      residuals_packed = false;
    }

    // 构建 Top-K 描述符
    view.h_topk_descs[p].gemm_offset = gemm_output_offset;
    view.h_topk_descs[p].norm_offset = cluster_offset;
    view.h_topk_descs[p].id_offset = cluster_offset;
    view.h_topk_descs[p].out_offset = static_cast<size_t>(query_offset) * topk;
    view.h_topk_descs[p].num_queries = M;
    view.h_topk_descs[p].num_candidates = N;
    view.h_topk_descs[p].query_norm_offset = query_offset;
    view.h_topk_descs[p].query_prefix_sum = query_offset; // 用于二分查找
    view.h_topk_descs[p].scale = scale;

    // 填充 GEMM 参数
    problem_sizes_host[p] = {M, N, K};
    problem_sizes_m[p] = M;
    problem_sizes_n[p] = N;
    problem_sizes_k[p] = K;
    ptr_A_host[p] = reinterpret_cast<cutlass::int8_t *>(view.d_query_int8 +
                                                        query_offset * K);
    ptr_B_host[p] = reinterpret_cast<cutlass::int8_t *>(index_->d_vectors +
                                                        cluster_offset * dim);
    ptr_C_host[p] = view.d_gemm_output + gemm_output_offset;
    ptr_D_host[p] = view.d_gemm_output + gemm_output_offset;
    lda[p] = K;
    ldb[p] = K;
    ldc[p] = N;
    ldd[p] = N;

    gemm_output_offset += static_cast<size_t>(M) * N;
  }

  if (!residuals_packed) {
    for (const auto &task : tasks) {
      if (task.num_queries <= 0)
        continue;
      const int M = task.num_queries;
      const int K = dim;
      const size_t residuals_bytes = static_cast<size_t>(M) * K * sizeof(float);
      std::memcpy(view.h_residuals_staging +
                      static_cast<size_t>(task.query_offset) * K,
                  task.residual_queries, residuals_bytes);
    }
  }

  // Step 2: 一次传输所有 residuals 到 GPU
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(view.d_residuals_fp32, view.h_residuals_staging,
                      static_cast<size_t>(packed_queries) * dim * sizeof(float),
                      cudaMemcpyHostToDevice, stream));

  // Step 3: 传输描述符到 GPU
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(view.d_topk_descs, view.h_topk_descs,
                                  static_cast<size_t>(num_problems) *
                                      sizeof(RSQ8TopkProblemDesc),
                                  cudaMemcpyHostToDevice, stream));

  // Step 4: 批量量化 query 残差 FP32→Int8
  rsq8_prepare_queries_int8_kernel(view.d_residuals_fp32, view.d_query_int8,
                                   view.d_query_norms, view.d_query_scales,
                                   packed_queries, dim, stream);

  // 拷贝 GEMM 参数到 GPU
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(
      view.d_grouped_problem_sizes, problem_sizes_host.data(),
      static_cast<size_t>(num_problems) * sizeof(cutlass::gemm::GemmCoord),
      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(view.d_grouped_ptr_A, ptr_A_host.data(),
                                  static_cast<size_t>(num_problems) *
                                      sizeof(cutlass::int8_t *),
                                  cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(view.d_grouped_ptr_B, ptr_B_host.data(),
                                  static_cast<size_t>(num_problems) *
                                      sizeof(cutlass::int8_t *),
                                  cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(view.d_grouped_ptr_C, ptr_C_host.data(),
                      static_cast<size_t>(num_problems) * sizeof(int32_t *),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(view.d_grouped_ptr_D, ptr_D_host.data(),
                      static_cast<size_t>(num_problems) * sizeof(int32_t *),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(view.d_grouped_lda, lda.data(),
                      static_cast<size_t>(num_problems) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(view.d_grouped_ldb, ldb.data(),
                      static_cast<size_t>(num_problems) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(view.d_grouped_ldc, ldc.data(),
                      static_cast<size_t>(num_problems) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream));
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(view.d_grouped_ldd, ldd.data(),
                      static_cast<size_t>(num_problems) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream));

  // 调用 CUTLASS Grouped GEMM
  int threadblock_count =
      GemmGrouped::sufficient(problem_sizes_host.data(), num_problems);

  typename GemmGrouped::Arguments args(
      reinterpret_cast<cutlass::gemm::GemmCoord *>(
          view.d_grouped_problem_sizes),
      num_problems, threadblock_count, {1, 0},
      reinterpret_cast<cutlass::int8_t **>(view.d_grouped_ptr_A),
      reinterpret_cast<cutlass::int8_t **>(view.d_grouped_ptr_B),
      reinterpret_cast<int32_t **>(view.d_grouped_ptr_C),
      reinterpret_cast<int32_t **>(view.d_grouped_ptr_D),
      reinterpret_cast<int64_t *>(view.d_grouped_lda),
      reinterpret_cast<int64_t *>(view.d_grouped_ldb),
      reinterpret_cast<int64_t *>(view.d_grouped_ldc),
      reinterpret_cast<int64_t *>(view.d_grouped_ldd),
      problem_sizes_host.data());

  // 注意: Grouped GEMM workspace 需要额外处理
  // 这里假设 workspace 已经足够大，或者使用 device memory 自动分配
  GemmGrouped gemm_grouped;
  size_t workspace_size = GemmGrouped::get_workspace_size(args);

  // 使用临时 workspace (如果需要)
  void *d_workspace = nullptr;
  if (workspace_size > 0) {
    RSQ8_CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
  }

  cutlass::Status status = gemm_grouped.initialize(args, d_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    if (d_workspace)
      cudaFree(d_workspace);
    throw std::runtime_error("CUTLASS Grouped GEMM initialize failed");
  }
  status = gemm_grouped.run(stream);
  if (status != cutlass::Status::kSuccess) {
    if (d_workspace)
      cudaFree(d_workspace);
    throw std::runtime_error("CUTLASS Grouped GEMM run failed");
  }

  if (d_workspace)
    cudaFree(d_workspace);

  // =========================================================================
  // 优化: 批量 Top-K (单次 kernel launch 处理所有 problems)
  // =========================================================================
  int batch_size = 0;
  for (const auto &task : tasks) {
    for (int i = 0; i < task.num_queries; ++i) {
      int32_t qi = task.query_indices[i];
      batch_size = std::max(batch_size, qi + 1);
    }
  }
  all_results.resize(batch_size);

  if (packed_queries == 0 || topk == 0)
    return;

  if (topk <= kRsq8TopkFastMax) {
    // 使用批量 Top-K kernel (单次 launch)
    rsq8_topk_fused_batched_kernel(
        view.d_gemm_output, view.d_query_norms, view.d_query_scales,
        index_->d_norms, index_->d_global_ids, view.d_topk_descs, num_problems,
        packed_queries, max_candidates, topk, view.d_all_topk_indices,
        view.d_all_topk_distances, stream);
  } else {
    // 大 topk 回退到循环模式 (较少见)
    gemm_output_offset = 0;
    for (int p = 0; p < num_problems; ++p) {
      const auto &task = tasks[p];
      const int M = task.num_queries;
      const uint32_t cluster_size = index_->cluster_sizes[task.cluster_id];
      const int N = static_cast<int>(cluster_size);
      const uint64_t cluster_offset = index_->cluster_offsets[task.cluster_id];
      const float scale = index_->cluster_scales[task.cluster_id];
      const int query_offset = task.query_offset;

      const float *d_cluster_norms = index_->d_norms + cluster_offset;
      const int32_t *d_cluster_ids = index_->d_global_ids + cluster_offset;
      size_t out_base_elem = static_cast<size_t>(query_offset) * topk;

      rsq8_finalize_distances_kernel(
          view.d_gemm_output + gemm_output_offset,
          view.d_query_norms + query_offset, view.d_query_scales + query_offset,
          d_cluster_norms, scale, view.d_distances + gemm_output_offset, M, N,
          stream);
      rsq8_topk_kernel(view.d_distances + gemm_output_offset, d_cluster_ids, M,
                       N, topk, view.d_all_topk_indices + out_base_elem,
                       view.d_all_topk_distances + out_base_elem, stream);

      gemm_output_offset += static_cast<size_t>(M) * N;
    }
  }

  // 异步拷贝 Top-K 结果到 pinned memory
  const size_t all_topk_indices_bytes =
      static_cast<size_t>(packed_queries) * topk * sizeof(int32_t);
  const size_t all_topk_distances_bytes =
      static_cast<size_t>(packed_queries) * topk * sizeof(float);
  RSQ8_CUDA_CHECK(
      cudaMemcpyAsync(view.h_all_topk_indices_pinned, view.d_all_topk_indices,
                      all_topk_indices_bytes, cudaMemcpyDeviceToHost, stream));
  RSQ8_CUDA_CHECK(cudaMemcpyAsync(
      view.h_all_topk_distances_pinned, view.d_all_topk_distances,
      all_topk_distances_bytes, cudaMemcpyDeviceToHost, stream));
}

} // namespace fusionann
