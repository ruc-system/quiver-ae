#include "common/nvtx_utils.h"
#include "gpu_kernels.cuh"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cuda/stream_ref>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include <cooperative_groups.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/std/functional>
#include <faiss/Index.h>
#include <thrust/device_ptr.h>

#include <cub/device/device_select.cuh>

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    cudaError_t e = (err);                                                     \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error on line %d: %s\n", __LINE__, cudaGetErrorString(e));  \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  }

void compute_cub_workspace_sizes(int num_items, size_t &select_bytes,
                                 size_t &sort_bytes) {
  FUSIONANNS_NVTX_RANGE_COLOR(
      "CUBWorkspaceProbe",
      0xFF607D8Bu); // 探测CUB工作区 / CUB workspace probe
  select_bytes = 0;
  sort_bytes = 0;
  if (num_items <= 0)
    return;

  long long *keys_in = nullptr;
  long long *keys_out = nullptr;
  float *values = nullptr;
  int *num_selected = nullptr;
  void *tmp = nullptr;

  CUDA_CHECK(
      cudaMalloc(&keys_in, static_cast<size_t>(num_items) * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&keys_out,
                        static_cast<size_t>(num_items) * sizeof(long long)));
  CUDA_CHECK(
      cudaMalloc(&values, static_cast<size_t>(num_items) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&num_selected, sizeof(int)));

  size_t sort_keys_bytes = 0;
  size_t unique_bytes = 0;
  size_t sort_pairs_bytes = 0;

  cub::DeviceRadixSort::SortKeys(nullptr, sort_keys_bytes, keys_in, keys_out,
                                 num_items, 0,
                                 static_cast<int>(sizeof(long long) * 8));

  cub::DeviceSelect::Unique(nullptr, unique_bytes, keys_out, keys_out,
                            num_selected, num_items);

  cub::DeviceRadixSort::SortPairs(nullptr, sort_pairs_bytes, values, values,
                                  keys_in, keys_out, num_items);

  select_bytes = std::max(sort_keys_bytes, unique_bytes);
  sort_bytes = sort_pairs_bytes;
  size_t scratch_bytes = std::max(select_bytes, sort_bytes);

  CUDA_CHECK(cudaMalloc(&tmp, scratch_bytes));
  CUDA_CHECK(cudaFree(keys_in));
  CUDA_CHECK(cudaFree(keys_out));
  CUDA_CHECK(cudaFree(values));
  CUDA_CHECK(cudaFree(num_selected));
  CUDA_CHECK(cudaFree(tmp));
}

__global__ void build_distance_table_kernel(const float *__restrict__ query,
                                            const float *__restrict__ codebooks,
                                            float *__restrict__ table, int M,
                                            int ksub, int dsub) {
  int m = blockIdx.x;
  if (m >= M)
    return;

  extern __shared__ float q_shared[];
  const float *q = query + m * dsub;
  for (int t = threadIdx.x; t < dsub; t += blockDim.x) {
    q_shared[t] = q[t];
  }
  __syncthreads();

  for (int c = threadIdx.x; c < ksub; c += blockDim.x) {
    const float *centroid = codebooks + (m * ksub + c) * dsub;
    float acc = 0.f;
    for (int t = 0; t < dsub; ++t) {
      float diff = q_shared[t] - centroid[t];
      acc += diff * diff;
    }
    table[m * ksub + c] = acc;
  }
}

__device__ __forceinline__ void accumulate_packed_4(uint32_t packed_code,
                                                    int m_start, int ksub,
                                                    const float *s_dist_table,
                                                    float &acc) {
  // 利用位运算快速解包
  uint8_t c0 = packed_code & 0xFF;
  uint8_t c1 = (packed_code >> 8) & 0xFF;
  uint8_t c2 = (packed_code >> 16) & 0xFF;
  uint8_t c3 = (packed_code >> 24) & 0xFF;

  acc += s_dist_table[(m_start + 0) * ksub + c0];
  acc += s_dist_table[(m_start + 1) * ksub + c1];
  acc += s_dist_table[(m_start + 2) * ksub + c2];
  acc += s_dist_table[(m_start + 3) * ksub + c3];
}

template <int M_VAL = 0>
__global__ void accumulate_pq_dists_kernel(
    const float *__restrict__ dist_table, const uint8_t *__restrict__ pq_codes,
    const long long *__restrict__ candidate_ids, int active,
    int M_dynamic, // 仅在 M_VAL=0 时使用
    int ksub, float *__restrict__ out_dists) {
  // 确定当前的 M
  const int M = (M_VAL > 0) ? M_VAL : M_dynamic;

  // 1. 加载 Distance Table 到 Shared Memory
  extern __shared__ float s_dist_table[];
  int table_size = M * ksub;
  for (int i = threadIdx.x; i < table_size; i += blockDim.x) {
    s_dist_table[i] = dist_table[i];
  }
  __syncthreads();

  // 2. Grid-Stride Loop 处理候选向量
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < active; i += stride) {
    long long vec_id = candidate_ids[i];

    if (vec_id < 0) {
      out_dists[i] = FLT_MAX;
      continue;
    }

    size_t offset = static_cast<size_t>(vec_id) * M;
    float acc = 0.f;

    // ------------------------------------------------------
    // 分支 1: PQ16 特化 (极致优化)
    // ------------------------------------------------------
    if constexpr (M_VAL == 16) {
      // 使用 int4 (128-bit) 一次加载 16 字节
      // 前提：pq_codes 的起始地址是 16 字节对齐的 (cudaMalloc 保证)
      // 且 M=16 保证了每个向量的 offset 也是 16 字节对齐的
      const uint4 *ptr = reinterpret_cast<const uint4 *>(pq_codes + offset);
      uint4 pack = *ptr; // 1 个 Transaction 读取整个向量

      accumulate_packed_4(pack.x, 0, ksub, s_dist_table, acc);
      accumulate_packed_4(pack.y, 4, ksub, s_dist_table, acc);
      accumulate_packed_4(pack.z, 8, ksub, s_dist_table, acc);
      accumulate_packed_4(pack.w, 12, ksub, s_dist_table, acc);
    }
    // ------------------------------------------------------
    // 分支 2: PQ32 特化
    // ------------------------------------------------------
    else if constexpr (M_VAL == 32) {
      // 使用两个 int4 加载 32 字节
      const uint4 *ptr = reinterpret_cast<const uint4 *>(pq_codes + offset);
      uint4 pack0 = ptr[0]; // Load 0-15 bytes
      uint4 pack1 = ptr[1]; // Load 16-31 bytes

      accumulate_packed_4(pack0.x, 0, ksub, s_dist_table, acc);
      accumulate_packed_4(pack0.y, 4, ksub, s_dist_table, acc);
      accumulate_packed_4(pack0.z, 8, ksub, s_dist_table, acc);
      accumulate_packed_4(pack0.w, 12, ksub, s_dist_table, acc);

      accumulate_packed_4(pack1.x, 16, ksub, s_dist_table, acc);
      accumulate_packed_4(pack1.y, 20, ksub, s_dist_table, acc);
      accumulate_packed_4(pack1.z, 24, ksub, s_dist_table, acc);
      accumulate_packed_4(pack1.w, 28, ksub, s_dist_table, acc);
    }
    // ------------------------------------------------------
    // 分支 3: 通用处理 (Fallback)
    // ------------------------------------------------------
    else {
      const uint8_t *ptr_base = pq_codes + offset;
      int m = 0;
      // 尝试使用 4 字节加载优化通用情况
      const uint32_t *ptr_32 = reinterpret_cast<const uint32_t *>(ptr_base);

      // Unroll loop for vectorizable parts
      for (; m <= M - 4; m += 4) {
        uint32_t packed = ptr_32[m / 4];
        accumulate_packed_4(packed, m, ksub, s_dist_table, acc);
      }
      // Handle remainder
      for (; m < M; ++m) {
        acc += s_dist_table[m * ksub + ptr_base[m]];
      }
    }

    out_dists[i] = acc;
  }
}

void copy_pq_results_to_host(int num_candidates, int device_capacity,
                             const float *d_dists, const long long *d_ids,
                             cudaStream_t stream_comp, ThreadGpuEvents &events,
                             GpuTimings *timings) {
  FUSIONANNS_NVTX_RANGE_COLOR(
      "PQDownload",
      0xFF795548u); // PQ结果下载 / download PQ results to host
  events.pending_copy_count = 0;

  if (timings) {
    timings->candidates_in = num_candidates;
    timings->candidates_unique = num_candidates;
    timings->topk_requested = num_candidates;
    timings->d2h_topn_ms = 0.0f;
    timings->sort_ms = 0.0f;
  }

  if (num_candidates <= 0) {
    return;
  }

  if (num_candidates > device_capacity) {
    throw std::runtime_error("Candidate count exceeds GPU block capacity");
  }

  if (!events.h_dists || !events.h_ids ||
      events.host_capacity < num_candidates) {
    if (events.h_dists) {
      cudaFreeHost(events.h_dists);
      events.h_dists = nullptr;
    }
    if (events.h_ids) {
      cudaFreeHost(events.h_ids);
      events.h_ids = nullptr;
    }
    unsigned int host_flags = cudaHostAllocPortable | cudaHostAllocMapped;
    CUDA_CHECK(cudaHostAlloc(
        &events.h_dists, sizeof(float) * static_cast<size_t>(num_candidates),
        host_flags));
    CUDA_CHECK(cudaHostAlloc(
        &events.h_ids, sizeof(long long) * static_cast<size_t>(num_candidates),
        host_flags));
    events.host_capacity = num_candidates;
    events.d_dists_mapped = nullptr;
    events.d_ids_mapped = nullptr;
  }
  if (!events.d_dists_mapped && events.h_dists) {
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&events.d_dists_mapped), events.h_dists, 0));
  }
  if (!events.d_ids_mapped && events.h_ids) {
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&events.d_ids_mapped), events.h_ids, 0));
  }

  CUDA_CHECK(cudaEventRecord(events.d2h_start, stream_comp));
  events.recorded_d2h = timings != nullptr;
  CUDA_CHECK(
      cudaMemcpyAsync(events.h_dists, d_dists,
                      static_cast<size_t>(num_candidates) * sizeof(float),
                      cudaMemcpyDeviceToHost, stream_comp));
  CUDA_CHECK(
      cudaMemcpyAsync(events.h_ids, d_ids,
                      static_cast<size_t>(num_candidates) * sizeof(long long),
                      cudaMemcpyDeviceToHost, stream_comp));
  CUDA_CHECK(cudaEventRecord(events.d2h_stop, stream_comp));

  events.pending_copy_count = num_candidates;
  events.host_results_current = false;
}

void compute_all_pq_distances(const uint8_t *all_pq_codes_gpu,
                              const long long *candidate_ids,
                              const float *dist_table, float *out_dists, int M,
                              int ksub, int device_capacity, int num_candidates,
                              cudaStream_t stream) {
  FUSIONANNS_NVTX_RANGE_COLOR(
      "PQDistanceAccum",
      0xFF388E3Cu); // PQ距离累积 / accumulate PQ distances
  if (num_candidates > device_capacity) {
    throw std::runtime_error("Candidate count exceeds GPU block capacity");
  }
  int active = std::max(0, std::min(num_candidates, device_capacity));
  size_t smem_size = M * ksub * sizeof(float);
  constexpr int kBlockSize = 256;
  int block_size = kBlockSize;
  int grid = (active + block_size - 1) / block_size;
  if (grid <= 0)
    grid = 1;
  constexpr int kMaxGrid = 65535;
  grid = std::min(grid, kMaxGrid);

  switch (M) {
  case 16:
    // PQ16 特化路径
    accumulate_pq_dists_kernel<16><<<grid, kBlockSize, smem_size, stream>>>(
        dist_table, all_pq_codes_gpu, candidate_ids, active, 16, ksub,
        out_dists);
    break;

  case 32:
    // PQ32 特化路径
    accumulate_pq_dists_kernel<32><<<grid, kBlockSize, smem_size, stream>>>(
        dist_table, all_pq_codes_gpu, candidate_ids, active, 32, ksub,
        out_dists);
    break;

  default:
    // 通用路径 (M_VAL = 0)
    accumulate_pq_dists_kernel<0><<<grid, kBlockSize, smem_size, stream>>>(
        dist_table, all_pq_codes_gpu, candidate_ids, active, M, ksub,
        out_dists);
    break;
  }
  // CUDA_CHECK(cudaGetLastError());
}

void upload_pq_codebooks_to_gpu(const faiss::ProductQuantizer &pq,
                                float **d_codebooks, int M, int ksub,
                                int dsub) {
  FUSIONANNS_NVTX_RANGE_COLOR("UploadPQCodebooks",
                              0xFF00ACC1u); // 上传PQ码本 / upload PQ codebooks
  if (!d_codebooks) {
    throw std::invalid_argument("d_codebooks pointer must not be null");
  }
  const size_t expected = static_cast<size_t>(M) * static_cast<size_t>(ksub) *
                          static_cast<size_t>(dsub);
  if (pq.centroids.size() != expected) {
    throw std::runtime_error("PQ centroid table size mismatch");
  }
  const size_t bytes = expected * sizeof(float);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(d_codebooks), bytes));
  CUDA_CHECK(cudaMemcpy(*d_codebooks, pq.centroids.data(), bytes,
                        cudaMemcpyHostToDevice));
}

void build_distance_table_gpu(const float *d_query, const float *d_codebooks,
                              float *d_table, int M, int ksub, int dsub,
                              cudaStream_t stream) {
  FUSIONANNS_NVTX_RANGE_COLOR(
      "LaunchDistanceTable",
      0xFF5E35B1u); // 启动距离表kernel / launch distance table kernel
  if (!d_query || !d_codebooks || !d_table) {
    throw std::invalid_argument(
        "Null device pointer passed to distance table builder");
  }
  if (M <= 0 || ksub <= 0 || dsub <= 0) {
    throw std::invalid_argument(
        "Invalid PQ parameters for distance table build");
  }
  const int blocks = M;
  const int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(dsub) * sizeof(float);
  build_distance_table_kernel<<<blocks, threads, shared_bytes, stream>>>(
      d_query, d_codebooks, d_table, M, ksub, dsub);
  // CUDA_CHECK(cudaGetLastError());
}

#ifdef FUSIONANNS_USE_CUVS_SELECT_K
#include <cuvs/selection/select_k.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

int select_top_k_gpu(const float *d_dists, const long long *d_ids,
                     int num_candidates, int k, float *d_out_dists,
                     long long *d_out_ids, cudaStream_t stream) {
  if (num_candidates <= 0 || k <= 0 || !d_dists || !d_ids || !d_out_dists ||
      !d_out_ids) {
    return 0;
  }
  const int actual_k = std::min(k, num_candidates);
  raft::device_resources handle;
  raft::resource::set_cuda_stream(handle, stream);

  auto in_val = raft::make_device_matrix_view<const float, int64_t>(
      d_dists, 1, static_cast<int64_t>(num_candidates));
  auto in_idx = raft::make_device_matrix_view<const int64_t, int64_t>(
      reinterpret_cast<const int64_t *>(d_ids), 1,
      static_cast<int64_t>(num_candidates));
  auto out_val = raft::make_device_matrix_view<float, int64_t>(
      d_out_dists, 1, static_cast<int64_t>(actual_k));
  auto out_idx = raft::make_device_matrix_view<int64_t, int64_t>(
      reinterpret_cast<int64_t *>(d_out_ids), 1,
      static_cast<int64_t>(actual_k));

  cuvs::selection::select_k(handle, in_val, in_idx, out_val, out_idx,
                            true /* select_min */, true /* sorted */);
  return actual_k;
}
#endif
