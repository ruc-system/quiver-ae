#pragma once

#include <cstdint>
#include <math.h>

#include "cuda_utils.cuh"

namespace shared {

__inline__ __device__ float warp_reduce_sum(float val) {
#if __CUDACC_VER_MAJOR__ >= 9
  unsigned int active = __activemask();
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = val + __shfl_down_sync(active, val, offset);
  }
#else
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = val + __shfl_down(val, offset);
  }
#endif
  return val;
}

template <class T>
__inline__ __device__ float square_sum_32(const float *a, T *b,
                                          const int num_dims) {
  __syncwarp();

  int lane = threadIdx.x % 32;
  float val = 0;

  for (int i = lane; i < num_dims; i += 32) {
    float diff = a[i] - static_cast<float>(b[i]);
    val += diff * diff;
  }
  __syncwarp();
#pragma unroll
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return __shfl_sync(0xffffffff, val, 0);
}

template <class T>
__inline__ __device__ float dot_sum_32(const float *a, T *b,
                                       const int num_dims) {
  __syncwarp();

  int lane = threadIdx.x % 32;
  float val = 0;

  for (int i = lane; i < num_dims; i += 32) {
    val += a[i] * static_cast<float>(b[i]);
  }
  __syncwarp();
#pragma unroll
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return __shfl_sync(0xffffffff, val, 0);
}

template <class T>
__inline__ __device__ float cos_similarity_32(const float *a, T *b,
                                              const int num_dims) {
  __syncwarp();

  int lane = threadIdx.x % 32;

  float dot_sum = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;

  for (int i = lane; i < num_dims; i += 32) {
    float val_a = a[i];
    float val_b = static_cast<float>(b[i]);

    dot_sum += val_a * val_b;
    norm_a += val_a * val_a;
    norm_b += val_b * val_b;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    dot_sum += __shfl_down_sync(0xffffffff, dot_sum, offset);
    norm_a += __shfl_down_sync(0xffffffff, norm_a, offset);
    norm_b += __shfl_down_sync(0xffffffff, norm_b, offset);
  }

  float final_dot = __shfl_sync(0xffffffff, dot_sum, 0);
  float final_norm_a = __shfl_sync(0xffffffff, norm_a, 0);
  float final_norm_b = __shfl_sync(0xffffffff, norm_b, 0);

  return final_dot / (sqrtf(final_norm_a) * sqrtf(final_norm_b) + 1e-6f);
}

__inline__ __device__ void retset_push_32(float *distance, int *idx, int &size,
                                          int max_size, float value,
                                          int value_idx) {
  int lane = threadIdx.x % 32;
  bool found_flag = false;
  for (int i = 0; i < size; i += 32) {
    int p = size - i - 1 - lane;
    bool flag = p < size && p >= 0;
    __syncwarp();
    float tmp_d = flag ? distance[p] : 0;
    int tmp_i = flag ? idx[p] : 0;
    __syncwarp();
    if (flag && tmp_d > value && p + 1 < max_size) {
      distance[p + 1] = tmp_d;
      idx[p + 1] = tmp_i;
    }
    __syncwarp();
    unsigned int mask = __ballot_sync(0xffffffff, flag && tmp_d > value);
    __syncwarp();

    if ((mask + 1) == (1u << lane)) {
      if (p + 1 < max_size) {
        distance[p + 1] = value;
        idx[p + 1] = value_idx;
      }
      found_flag = 1;
    }
    found_flag = __any_sync(0xffffffff, found_flag);
    if (found_flag) {
      break;
    }
  }
  if (!found_flag && lane == 0) {
    distance[0] = value;
    idx[0] = value_idx;
  }
  if (size + 1 <= max_size && lane == 0) {
    size++;
  }
}

__inline__ __device__ int lower_bound(float *dist_arr, uint32_t *id_arr, int l,
                                      int r, float d, int x) {
  x = x & 0x7fffffff;
  while (l < r) {
    int mid = (l + r) / 2;
    if (dist_arr[mid] < d ||
        (dist_arr[mid] == d && (id_arr[mid] & 0x7fffffff) < x)) {
      l = mid + 1;
    } else {
      r = mid;
    }
  }
  return l;
}

}  // namespace shared
