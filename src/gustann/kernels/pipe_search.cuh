#pragma once

#include "../shared/common/gpu_primitives.cuh"
#include "../shared/common/search_state.cuh"
#include "../shared/index/pq_device.cuh"

namespace gustann {

using shared::Data;
using shared::PQSearchData;
using shared::WARP_SIZE;

#define PQ_HEAPED

__inline__ __device__ void update_pq_heaped(uint32_t *pq_id, float *pq_dist,
                                            int &size, int capacity,
                                            float dist, int id) {
  int tid = threadIdx.x % 32;
  int bsz = 32;

  if (size == capacity && dist >= pq_dist[0]) {
    return;
  }
  if (size < capacity) {
    bool exists = false;
    for (int i = tid; i < size; i += bsz) {
      exists |= (pq_id[i] & 0x7fffffffu) == (uint32_t)(id);
    }
    if (__any_sync(0xffffffff, exists)) {
      return;
    }
    if (tid == 0) {
      if (size == 0 || pq_dist[0] >= dist) {
        pq_id[size] = id;
        pq_dist[size] = dist;
      } else {
        pq_id[size] = pq_id[0];
        pq_dist[size] = pq_dist[0];
        pq_id[0] = id;
        pq_dist[0] = dist;
      }
      size++;
    }
    return;
  }

  bool exists = false;
  int cid = -1;
  float max = -INFINITY;
  for (int i = tid; i < size; i += bsz) {
    exists |= (pq_id[i] & 0x7fffffffu) == (uint32_t)(id);

    if (i != 0 && pq_dist[i] > max) {
      cid = i;
      max = pq_dist[i];
    }
  }

  if (__any_sync(0xffffffff, exists)) {
    return;
  }
#pragma unroll
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    int _id = __shfl_down_sync(0xffffffff, cid, offset);
    float _max = __shfl_down_sync(0xffffffff, max, offset);
    if (_max > max) {
      max = _max;
      cid = _id;
    }
  }

  if (tid == 0) {
    if (max > dist) {
      pq_id[0] = pq_id[cid];
      pq_dist[0] = max;
      pq_id[cid] = id;
      pq_dist[cid] = dist;
    } else {
      pq_id[0] = id;
      pq_dist[0] = dist;
    }
  }
}

__inline__ __device__ void update_pq_insertion(uint32_t *pq_id, float *pq_dist,
                                               int &size, int capacity,
                                               float dist, int id) {
  if (size == capacity && dist >= pq_dist[0]) {
    return;
  }

  int tid = threadIdx.x % 32;
  int bsz = 32;
  bool exists = false;

  for (int i = tid; i < size; i += bsz) {
    exists |= (pq_id[i] & 0x7fffffffu) == (uint32_t)(id);
  }

  if (__any_sync(0xffffffff, exists)) {
    return;
  }

  if (tid == 0) {
    if (size == capacity) {
      int i = 0, j = 1;
      while (j < size) {
        if (j + 1 < size && pq_dist[j + 1] > pq_dist[j]) {
          j++;
        }
        if (pq_dist[j] < dist) {
          break;
        }
        pq_id[i] = pq_id[j];
        pq_dist[i] = pq_dist[j];
        i = j;
        j = j * 2 + 1;
      }
      pq_dist[i] = dist;
      pq_id[i] = id;
    } else {
      int i = size++;
      while (i > 0) {
        int j = (i + 1) / 2 - 1;
        if (pq_dist[j] > dist) {
          break;
        }
        pq_dist[i] = pq_dist[j];
        pq_id[i] = pq_id[j];
        i = j;
      }
      pq_dist[i] = dist;
      pq_id[i] = id;
    }
  }
}

__inline__ __device__ void update_pq(uint32_t *pq_id, float *pq_dist, int &size,
                                     int capacity, float dist, int id) {
#ifdef PQ_HEAPED
  update_pq_heaped
#else
  update_pq_insertion
#endif
      (pq_id, pq_dist, size, capacity, dist, id);
}

__inline__ __device__ int get_cand_id(uint32_t *pq_id, float *pq_dist,
                                      int size) {
  int cand = -1;
  float dist = INFINITY;
  int tid = threadIdx.x % 32;
  int tsz = 32;
  for (int i = tid; i < size; i += tsz) {
    if ((pq_id[i] >> 31) == 0 && pq_dist[i] < dist) {
      cand = i;
      dist = pq_dist[i];
    }
  }
#pragma unroll
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    int _cand = __shfl_down_sync(0xffffffff, cand, offset);
    float _dist = __shfl_down_sync(0xffffffff, dist, offset);
    if (_dist < dist) {
      dist = _dist;
      cand = _cand;
    }
  }
  cand = __shfl_sync(0xffffffff, cand, 0);
  return cand;
}

template <class data_type>
__global__ void update_kernel(float *qdata, uint8_t *buffer, int32_t *request,
                              float *tmp_dist, int *tmp_id, int num_nodes,
                              const int num_dims, const int max_m,
                              const int ef_search, const int topk, int *nns,
                              float *distances, int *found_cnt,
                              int64_t *acc_visited_cnt, uint32_t *neighbor_id,
                              float *neighbor_dist, int nodes_per_page,
                              int node_len, int data_len, Data *data,
                              int qcnt) {
  int buffer_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (buffer_id >= qcnt) {
    return;
  }
  int tid = threadIdx.x % 32;
  float *src_vec = qdata + num_dims * buffer_id;

  int node_u = request[buffer_id];
  if (node_u == -1) {
    return;
  }
  int offset = node_u % nodes_per_page * node_len;

  data_type *buffer_u = (data_type *)(buffer + buffer_id * 4096 + offset);
  float dist = shared::square_sum_32(src_vec, buffer_u, num_dims);
  shared::retset_push_32(distances + buffer_id * topk, nns + buffer_id * topk,
                         found_cnt[buffer_id], topk, dist, node_u);
  int *buffer_edge = tmp_id + buffer_id * max_m;
  int edge_deg = max_m;

  Data &ctx = data[buffer_id];
  int aligned_ef = (ef_search + 31) / 32 * 32;
  uint32_t *ef_search_pq_id = neighbor_id + aligned_ef * buffer_id;
  float *ef_search_pq_dist = neighbor_dist + aligned_ef * buffer_id;
  __syncwarp();

  __shared__ int sh_dst[64];
  int *loc_dst = sh_dst + 32 * (threadIdx.x / 32);
#if 1
  for (int j = 0; j < edge_deg; j++) {
    if (j % 32 == 0) {
      __syncwarp();
      loc_dst[tid] = buffer_edge[j + tid];
      __syncwarp();
    }

    int dstid = loc_dst[j % 32];
    if (dstid == -1) {
      break;
    }
    update_pq(ef_search_pq_id, ef_search_pq_dist, ctx.size, ef_search,
              tmp_dist[j + buffer_id * max_m], dstid);
    __syncwarp();
  }
#endif
  __syncwarp();
  int idx = get_cand_id(ef_search_pq_id, ef_search_pq_dist, ctx.size);

  if (idx == -1) {
    if (tid == 0) {
      request[buffer_id] = -1;
      ctx.size = 0;
      ctx.visited_cnt = 0;
    }
  } else {
    if (tid == 0) {
      request[buffer_id] = ef_search_pq_id[idx];
      ef_search_pq_id[idx] |= 0x80000000u;
    }
  }
}

__global__ void get_pq_dist_kernel(uint8_t *buffer, int32_t *request,
                                   float *tmp_dist, int *tmp_id,
                                   PQSearchData *pq_data, int nodes_per_page,
                                   int node_len, int data_len, int pq_offset,
                                   int max_m0) {
  int buffer_id = blockIdx.x;
  int nodeid_u = request[buffer_id];
  if (nodeid_u == -1) {
    return;
  }
  int offset = nodeid_u % nodes_per_page * node_len + data_len;
  int *buffer_u = (int *)(buffer + buffer_id * 4096 + offset);

  if (threadIdx.x >= buffer_u[0]) {
    if (threadIdx.x < max_m0) {
      tmp_id[blockIdx.x * max_m0 + threadIdx.x] = -1;
      tmp_dist[blockIdx.x * max_m0 + threadIdx.x] = INFINITY;
    }
    return;
  }
  int nodeid_v =
      tmp_id[blockIdx.x * max_m0 + threadIdx.x] = buffer_u[threadIdx.x + 1];
  float *dist_vec =
      pq_data->pq_dists +
      (pq_offset + buffer_id) * pq_data->num_pivots * pq_data->num_chunks;
  float dist = 0;
  uint8_t *data = pq_data->compressed_data + (long)nodeid_v * pq_data->num_chunks;
  for (int j = 0; j < pq_data->num_chunks; j++) {
    dist += dist_vec[j * pq_data->num_pivots + data[j]];
  }
  tmp_dist[blockIdx.x * max_m0 + threadIdx.x] = dist;
}

}  // namespace gustann
