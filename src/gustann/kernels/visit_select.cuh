#pragma once

#include "../shared/common/gpu_primitives.cuh"
#include "../shared/common/search_state.cuh"
#include "../probe.cuh"

namespace gustann {

using shared::Data;

template <class T>
__global__ void visit_exact_dist_kernel(float *qdata, uint8_t *buffer,
                                        int32_t *request, const int num_dims,
                                        const int topk, int *nns,
                                        float *distances, int *found_cnt,
                                        int nodes_per_page, int node_len,
                                        int data_len, int qcnt) {
  int bid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (bid >= qcnt || request[bid] == -1) {
    return;
  }

  int tid = threadIdx.x % 32;
  (void)tid;

  int buffer_offset = request[bid] % nodes_per_page * node_len;
  T *buffer_u = (T *)(buffer + bid * 4096 + buffer_offset);
  float dist =
      shared::square_sum_32(qdata + num_dims * bid, buffer_u, num_dims);
  shared::retset_push_32(distances + bid * topk, nns + bid * topk,
                         found_cnt[bid], topk, dist, request[bid]);
  (void)data_len;
}

__global__ void clamp_ctx_size_kernel(Data *data, int max_size, int qcnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < qcnt) {
    data[i].size = min(data[i].size, max_size);
  }
}

__global__ void select_next_spe_kernel(const int max_m, const int ef_search,
                                       uint32_t *neighbor_id,
                                       float *neighbor_dist, Data *data,
                                       int32_t *request, int qcnt,
                                       int spe_width,
                                       int32_t *visited_ids = nullptr,
                                       int *n_visited = nullptr,
                                       int max_visited = 0) {
  int bid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (bid >= qcnt) {
    return;
  }

  int tid = threadIdx.x % 32;
  int offset = (max_m + ef_search + 31) / 32 * 32 * bid;
  Data &ctx = data[bid];
  int sz = ctx.size;
  int target = (sz + 31) / 32 * 32;
  int count = 0;

  for (int i = tid; i < target; i += 32) {
    bool flag =
        (i != 0) && (i < sz) &&
        (((neighbor_id[offset + i] ^ neighbor_id[offset + i - 1]) & 0x7fffffff) ==
         0);
    unsigned mask = __ballot_sync(0xffffffff, flag);
    int mv = i - count - __popc(mask & ((1u << tid) - 1));
    uint32_t tmp_id = neighbor_id[offset + i];
    float tmp_dist = neighbor_dist[offset + i];
    __syncwarp();

    if (i < sz && !flag) {
      neighbor_id[offset + mv] = tmp_id;
      neighbor_dist[offset + mv] = tmp_dist;
    }
    __syncwarp();
    count += __popc(mask);
    __syncwarp();
  }

  sz = sz - count;

  for (int w = 0; w < spe_width; ++w) {
    int id = sz + 1;
    for (int i = tid; i < sz; i += 32) {
      if ((neighbor_id[offset + i] & 0x80000000u) == 0) {
        id = min(id, i);
      }
    }
    __syncwarp();
#pragma unroll
    for (int shfl_off = 32 / 2; shfl_off > 0; shfl_off /= 2) {
      int _id = __shfl_down_sync(0xffffffff, id, shfl_off);
      id = min(_id, id);
    }
    id = __shfl_sync(0xffffffff, id, 0);

    if (tid == 0) {
      if (id >= ef_search || id >= sz) {
        request[w * qcnt + bid] = -1;
      } else {
        int32_t node_id =
            static_cast<int32_t>(neighbor_id[offset + id] & 0x7fffffff);
        request[w * qcnt + bid] = node_id;
        neighbor_id[offset + id] |= 0x80000000u;
        if (visited_ids && n_visited) {
          int vi = atomicAdd(&n_visited[bid], 1);
          if (vi < max_visited) {
            visited_ids[bid * max_visited + vi] = node_id;
          }
        }
      }
    }
    __syncwarp();
  }

  if (tid == 0) {
    bool should_stop = false;
    if (n_visited) {
      should_stop = (n_visited[bid] >= ef_search);
    }
    if (!should_stop) {
      bool all_done = true;
      for (int w = 0; w < spe_width; ++w) {
        if (request[w * qcnt + bid] != -1) {
          all_done = false;
          break;
        }
      }
      if (all_done) {
        should_stop = true;
      }
    }
    if (should_stop) {
      ctx.size = 0;
      for (int w = 0; w < spe_width; ++w) {
        request[w * qcnt + bid] = -1;
      }
    } else {
      ctx.size = min(sz, ef_search);
    }
  }
  __syncwarp();
}

__global__ void pipe_select_next_kernel(const int max_m, const int ef_search,
                                        uint32_t *neighbor_id,
                                        float *neighbor_dist, Data *data,
                                        int32_t *request, int qcnt) {
  int bid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (bid >= qcnt) {
    return;
  }
  int tid = threadIdx.x % 32;

  int offset = (max_m + ef_search + 31) / 32 * 32 * bid;
  Data &ctx = data[bid];
  int sz = ctx.size;
  if (sz == 0) {
    if (tid == 0) {
      request[bid] = -1;
    }
    return;
  }

  int target = (sz + 31) / 32 * 32;
  int count = 0;
  for (int i = tid; i < target; i += 32) {
    bool flag =
        (i != 0) && (i < sz) &&
        (((neighbor_id[offset + i] ^ neighbor_id[offset + i - 1]) & 0x7fffffff) ==
         0);
    unsigned mask = __ballot_sync(0xffffffff, flag);
    int mv = i - count - __popc(mask & ((1u << tid) - 1));
    uint32_t tmp_id = neighbor_id[offset + i];
    float tmp_dist = neighbor_dist[offset + i];
    __syncwarp();
    if (i < sz && !flag) {
      neighbor_id[offset + mv] = tmp_id;
      neighbor_dist[offset + mv] = tmp_dist;
    }
    __syncwarp();
    count += __popc(mask);
    __syncwarp();
  }
  sz -= count;

  int id = sz + 1;
  for (int i = tid; i < sz; i += 32) {
    if ((neighbor_id[offset + i] & 0x80000000u) == 0) {
      id = min(id, i);
    }
  }
  __syncwarp();
#pragma unroll
  for (int shfl_off = 16; shfl_off > 0; shfl_off /= 2) {
    int _id = __shfl_down_sync(0xffffffff, id, shfl_off);
    id = min(_id, id);
  }
  id = __shfl_sync(0xffffffff, id, 0);

  if (tid == 0) {
    if (id >= ef_search || id >= sz) {
      request[bid] = -1;
    } else {
      request[bid] = neighbor_id[offset + id] & 0x7fffffff;
      neighbor_id[offset + id] |= 0x80000000u;
    }
    ctx.size = min(sz, ef_search);
  }
}

template <class T>
__global__ void pipe_visit_select_kernel(float *qdata, uint8_t *buffer,
                                         int32_t *request, const int num_dims,
                                         const int topk, int *nns,
                                         float *distances, int *found_cnt,
                                         int nodes_per_page, int node_len,
                                         const int max_m, const int ef_search,
                                         uint32_t *neighbor_id,
                                         float *neighbor_dist, Data *data,
                                         int qcnt
#ifdef GUSTANN_LATENCY_PROBE
                                         ,
                                         GustannProbeQueryTrace *probe_traces,
                                         int *probe_active_samples,
                                         int probe_stride, int probe_sublane
#endif
                                         ) {
  int bid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (bid >= qcnt) {
    return;
  }
  int tid = threadIdx.x % 32;

#ifdef GUSTANN_LATENCY_PROBE
  int probe_sample_idx = -1;
  if (tid == 0 && request[bid] != -1) {
    probe_sample_idx = probe_active_samples[probe_sublane * probe_stride + bid];
    if (probe_sample_idx >= 0 &&
        probe_sample_idx < GUSTANN_PROBE_MAX_SAMPLES) {
      probe_traces[bid].samples[probe_sample_idx].t_visit_start =
          gustann_globaltimer_ns();
    }
  }
#endif

  if (request[bid] != -1) {
    int buffer_offset = request[bid] % nodes_per_page * node_len;
    T *buffer_u = (T *)(buffer + bid * 4096 + buffer_offset);
    float dist =
        shared::square_sum_32(qdata + num_dims * bid, buffer_u, num_dims);
    shared::retset_push_32(distances + bid * topk, nns + bid * topk,
                           found_cnt[bid], topk, dist, request[bid]);
  }

  int offset = (max_m + ef_search + 31) / 32 * 32 * bid;
  Data &ctx = data[bid];
  int sz = ctx.size;
  if (sz == 0) {
    if (tid == 0) {
      request[bid] = -1;
#ifdef GUSTANN_LATENCY_PROBE
      if (probe_sample_idx >= 0 &&
          probe_sample_idx < GUSTANN_PROBE_MAX_SAMPLES) {
        probe_traces[bid].samples[probe_sample_idx].t_compute_done =
            gustann_globaltimer_ns();
      }
#endif
    }
    return;
  }

  int target = (sz + 31) / 32 * 32;
  int count = 0;
  for (int i = tid; i < target; i += 32) {
    bool flag =
        (i != 0) && (i < sz) &&
        (((neighbor_id[offset + i] ^ neighbor_id[offset + i - 1]) & 0x7fffffff) ==
         0);
    unsigned mask = __ballot_sync(0xffffffff, flag);
    int mv = i - count - __popc(mask & ((1u << tid) - 1));
    uint32_t tmp_id = neighbor_id[offset + i];
    float tmp_dist = neighbor_dist[offset + i];
    __syncwarp();
    if (i < sz && !flag) {
      neighbor_id[offset + mv] = tmp_id;
      neighbor_dist[offset + mv] = tmp_dist;
    }
    __syncwarp();
    count += __popc(mask);
    __syncwarp();
  }
  sz -= count;

  int id = sz + 1;
  for (int i = tid; i < sz; i += 32) {
    if ((neighbor_id[offset + i] & 0x80000000u) == 0) {
      id = min(id, i);
    }
  }
  __syncwarp();
#pragma unroll
  for (int shfl_off = 16; shfl_off > 0; shfl_off /= 2) {
    int _id = __shfl_down_sync(0xffffffff, id, shfl_off);
    id = min(_id, id);
  }
  id = __shfl_sync(0xffffffff, id, 0);

  if (tid == 0) {
    if (id >= ef_search || id >= sz) {
      request[bid] = -1;
    } else {
      request[bid] = neighbor_id[offset + id] & 0x7fffffff;
      neighbor_id[offset + id] |= 0x80000000u;
    }
    ctx.size = min(sz, ef_search);
#ifdef GUSTANN_LATENCY_PROBE
    if (probe_sample_idx >= 0 &&
        probe_sample_idx < GUSTANN_PROBE_MAX_SAMPLES) {
      probe_traces[bid].samples[probe_sample_idx].t_compute_done =
          gustann_globaltimer_ns();
    }
#endif
  }
}

}  // namespace gustann
