#pragma once

#include "../shared/common/gpu_primitives.cuh"
#include "../shared/common/search_state.cuh"
#include "../shared/index/pq_search.hpp"
#include "../probe.cuh"

namespace gustann {

using shared::Data;
using shared::PQSearchData;
using shared::lower_bound;

__global__ void __launch_bounds__(128, 14) merge_data_kernel(
    uint8_t *buffer, int32_t *request, int num_chunks, float *pq_dists,
    uint8_t *compressed_data, int nodes_per_page, int node_len, int data_len,
    int pq_offset, const int max_m, const int ef_search, uint32_t *neighbor_id,
    float *neighbor_dist, Data *data
#ifdef GUSTANN_LATENCY_PROBE
    ,
    GustannProbeQueryTrace *probe_traces, int *probe_active_samples,
    int probe_stride, int probe_sublane, int probe_hop_index,
    int probe_query_base, const int64_t *probe_kernel_start
#endif
    ) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int nodeid_u = request[bid];
  if (nodeid_u == -1) {
    return;
  }

#ifdef GUSTANN_LATENCY_PROBE
  int probe_sample_idx = -1;
  if (tid == 0) {
    GustannProbeQueryTrace &trace = probe_traces[bid];
    trace.query_id = probe_query_base + bid;
    int slot = atomicAdd(&trace.num_samples, 1);
    if (slot < GUSTANN_PROBE_MAX_SAMPLES) {
      probe_sample_idx = slot;
      GustannProbeSample &sample = trace.samples[slot];
      sample.hop_index = probe_hop_index;
      sample.sublane = probe_sublane;
      sample.t_kernel_start = *probe_kernel_start;
      sample.t_merge_start = gustann_globaltimer_ns();
      sample.t_merge_done = 0;
      sample.t_visit_start = 0;
      sample.t_compute_done = 0;
      probe_active_samples[probe_sublane * probe_stride + bid] = slot;
    } else {
      probe_active_samples[probe_sublane * probe_stride + bid] = -1;
    }
  }
#endif

  extern __shared__ uint8_t shm_pool[];
  int *mv_pos = (int *)shm_pool;
  uint32_t *mv_id = (uint32_t *)(mv_pos + ef_search + max_m);
  float *mv_dist = (float *)(mv_id + ef_search + max_m);
  uint32_t *tmp_id = (uint32_t *)(mv_dist + ef_search + max_m);
  float *tmp_dist = (float *)(tmp_id + ef_search + max_m);

  int offset = (max_m + ef_search + 31) / 32 * 32 * bid;
  Data &ctx = data[bid];
  int sz = ctx.size;
  int edge_offset = nodeid_u % nodes_per_page * node_len + data_len;
  int *buffer_u = (int *)(buffer + bid * 4096 + edge_offset);

  for (int i = tid; i < sz; i += blockDim.x) {
    tmp_id[i] = neighbor_id[offset + i];
    tmp_dist[i] = neighbor_dist[offset + i];
  }
  __syncthreads();

  int deg = buffer_u[0];

#if 1
  if (tid >= deg) {
    if (tid < max_m) {
      tmp_id[sz + tid] = 0xffffffff - tid;
      tmp_dist[sz + tid] = INFINITY;
    }
  } else {
    int nodeid_v = tmp_id[sz + tid] = buffer_u[threadIdx.x + 1];
    float *dist_vec =
        pq_dists + (pq_offset + bid) * PQSearchData::num_pivots * num_chunks;
    float dist = 0;
    uint8_t *data = compressed_data + (long)nodeid_v * num_chunks;

#pragma unroll 32
    for (int j = 0; j < num_chunks; j++) {
      dist += dist_vec[j * PQSearchData::num_pivots + data[j]];
    }

    tmp_dist[sz + tid] = dist;
  }
#elif 0
  if (tid >= deg) {
    if (tid < max_m) {
      tmp_id[sz + tid] = 0xffffffff - tid;
      tmp_dist[sz + tid] = INFINITY;
    }
  } else {
    int nodeid_v = tmp_id[sz + tid] = buffer_u[threadIdx.x + 1];
  }
  __syncthreads();

  uint8_t *data_shm = (uint8_t *)(tmp_dist + ef_search + max_m);
  int subtask = blockDim.x / num_chunks;
  int subid = tid / num_chunks;
  int xid = tid % num_chunks;
  if (subid < subtask) {
    for (int i = subid; i < deg; i += subtask) {
      uint8_t *data = compressed_data + (long)tmp_id[sz + i] * num_chunks;
      data_shm[i * (num_chunks + 1) + xid] = data[xid];
    }
  }
  __syncthreads();

  float dist = 0;
  if (tid < deg) {
    uint8_t *data = data_shm + tid * (num_chunks + 1);
    float *dist_vec =
        pq_dists + (pq_offset + bid) * PQSearchData::num_pivots * num_chunks;
    for (int j = 0; j < num_chunks; j++) {
      dist += dist_vec[j * PQSearchData::num_pivots + data[j]];
    }
    tmp_dist[sz + tid] = dist;
  }
#else
  if (tid >= deg) {
    if (tid < max_m) {
      tmp_id[sz + tid] = 0xffffffff - tid;
      tmp_dist[sz + tid] = INFINITY;
    }
  } else {
    int nodeid_v = tmp_id[sz + tid] = buffer_u[threadIdx.x + 1];
  }
  __syncthreads();
  assert(num_chunks == 32 && (size_t)(compressed_data) % 4 == 0);
  const int np = 8;
  int subtask = blockDim.x / np;
  int subid = tid / np;
  int xid = tid % np;
  float *dist_vec =
      pq_dists + (pq_offset + bid) * PQSearchData::num_pivots * num_chunks;
  for (int i = subid; i < (deg + np - 1) / np * np; i += subtask) {
    float dist;
    if (i < deg) {
      uint32_t val =
          *((uint32_t *)(compressed_data + (long)(tmp_id[sz + i]) * num_chunks) +
            xid);

      int x = val & 0xff;
      int y = (val >> 8) & 0xff;
      int z = (val >> 16) & 0xff;
      int w = (val >> 24) & 0xff;
      float dist1 = dist_vec[(xid * 4 + 0) * PQSearchData::num_pivots + x];
      float dist2 = dist_vec[(xid * 4 + 1) * PQSearchData::num_pivots + y];
      float dist3 = dist_vec[(xid * 4 + 2) * PQSearchData::num_pivots + z];
      float dist4 = dist_vec[(xid * 4 + 3) * PQSearchData::num_pivots + w];
      dist = (dist1 + dist2) + (dist3 + dist4);
    } else {
      dist = 0;
    }
#pragma unroll
    for (int offset = np / 2; offset > 0; offset /= 2) {
      dist += __shfl_down_sync(0xffffffff, dist, offset);
    }
    if (xid == 0 && i < deg) {
      tmp_dist[sz + i] = dist;
    }
  }
#endif

  offset += sz;
  __syncthreads();

  for (int len = 2; len < 2 * deg; len *= 2) {
    int array_id = tid / len;
    int start = array_id * len;
    int mid = min(start + len / 2, deg);
    int end = min(start + len, deg);
    if (tid >= start && tid < mid) {
      int id = lower_bound(tmp_dist + sz + mid, tmp_id + sz + mid, 0,
                           end - mid, tmp_dist[sz + tid], tmp_id[sz + tid]);
      mv_pos[tid] = id + tid;
    }
    if (tid >= mid && tid < end) {
      int id = lower_bound(tmp_dist + sz, tmp_id + sz, start, mid,
                           tmp_dist[sz + tid], tmp_id[sz + tid]);
      mv_pos[tid] = id + tid - mid;
    }
    __syncthreads();
    __threadfence_block();

    if (tid < deg) {
      mv_id[mv_pos[tid]] = tmp_id[sz + tid];
      mv_dist[mv_pos[tid]] = tmp_dist[sz + tid];
    }

    __syncthreads();
    if (tid < deg) {
      tmp_id[sz + tid] = mv_id[tid];
      tmp_dist[sz + tid] = mv_dist[tid];
    }
    __syncthreads();
  }

  offset -= sz;
  if (tid < deg) {
    int id = lower_bound(tmp_dist, tmp_id, 0, sz, tmp_dist[sz + tid],
                         tmp_id[sz + tid]);
    if (id != sz && ((tmp_id[id] ^ tmp_id[sz + tid]) & 0x7fffffff) == 0) {
      id++;
    }
    mv_pos[sz + tid] = id + tid;
  }
  for (int i = tid; i < sz; i += blockDim.x) {
    int id =
        lower_bound(tmp_dist + sz, tmp_id + sz, 0, deg, tmp_dist[i], tmp_id[i]);
    mv_pos[i] = id + i;
  }
  __syncthreads();

  for (int i = tid; i < sz + deg; i += blockDim.x) {
    mv_id[mv_pos[i]] = tmp_id[i];
    mv_dist[mv_pos[i]] = tmp_dist[i];
  }
  __syncthreads();

  int target = min(sz + deg, 2 * ef_search);
  for (int i = tid; i < target; i += blockDim.x) {
    neighbor_id[offset + i] = mv_id[i];
    neighbor_dist[offset + i] = mv_dist[i];
  }
  if (threadIdx.x == 0) {
    ctx.size = target;
  }

#ifdef GUSTANN_LATENCY_PROBE
  if (tid == 0 && probe_sample_idx >= 0) {
    probe_traces[bid].samples[probe_sample_idx].t_merge_done =
        gustann_globaltimer_ns();
  }
#endif
}

template <class T>
__global__ void unify_kernel(float *qdata, uint8_t *buffer, int32_t *request,
                             const int num_dims, const int max_m,
                             const int ef_search, const int topk, int *nns,
                             float *distances, int *found_cnt,
                             uint32_t *neighbor_id, float *neighbor_dist,
                             int nodes_per_page, int node_len, int data_len,
                             Data *data, int qcnt) {
  int bid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (bid >= qcnt) {
    return;
  }
  int tid = threadIdx.x % 32;
  int offset = (max_m + ef_search + 31) / 32 * 32 * bid;
  float *src_vec = qdata + num_dims * bid;

  int node_u = request[bid];
  if (node_u == -1) {
    return;
  }
  int buffer_offset = node_u % nodes_per_page * node_len;

  T *buffer_u = (T *)(buffer + bid * 4096 + buffer_offset);
  float dist = shared::square_sum_32(src_vec, buffer_u, num_dims);
  shared::retset_push_32(distances + bid * topk, nns + bid * topk,
                         found_cnt[bid], topk, dist, node_u);

  Data &ctx = data[bid];
  int sz = ctx.size;
  int id = sz + 1;
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
    if (!flag && id > mv && ((tmp_id & 0x80000000u) == 0)) {
      id = mv;
    }
    count += __popc(mask);
    __syncwarp();
  }

#pragma unroll
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    int _id = __shfl_down_sync(0xffffffff, id, offset);
    id = min(_id, id);
  }
  if (tid == 0) {
    if (id >= ef_search) {
      request[bid] = -1;
      ctx.size = 0;
    } else {
      request[bid] = neighbor_id[offset + id];
      ctx.size = min(sz - count, ef_search);
      neighbor_id[offset + id] |= 0x80000000u;
    }
  }
}

}  // namespace gustann
