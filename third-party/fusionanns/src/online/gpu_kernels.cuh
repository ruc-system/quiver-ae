#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#include <faiss/impl/ProductQuantizer.h>

struct ResultPair {
  float dist;
  long id;
};

struct GpuTimings {
  float h2d_ids_ms = 0.0f;
  float dedup_ms = 0.0f;
  float pq_ms = 0.0f;
  float sort_ms = 0.0f;
  float d2h_topn_ms = 0.0f;
  int candidates_in = 0;
  int candidates_unique = 0;
  int topk_requested = 0;
  size_t tmp_storage_bytes = 0;
};

struct ThreadGpuEvents {
  cudaEvent_t query_ready = nullptr;
  cudaEvent_t dt_start = nullptr;
  cudaEvent_t dt_done = nullptr;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_stop = nullptr;
  cudaEvent_t dedup_start = nullptr;
  cudaEvent_t dedup_stop = nullptr;
  cudaEvent_t pq_start = nullptr;
  cudaEvent_t pq_stop = nullptr;
  cudaEvent_t d2h_start = nullptr;
  cudaEvent_t d2h_stop = nullptr;
  float *h_dists = nullptr;
  long long *h_ids = nullptr;
  float *d_dists_mapped = nullptr;
  long long *d_ids_mapped = nullptr;
  int host_capacity = 0;
  int *h_unique_count = nullptr;
  int pending_copy_count = 0;
  bool recorded_h2d = false;
  bool recorded_dedup = false;
  bool recorded_pq = false;
  bool recorded_d2h = false;
  bool host_results_current = false;
};

struct GpuDedupSet {
  void *handle = nullptr;
  int capacity = 0;
};

// 将 GPU 侧计算完成的 (id, dist) 候选复制到主机端缓冲
void copy_pq_results_to_host(int num_candidates, int device_capacity,
                             const float *d_dists, const long long *d_ids,
                             cudaStream_t stream_comp, ThreadGpuEvents &events,
                             GpuTimings *timings = nullptr);

void compute_all_pq_distances(const uint8_t *all_pq_codes_gpu,
                              const long long *candidate_ids,
                              const float *dist_table, float *out_dists, int M,
                              int ksub, int device_capacity, int num_candidates,
                              cudaStream_t stream);

void upload_pq_codebooks_to_gpu(const faiss::ProductQuantizer &pq,
                                float **d_codebooks, int M, int ksub, int dsub);

void build_distance_table_gpu(const float *d_query, const float *d_codebooks,
                              float *d_table, int M, int ksub, int dsub,
                              cudaStream_t stream = 0);

void compute_cub_workspace_sizes(int num_items, size_t &select_bytes,
                                 size_t &sort_bytes);

#ifdef FUSIONANNS_USE_CUVS_SELECT_K
// GPU TopK via cuVS select_k; writes top k to d_out_dists, d_out_ids.
// Returns number written (min(num_candidates, k)).
int select_top_k_gpu(const float *d_dists, const long long *d_ids,
                     int num_candidates, int k, float *d_out_dists,
                     long long *d_out_ids, cudaStream_t stream);
#endif
