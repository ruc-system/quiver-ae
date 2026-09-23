#pragma once

#include "common/nvtx_utils.h"
#include "online/gpu_kernels.cuh"
#include "online/gpu_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/ProductQuantizer.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      throw std::runtime_error(cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)
#endif

namespace fusionann::online {

struct PqWorkspacePlan {
  int M = 0;
  int ksub = 0;
  int dsub = 0;
  int dim = 0;
  int c_max = 0;
  size_t tmp_storage_bytes = 0;
  size_t bytes_per_block = 0;
};

struct PqWorkspaceView {
  float *d_dist_table = nullptr;
  long long *d_ids_in = nullptr;
  long long *d_ids_unique = nullptr;
  float *d_dists = nullptr;
  long long *d_ids_alt = nullptr;
  float *d_dists_alt = nullptr;
  faiss::idx_t *d_topk_indices = nullptr;
  float *d_query = nullptr;
  int *d_unique_count = nullptr;
  void *d_tmp_scratch = nullptr;
  size_t tmp_storage_bytes = 0;
};

class PqLutScorer {
public:
  PqLutScorer(const faiss::ProductQuantizer &pq, const uint8_t *d_codes,
              float *d_codebooks, int query_dim, int c_max);

  size_t bytes_per_block() const { return plan_.bytes_per_block; }
  int candidate_capacity() const { return plan_.c_max; }
  int query_dim() const { return plan_.dim; }
  size_t tmp_storage_bytes() const { return plan_.tmp_storage_bytes; }

  PqWorkspaceView prepare_workspace(GpuBlock &block) const;

  void prepare_query(const float *h_query, int query_dim,
                     PqWorkspaceView &workspace, cudaStream_t stream_xfer,
                     cudaStream_t stream_comp, ThreadGpuEvents &events,
                     bool timing_enabled,
                     const float *d_query_mapped = nullptr) const;

  void score_candidates(int unique_candidate_count,
                        const long long *h_candidate_pinned,
                        PqWorkspaceView &workspace, cudaStream_t stream_xfer,
                        cudaStream_t stream_comp, ThreadGpuEvents &events,
                        bool timing_enabled,
                        const long long *d_candidate_pinned = nullptr,
                        bool use_gpu_topk = true) const;

  void fetch_results(int unique_candidate_count, PqWorkspaceView &workspace,
                     cudaStream_t stream_comp, ThreadGpuEvents &events,
                     GpuTimings *timings_ptr,
                     std::vector<ResultPair> &out_results,
                     const long long *candidate_ids_host = nullptr,
                     double *wait_cpu_ms = nullptr, int top_k = 0,
                     bool use_gpu_topk = true,
                     const long long *d_candidate_pinned = nullptr) const;

private:
  static PqWorkspacePlan make_plan(const faiss::ProductQuantizer &pq, int dim,
                                   int c_max);
  PqWorkspaceView allocate_workspace(GpuBlock &block) const;

  PqWorkspacePlan plan_;
  const uint8_t *d_codes_ = nullptr;
  float *d_codebooks_ = nullptr;
};

inline PqWorkspacePlan PqLutScorer::make_plan(const faiss::ProductQuantizer &pq,
                                              int dim, int c_max) {
  if (dim <= 0 || c_max <= 0) {
    throw std::invalid_argument("Invalid PQ scorer configuration");
  }
  PqWorkspacePlan plan;
  plan.M = static_cast<int>(pq.M);
  plan.ksub = static_cast<int>(pq.ksub);
  plan.dsub = static_cast<int>(pq.dsub);
  plan.dim = dim;
  plan.c_max = c_max;

  const size_t bytes_dist = static_cast<size_t>(plan.M) *
                            static_cast<size_t>(plan.ksub) * sizeof(float);
  const size_t bytes_ids = static_cast<size_t>(plan.c_max) * sizeof(long long);
  const size_t bytes_dists = static_cast<size_t>(plan.c_max) * sizeof(float);
  const size_t bytes_topk_indices =
      static_cast<size_t>(plan.c_max) * sizeof(faiss::idx_t);
  const size_t bytes_query = static_cast<size_t>(plan.dim) * sizeof(float);
  const size_t bytes_unique_count = sizeof(int);

  size_t tmp_select = 0;
  size_t tmp_sort = 0;
  compute_cub_workspace_sizes(plan.c_max, tmp_select, tmp_sort);
  plan.tmp_storage_bytes = std::max(tmp_select, tmp_sort);

  size_t offset = 0;
  auto reserve = [&](size_t size, size_t alignment) {
    offset = align_up_h(offset, alignment);
    offset += size;
  };
  reserve(bytes_dist, alignof(float));
  reserve(bytes_ids, alignof(long long)); // d_ids_in
  reserve(bytes_ids, alignof(long long)); // d_ids_unique
  reserve(bytes_dists, alignof(float));   // d_dists
  reserve(bytes_ids, alignof(long long)); // d_ids_alt
  reserve(bytes_dists, alignof(float));   // d_dists_alt
  reserve(bytes_topk_indices, alignof(faiss::idx_t));
  reserve(bytes_query, alignof(float));
  reserve(bytes_unique_count, alignof(int));
  if (plan.tmp_storage_bytes > 0) {
    reserve(plan.tmp_storage_bytes,
            std::max<size_t>(8, alignof(std::max_align_t)));
  }
  plan.bytes_per_block = align_up_h(offset, 256);
  return plan;
}

inline PqLutScorer::PqLutScorer(const faiss::ProductQuantizer &pq,
                                const uint8_t *d_codes, float *d_codebooks,
                                int query_dim, int c_max)
    : plan_(make_plan(pq, query_dim, c_max)), d_codes_(d_codes),
      d_codebooks_(d_codebooks) {
  if (!d_codes_ || !d_codebooks_) {
    throw std::invalid_argument("PQ scorer requires valid GPU buffers");
  }
}

inline PqWorkspaceView PqLutScorer::allocate_workspace(GpuBlock &block) const {
  block.reset();
  PqWorkspaceView view{};
  view.d_dist_table = block.allocate<float>(static_cast<size_t>(plan_.M) *
                                                static_cast<size_t>(plan_.ksub),
                                            alignof(float));
  const size_t ids_count = static_cast<size_t>(plan_.c_max);
  view.d_ids_in = block.allocate<long long>(ids_count, alignof(long long));
  view.d_ids_unique = block.allocate<long long>(ids_count, alignof(long long));
  view.d_dists = block.allocate<float>(ids_count, alignof(float));
  view.d_ids_alt = block.allocate<long long>(ids_count, alignof(long long));
  view.d_dists_alt = block.allocate<float>(ids_count, alignof(float));
  view.d_topk_indices =
      block.allocate<faiss::idx_t>(ids_count, alignof(faiss::idx_t));
  view.d_query =
      block.allocate<float>(static_cast<size_t>(plan_.dim), alignof(float));
  view.d_unique_count = block.allocate<int>(1, alignof(int));
  if (plan_.tmp_storage_bytes > 0) {
    view.d_tmp_scratch = static_cast<void *>(block.allocate<std::uint8_t>(
        plan_.tmp_storage_bytes,
        std::max<size_t>(8, alignof(std::max_align_t))));
  }
  view.tmp_storage_bytes = plan_.tmp_storage_bytes;
  return view;
}

inline PqWorkspaceView PqLutScorer::prepare_workspace(GpuBlock &block) const {
  return allocate_workspace(block);
}

inline void PqLutScorer::prepare_query(
    const float *h_query, int query_dim, PqWorkspaceView &workspace,
    cudaStream_t stream_xfer, cudaStream_t stream_comp, ThreadGpuEvents &events,
    bool timing_enabled, const float *d_query_mapped) const {
  if (!h_query || !workspace.d_dist_table) {
    throw std::runtime_error("PQ scorer workspace not initialized");
  }
  if (query_dim != plan_.dim) {
    throw std::runtime_error("Query dimension mismatch for PQ scorer");
  }

  const float *query_device_ptr = d_query_mapped;
  if (!query_device_ptr) {
    if (!workspace.d_query) {
      throw std::runtime_error("PQ scorer workspace missing query buffer");
    }
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_query, h_query,
                               static_cast<size_t>(plan_.dim) * sizeof(float),
                               cudaMemcpyHostToDevice, stream_xfer));
    CUDA_CHECK(cudaEventRecord(events.query_ready, stream_xfer));
    CUDA_CHECK(cudaStreamWaitEvent(stream_comp, events.query_ready, 0));
    query_device_ptr = workspace.d_query;
  }

  if (timing_enabled) {
    CUDA_CHECK(cudaEventRecord(events.dt_start, stream_comp));
  }
  {
    FUSIONANNS_NVTX_RANGE_COLOR("BuildDistanceTable",
                                0xFF009688u); // PQ距离表 / distance table build
    build_distance_table_gpu(query_device_ptr, d_codebooks_,
                             workspace.d_dist_table, plan_.M, plan_.ksub,
                             plan_.dsub, stream_comp);
  }
  if (timing_enabled) {
    CUDA_CHECK(cudaEventRecord(events.dt_done, stream_comp));
  }
}

inline void PqLutScorer::score_candidates(
    int unique_candidate_count, const long long *h_candidate_pinned,
    PqWorkspaceView &workspace, cudaStream_t stream_xfer,
    cudaStream_t stream_comp, ThreadGpuEvents &events, bool timing_enabled,
    const long long *d_candidate_pinned, bool use_gpu_topk) const {
  if (unique_candidate_count <= 0) {
    events.recorded_h2d = false;
    events.recorded_pq = false;
    return;
  }
  if (unique_candidate_count > plan_.c_max) {
    throw std::runtime_error("PQ scorer candidate count exceeds capacity");
  }
  if (!h_candidate_pinned || !workspace.d_ids_unique || !workspace.d_dists) {
    throw std::runtime_error("Pinned host candidate buffer is null");
  }

  // host 直写为零拷贝优化，比 device+D2H 快得多；use_gpu_topk 时若命中 host 路径会静默用 CPU TopK
  const bool host_output_available =
      events.h_dists && events.host_capacity >= unique_candidate_count;
  float *distance_out = workspace.d_dists;
  if (host_output_available) {
    if (!events.d_dists_mapped) {
      CUDA_CHECK(cudaHostGetDevicePointer(
          reinterpret_cast<void **>(&events.d_dists_mapped), events.h_dists,
          0));
    }
    distance_out = events.d_dists_mapped;
  }
  events.host_results_current = host_output_available;

  const bool require_device_ids = !host_output_available;
  const size_t bytes_ids =
      static_cast<size_t>(unique_candidate_count) * sizeof(long long);
  const long long *device_ids = d_candidate_pinned;
  if (!device_ids || require_device_ids) {
    if (!workspace.d_ids_unique) {
      throw std::runtime_error("PQ scorer workspace missing id buffer");
    }
    FUSIONANNS_NVTX_RANGE_COLOR(
        "UploadCandidates",
        0xFF8BC34Au); // 候选上传与预评分 / candidate upload & pre-score
    events.recorded_h2d = timing_enabled;
    if (timing_enabled) {
      CUDA_CHECK(cudaEventRecord(events.h2d_start, stream_xfer));
    }
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_ids_unique, h_candidate_pinned,
                               bytes_ids, cudaMemcpyHostToDevice, stream_xfer));
    CUDA_CHECK(cudaEventRecord(events.h2d_stop, stream_xfer));
    CUDA_CHECK(cudaStreamWaitEvent(stream_comp, events.h2d_stop, 0));
    device_ids = workspace.d_ids_unique;
  } else {
    events.recorded_h2d = false;
  }

  events.recorded_pq = timing_enabled;
  if (timing_enabled) {
    CUDA_CHECK(cudaEventRecord(events.pq_start, stream_comp));
  }
  compute_all_pq_distances(d_codes_, device_ids, workspace.d_dist_table,
                           distance_out, plan_.M, plan_.ksub, plan_.c_max,
                           unique_candidate_count, stream_comp);
  if (timing_enabled) {
    CUDA_CHECK(cudaEventRecord(events.pq_stop, stream_comp));
  }

  if (!host_output_available) {
    events.host_results_current = false;
  }
}

inline void PqLutScorer::fetch_results(
    int unique_candidate_count, PqWorkspaceView &workspace,
    cudaStream_t stream_comp, ThreadGpuEvents &events, GpuTimings *timings_ptr,
    std::vector<ResultPair> &out_results, const long long *candidate_ids_host,
    double *wait_cpu_ms, int top_k, bool use_gpu_topk,
    const long long *d_candidate_pinned) const {
  auto timed_sync = [&](const char *label, auto &&fn) {
    if (!wait_cpu_ms) {
      CUDA_CHECK(fn());
      return;
    }
    FUSIONANNS_NVTX_RANGE_COLOR(label, 0xFF9E9E9Eu);
    auto begin = std::chrono::steady_clock::now();
    CUDA_CHECK(fn());
    auto end = std::chrono::steady_clock::now();
    *wait_cpu_ms +=
        std::chrono::duration<double, std::milli>(end - begin).count();
  };
  if (unique_candidate_count <= 0) {
    out_results.clear();
    events.pending_copy_count = 0;
    events.recorded_d2h = false;
    events.recorded_pq = false;
    return;
  }

  const bool host_results_ready =
      events.host_results_current && events.h_dists &&
      events.host_capacity >= unique_candidate_count &&
      candidate_ids_host != nullptr;
  if (host_results_ready) {
    timed_sync("WaitCompStream",
               [&] { return cudaStreamSynchronize(stream_comp); });
#ifdef FUSIONANNS_USE_CUVS_SELECT_K
    if (use_gpu_topk && top_k > 0 && unique_candidate_count > top_k &&
        d_candidate_pinned != nullptr && events.h_ids &&
        events.host_capacity >= top_k) {
      int actual_k = select_top_k_gpu(
          events.d_dists_mapped, d_candidate_pinned, unique_candidate_count,
          top_k, workspace.d_dists_alt, workspace.d_ids_alt, stream_comp);
      copy_pq_results_to_host(actual_k, plan_.c_max, workspace.d_dists_alt,
                              workspace.d_ids_alt, stream_comp, events,
                              timings_ptr);
      if (events.pending_copy_count > 0) {
        timed_sync("WaitD2HCopy",
                  [&] { return cudaEventSynchronize(events.d2h_stop); });
      }
      out_results.clear();
      out_results.reserve(static_cast<size_t>(events.pending_copy_count));
      for (int i = 0; i < events.pending_copy_count; ++i) {
        long long id = events.h_ids[i];
        float dist = events.h_dists[i];
        if (id < 0 || !std::isfinite(dist))
          continue;
        out_results.push_back({dist, static_cast<long>(id)});
      }
      events.host_results_current = false;
      return;
    }
#endif
    out_results.clear();
    out_results.reserve(static_cast<size_t>(unique_candidate_count));
    for (int i = 0; i < unique_candidate_count; ++i) {
      long long id = candidate_ids_host[i];
      float dist = events.h_dists[i];
      if (id < 0 || !std::isfinite(dist))
        continue;
      out_results.push_back({dist, static_cast<long>(id)});
    }
    if (top_k > 0 &&
        static_cast<int>(out_results.size()) > top_k) {
      auto nth = out_results.begin() +
                 static_cast<std::vector<ResultPair>::difference_type>(top_k);
      std::nth_element(out_results.begin(), nth, out_results.end(),
                       [](const ResultPair &a, const ResultPair &b) {
                         return a.dist < b.dist;
                       });
      out_results.resize(static_cast<size_t>(top_k));
      std::sort(out_results.begin(), out_results.end(),
                [](const ResultPair &a, const ResultPair &b) {
                  return a.dist < b.dist;
                });
    }
    events.pending_copy_count = static_cast<int>(out_results.size());
    events.recorded_d2h = false;
    events.host_results_current = false;
    if (timings_ptr) {
      timings_ptr->d2h_topn_ms = 0.0f;
    }
    return;
  }

  events.host_results_current = false;

  int copy_count = unique_candidate_count;
  const float *d_dists_src = workspace.d_dists;
  const long long *d_ids_src = workspace.d_ids_unique;
#ifdef FUSIONANNS_USE_CUVS_SELECT_K
  if (use_gpu_topk && top_k > 0 && unique_candidate_count > top_k) {
    int actual_k =
        select_top_k_gpu(workspace.d_dists, workspace.d_ids_unique,
                         unique_candidate_count, top_k, workspace.d_dists_alt,
                         workspace.d_ids_alt, stream_comp);
    copy_count = actual_k;
    d_dists_src = workspace.d_dists_alt;
    d_ids_src = workspace.d_ids_alt;
  }
#endif
  copy_pq_results_to_host(copy_count, plan_.c_max, d_dists_src, d_ids_src,
                          stream_comp, events, timings_ptr);

  if (events.pending_copy_count > 0) {
    timed_sync("WaitD2HCopy",
               [&] { return cudaEventSynchronize(events.d2h_stop); });
  }

  const int result_count = events.pending_copy_count;
  out_results.clear();
  out_results.reserve(static_cast<size_t>(result_count));
  for (int i = 0; i < result_count; ++i) {
    long long id = events.h_ids[i];
    float dist = events.h_dists[i];
    if (id < 0 || !std::isfinite(dist))
      continue;
    out_results.push_back({dist, static_cast<long>(id)});
  }
}

} // namespace fusionann::online
