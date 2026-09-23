#pragma once

#include "common/nvtx_utils.h"
#include "online/gpu_kernels.cuh"
#include "online/gpu_pool.h"
#include "online/io_manager.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <string>
#include <vector>

namespace fusionann::online {

struct Int8Corpus {
  int dim = 0;
  size_t vector_count = 0;
  float scale = 1.0f;
  int8_t *d_vectors = nullptr;
  float *d_norms = nullptr;
  ~Int8Corpus();
};

std::shared_ptr<Int8Corpus>
acquire_int8_corpus(const std::string &cache_key, IOManager &io_manager,
                    int dim, int preferred_threads = 0);

class Int8GemmScorer {
public:
  Int8GemmScorer(std::shared_ptr<Int8Corpus> corpus, int c_max);
  ~Int8GemmScorer();

  size_t bytes_per_block() const { return plan_.bytes_per_block; }
  int candidate_capacity() const { return plan_.c_max; }
  int query_dim() const { return plan_.dim; }
  size_t tmp_storage_bytes() const { return 0; }

  struct Workspace {
    int8_t *d_query_int8 = nullptr;
    int8_t *d_candidate_matrix = nullptr;
    float *d_candidate_norms = nullptr;
    int32_t *d_dot_int32 = nullptr;
    float query_norm = 0.0f;
    cublasHandle_t handle = nullptr;
  };

  Workspace prepare_workspace(GpuBlock &block);
  void prepare_query(const float *h_query, int query_dim, Workspace &workspace,
                     cudaStream_t stream_xfer, cudaStream_t stream_comp,
                     ThreadGpuEvents &events, bool timing_enabled,
                     const float *d_query_mapped = nullptr);
  void score_candidates(int unique_candidate_count,
                        const long long *h_candidate_pinned,
                        Workspace &workspace, cudaStream_t stream_xfer,
                        cudaStream_t stream_comp, ThreadGpuEvents &events,
                        bool timing_enabled,
                        const long long *d_candidate_pinned = nullptr,
                        bool use_gpu_topk = true);
  void fetch_results(int unique_candidate_count, Workspace &workspace,
                     cudaStream_t stream_comp, ThreadGpuEvents &events,
                     GpuTimings *timings,
                     std::vector<ResultPair> &out_results,
                     const long long *candidate_ids_host = nullptr,
                     double *wait_cpu_ms = nullptr, int top_k = 0,
                     bool use_gpu_topk = true,
                     const long long *d_candidate_pinned = nullptr);

private:
  struct Plan {
    int dim = 0;
    int c_max = 0;
    size_t bytes_per_block = 0;
  };

  cublasHandle_t ensure_handle(GpuBlock *block);

  std::shared_ptr<Int8Corpus> corpus_;
  Plan plan_;
  float scale_sq_ = 1.0f;
};

} // namespace fusionann::online
