#include "online/int8_gemm_scorer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _cerr = (expr);                                                \
    if (_cerr != cudaSuccess) {                                                \
      throw std::runtime_error(cudaGetErrorString(_cerr));                     \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t _stat = (expr);                                             \
    if (_stat != CUBLAS_STATUS_SUCCESS) {                                      \
      throw std::runtime_error("cublas error code " +                          \
                               std::to_string(static_cast<int>(_stat)));       \
    }                                                                          \
  } while (0)

namespace fusionann::online {

namespace {

__global__ void quantize_fp32_to_int8_kernel(const float *src, int8_t *dst,
                                             int dim, float inv_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dim)
    return;
  float scaled = src[idx] * inv_scale;
  scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
  dst[idx] = static_cast<int8_t>(lrintf(scaled));
}

__global__ void gather_int8_vectors_kernel(
    const int8_t *__restrict__ corpus, const float *__restrict__ norms,
    size_t total_vectors, const long long *__restrict__ ids,
    int8_t *__restrict__ out_matrix, float *__restrict__ out_norms, int dim,
    int num_candidates) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_candidates * dim;
  if (idx >= total)
    return;
  int col = idx / dim;
  int row = idx % dim;
  long long vec_id = ids[col];
  if (vec_id < 0 || static_cast<size_t>(vec_id) >= total_vectors) {
    if (row == 0 && out_norms) {
      out_norms[col] = CUDART_INF_F;
    }
    out_matrix[row + col * dim] = 0;
    return;
  }
  out_matrix[row + col * dim] = corpus[static_cast<size_t>(vec_id) * dim + row];
  if (row == 0 && out_norms) {
    out_norms[col] = norms[vec_id];
  }
}

__global__ void finalize_distances_kernel(int num_candidates,
                                          const int32_t *__restrict__ dots,
                                          const float *__restrict__ cand_norms,
                                          float query_norm, float scale_sq,
                                          float *__restrict__ distances) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_candidates)
    return;
  float dot_real = static_cast<float>(dots[idx]) * scale_sq;
  float dist = query_norm + cand_norms[idx] - 2.0f * dot_real;
  distances[idx] = dist;
}

struct CorpusCache {
  std::mutex mutex;
  std::unordered_map<std::string, std::weak_ptr<Int8Corpus>> entries;
};

CorpusCache &corpus_cache() {
  static CorpusCache cache;
  return cache;
}

int parse_positive_env(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    return 0;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return 0;
  }
  if (parsed > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(parsed);
}

std::shared_ptr<Int8Corpus> build_int8_corpus(IOManager &io_manager, int dim,
                                              const std::string &label,
                                              int preferred_threads) {
  const size_t total = io_manager.vector_count();
  if (total == 0) {
    throw std::runtime_error("IOManager reports zero vectors; cannot build "
                             "int8 corpus for " +
                             label);
  }
  auto corpus = std::make_shared<Int8Corpus>();
  corpus->dim = dim;
  corpus->vector_count = total;

  std::cout << "  > [Int8] Quantizing " << total
            << " vectors for Tensor Core scorer..." << std::endl;

  const size_t chunk = 4096;
  const size_t num_chunks = (total + chunk - 1) / chunk;
  const size_t chunk_elems = chunk * static_cast<size_t>(dim);
  const size_t chunk_vec_bytes = chunk_elems * sizeof(int8_t);
  const size_t chunk_norm_bytes = chunk * sizeof(float);
  constexpr int kPinnedBufferCount = 2;

  std::mutex error_mutex;
  std::exception_ptr worker_error;
  std::atomic<bool> error_flag{false};

  int quant_threads = parse_positive_env("FUSIONANNS_INT8_QUANT_THREADS");
  if (quant_threads <= 0 && preferred_threads > 0) {
    quant_threads = preferred_threads;
  }
  if (quant_threads <= 0) {
    quant_threads = omp_get_max_threads();
  }
  if (quant_threads <= 1) {
    int omp_env = parse_positive_env("OMP_NUM_THREADS");
    if (omp_env > 0) {
      quant_threads = omp_env;
    }
  }
  if (quant_threads <= 1) {
    quant_threads = omp_get_num_procs();
  }
  if (quant_threads <= 0) {
    quant_threads = 1;
  }

  omp_set_dynamic(0);
  omp_set_num_threads(quant_threads);
  std::cout << "  > [Int8] Quantization OMP threads=" << quant_threads
            << std::endl;

  float max_abs = 0.0f;
#pragma omp parallel reduction(max : max_abs)
  {
    std::vector<long> ids_local;
    ids_local.reserve(chunk);
    std::vector<float> floats_local;
    floats_local.reserve(chunk * static_cast<size_t>(dim));

#pragma omp single
    {
      std::cout << "  > [Int8] Quantize pass 1 threads="
                << omp_get_num_threads() << std::endl;
    }

#pragma omp for schedule(dynamic)
    for (long long chunk_idx = 0;
         chunk_idx < static_cast<long long>(num_chunks); ++chunk_idx) {
      if (error_flag.load(std::memory_order_acquire))
        continue;
      size_t offset = static_cast<size_t>(chunk_idx) * chunk;
      size_t take = std::min(chunk, total - offset);
      ids_local.resize(take);
      for (size_t i = 0; i < take; ++i) {
        ids_local[i] = static_cast<long>(offset + i);
      }
      try {
        io_manager.get_vectors(ids_local, floats_local, nullptr);
      } catch (...) {
        if (!error_flag.exchange(true)) {
          std::lock_guard<std::mutex> lk(error_mutex);
          worker_error = std::current_exception();
        }
        continue;
      }
      size_t elems = take * static_cast<size_t>(dim);
      for (size_t j = 0; j < elems; ++j) {
        max_abs = std::max(max_abs, std::fabs(floats_local[j]));
      }
    }
  }
  if (worker_error) {
    std::rethrow_exception(worker_error);
  }

  if (max_abs <= 0.0f) {
    max_abs = 1.0f;
  }
  corpus->scale = max_abs / 127.0f;
  float inv_scale = 1.0f / corpus->scale;

  CUDA_CHECK(cudaMalloc(&corpus->d_vectors,
                        corpus->vector_count * static_cast<size_t>(dim) *
                            sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(
      &corpus->d_norms, corpus->vector_count * sizeof(float)));

  error_flag.store(false, std::memory_order_release);
  worker_error = nullptr;

#pragma omp parallel
  {
    struct StagingBuffer {
      int8_t *quantized = nullptr;
      float *norms = nullptr;
      cudaEvent_t event = nullptr;
      bool busy = false;
    };

    std::array<StagingBuffer, kPinnedBufferCount> staging{};
    for (auto &buf : staging) {
      CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&buf.quantized),
                               chunk_vec_bytes, cudaHostAllocPortable));
      CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&buf.norms),
                               chunk_norm_bytes, cudaHostAllocPortable));
      CUDA_CHECK(
          cudaEventCreateWithFlags(&buf.event, cudaEventDisableTiming));
      buf.busy = false;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto wait_for_buffer = [&](StagingBuffer &buf) {
      if (buf.busy) {
        CUDA_CHECK(cudaEventSynchronize(buf.event));
        buf.busy = false;
      }
    };

    std::vector<long> ids_local;
    ids_local.reserve(chunk);
    std::vector<float> floats_local;
    floats_local.reserve(chunk * static_cast<size_t>(dim));
    size_t next_buffer = 0;

#pragma omp single
    {
      std::cout << "  > [Int8] Quantize pass 2 threads="
                << omp_get_num_threads() << std::endl;
    }

#pragma omp for schedule(dynamic)
    for (long long chunk_idx = 0;
         chunk_idx < static_cast<long long>(num_chunks); ++chunk_idx) {
      if (error_flag.load(std::memory_order_acquire))
        continue;
      size_t offset = static_cast<size_t>(chunk_idx) * chunk;
      size_t take = std::min(chunk, total - offset);
      ids_local.resize(take);
      for (size_t i = 0; i < take; ++i) {
        ids_local[i] = static_cast<long>(offset + i);
      }
      try {
        io_manager.get_vectors(ids_local, floats_local, nullptr);
      } catch (...) {
        if (!error_flag.exchange(true)) {
          std::lock_guard<std::mutex> lk(error_mutex);
          worker_error = std::current_exception();
        }
        continue;
      }

      StagingBuffer &buf = staging[next_buffer];
      next_buffer = (next_buffer + 1) % staging.size();
      wait_for_buffer(buf);

      int8_t *quantized_dst = buf.quantized;
      float *norm_dst = buf.norms;
      for (size_t i = 0; i < take; ++i) {
        float norm = 0.0f;
        float *vec = floats_local.data() + i * static_cast<size_t>(dim);
        int8_t *dst = quantized_dst + i * static_cast<size_t>(dim);
        for (int d = 0; d < dim; ++d) {
          float value = vec[d];
          norm += value * value;
          float q = value * inv_scale;
          q = std::max(-127.0f, std::min(127.0f, std::nearbyint(q)));
          dst[d] = static_cast<int8_t>(q);
        }
        norm_dst[i] = norm;
      }

      size_t vec_bytes = take * static_cast<size_t>(dim) * sizeof(int8_t);
      size_t norm_bytes = take * sizeof(float);
      CUDA_CHECK(cudaMemcpyAsync(
          corpus->d_vectors + offset * static_cast<size_t>(dim), quantized_dst,
          vec_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(corpus->d_norms + offset, norm_dst, norm_bytes,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaEventRecord(buf.event, stream));
      buf.busy = true;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (auto &buf : staging) {
      if (buf.busy) {
        CUDA_CHECK(cudaEventSynchronize(buf.event));
      }
      if (buf.event) {
        cudaEventDestroy(buf.event);
        buf.event = nullptr;
      }
      if (buf.quantized) {
        cudaFreeHost(buf.quantized);
        buf.quantized = nullptr;
      }
      if (buf.norms) {
        cudaFreeHost(buf.norms);
        buf.norms = nullptr;
      }
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
  if (worker_error) {
    std::rethrow_exception(worker_error);
  }
  std::cout << "  > [Int8] Quantization complete. scale=" << corpus->scale
            << std::endl;
  return corpus;
}

} // namespace

Int8Corpus::~Int8Corpus() {
  if (d_vectors) {
    cudaFree(d_vectors);
    d_vectors = nullptr;
  }
  if (d_norms) {
    cudaFree(d_norms);
    d_norms = nullptr;
  }
}

std::shared_ptr<Int8Corpus>
acquire_int8_corpus(const std::string &cache_key, IOManager &io_manager,
                    int dim, int preferred_threads) {
  CorpusCache &cache = corpus_cache();
  std::shared_ptr<Int8Corpus> corpus;
  {
    std::lock_guard<std::mutex> lk(cache.mutex);
    auto it = cache.entries.find(cache_key);
    if (it != cache.entries.end()) {
      corpus = it->second.lock();
    }
  }
  if (corpus) {
    return corpus;
  }

  corpus = build_int8_corpus(io_manager, dim, cache_key, preferred_threads);
  {
    std::lock_guard<std::mutex> lk(cache.mutex);
    cache.entries[cache_key] = corpus;
  }
  return corpus;
}

Int8GemmScorer::Int8GemmScorer(std::shared_ptr<Int8Corpus> corpus, int c_max)
    : corpus_(std::move(corpus)) {
  if (!corpus_) {
    throw std::invalid_argument("Int8GemmScorer requires a valid corpus");
  }
  plan_.dim = corpus_->dim;
  plan_.c_max = std::max(1, c_max);
  size_t offset = 0;
  auto reserve = [&](size_t bytes, size_t alignment) {
    offset = align_up_h(offset, alignment);
    offset += bytes;
  };
  reserve(static_cast<size_t>(plan_.dim) * sizeof(int8_t), alignof(int8_t));
  reserve(static_cast<size_t>(plan_.dim) * static_cast<size_t>(plan_.c_max) *
              sizeof(int8_t),
          alignof(int8_t));
  reserve(static_cast<size_t>(plan_.c_max) * sizeof(float), alignof(float));
  reserve(static_cast<size_t>(plan_.c_max) * sizeof(int32_t), alignof(int32_t));
  plan_.bytes_per_block = align_up_h(offset, 256);
  scale_sq_ = corpus_->scale * corpus_->scale;
}

Int8GemmScorer::~Int8GemmScorer() = default;

namespace {

struct ThreadLocalCublasHandle {
  ThreadLocalCublasHandle() {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  }

  ~ThreadLocalCublasHandle() {
    if (handle) {
      cublasDestroy(handle);
      handle = nullptr;
    }
  }

  cublasHandle_t handle = nullptr;
};

} // namespace

cublasHandle_t Int8GemmScorer::ensure_handle(GpuBlock * /*block*/) {
  static thread_local ThreadLocalCublasHandle tls_handle;
  return tls_handle.handle;
}

Int8GemmScorer::Workspace Int8GemmScorer::prepare_workspace(GpuBlock &block) {
  Workspace ws{};
  block.reset();
  ws.d_query_int8 =
      block.allocate<int8_t>(static_cast<size_t>(plan_.dim), alignof(int8_t));
  ws.d_candidate_matrix = block.allocate<int8_t>(
      static_cast<size_t>(plan_.dim) * static_cast<size_t>(plan_.c_max),
      alignof(int8_t));
  ws.d_candidate_norms =
      block.allocate<float>(static_cast<size_t>(plan_.c_max), alignof(float));
  ws.d_dot_int32 =
      block.allocate<int32_t>(static_cast<size_t>(plan_.c_max), alignof(int32_t));
  ws.handle = ensure_handle(nullptr);
  return ws;
}

void Int8GemmScorer::prepare_query(const float *h_query, int query_dim,
                                   Workspace &workspace,
                                   cudaStream_t stream_xfer,
                                   cudaStream_t stream_comp,
                                   ThreadGpuEvents &events,
                                   bool timing_enabled,
                                   const float *d_query_mapped) {
  (void)stream_xfer;
  if (query_dim != plan_.dim) {
    throw std::runtime_error("Query dimension mismatch for Int8GemmScorer");
  }
  if (!h_query) {
    throw std::runtime_error("Null mapped query buffer for Int8GemmScorer");
  }

  float norm = 0.0f;
  for (int i = 0; i < plan_.dim; ++i) {
    float v = h_query[i];
    norm += v * v;
  }
  workspace.query_norm = norm;

  const float *device_query = d_query_mapped;
  if (!device_query) {
    float *mapped_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&mapped_ptr), const_cast<float *>(h_query),
        0));
    device_query = mapped_ptr;
  }

  if (timing_enabled) {
    CUDA_CHECK(cudaEventRecord(events.dt_start, stream_comp));
  }
  int threads = 256;
  int blocks = (plan_.dim + threads - 1) / threads;
  quantize_fp32_to_int8_kernel<<<blocks, threads, 0, stream_comp>>>(
      device_query, workspace.d_query_int8, plan_.dim, 1.0f / corpus_->scale);
  if (timing_enabled) {
    CUDA_CHECK(cudaEventRecord(events.dt_done, stream_comp));
  }
}

void Int8GemmScorer::score_candidates(
    int unique_candidate_count, const long long *h_candidate_pinned,
    Workspace &workspace, cudaStream_t stream_xfer,
    cudaStream_t stream_comp, ThreadGpuEvents &events, bool timing_enabled,
    const long long *d_candidate_pinned, bool /* use_gpu_topk */) {
  (void)stream_xfer;
  if (unique_candidate_count <= 0) {
    events.recorded_h2d = false;
    events.recorded_pq = false;
    return;
  }
  if (unique_candidate_count > plan_.c_max) {
    throw std::runtime_error("Candidate count exceeds Int8 workspace capacity");
  }

  if (!h_candidate_pinned) {
    throw std::runtime_error("Null candidate buffer for Int8GemmScorer");
  }
  events.recorded_h2d = false;

  const long long *device_ids = d_candidate_pinned;
  if (!device_ids) {
    long long *mapped_ids = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&mapped_ids),
        const_cast<long long *>(h_candidate_pinned), 0));
    device_ids = mapped_ids;
  }

  int threads = 256;
  int blocks = (unique_candidate_count * plan_.dim + threads - 1) / threads;
  gather_int8_vectors_kernel<<<blocks, threads, 0, stream_comp>>>(
      corpus_->d_vectors, corpus_->d_norms, corpus_->vector_count,
      device_ids, workspace.d_candidate_matrix, workspace.d_candidate_norms,
      plan_.dim, unique_candidate_count);

  events.recorded_pq = timing_enabled;
  if (timing_enabled) {
    CUDA_CHECK(cudaEventRecord(events.pq_start, stream_comp));
  }
  CUBLAS_CHECK(cublasSetStream(workspace.handle, stream_comp));
  int m = unique_candidate_count;
  int n = 1;
  int k = plan_.dim;
  const int32_t alpha = 1;
  const int32_t beta = 0;
  CUBLAS_CHECK(cublasGemmEx(
      workspace.handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
      workspace.d_candidate_matrix, CUDA_R_8I, k, workspace.d_query_int8,
      CUDA_R_8I, k, &beta, workspace.d_dot_int32, CUDA_R_32I, m,
      CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (timing_enabled) {
    CUDA_CHECK(cudaEventRecord(events.pq_stop, stream_comp));
  }

  if (!events.h_dists || events.host_capacity < unique_candidate_count) {
    throw std::runtime_error("Insufficient host distance buffer for Int8 scorer");
  }
  if (!events.d_dists_mapped) {
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&events.d_dists_mapped), events.h_dists, 0));
  }

  threads = 256;
  blocks = (unique_candidate_count + threads - 1) / threads;
  finalize_distances_kernel<<<blocks, threads, 0, stream_comp>>>(
      unique_candidate_count, workspace.d_dot_int32,
      workspace.d_candidate_norms, workspace.query_norm, scale_sq_,
      events.d_dists_mapped);
}

void Int8GemmScorer::fetch_results(
    int unique_candidate_count, Workspace &workspace,
    cudaStream_t stream_comp, ThreadGpuEvents &events, GpuTimings *timings,
    std::vector<ResultPair> &out_results, const long long *candidate_ids_host,
    double *wait_cpu_ms, int /* top_k */, bool /* use_gpu_topk */,
    const long long * /* d_candidate_pinned */) {
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
  out_results.clear();
  if (unique_candidate_count <= 0) {
    events.pending_copy_count = 0;
    events.recorded_d2h = false;
    return;
  }
  if (!candidate_ids_host) {
    throw std::runtime_error("Missing host candidate ids for Int8GemmScorer");
  }
  events.recorded_d2h = false;
  timed_sync("WaitCompStream",
             [&] { return cudaStreamSynchronize(stream_comp); });
  events.pending_copy_count = unique_candidate_count;
  const int copy_count = events.pending_copy_count;
  out_results.reserve(static_cast<size_t>(copy_count));
  for (int i = 0; i < copy_count; ++i) {
    long long id = candidate_ids_host[i];
    float dist = events.h_dists[i];
    if (id < 0 || !std::isfinite(dist))
      continue;
    out_results.push_back({dist, static_cast<long>(id)});
  }
}

} // namespace fusionann::online
