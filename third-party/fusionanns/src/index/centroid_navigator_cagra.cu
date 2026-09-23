#ifdef FUSIONANNS_USE_CUVS_CAGRA

#ifndef RAFT_SYSTEM_LITTLE_ENDIAN
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define RAFT_SYSTEM_LITTLE_ENDIAN 1
#elif defined(_WIN32) || defined(__x86_64__) || defined(__i386__)
#define RAFT_SYSTEM_LITTLE_ENDIAN 1
#else
#define RAFT_SYSTEM_LITTLE_ENDIAN 0
#endif
#endif

#include "index/centroid_navigator_cagra.h"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>

#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)
#endif

using hclock = std::chrono::steady_clock;

inline double us_between(hclock::time_point a, hclock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

/// 原子地更新 max 值（lock-free）
inline void atomic_max(std::atomic<double> &target, double value) {
  double prev = target.load(std::memory_order_relaxed);
  while (value > prev &&
         !target.compare_exchange_weak(prev, value, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

/// 原子累加 double（lock-free）
inline void atomic_add(std::atomic<double> &target, double value) {
  double prev = target.load(std::memory_order_relaxed);
  while (!target.compare_exchange_weak(prev, prev + value,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

} // namespace

// ============================================================================
// GPU globaltimer probe kernel
// ============================================================================

__device__ __forceinline__ unsigned long long globaltimer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void record_gpu_timestamp(unsigned long long *d_out, int slot) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_out[slot] = globaltimer_ns();
  }
}

// ============================================================================
// CagraProfileStats 实现（lock-free atomics）
// ============================================================================

void CagraProfileStats::accumulate(const CagraSearchProfile &p) {
  count.fetch_add(1, std::memory_order_relaxed);
  atomic_add(sum_acquire_slot_us, p.acquire_slot_us);
  atomic_add(sum_alloc_query_us, p.alloc_query_us);
  atomic_add(sum_h2d_query_us, p.h2d_query_us);
  atomic_add(sum_alloc_output_us, p.alloc_output_us);
  atomic_add(sum_kernel_launch_us, p.kernel_launch_us);
  atomic_add(sum_stream_sync_us, p.stream_sync_us);
  atomic_add(sum_d2h_result_us, p.d2h_result_us);
  atomic_add(sum_assemble_us, p.assemble_us);
  atomic_add(sum_total_us, p.total_us);
  atomic_add(sum_gpu_kernel_us, p.gpu_kernel_us);
  atomic_add(sum_collect_wait_us, p.collect_wait_us);
  atomic_max(max_acquire_slot_us, p.acquire_slot_us);
  atomic_max(max_total_us, p.total_us);
  sum_batch_size.fetch_add(p.batch_size, std::memory_order_relaxed);
  batch_count.fetch_add(1, std::memory_order_relaxed);
}

void CagraProfileStats::reset() {
  count.store(0, std::memory_order_relaxed);
  sum_acquire_slot_us.store(0.0, std::memory_order_relaxed);
  sum_alloc_query_us.store(0.0, std::memory_order_relaxed);
  sum_h2d_query_us.store(0.0, std::memory_order_relaxed);
  sum_alloc_output_us.store(0.0, std::memory_order_relaxed);
  sum_kernel_launch_us.store(0.0, std::memory_order_relaxed);
  sum_stream_sync_us.store(0.0, std::memory_order_relaxed);
  sum_d2h_result_us.store(0.0, std::memory_order_relaxed);
  sum_assemble_us.store(0.0, std::memory_order_relaxed);
  sum_total_us.store(0.0, std::memory_order_relaxed);
  sum_gpu_kernel_us.store(0.0, std::memory_order_relaxed);
  sum_collect_wait_us.store(0.0, std::memory_order_relaxed);
  max_acquire_slot_us.store(0.0, std::memory_order_relaxed);
  max_total_us.store(0.0, std::memory_order_relaxed);
  sum_batch_size.store(0, std::memory_order_relaxed);
  batch_count.store(0, std::memory_order_relaxed);
}

std::string CagraProfileStats::to_string() const {
  int c = count.load(std::memory_order_relaxed);
  if (c == 0) return "[CagraProfile] no data collected\n";
  double n = static_cast<double>(c);
  int bc = batch_count.load(std::memory_order_relaxed);
  double avg_bs = bc > 0 ? static_cast<double>(sum_batch_size.load()) / bc : 1.0;
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1);
  ss << "=== CAGRA Search Latency Breakdown (" << c << " calls, "
     << bc << " batches, avg_batch_size=" << avg_bs << ") ===\n";
  ss << "  collect_wait: avg=" << (sum_collect_wait_us.load() / n) << " us\n";
  ss << "  acquire_slot: avg=" << (sum_acquire_slot_us.load() / n)
     << " us,  max=" << max_acquire_slot_us.load() << " us\n";
  ss << "  alloc_query:  avg=" << (sum_alloc_query_us.load() / n) << " us\n";
  ss << "  h2d_query  :  avg=" << (sum_h2d_query_us.load() / n) << " us\n";
  ss << "  alloc_out  :  avg=" << (sum_alloc_output_us.load() / n) << " us\n";
  ss << "  kernel     :  avg=" << (sum_kernel_launch_us.load() / n)
     << " us  (host-side, includes launch overhead)\n";
  ss << "  gpu_actual :  avg=" << (sum_gpu_kernel_us.load() / n)
     << " us  (globaltimer on-GPU)\n";
  ss << "  sync       :  avg=" << (sum_stream_sync_us.load() / n) << " us\n";
  ss << "  d2h_result :  avg=" << (sum_d2h_result_us.load() / n) << " us\n";
  ss << "  assemble   :  avg=" << (sum_assemble_us.load() / n) << " us\n";
  ss << "  TOTAL      :  avg=" << (sum_total_us.load() / n)
     << " us,  max=" << max_total_us.load() << " us\n";
  // 占比分析（不含 collect_wait 和 acquire_slot）
  double sum_inner = sum_alloc_query_us.load() + sum_h2d_query_us.load() +
                     sum_alloc_output_us.load() + sum_kernel_launch_us.load() +
                     sum_stream_sync_us.load() + sum_d2h_result_us.load() +
                     sum_assemble_us.load();
  if (sum_inner > 0) {
    ss << "  --- Breakdown (excluding collect_wait + acquire_slot) ---\n";
    auto pct = [&](double v) { return 100.0 * v / sum_inner; };
    ss << "    alloc_query: " << pct(sum_alloc_query_us.load()) << "%\n";
    ss << "    h2d_query  : " << pct(sum_h2d_query_us.load()) << "%\n";
    ss << "    alloc_out  : " << pct(sum_alloc_output_us.load()) << "%\n";
    ss << "    kernel     : " << pct(sum_kernel_launch_us.load()) << "%\n";
    ss << "    sync       : " << pct(sum_stream_sync_us.load()) << "%\n";
    ss << "    d2h_result : " << pct(sum_d2h_result_us.load()) << "%\n";
    ss << "    assemble   : " << pct(sum_assemble_us.load()) << "%\n";
  }
  return ss.str();
}

// ============================================================================
// 构造 / 析构
// ============================================================================

CAGRACentroidNavigator::CAGRACentroidNavigator() = default;

CAGRACentroidNavigator::~CAGRACentroidNavigator() {
  // 先清理 resource pool
  for (auto &slot : resource_pool_) {
    if (slot) {
      slot->res.reset();
      if (slot->d_query) { cudaFree(slot->d_query); slot->d_query = nullptr; }
      if (slot->d_neighbors) { cudaFree(slot->d_neighbors); slot->d_neighbors = nullptr; }
      if (slot->d_distances) { cudaFree(slot->d_distances); slot->d_distances = nullptr; }
      if (slot->d_timer_buf) { cudaFree(slot->d_timer_buf); slot->d_timer_buf = nullptr; }
      if (slot->stream) { cudaStreamDestroy(slot->stream); slot->stream = nullptr; }
    }
  }
  resource_pool_.clear();
  pool_size_ = 0;

  // 然后释放 index 和主资源
  index_.reset();
  res_.reset();
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void CAGRACentroidNavigator::init_resources() {
  if (!stream_) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }
  if (!res_) {
    res_ = std::make_unique<raft::device_resources>(stream_);
  }
}

// ============================================================================
// Resource Pool：init / acquire / release
// ============================================================================

void CAGRACentroidNavigator::init_resource_pool(int num_slots, int max_nprobe,
                                                int max_batch_size) {
  if (num_slots <= 0) num_slots = 1;
  if (max_nprobe <= 0) max_nprobe = 256;
  if (max_batch_size <= 0) max_batch_size = 1;

  // 先清理旧的 pool
  for (auto &slot : resource_pool_) {
    if (slot) {
      slot->res.reset();
      if (slot->d_query) { cudaFree(slot->d_query); }
      if (slot->d_neighbors) { cudaFree(slot->d_neighbors); }
      if (slot->d_distances) { cudaFree(slot->d_distances); }
      if (slot->d_timer_buf) { cudaFree(slot->d_timer_buf); }
      if (slot->stream) { cudaStreamDestroy(slot->stream); }
    }
  }
  resource_pool_.clear();

  resource_pool_.reserve(static_cast<size_t>(num_slots));
  for (int i = 0; i < num_slots; ++i) {
    auto slot = std::make_unique<ResourceSlot>();
    CUDA_CHECK(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking));
    slot->res = std::make_unique<raft::device_resources>(slot->stream);
    // 预分配 device buffer（支持 batch 查询）
    slot->max_nprobe = max_nprobe;
    slot->max_batch_size = max_batch_size;
    size_t query_bytes = static_cast<size_t>(max_batch_size) * dim_ * sizeof(float);
    size_t neighbors_bytes = static_cast<size_t>(max_batch_size) * max_nprobe * sizeof(uint32_t);
    size_t distances_bytes = static_cast<size_t>(max_batch_size) * max_nprobe * sizeof(float);
    CUDA_CHECK(cudaMalloc(&slot->d_query, query_bytes));
    CUDA_CHECK(cudaMalloc(&slot->d_neighbors, neighbors_bytes));
    CUDA_CHECK(cudaMalloc(&slot->d_distances, distances_bytes));
    // 分配 globaltimer probe buffer
    CUDA_CHECK(cudaMalloc(&slot->d_timer_buf, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(slot->d_timer_buf, 0, 2 * sizeof(unsigned long long)));
    slot->in_use.store(false, std::memory_order_relaxed);
    resource_pool_.push_back(std::move(slot));
  }
  pool_size_ = num_slots;
  std::cout << "  [CAGRA] Resource pool initialized: " << num_slots
            << " slots, max_nprobe=" << max_nprobe
            << ", max_batch=" << max_batch_size
            << " (lock-free, pre-allocated)" << std::endl;
}

CAGRACentroidNavigator::ResourceSlot *
CAGRACentroidNavigator::acquire_slot() const {
  if (pool_size_ == 0) {
    throw std::runtime_error(
        "CAGRACentroidNavigator: resource pool not initialized. "
        "Call init_resource_pool() before searching.");
  }

  // Lock-free spin：尝试获取一个空闲 slot
  static thread_local int hint = 0;
  const int n = pool_size_;
  for (;;) {
    for (int i = 0; i < n; ++i) {
      int idx = (hint + i) % n;
      bool expected = false;
      if (resource_pool_[idx]->in_use.compare_exchange_weak(
              expected, true, std::memory_order_acquire,
              std::memory_order_relaxed)) {
        hint = (idx + 1) % n;
        return resource_pool_[idx].get();
      }
    }
    std::this_thread::yield();
  }
}

void CAGRACentroidNavigator::release_slot(ResourceSlot *slot) const {
  slot->in_use.store(false, std::memory_order_release);
}

std::unique_ptr<CagraBatchCollector>
CAGRACentroidNavigator::create_batch_collector(int batch_size, int nprobe,
                                               int flush_timeout_us) {
  return std::make_unique<CagraBatchCollector>(*this, batch_size, nprobe,
                                               flush_timeout_us);
}

// ============================================================================
// build / save / load
// ============================================================================

void CAGRACentroidNavigator::build(const float *centroids, int nlist, int dim,
                                   int num_threads) {
  (void)num_threads;
  if (centroids == nullptr || nlist <= 0 || dim <= 0) {
    throw std::invalid_argument(
        "Invalid input to CAGRACentroidNavigator::build");
  }
  dim_ = dim;
  init_resources();

  auto dataset_view =
      raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
          centroids, static_cast<int64_t>(nlist), static_cast<int64_t>(dim));

  cuvs::neighbors::cagra::index_params index_params;
  index_params.graph_degree = 64;
  index_params.intermediate_graph_degree = 128;
  index_params.attach_dataset_on_build = true;

  index_ = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(
      cuvs::neighbors::cagra::build(*res_, index_params, dataset_view));
}

void CAGRACentroidNavigator::save(const std::string &path) const {
  ensure_ready();
  raft::device_resources tmp_res;
  cuvs::neighbors::cagra::serialize(tmp_res, path, *index_, true);
}

void CAGRACentroidNavigator::load(const std::string &path) {
  std::filesystem::path p(path);
  if (!std::filesystem::exists(p)) {
    throw std::runtime_error("CAGRA index path does not exist: " + path);
  }
  init_resources();

  index_ = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(
      *res_, cuvs::distance::DistanceType::L2Expanded);
  cuvs::neighbors::cagra::deserialize(*res_, path, index_.get());
  dim_ = static_cast<int>(index_->dim());
}

// ============================================================================
// search_one（无锁：从 pool 获取 slot）
// ============================================================================

ProbeResult CAGRACentroidNavigator::search_one(const float *query,
                                               int nprobe) const {
  if (query == nullptr || nprobe <= 0) {
    throw std::invalid_argument(
        "Invalid input to CAGRACentroidNavigator::search_one");
  }
  auto results = search_batch(query, 1, nprobe);
  return std::move(results[0]);
}

// ============================================================================
// search_batch —— 无锁并发搜索（带完整 profiling）
// ============================================================================

std::vector<ProbeResult>
CAGRACentroidNavigator::search_batch(const float *queries, int num_queries,
                                     int nprobe) const {
  ensure_ready();
  if (queries == nullptr || num_queries <= 0 || nprobe <= 0) {
    throw std::invalid_argument(
        "Invalid input to CAGRACentroidNavigator::search_batch");
  }

  const bool profiling = profiling_enabled_.load(std::memory_order_relaxed);

  // [阶段0] 从 resource pool 获取 slot（lock-free spin）
  auto t0_acquire = hclock::now();
  ResourceSlot *slot = acquire_slot();
  auto t1_acquire = hclock::now();

  // RAII guard：确保 slot 一定被释放
  struct SlotGuard {
    const CAGRACentroidNavigator *nav;
    ResourceSlot *slot;
    ~SlotGuard() { nav->release_slot(slot); }
  } guard{this, slot};

  CagraSearchProfile prof;
  auto result = search_batch_with_slot(slot, queries, num_queries, nprobe,
                                       profiling ? &prof : nullptr);

  // 记录 profiling（合并 acquire_slot 和 total，一次性 accumulate）
  if (profiling) {
    auto t_end = hclock::now();
    prof.acquire_slot_us = us_between(t0_acquire, t1_acquire);
    prof.total_us = us_between(t0_acquire, t_end);
    prof.num_queries = num_queries;
    prof.batch_size = num_queries;
    profile_stats_.accumulate(prof);
  }

  return result;
}

std::vector<ProbeResult>
CAGRACentroidNavigator::search_batch_with_slot(ResourceSlot *slot,
                                               const float *queries,
                                               int num_queries,
                                               int nprobe,
                                               CagraSearchProfile *prof_out) const {
  const bool profiling = (prof_out != nullptr);

  auto &res = *slot->res;
  cudaStream_t stream = slot->stream;
  unsigned long long *d_timer = slot->d_timer_buf;

  const int64_t n = static_cast<int64_t>(num_queries);
  const int64_t k = static_cast<int64_t>(nprobe);

  // 使用预分配 buffer（当 num_queries <= max_batch_size 且 nprobe <= max_nprobe）
  const bool use_prealloc = (num_queries <= slot->max_batch_size &&
                             nprobe <= slot->max_nprobe &&
                             slot->d_query && slot->d_neighbors && slot->d_distances);

  // [阶段1] query buffer（预分配模式跳过 alloc）
  auto t0_alloc_q = hclock::now();
  float *d_query_ptr = nullptr;
  if (use_prealloc) {
    d_query_ptr = slot->d_query;
  } else {
    // fallback: 动态分配
    CUDA_CHECK(cudaMalloc(&d_query_ptr,
                          static_cast<size_t>(n) * dim_ * sizeof(float)));
  }
  auto t1_alloc_q = hclock::now();
  if (profiling) prof_out->alloc_query_us = us_between(t0_alloc_q, t1_alloc_q);

  // [阶段2] H2D query memcpy
  auto t0_h2d = hclock::now();
  CUDA_CHECK(cudaMemcpyAsync(d_query_ptr, queries,
                             static_cast<size_t>(n) * dim_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  auto t1_h2d = hclock::now();
  if (profiling) prof_out->h2d_query_us = us_between(t0_h2d, t1_h2d);

  // [阶段3] output buffer（预分配模式跳过 alloc）
  auto t0_alloc_out = hclock::now();
  uint32_t *d_neighbors_ptr = nullptr;
  float *d_distances_ptr = nullptr;
  if (use_prealloc) {
    d_neighbors_ptr = slot->d_neighbors;
    d_distances_ptr = slot->d_distances;
  } else {
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(k);
    CUDA_CHECK(cudaMalloc(&d_neighbors_ptr, total * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_distances_ptr, total * sizeof(float)));
  }
  auto t1_alloc_out = hclock::now();
  if (profiling) prof_out->alloc_output_us = us_between(t0_alloc_out, t1_alloc_out);

  // [阶段4] CAGRA search kernel
  cuvs::neighbors::cagra::search_params search_params;
  search_params.itopk_size = std::max(512, nprobe * 4);

  // 构造 raft device_matrix_view 包装 buffer
  auto query_view = raft::make_device_matrix_view<const float, int64_t>(
      d_query_ptr, n, static_cast<int64_t>(dim_));
  auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(
      d_neighbors_ptr, n, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(
      d_distances_ptr, n, k);

  auto t0_kernel = hclock::now();
  if (profiling && d_timer) {
    record_gpu_timestamp<<<1, 1, 0, stream>>>(d_timer, 0);
  }

  cuvs::neighbors::cagra::search(res, search_params, *index_,
                                 query_view, neighbors_view, distances_view);

  if (profiling && d_timer) {
    record_gpu_timestamp<<<1, 1, 0, stream>>>(d_timer, 1);
  }
  auto t1_kernel = hclock::now();
  if (profiling) prof_out->kernel_launch_us = us_between(t0_kernel, t1_kernel);

  // [阶段5] 同步流
  auto t0_sync = hclock::now();
  raft::resource::sync_stream(res);
  auto t1_sync = hclock::now();
  if (profiling) prof_out->stream_sync_us = us_between(t0_sync, t1_sync);

  // 读取 GPU globaltimer 时间戳
  if (profiling && d_timer) {
    unsigned long long h_timestamps[2];
    CUDA_CHECK(cudaMemcpy(h_timestamps, d_timer,
                          2 * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    prof_out->gpu_kernel_us =
        static_cast<double>(h_timestamps[1] - h_timestamps[0]) / 1000.0;
  }

  // [阶段6] D2H 拷贝结果
  auto t0_d2h = hclock::now();
  const size_t total = static_cast<size_t>(n) * static_cast<size_t>(k);
  std::vector<uint32_t> h_neighbors(total);
  std::vector<float> h_distances(total);
  CUDA_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors_ptr,
                        total * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances_ptr,
                        total * sizeof(float), cudaMemcpyDeviceToHost));
  auto t1_d2h = hclock::now();
  if (profiling) prof_out->d2h_result_us = us_between(t0_d2h, t1_d2h);

  // 清理 fallback 分配的内存
  if (!use_prealloc) {
    cudaFree(d_query_ptr);
    cudaFree(d_neighbors_ptr);
    cudaFree(d_distances_ptr);
  }

  // [阶段7] 结果组装
  auto t0_asm = hclock::now();
  std::vector<ProbeResult> results(static_cast<size_t>(num_queries));
  for (int q = 0; q < num_queries; ++q) {
    ProbeResult &out = results[q];
    out.ids.reserve(static_cast<size_t>(nprobe));
    out.dists.reserve(static_cast<size_t>(nprobe));
    const size_t base = static_cast<size_t>(q) * static_cast<size_t>(nprobe);
    for (int i = 0; i < nprobe; ++i) {
      out.ids.push_back(
          static_cast<long>(h_neighbors[base + static_cast<size_t>(i)]));
      out.dists.push_back(h_distances[base + static_cast<size_t>(i)]);
    }
  }
  auto t1_asm = hclock::now();
  if (profiling) prof_out->assemble_us = us_between(t0_asm, t1_asm);

  return results;
}

// ============================================================================
// search_batch_flat —— flat 输出版本（无锁）
// ============================================================================

void CAGRACentroidNavigator::search_batch_flat(
    const float *queries, int num_queries, int nprobe,
    std::vector<int32_t> &out_ids, std::vector<float> &out_dists) const {
  ensure_ready();
  if (queries == nullptr || num_queries <= 0 || nprobe <= 0) {
    throw std::invalid_argument(
        "Invalid input to CAGRACentroidNavigator::search_batch_flat");
  }

  ResourceSlot *slot = acquire_slot();
  struct SlotGuard {
    const CAGRACentroidNavigator *nav;
    ResourceSlot *slot;
    ~SlotGuard() { nav->release_slot(slot); }
  } guard{this, slot};

  auto &res = *slot->res;
  cudaStream_t stream = slot->stream;

  const int64_t n = static_cast<int64_t>(num_queries);
  const int64_t k = static_cast<int64_t>(nprobe);

  // 尝试使用预分配 buffer
  const bool use_prealloc = (num_queries <= slot->max_batch_size &&
                             nprobe <= slot->max_nprobe &&
                             slot->d_query && slot->d_neighbors && slot->d_distances);

  float *d_q = use_prealloc ? slot->d_query : nullptr;
  uint32_t *d_n = use_prealloc ? slot->d_neighbors : nullptr;
  float *d_d = use_prealloc ? slot->d_distances : nullptr;

  if (!use_prealloc) {
    CUDA_CHECK(cudaMalloc(&d_q, static_cast<size_t>(n) * dim_ * sizeof(float)));
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(k);
    CUDA_CHECK(cudaMalloc(&d_n, total * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_d, total * sizeof(float)));
  }

  CUDA_CHECK(cudaMemcpyAsync(d_q, queries,
                             static_cast<size_t>(n) * dim_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  cuvs::neighbors::cagra::search_params search_params;
  search_params.itopk_size = std::max(512, nprobe * 4);

  auto query_view = raft::make_device_matrix_view<const float, int64_t>(
      d_q, n, static_cast<int64_t>(dim_));
  auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(d_n, n, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(d_d, n, k);

  cuvs::neighbors::cagra::search(res, search_params, *index_,
                                 query_view, neighbors_view, distances_view);

  raft::resource::sync_stream(res);

  const size_t total = static_cast<size_t>(n) * static_cast<size_t>(k);
  std::vector<uint32_t> h_neighbors(total);
  std::vector<float> h_distances(total);
  CUDA_CHECK(cudaMemcpy(h_neighbors.data(), d_n,
                        total * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_distances.data(), d_d,
                        total * sizeof(float), cudaMemcpyDeviceToHost));

  if (!use_prealloc) {
    cudaFree(d_q);
    cudaFree(d_n);
    cudaFree(d_d);
  }

  out_ids.resize(total);
  out_dists.resize(total);
  for (size_t i = 0; i < total; ++i) {
    out_ids[i] = static_cast<int32_t>(h_neighbors[i]);
    out_dists[i] = h_distances[i];
  }
}

// ============================================================================
// 工具方法
// ============================================================================

void CAGRACentroidNavigator::ensure_ready() const {
  if (!index_) {
    throw std::runtime_error(
        "CAGRACentroidNavigator: index not built or loaded");
  }
}

// ============================================================================
// CagraBatchCollector 实现
// ============================================================================

CagraBatchCollector::CagraBatchCollector(CAGRACentroidNavigator &navigator,
                                         int batch_size, int nprobe,
                                         int flush_timeout_us)
    : navigator_(navigator), batch_size_(std::max(1, batch_size)),
      nprobe_(nprobe), flush_timeout_us_(flush_timeout_us) {
  pending_.reserve(static_cast<size_t>(batch_size_));
  // 启动后台 flush 线程
  flush_thread_ = std::thread(&CagraBatchCollector::flush_loop, this);
}

CagraBatchCollector::~CagraBatchCollector() {
  shutdown();
}

std::future<ProbeResult> CagraBatchCollector::submit(const float *query) {
  std::promise<ProbeResult> promise;
  auto future = promise.get_future();

  const int dim = navigator_.get_dim();
  PendingQuery pq;
  pq.query_data.assign(query, query + dim);
  pq.promise = std::move(promise);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(std::move(pq));
  }
  cv_.notify_one();  // 唤醒 flush 线程

  return future;
}

void CagraBatchCollector::flush() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (pending_.empty()) return;
  flush_pending(lock);
}

void CagraBatchCollector::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_) return;
    shutdown_ = true;
  }
  cv_.notify_all();
  if (flush_thread_.joinable()) {
    flush_thread_.join();
  }
  // flush 剩余查询
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!pending_.empty()) {
      flush_pending(lock);
    }
  }
}

void CagraBatchCollector::flush_loop() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!shutdown_) {
    // 等待：有查询到达或 shutdown
    cv_.wait(lock, [this] {
      return !pending_.empty() || shutdown_;
    });

    if (shutdown_ && pending_.empty()) break;

    if (pending_.empty()) continue;

    // 如果 pending 已满，立即 flush
    if (static_cast<int>(pending_.size()) >= batch_size_) {
      flush_pending(lock);
      continue;
    }

    // pending 未满，等待超时或更多查询到达
    if (flush_timeout_us_ > 0) {
      auto timeout = std::chrono::microseconds(flush_timeout_us_);
      cv_.wait_for(lock, timeout, [this] {
        return static_cast<int>(pending_.size()) >= batch_size_ || shutdown_;
      });
    }

    // 超时或满了，flush
    if (!pending_.empty()) {
      flush_pending(lock);
    }
  }

  // shutdown 时 flush 剩余
  if (!pending_.empty()) {
    flush_pending(lock);
  }
}

void CagraBatchCollector::flush_pending(std::unique_lock<std::mutex> &lock) {
  // 取出当前所有 pending 查询（swap 出来，减少锁持有时间）
  std::vector<PendingQuery> batch;
  batch.swap(pending_);
  pending_.reserve(static_cast<size_t>(batch_size_));
  lock.unlock();

  const int dim = navigator_.get_dim();
  const int num_queries = static_cast<int>(batch.size());
  const bool profiling = navigator_.profiling_enabled();

  // 拼接所有查询向量为连续 buffer
  std::vector<float> packed_queries(static_cast<size_t>(num_queries) * dim);
  for (int i = 0; i < num_queries; ++i) {
    std::memcpy(packed_queries.data() + static_cast<size_t>(i) * dim,
                batch[i].query_data.data(), dim * sizeof(float));
  }

  // 获取 slot + 执行 batch search
  auto t0_acquire = hclock::now();
  auto *slot = navigator_.acquire_slot();
  auto t1_acquire = hclock::now();

  // RAII guard
  struct SlotGuard {
    const CAGRACentroidNavigator *nav;
    CAGRACentroidNavigator::ResourceSlot *slot;
    ~SlotGuard() { nav->release_slot(slot); }
  } guard{&navigator_, slot};

  CagraSearchProfile prof;
  auto results = navigator_.search_batch_with_slot(
      slot, packed_queries.data(), num_queries, nprobe_,
      profiling ? &prof : nullptr);

  // 记录 profiling
  if (profiling) {
    auto t_end = hclock::now();
    prof.acquire_slot_us = us_between(t0_acquire, t1_acquire);
    prof.total_us = us_between(t0_acquire, t_end);
    prof.num_queries = num_queries;
    prof.batch_size = num_queries;
    // 注意：count 会加 1 次（以 batch 为单位），用 num_queries 来统计逻辑查询数
    navigator_.profile_stats_.accumulate(prof);
  }

  // 分发结果给各个 worker 线程
  for (int i = 0; i < num_queries; ++i) {
    if (i < static_cast<int>(results.size())) {
      batch[i].promise.set_value(std::move(results[i]));
    } else {
      // 不应该发生，但保险起见
      batch[i].promise.set_value(ProbeResult{});
    }
  }

  lock.lock();
  flush_cv_.notify_all();
}

#endif // FUSIONANNS_USE_CUVS_CAGRA
