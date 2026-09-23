#include "index/hbc_builder.h"
#include "index/hbc_progress.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <queue>
#include <random>
#include <stack>
#include <stdexcept>
#include <thread>
#include <utility>

#include <faiss/utils/distances.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__) && defined(_OPENMP)
#include <parallel/algorithm>
#endif

namespace {
constexpr float kMaxDist = 1e30f;

inline float l2sqr(const float *a, const float *b, int dim) {
  return faiss::fvec_L2sqr(a, b, dim);
}

double bytes_to_gib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

class DatasetAccessor {
public:
  DatasetAccessor(const HbcBuilderOptions &opt,
                  const HbcBuilder::GatherFn &gather)
      : gather_(gather), rows_(opt.total_vectors), dim_(opt.dim) {
    if (!gather_) {
      throw std::invalid_argument(
          "DatasetAccessor requires a valid gather callback");
    }
    if (rows_ == 0 || dim_ <= 0) {
      throw std::invalid_argument(
          "DatasetAccessor requires non-empty dataset dimensions");
    }
    chunk_size_ =
        std::max<size_t>(1, std::min(opt.chunk_size, opt.total_vectors));
    stream_block_size_ = compute_stream_block_(opt);
    attempt_preload_(opt);
  }

  size_t rows() const { return rows_; }
  int dim() const { return dim_; }

  const float *row(size_t idx) const {
    if (idx >= rows_) {
      throw std::out_of_range("dataset row index out of range");
    }
    if (preloaded_) {
      return storage_.data() + idx * stride_();
    }
    return fetch_streaming_(idx);
  }

  void read_batch(const int32_t *ids, size_t count, float *out) const {
    if (count == 0 || !out) {
      return;
    }
    const size_t stride = stride_();
    if (preloaded_) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < count; ++i) {
        size_t idx = static_cast<size_t>(ids[i]);
        if (idx < rows_) {
          std::memcpy(out + i * stride, storage_.data() + idx * stride,
                      stride * sizeof(float));
        }
      }
      return;
    }
    gather_(ids, count, out);
  }

private:
  size_t stride_() const { return static_cast<size_t>(dim_); }

  size_t compute_stream_block_(const HbcBuilderOptions &opt) const {
    size_t target_bytes = opt.stream_block_bytes == 0 ? (32ull * 1024 * 1024)
                                                      : opt.stream_block_bytes;
    target_bytes = std::max<size_t>(stride_() * sizeof(float), target_bytes);
    size_t vectors = target_bytes / (stride_() * sizeof(float));
    if (vectors == 0) {
      vectors = 1;
    }
    if (opt.chunk_size > 0) {
      vectors = std::min(vectors, opt.chunk_size);
    }
    vectors = std::min(vectors, rows_);
    return std::max<size_t>(1, vectors);
  }

  void attempt_preload_(const HbcBuilderOptions &opt) {
    const size_t total_bytes = rows_ * stride_() * sizeof(float);
    if (opt.dataset_preload_threshold_bytes == 0 ||
        total_bytes > opt.dataset_preload_threshold_bytes) {
      HBC_LOG("    [dataset] streaming HBC build (block="
              << stream_block_size_ << " vectors, " << std::fixed
              << std::setprecision(2)
              << bytes_to_gib(stream_block_size_ * stride_() * sizeof(float))
              << " GiB/thread)" << std::defaultfloat << std::endl);
      return;
    }
    try {
      storage_.resize(rows_ * stride_());
    } catch (const std::bad_alloc &) {
      storage_.clear();
      HBC_LOG("    [dataset] preload allocation failed; falling back to "
              "streaming"
              << std::endl);
      return;
    }

    std::vector<int32_t> ids(chunk_size_);
    std::vector<float> buffer(chunk_size_ * stride_());
    size_t offset = 0;
    while (offset < rows_) {
      size_t current = std::min(chunk_size_, rows_ - offset);
      for (size_t i = 0; i < current; ++i) {
        ids[i] = static_cast<int32_t>(offset + i);
      }
      gather_(ids.data(), current, buffer.data());
      std::memcpy(storage_.data() + offset * stride_(), buffer.data(),
                  current * stride_() * sizeof(float));
      offset += current;
    }

    preloaded_ = true;
    HBC_LOG("    [dataset] preloaded " << rows_ << " vectors for HBC ("
                                       << std::fixed << std::setprecision(2)
                                       << bytes_to_gib(total_bytes) << " GiB)"
                                       << std::defaultfloat << std::endl);
  }

  const float *fetch_streaming_(size_t idx) const {
    struct ThreadCache {
      const DatasetAccessor *owner = nullptr;
      size_t block_start = 0;
      size_t block_count = 0;
      std::vector<float> block;
      std::vector<int32_t> ids;

      void reset(const DatasetAccessor *parent) {
        owner = parent;
        block_start = 0;
        block_count = 0;
        block.clear();
        ids.clear();
      }
    };

    thread_local ThreadCache cache;
    if (cache.owner != this) {
      cache.reset(this);
    }
    if (idx < cache.block_start ||
        idx >= cache.block_start + cache.block_count) {
      size_t block_start = (idx / stream_block_size_) * stream_block_size_;
      size_t remaining = rows_ - block_start;
      size_t count = std::min(stream_block_size_, remaining);
      cache.block.resize(count * stride_());
      cache.ids.resize(count);
      for (size_t i = 0; i < count; ++i) {
        cache.ids[i] = static_cast<int32_t>(block_start + i);
      }
      gather_(cache.ids.data(), count, cache.block.data());
      cache.block_start = block_start;
      cache.block_count = count;
    }
    size_t offset = idx - cache.block_start;
    return cache.block.data() + offset * stride_();
  }

private:
  HbcBuilder::GatherFn gather_;
  size_t rows_ = 0;
  int dim_ = 0;
  size_t chunk_size_ = 1;
  size_t stream_block_size_ = 1;
  bool preloaded_ = false;
  std::vector<float> storage_;
};

struct DatasetView {
  const DatasetAccessor *accessor = nullptr;

  const float *operator[](size_t idx) const {
    if (!accessor) {
      throw std::runtime_error("dataset view is uninitialized");
    }
    return accessor->row(idx);
  }

  size_t R() const { return accessor ? accessor->rows() : 0; }
  int C() const { return accessor ? accessor->dim() : 0; }

  void read_batch(const int32_t *ids, size_t count, float *out) const {
    if (!accessor) {
      throw std::runtime_error("dataset view is uninitialized");
    }
    accessor->read_batch(ids, count, out);
  }
};

struct KmeansArgs {
  int K = 0;
  int DK = 0;
  int dim = 0;
  int threads = 1;
  size_t total_points = 0;

  std::vector<float> centers;
  std::vector<float> temp_centers;
  std::vector<int> counts;

  std::vector<float> new_centers;
  std::vector<int> new_counts;
  std::vector<int> labels;
  std::vector<int32_t> cluster_idx;
  std::vector<float> cluster_dist;
  std::vector<float> weighted_counts;
  std::vector<float> new_weighted_counts;
  std::vector<int32_t> scratch_indices;

  KmeansArgs(int k, int dim_in, size_t total, int thread_count)
      : K(k), DK(k), dim(dim_in), threads(std::max(1, thread_count)),
        total_points(total),
        centers(static_cast<size_t>(k) * static_cast<size_t>(dim_in), 0.0f),
        temp_centers(static_cast<size_t>(k) * static_cast<size_t>(dim_in),
                     0.0f),
        counts(static_cast<size_t>(k), 0),
        new_centers(static_cast<size_t>(threads) * static_cast<size_t>(k) *
                        static_cast<size_t>(dim_in),
                    0.0f),
        new_counts(static_cast<size_t>(threads) * static_cast<size_t>(k), 0),
        labels(total, 0),
        cluster_idx(static_cast<size_t>(threads) * static_cast<size_t>(k), -1),
        cluster_dist(static_cast<size_t>(threads) * static_cast<size_t>(k),
                     0.0f),
        weighted_counts(static_cast<size_t>(k), 0.0f),
        new_weighted_counts(
            static_cast<size_t>(threads) * static_cast<size_t>(k), 0.0f) {}

  void ClearCounts() {
    std::fill(new_counts.begin(), new_counts.end(), 0);
    std::fill(new_weighted_counts.begin(), new_weighted_counts.end(), 0.0f);
  }

  void ClearCenters() {
    std::fill(new_centers.begin(), new_centers.end(), 0.0f);
  }

  void ClearDists(float value) {
    std::fill(cluster_idx.begin(), cluster_idx.end(), -1);
    std::fill(cluster_dist.begin(), cluster_dist.end(), value);
  }
};

inline size_t rand_index(size_t first, size_t last, std::mt19937_64 &rng) {
  std::uniform_int_distribution<size_t> dist(first, last - 1);
  return dist(rng);
}

class BatchPrefetcher {
public:
  struct Batch {
    std::vector<float> data;
    size_t start_idx = 0;
    size_t count = 0;
  };

  BatchPrefetcher(const DatasetView &data, const std::vector<int32_t> &indices,
                  size_t first, size_t last, size_t batch_cap, int dim)
      : data_(data), indices_(indices), next_idx_(first), end_idx_(last),
        batch_cap_(batch_cap), dim_(dim) {
    // Pre-allocate buffers (3 buffers for triple buffering)
    for (int i = 0; i < 3; ++i) {
      auto batch = std::make_unique<Batch>();
      batch->data.resize(batch_cap * static_cast<size_t>(dim));
      free_queue_.push(std::move(batch));
    }
    worker_ = std::thread([this] { run(); });
  }

  ~BatchPrefetcher() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_free_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  std::unique_ptr<Batch> next() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_ready_.wait(lock, [this] { return !ready_queue_.empty() || done_; });
    if (exception_) {
      std::rethrow_exception(exception_);
    }
    if (ready_queue_.empty()) {
      return nullptr;
    }
    auto batch = std::move(ready_queue_.front());
    ready_queue_.pop();
    return batch;
  }

  void recycle(std::unique_ptr<Batch> batch) {
    if (!batch)
      return;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      free_queue_.push(std::move(batch));
    }
    cv_free_.notify_one();
  }

private:
  void run() {
#ifdef _OPENMP
    // Prevent thread oversubscription: prefetcher is memory-bound, so we don't
    // want it to spawn 96 threads that compete with the 96 compute threads.
    omp_set_num_threads(4);
#endif
    try {
      while (true) {
        std::unique_ptr<Batch> batch;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_free_.wait(lock, [this] { return !free_queue_.empty() || stop_; });
          if (stop_)
            return;
          batch = std::move(free_queue_.front());
          free_queue_.pop();
        }

        size_t current_start = next_idx_;
        if (current_start >= end_idx_) {
          std::lock_guard<std::mutex> lock(mutex_);
          done_ = true;
          cv_ready_.notify_all();
          return;
        }

        size_t count = std::min(batch_cap_, end_idx_ - current_start);
        next_idx_ += count;

        // Do I/O without lock
        batch->start_idx = current_start;
        batch->count = count;
        data_.read_batch(indices_.data() + current_start, count,
                         batch->data.data());

        {
          std::lock_guard<std::mutex> lock(mutex_);
          ready_queue_.push(std::move(batch));
        }
        cv_ready_.notify_one();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      exception_ = std::current_exception();
      done_ = true;
      cv_ready_.notify_all();
    }
  }

  const DatasetView &data_;
  const std::vector<int32_t> &indices_;
  size_t next_idx_;
  size_t end_idx_;
  size_t batch_cap_;
  int dim_;

  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_ready_;
  std::condition_variable cv_free_;
  std::queue<std::unique_ptr<Batch>> ready_queue_;
  std::queue<std::unique_ptr<Batch>> free_queue_;
  std::exception_ptr exception_;
  bool stop_ = false;
  bool done_ = false;
};

float KmeansAssign(const DatasetView &data, std::vector<int32_t> &indices,
                   size_t first, size_t last, KmeansArgs &args,
                   bool update_centers, float lambda) {
  if (last <= first) {
    return 0.0f;
  }
  struct ProgressGuard {
    ProgressGuard(const std::string &label, size_t total) {
      if (total > 100000) {
        active = true;
        HbcProgressStartLabeled(label, total);
      }
    }
    ~ProgressGuard() {
      if (active) {
        HbcProgressFinish();
      }
    }
    bool active = false;
  };

  const size_t total = last - first;
  const size_t stride = static_cast<size_t>(args.dim);
  const size_t bytes_per_vec = std::max<size_t>(1, stride * sizeof(float));
  const size_t target_batch_bytes = 128ull * 1024ull * 1024ull;
  size_t batch_cap = target_batch_bytes / bytes_per_vec;
  if (batch_cap == 0) {
    batch_cap = 1;
  }
  batch_cap = std::min(batch_cap, total);

  std::string progress_label = "kmeans(" + std::to_string(total) + ")";
  ProgressGuard progress(progress_label, total);

  // Precompute centroid norms and biases
  // dist(x, c) = ||x||^2 + ||c||^2 - 2 <x, c>
  // We want to minimize dist(x, c) + lambda * count
  // Equivalent to minimizing: ||c||^2 - 2 <x, c> + lambda * count
  // (||x||^2 is constant for all clusters)
  std::vector<float> centroid_norms(args.DK);
  for (int k = 0; k < args.DK; ++k) {
    float norm_sq = faiss::fvec_norm_L2sqr(
        args.centers.data() + static_cast<size_t>(k) * stride, args.dim);
    centroid_norms[k] = norm_sq + lambda * static_cast<float>(args.counts[k]);
  }

  float curr_dist = 0.0f;

  BatchPrefetcher prefetcher(data, indices, first, last, batch_cap, args.dim);

  while (auto batch = prefetcher.next()) {
    const float *buffer_ptr = batch->data.data();
    size_t batch_size_local = batch->count;
    size_t batch_start_local = batch->start_idx;

#pragma omp parallel num_threads(args.threads) reduction(+ : curr_dist)
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      int *local_counts =
          args.new_counts.data() + static_cast<size_t>(tid) * args.K;
      float *local_centers =
          args.new_centers.data() +
          static_cast<size_t>(tid) * static_cast<size_t>(args.K) * stride;
      int32_t *local_cluster_idx =
          args.cluster_idx.data() + static_cast<size_t>(tid) * args.K;
      float *local_cluster_dist =
          args.cluster_dist.data() + static_cast<size_t>(tid) * args.K;
      float *local_weighted =
          args.new_weighted_counts.data() + static_cast<size_t>(tid) * args.K;

      // Initialize local buffers if this is the first batch
      // Note: We can't easily detect "first batch" inside parallel region
      // across loops. But we can just rely on the fact that we accumulate. The
      // caller (KmeansArgs) is responsible for clearing before the function
      // call. However, we need to initialize local_cluster_idx/dist for the
      // *current* batch? No, they track the best *global* cluster for each
      // center. Wait, args.cluster_idx[k] stores the index of the vector
      // closest to center k. This needs to be maintained across batches. But we
      // need to initialize them to "empty" at the start of KmeansAssign.
      // KmeansArgs constructor initializes them.
      // But inside the loop?
      // The previous code initialized them inside the parallel region:
      // for (int k = 0; k < args.K; ++k) {
      //   local_cluster_idx[k] = -1;
      //   local_cluster_dist[k] = update_centers ? -kMaxDist : kMaxDist;
      // }
      // This was done ONCE per thread at start of parallel region.
      // Now we have multiple parallel regions.
      // We must NOT reset them in every batch!
      // But wait, if we don't reset, they contain values from previous batch.
      // That is CORRECT. We want the best vector across ALL batches.
      // So we should NOT reset them inside the loop.
      // BUT, they need to be initialized somewhere.
      // They are initialized in KmeansArgs constructor or ClearDists.
      // But KmeansArgs::ClearDists clears the *global* arrays?
      // No, KmeansArgs has `cluster_idx` sized `threads * K`.
      // So they are persistent.
      // BUT, the previous code did:
      // for (int k = 0; k < args.K; ++k) { local_cluster_idx[k] = -1; ... }
      // inside the parallel region, BEFORE the while(true) loop.
      // So they were reset once per KmeansAssign call.
      // Here, we enter parallel region multiple times.
      // If we don't reset, we are fine, we just keep updating.
      // BUT, we must ensure they are initialized correctly at the start of
      // KmeansAssign.
      // KmeansArgs::ClearDists does that?
      // Let's check KmeansAssign caller.
      // TryClustering calls args.ClearDists(-kMaxDist).
      // KmeansArgs::ClearDists fills `cluster_idx` with -1.
      // So they are already initialized!
      // So we don't need to reset them inside the parallel region.
      // EXCEPT: The previous code did it inside the parallel region.
      // Why? Maybe because ClearDists only clears the first K elements?
      // Let's check KmeansArgs::ClearDists.
      // "std::fill(cluster_idx.begin(), cluster_idx.end(), -1);"
      // It clears the WHOLE vector (threads * K).
      // So we are safe. We don't need to reset in the loop.

      // Use thread_local to avoid reallocation overhead in every batch
      static thread_local std::vector<float> ip_results;
      if (ip_results.size() < static_cast<size_t>(args.DK)) {
        ip_results.resize(args.DK);
      }

#pragma omp for schedule(static)
      for (size_t i = 0; i < batch_size_local; ++i) {
        const float *vec = buffer_ptr + i * stride;
        const size_t global_idx = batch_start_local + i;

        // Compute inner products: IP[k] = <vec, center_k>
        faiss::fvec_inner_products_ny(ip_results.data(), vec,
                                      args.centers.data(), args.dim, args.DK);

        float vec_norm_sq = faiss::fvec_norm_L2sqr(vec, args.dim);

        int best = 0;
        float best_val = centroid_norms[0] - 2 * ip_results[0];

        for (int k = 1; k < args.DK; ++k) {
          float val = centroid_norms[k] - 2 * ip_results[k];
          if (val < best_val) {
            best_val = val;
            best = k;
          }
        }

        float real_dist = best_val + vec_norm_sq;

        args.labels[global_idx] = best;
        local_counts[best] += 1;
        local_weighted[best] += real_dist;
        curr_dist += real_dist;

        if (update_centers) {
          float *dst = local_centers + static_cast<size_t>(best) * stride;
          for (size_t d = 0; d < stride; ++d) {
            dst[d] += vec[d];
          }
          if (real_dist > local_cluster_dist[best]) {
            local_cluster_dist[best] = real_dist;
            local_cluster_idx[best] = indices[global_idx];
          }
        } else if (real_dist <= local_cluster_dist[best]) {
          local_cluster_dist[best] = real_dist;
          local_cluster_idx[best] = indices[global_idx];
        }
      }
    }

    if (progress.active) {
      HbcProgressAdvance(batch_size_local);
    }
    prefetcher.recycle(std::move(batch));
  }

  for (int t = 1; t < args.threads; ++t) {
    for (int k = 0; k < args.K; ++k) {
      args.new_counts[static_cast<size_t>(k)] +=
          args.new_counts[static_cast<size_t>(t) * static_cast<size_t>(args.K) +
                          static_cast<size_t>(k)];
      args.new_weighted_counts[static_cast<size_t>(k)] +=
          args.new_weighted_counts[static_cast<size_t>(t) *
                                       static_cast<size_t>(args.K) +
                                   static_cast<size_t>(k)];
    }
  }

  if (update_centers) {
    for (int t = 1; t < args.threads; ++t) {
      float *src = args.new_centers.data() + static_cast<size_t>(t) *
                                                 static_cast<size_t>(args.K) *
                                                 stride;
      for (size_t j = 0; j < static_cast<size_t>(args.DK) * stride; ++j) {
        args.new_centers[j] += src[j];
      }
      for (int k = 0; k < args.DK; ++k) {
        float dist = args.cluster_dist[static_cast<size_t>(t) *
                                           static_cast<size_t>(args.K) +
                                       static_cast<size_t>(k)];
        if (dist > args.cluster_dist[static_cast<size_t>(k)]) {
          args.cluster_dist[static_cast<size_t>(k)] = dist;
          args.cluster_idx[static_cast<size_t>(k)] =
              args.cluster_idx[static_cast<size_t>(t) *
                                   static_cast<size_t>(args.K) +
                               static_cast<size_t>(k)];
        }
      }
    }
  } else {
    for (int t = 1; t < args.threads; ++t) {
      for (int k = 0; k < args.DK; ++k) {
        float dist = args.cluster_dist[static_cast<size_t>(t) *
                                           static_cast<size_t>(args.K) +
                                       static_cast<size_t>(k)];
        if (dist <= args.cluster_dist[static_cast<size_t>(k)]) {
          args.cluster_dist[static_cast<size_t>(k)] = dist;
          args.cluster_idx[static_cast<size_t>(k)] =
              args.cluster_idx[static_cast<size_t>(t) *
                                   static_cast<size_t>(args.K) +
                               static_cast<size_t>(k)];
        }
      }
    }
  }

  return curr_dist;
}

float InitCenters(const DatasetView &data, std::vector<int32_t> &indices,
                  size_t first, size_t last, KmeansArgs &args, int samples,
                  int trials, std::mt19937_64 &rng) {
  size_t batch_end = std::min(first + static_cast<size_t>(samples), last);
  float lambda = 0.0f;
  float min_cluster_dist = kMaxDist;
  for (int attempt = 0; attempt < trials; ++attempt) {
    for (int k = 0; k < args.DK; ++k) {
      size_t rid = rand_index(first, last, rng);
      std::memcpy(args.centers.data() +
                      static_cast<size_t>(k) * static_cast<size_t>(args.dim),
                  data[indices[rid]],
                  sizeof(float) * static_cast<size_t>(args.dim));
    }
    args.ClearCounts();
    args.ClearDists(-kMaxDist);
    float dist =
        KmeansAssign(data, indices, first, batch_end, args, true, 0.0f);
    if (dist < min_cluster_dist) {
      min_cluster_dist = dist;
      std::memcpy(args.temp_centers.data(), args.centers.data(),
                  sizeof(float) * static_cast<size_t>(args.DK) *
                      static_cast<size_t>(args.dim));
      std::copy(args.new_counts.begin(), args.new_counts.begin() + args.K,
                args.counts.begin());
      lambda = 0.0f;
      int maxcluster = -1;
      int maxcount = 0;
      for (int k = 0; k < args.DK; ++k) {
        if (args.counts[k] > maxcount && args.new_counts[k] > 0) {
          maxcluster = k;
          maxcount = args.counts[k];
        }
      }
      if (maxcluster >= 0 && args.new_counts[maxcluster] > 0) {
        float avg = args.new_weighted_counts[maxcluster] /
                    static_cast<float>(args.new_counts[maxcluster]);
        lambda = (args.cluster_dist[maxcluster] - avg) /
                 static_cast<float>(batch_end - first);
        if (lambda < 0.0f) {
          lambda = 0.0f;
        }
      }
    }
  }
  return lambda;
}

void RefineLambda(KmeansArgs &args, float &lambda, size_t size) {
  int maxcluster = -1;
  int maxcount = 0;
  for (int k = 0; k < args.DK; ++k) {
    if (args.counts[k] > maxcount && args.new_counts[k] > 0) {
      maxcluster = k;
      maxcount = args.counts[k];
    }
  }
  if (maxcluster >= 0 && args.new_counts[maxcluster] > 0) {
    float avg = args.new_weighted_counts[maxcluster] /
                static_cast<float>(args.new_counts[maxcluster]);
    lambda = (args.cluster_dist[maxcluster] - avg) /
             static_cast<float>(std::max<size_t>(1, size));
    if (lambda < 0.0f) {
      lambda = 0.0f;
    }
  }
}

float RefineCenters(const DatasetView &data, KmeansArgs &args) {
  int maxcluster = -1;
  int maxcount = 0;
  for (int k = 0; k < args.DK; ++k) {
    if (args.counts[k] <= maxcount || args.new_counts[k] <= 0) {
      continue;
    }
    int32_t center_idx = args.cluster_idx[k];
    if (center_idx < 0 || static_cast<size_t>(center_idx) >= data.R()) {
      continue;
    }
    float dist = l2sqr(data[static_cast<size_t>(center_idx)],
                       args.centers.data() + static_cast<size_t>(k) *
                                                 static_cast<size_t>(args.dim),
                       args.dim);
    if (dist > 1e-6f) {
      maxcluster = k;
      maxcount = args.counts[k];
    }
  }

  float diff = 0.0f;
  for (int k = 0; k < args.DK; ++k) {
    float *dest = args.temp_centers.data() +
                  static_cast<size_t>(k) * static_cast<size_t>(args.dim);
    if (args.counts[k] == 0) {
      if (maxcluster >= 0) {
        int32_t fallback_idx = args.cluster_idx[maxcluster];
        if (fallback_idx >= 0 && static_cast<size_t>(fallback_idx) < data.R()) {
          std::memcpy(dest, data[static_cast<size_t>(fallback_idx)],
                      sizeof(float) * static_cast<size_t>(args.dim));
        } else {
          std::memcpy(dest,
                      args.centers.data() + static_cast<size_t>(k) *
                                                static_cast<size_t>(args.dim),
                      sizeof(float) * static_cast<size_t>(args.dim));
        }
      } else {
        std::memcpy(dest,
                    args.centers.data() +
                        static_cast<size_t>(k) * static_cast<size_t>(args.dim),
                    sizeof(float) * static_cast<size_t>(args.dim));
      }
    } else {
      float *sum = args.new_centers.data() +
                   static_cast<size_t>(k) * static_cast<size_t>(args.dim);
      for (int d = 0; d < args.dim; ++d) {
        dest[d] = sum[d] / static_cast<float>(args.counts[k]);
      }
    }
    diff += l2sqr(dest,
                  args.centers.data() +
                      static_cast<size_t>(k) * static_cast<size_t>(args.dim),
                  args.dim);
  }
  return diff;
}

float TryClustering(const DatasetView &data, std::vector<int32_t> &indices,
                    size_t first, size_t last, KmeansArgs &args, int samples,
                    float lambda_factor, bool debug, std::mt19937_64 &rng) {
  float adjusted_lambda =
      InitCenters(data, indices, first, last, args, samples, 3, rng);
  size_t batch_end = std::min(first + static_cast<size_t>(samples), last);
  float base = 1.0f;
  float original_lambda =
      base * base / lambda_factor /
      static_cast<float>(std::max<size_t>(1, batch_end - first));

  float min_cluster_dist = kMaxDist;
  int no_improvement = 0;
  auto it_begin = indices.begin() + static_cast<std::ptrdiff_t>(first);
  auto it_end = indices.begin() + static_cast<std::ptrdiff_t>(last);
#if defined(__GNUC__) && defined(_OPENMP)
  __gnu_parallel::sort(it_begin, it_end);
#else
  std::sort(it_begin, it_end);
#endif

  for (int iter = 0; iter < 100; ++iter) {
    std::memcpy(args.centers.data(), args.temp_centers.data(),
                sizeof(float) * static_cast<size_t>(args.DK) *
                    static_cast<size_t>(args.dim));

    args.ClearCenters();
    args.ClearCounts();
    args.ClearDists(-kMaxDist);
    float curr_dist = KmeansAssign(data, indices, first, batch_end, args, true,
                                   std::min(adjusted_lambda, original_lambda));
    std::copy(args.new_counts.begin(), args.new_counts.begin() + args.K,
              args.counts.begin());

    if (curr_dist < min_cluster_dist) {
      min_cluster_dist = curr_dist;
      no_improvement = 0;
    } else {
      ++no_improvement;
    }

    RefineLambda(args, adjusted_lambda, batch_end - first);
    float diff = RefineCenters(data, args);

    if (debug) {
      float avg = static_cast<float>(last - first) /
                  static_cast<float>(std::max(1, args.DK));
      int max_count = 0;
      int min_count = std::numeric_limits<int>::max();
      int active = 0;
      float variance = 0.0f;
      for (int k = 0; k < args.DK; ++k) {
        max_count = std::max(max_count, args.counts[k]);
        if (args.counts[k] > 0) {
          min_count = std::min(min_count, args.counts[k]);
          variance += (args.counts[k] - avg) * (args.counts[k] - avg);
          active++;
        }
      }
      float std_ratio = active > 0 ? std::sqrt(variance / args.DK) / avg : 0.0f;
      HBC_LOG("    [bkt] lambda=("
              << original_lambda << "," << adjusted_lambda << ") pop="
              << (last - first) << " max=" << max_count << " min=" << min_count
              << " std/avg=" << std::fixed << std::setprecision(4) << std_ratio
              << std::defaultfloat << std::endl);
    }

    if (diff < 1e-3f || no_improvement >= 5) {
      break;
    }
  }

  args.ClearCounts();
  args.ClearDists(kMaxDist);
  const size_t total_points = last - first;
  const size_t refine_threshold = 10'000'000;
  if (total_points <= refine_threshold) {
    KmeansAssign(data, indices, first, last, args, false, 0.0f);
    for (int k = 0; k < args.DK; ++k) {
      if (args.cluster_idx[k] >= 0 &&
          static_cast<size_t>(args.cluster_idx[k]) < data.R()) {
        std::memcpy(args.centers.data() +
                        static_cast<size_t>(k) * static_cast<size_t>(args.dim),
                    data[static_cast<size_t>(args.cluster_idx[k])],
                    sizeof(float) * static_cast<size_t>(args.dim));
      }
    }
    args.ClearCounts();
    args.ClearDists(kMaxDist);
  }
  KmeansAssign(data, indices, first, last, args, false, 0.0f);
  std::copy(args.new_counts.begin(), args.new_counts.begin() + args.K,
            args.counts.begin());
  return 0.0f;
}

float DynamicFactorSelect(const DatasetView &data,
                          std::vector<int32_t> &indices, size_t first,
                          size_t last, KmeansArgs &args, int samples,
                          std::mt19937_64 &rng) {
  float best_factor = 100.0f;
  float best_std = std::numeric_limits<float>::max();
  for (float factor = 0.001f; factor <= 1000.0f + 1e-3f; factor *= 10.0f) {
    TryClustering(data, indices, first, last, args, samples, factor, false,
                  rng);
    int active = 0;
    float avg = static_cast<float>(last - first) /
                static_cast<float>(std::max(1, args.DK));
    float variance = 0.0f;
    for (int k = 0; k < args.DK; ++k) {
      if (args.counts[k] > 0) {
        variance += (args.counts[k] - avg) * (args.counts[k] - avg);
        active++;
      }
    }
    float std_ratio = active > 0 ? std::sqrt(variance / args.DK) / avg
                                 : std::numeric_limits<float>::max();
    if (std_ratio < best_std) {
      best_std = std_ratio;
      best_factor = factor;
    }
  }
  HBC_LOG("    [bkt] best lambda factor=" << best_factor << std::endl);
  return best_factor;
}

void ShuffleClusters(std::vector<int32_t> &indices, size_t first, size_t last,
                     KmeansArgs &args) {
  if (last <= first) {
    return;
  }
  const size_t total = last - first;
  if (args.scratch_indices.size() < total) {
    args.scratch_indices.resize(total);
  }

  // 1. Compute write offsets for each cluster
  // writes[k] points to the next available slot for cluster k
  // ends[k] points to the last slot of cluster k (reserved for the center)
  std::vector<size_t> writes(args.K);
  std::vector<size_t> ends(args.K);

  size_t current_offset = 0;
  for (int k = 0; k < args.K; ++k) {
    size_t count = static_cast<size_t>(args.counts[k]);
    writes[k] = current_offset;
    ends[k] = current_offset + count - 1; // Last slot
    current_offset += count;
  }

  // 2. Scatter indices to scratch buffer
  // We don't need to update args.labels because they are not used subsequently
  // in BuildBkTree for this level.
  const int32_t *src_indices = indices.data() + first;
  const int *src_labels = args.labels.data() + first;
  int32_t *dst = args.scratch_indices.data();

  // The center vector ID for each cluster k is args.cluster_idx[k].
  // We need to place it exactly at ends[k].
  // All other vectors go to writes[k]++.

  // Optimization: caching cluster_idx for faster lookup might be useful,
  // but checking `val == center_id` inside loop is fast enough.

  for (size_t i = 0; i < total; ++i) {
    int32_t vid = src_indices[i];
    int label = src_labels[i];

    // Safety check for label range
    if (label < 0 || label >= args.K) {
      // access out of bounds protection?
      // Should not happen if logic is correct.
      // Fallback: just put it somewhere or skip?
      // Let's assume valid labels for speed.
      // But if Kmeans produced invalid label, we risk crash.
      // Given verified code, labels should be valid (0..DK-1).
      // args.labels initialized to 0.
      continue;
    }

    if (vid == args.cluster_idx[label]) {
      dst[ends[label]] = vid;
    } else {
      dst[writes[label]++] = vid;
    }
  }

  // 3. Copy back
  std::memcpy(indices.data() + first, dst, total * sizeof(int32_t));
}
int KmeansClustering(const DatasetView &data, std::vector<int32_t> &indices,
                     size_t first, size_t last, KmeansArgs &args, int samples,
                     float lambda_factor, std::mt19937_64 &rng) {
  TryClustering(data, indices, first, last, args, samples, lambda_factor, false,
                rng);
  int clusters = 0;
  for (int k = 0; k < args.DK; ++k) {
    if (args.counts[k] > 0) {
      clusters++;
    }
  }
  if (clusters <= 1) {
    return clusters;
  }
  ShuffleClusters(indices, first, last, args);
  return clusters;
}

struct BktBuildOptions {
  int k = 32;
  int leaf_size = 8;
  int samples = 1000;
  int threads = 1;
  float lambda_factor = 100.0f;
  bool dynamic_k = true;
};

struct BuildItem {
  int node_index = 0;
  size_t first = 0;
  size_t last = 0;
};

void BuildBkTree(const DatasetView &data, std::vector<int32_t> &indices,
                 const BktBuildOptions &opts, std::mt19937_64 &rng,
                 BKTree &out_tree) {
  out_tree.nodes.clear();
  out_tree.parent.clear();
  out_tree.tree_start.clear();
  out_tree.tree_start.push_back(0);

  BktNode root;
  root.centerid = static_cast<int32_t>(indices.size());
  root.child_start = -1;
  root.child_end = -1;
  root.pop = indices.size();
  root.begin = 0;
  root.end = indices.size();
  root.star = false;
  out_tree.nodes.push_back(root);
  out_tree.parent.push_back(-1);

  KmeansArgs args(opts.k, data.C(), indices.size(), opts.threads);

  float balance_factor = opts.lambda_factor;
  if (balance_factor <= 0.0f) {
    balance_factor = DynamicFactorSelect(data, indices, 0, indices.size(), args,
                                         opts.samples, rng);
  }

  std::stack<BuildItem> stack;
  stack.push(BuildItem{0, 0, indices.size()});

  const size_t total_points = indices.size();
  const size_t log_interval =
      std::max<size_t>(size_t(1), total_points / size_t(512));
  size_t processed_nodes = 0;

  HbcProgressStart(total_points);

  while (!stack.empty()) {
    BuildItem item = stack.top();
    stack.pop();

    BktNode &node_ref = out_tree.nodes[static_cast<size_t>(item.node_index)];
    node_ref.pop = item.last - item.first;
    node_ref.begin = item.first;
    node_ref.end = item.last;
    int32_t child_start = static_cast<int32_t>(out_tree.nodes.size());
    int32_t child_end = child_start;
    node_ref.child_start = child_start;
    node_ref.child_end = child_start;
    node_ref.star = false;

    size_t range = item.last - item.first;
    ++processed_nodes;

    if (range <= static_cast<size_t>(opts.leaf_size)) {
      for (size_t i = item.first; i < item.last; ++i) {
        BktNode leaf;
        leaf.centerid = indices[i];
        leaf.pop = 1;
        leaf.child_start = -1;
        leaf.child_end = -1;
        leaf.begin = i;
        leaf.end = i + 1;
        leaf.star = false;
        out_tree.nodes.push_back(leaf);
        out_tree.parent.push_back(item.node_index);
        ++child_end;
      }
      out_tree.nodes[static_cast<size_t>(item.node_index)].child_end =
          child_end;
      HbcProgressAdvance(range);
      continue;
    }

    if (opts.dynamic_k) {
      args.DK = std::min(
          opts.k, std::max(2, static_cast<int>(range / opts.leaf_size) + 1));
    } else {
      args.DK = opts.k;
    }

    const bool log_cluster =
        range >= static_cast<size_t>(opts.leaf_size) * 32 ||
        (processed_nodes % log_interval == 0);
    if (log_cluster) {
      HBC_LOG("    [bkt] clustering node (pop=" << range << ", dk=" << args.DK
                                                << ", pending=" << stack.size()
                                                << ")" << std::endl);
    }

    int clusters = KmeansClustering(data, indices, item.first, item.last, args,
                                    opts.samples, balance_factor, rng);
    if (log_cluster) {
      HBC_LOG("    [bkt] clustering node done (pop="
              << range << ", produced=" << clusters << ")" << std::endl);
    }
    if (clusters <= 1) {
      out_tree.nodes[static_cast<size_t>(item.node_index)].star = true;
      // if (log_this) {
      //   HBC_LOG("    [bkt] fallback to star pop="
      //           << range << " first=" << item.first << " last=" << item.last
      //           << std::endl);
      // }
      for (size_t i = item.first; i < item.last; ++i) {
        BktNode leaf;
        leaf.centerid = indices[i];
        leaf.pop = 1;
        leaf.child_start = -1;
        leaf.child_end = -1;
        leaf.begin = i;
        leaf.end = i + 1;
        leaf.star = false;
        out_tree.nodes.push_back(leaf);
        out_tree.parent.push_back(item.node_index);
        ++child_end;
      }
      out_tree.nodes[static_cast<size_t>(item.node_index)].child_end =
          child_end;
      HbcProgressAdvance(range);
      continue;
    }

    size_t cursor = item.first;
    for (int k = 0; k < args.DK; ++k) {
      int count = args.counts[k];
      if (count <= 0) {
        continue;
      }
      size_t begin = cursor;
      size_t end = cursor + static_cast<size_t>(count);
      BktNode child;
      child.centerid = indices[end - 1];
      child.child_start = -1;
      child.child_end = -1;
      child.pop = count;
      child.begin = begin;
      child.end = end;
      child.star = false;
      if (args.centers.size() >= static_cast<size_t>((k + 1) * args.dim)) {
        child.centroid.assign(
            args.centers.begin() +
                static_cast<size_t>(k) * static_cast<size_t>(args.dim),
            args.centers.begin() +
                static_cast<size_t>(k + 1) * static_cast<size_t>(args.dim));
      }
      out_tree.nodes.push_back(child);
      int child_index = static_cast<int>(out_tree.nodes.size() - 1);
      out_tree.parent.push_back(item.node_index);
      ++child_end;

      if (static_cast<size_t>(count) > static_cast<size_t>(opts.leaf_size)) {
        stack.push(BuildItem{child_index, begin, end});
      } else {
        HbcProgressAdvance(static_cast<size_t>(count));
      }
      cursor = end;
    }
    out_tree.nodes[static_cast<size_t>(item.node_index)].child_end = child_end;
  }

  HbcProgressFinish();
}

struct HeadSelectionOptions {
  int select_threshold = 6;
  int split_threshold = 25;
  int split_factor = 5;
  bool select_dynamically = true;
};

void AdjustHeadOptions(size_t vector_count, double ratio,
                       HeadSelectionOptions &opts) {
  if (opts.select_threshold <= 0) {
    opts.select_threshold =
        std::min<int>(static_cast<int>(vector_count) - 1,
                      std::max(2, static_cast<int>(1.0 / ratio)));
  }
  if (opts.split_threshold <= 0) {
    opts.split_threshold = std::min<int>(
        static_cast<int>(vector_count) - 1,
        std::max(opts.select_threshold * 2, opts.select_threshold + 1));
  }
  if (opts.split_factor <= 0) {
    opts.split_factor = std::min<int>(
        static_cast<int>(vector_count) - 1,
        std::max(2, static_cast<int>(std::round(1.0 / ratio) + 0.5)));
  }
}

size_t SelectHeadsInternal(const BKTree &tree, int node_idx,
                           const HeadSelectionOptions &opts, int root_centerid,
                           std::vector<int32_t> &selected) {
  using ChildPair = std::pair<int, size_t>;
  size_t children_size = 1;
  const BktNode &node = tree.nodes[static_cast<size_t>(node_idx)];

  std::vector<ChildPair> children;

  if (node.child_start >= 0 && node.child_start < node.child_end) {
    int count = node.child_end - node.child_start;
    std::vector<size_t> child_sizes(count, 0);

#pragma omp taskgroup
    {
      for (int i = 0; i < count; ++i) {
        int c = node.child_start + i;
#pragma omp task shared(tree, opts, root_centerid, selected, child_sizes)      \
    firstprivate(c, i)
        {
          child_sizes[i] =
              SelectHeadsInternal(tree, c, opts, root_centerid, selected);
        }
      }
    }

    for (int i = 0; i < count; ++i) {
      size_t cs = child_sizes[i];
      if (cs > 0) {
        children.emplace_back(node.child_start + i, cs);
        children_size += cs;
      }
    }
  }

  if (children_size >= static_cast<size_t>(opts.select_threshold)) {
    bool selected_node_added = false;
    if (node.centerid >= 0 && node.centerid < root_centerid) {
#pragma omp critical(select_heads_add)
      { selected.push_back(node_idx); }
      selected_node_added = true;
    }
    if (children_size > static_cast<size_t>(opts.split_threshold) &&
        !children.empty()) {
      std::sort(children.begin(), children.end(),
                [](const ChildPair &a, const ChildPair &b) {
                  return a.second > b.second;
                });
      size_t take = static_cast<size_t>(
          std::ceil(children_size * 1.0 / opts.split_factor) + 0.5);
#pragma omp critical(select_heads_add)
      {
        for (size_t i = 0; i < take && i < children.size(); ++i) {
          selected.push_back(children[i].first);
        }
      }
    }
    return 0;
  }

  return children_size;
}

std::vector<int32_t> SelectHeads(const BKTree &tree, size_t vector_count,
                                 size_t target_k, HeadSelectionOptions opts,
                                 std::mt19937_64 &rng) {
  std::vector<int32_t> heads;
  heads.reserve(target_k * 2);

  if (vector_count == 0) {
    return heads;
  }
  double ratio =
      target_k == 0 ? 0.0 : static_cast<double>(target_k) / vector_count;
  if (ratio >= 1.0) {
    for (size_t i = 0; i < vector_count; ++i) {
      heads.push_back(static_cast<int32_t>(i));
    }
    return heads;
  }

  AdjustHeadOptions(vector_count, ratio, opts);

  int root_centerid = tree.nodes.empty() ? -1 : tree.nodes.front().centerid;

  int best_select = opts.select_threshold;
  int best_split = opts.split_threshold;
  double min_diff = std::numeric_limits<double>::max();

#pragma omp parallel
#pragma omp single
  {
    if (opts.select_dynamically) {
      for (int select = 2; select <= opts.select_threshold; ++select) {
        HeadSelectionOptions trial_opts = opts;
        trial_opts.select_threshold = select;
        int l = opts.split_factor;
        int r = opts.split_threshold;
        while (l < r - 1) {
          trial_opts.split_threshold = (l + r) / 2;
          heads.clear();
          SelectHeadsInternal(tree, 0, trial_opts, root_centerid, heads);
          std::sort(heads.begin(), heads.end());
          heads.erase(std::unique(heads.begin(), heads.end()), heads.end());
          double diff =
              static_cast<double>(heads.size()) / vector_count - ratio;
          if (std::abs(diff) < min_diff) {
            min_diff = std::abs(diff);
            best_select = trial_opts.select_threshold;
            best_split = trial_opts.split_threshold;
          }
          if (diff > 0) {
            l = (l + r) / 2;
          } else {
            r = (l + r) / 2;
          }
        }
      }
    }

    heads.clear();
    HeadSelectionOptions final_opts = opts;
    final_opts.select_threshold = best_select;
    final_opts.split_threshold = best_split;
    SelectHeadsInternal(tree, 0, final_opts, root_centerid, heads);
  }
  std::sort(heads.begin(), heads.end());
  heads.erase(std::unique(heads.begin(), heads.end()), heads.end());

  if (heads.size() > target_k) {
    std::sort(heads.begin(), heads.end(), [&](int a, int b) {
      return tree.nodes[static_cast<size_t>(a)].pop >
             tree.nodes[static_cast<size_t>(b)].pop;
    });
    heads.resize(target_k);
  } else if (heads.size() < target_k) {
    std::vector<int32_t> candidates;
    candidates.reserve(tree.nodes.size());
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
      if (static_cast<int32_t>(i) == 0) {
        continue;
      }
      if (tree.nodes[i].centerid >= 0 &&
          tree.nodes[i].centerid < root_centerid) {
        candidates.push_back(static_cast<int32_t>(i));
      }
    }
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
      return tree.nodes[static_cast<size_t>(a)].pop >
             tree.nodes[static_cast<size_t>(b)].pop;
    });
    for (int idx : candidates) {
      if (std::find(heads.begin(), heads.end(), idx) == heads.end()) {
        heads.push_back(idx);
      }
      if (heads.size() >= target_k) {
        break;
      }
    }
  }

  std::sort(heads.begin(), heads.end());
  heads.erase(std::unique(heads.begin(), heads.end()), heads.end());
  if (heads.size() > target_k) {
    heads.resize(target_k);
  }
  return heads;
}

} // namespace

HbcBuilder::HbcBuilder(const HbcBuilderOptions &opt, GatherFn gather)
    : opt_(opt), gather_(std::move(gather)), rng_(opt.seed) {
  if (!gather_) {
    throw std::invalid_argument("HbcBuilder requires a valid gather reader");
  }
  if (opt_.dim <= 0 || opt_.total_vectors == 0) {
    throw std::invalid_argument("invalid HBC dimensions or dataset size");
  }
  if (opt_.bkt_kmeans_k < 2) {
    opt_.bkt_kmeans_k = 2;
  }
  if (opt_.bkt_leaf_size <= 0) {
    opt_.bkt_leaf_size = 8;
  }
  if (opt_.bkt_sample_size <= 0) {
    opt_.bkt_sample_size = 1000;
  }
  size_t computed = opt_.target_leaf_count;
  if (computed == 0 && opt_.target_leaf_size > 0) {
    computed = (opt_.total_vectors + opt_.target_leaf_size - 1) /
               opt_.target_leaf_size;
  }
  if (computed == 0) {
    computed = std::max<size_t>(1, opt_.total_vectors / 5);
  }
  target_leaf_count_ = std::max<size_t>(1, computed);

#ifdef _OPENMP
  if (opt_.num_threads > 0) {
    omp_set_num_threads(opt_.num_threads);
  }
#endif
}

HbcResult HbcBuilder::build() {
  HbcResult result;
  result.root = std::make_unique<HbcNode>();
  init_root_(*result.root);

  const size_t total = opt_.total_vectors;
  DatasetAccessor dataset(opt_, gather_);
  DatasetView data_view{&dataset};
  std::vector<int32_t> indices(total);
  std::iota(indices.begin(), indices.end(), 0);

  BktBuildOptions bopts;
  bopts.k = opt_.bkt_kmeans_k;
  bopts.leaf_size = opt_.bkt_leaf_size;
  bopts.samples = opt_.bkt_sample_size;
  bopts.dynamic_k = opt_.bkt_dynamic_k;
#ifdef _OPENMP
  bopts.threads =
      (opt_.num_threads > 0) ? opt_.num_threads : omp_get_max_threads();
#else
  bopts.threads = 1;
#endif
  bopts.lambda_factor = opt_.lambda_factor;

  HBC_LOG("    [bkt] build tree (N="
          << opt_.total_vectors << ", target_heads=" << target_leaf_count_
          << ", leaf_size=" << opt_.bkt_leaf_size << ")" << std::endl);

  BuildBkTree(data_view, indices, bopts, rng_, result.tree);

  HeadSelectionOptions head_opts;
  head_opts.select_threshold =
      opt_.select_threshold_hi > 0 ? opt_.select_threshold_hi : 6;
  head_opts.split_threshold =
      opt_.split_threshold_hi > 0 ? opt_.split_threshold_hi : 25;
  head_opts.split_factor = opt_.split_factor > 0 ? opt_.split_factor : 5;
  head_opts.select_dynamically = true;

  HBC_LOG("    [select] selecting heads..." << std::endl);
  std::vector<int32_t> head_nodes =
      SelectHeads(result.tree, total, target_leaf_count_, head_opts, rng_);

  HBC_LOG("    [bkt] built " << result.tree.nodes.size() << " nodes"
                             << std::endl);
  HBC_LOG("    [select] chosen "
          << head_nodes.size() << " posting centers out of "
          << result.tree.nodes.size() << " nodes" << std::endl);

  result.head_nodes = head_nodes;

  HBC_LOG("    [populate] populating HBC tree structure..." << std::endl);
  populate_hbc_tree_flat_(result.tree, head_nodes, indices, result);
  return result;
}

void HbcBuilder::init_root_(HbcNode &node) {
  node.center_id = -1;
  node.is_leaf = false;
  node.depth = 0;
  node.pop = opt_.total_vectors;
  node.centroid.assign(static_cast<size_t>(opt_.dim), 0.0f);
  node.ids.clear();
  node.children.clear();
}

void HbcBuilder::compute_centroid_for_ids_(const std::vector<int32_t> &ids,
                                           std::vector<float> &centroid) const {
  const int dim = opt_.dim;
  centroid.assign(static_cast<size_t>(dim), 0.0f);
  if (ids.empty()) {
    return;
  }
  std::vector<double> accum(static_cast<size_t>(dim), 0.0);
  const size_t block =
      std::max<size_t>(1, std::min(opt_.chunk_size, ids.size()));
  std::vector<int32_t> gather_ids(block);
  std::vector<float> buffer(block * static_cast<size_t>(dim));
  size_t processed = 0;
  while (processed < ids.size()) {
    size_t cur = std::min(block, ids.size() - processed);
    std::copy_n(ids.begin() + static_cast<std::ptrdiff_t>(processed), cur,
                gather_ids.begin());
    gather_(gather_ids.data(), cur, buffer.data());
    for (size_t i = 0; i < cur; ++i) {
      const float *vec = buffer.data() + i * static_cast<size_t>(dim);
      for (int d = 0; d < dim; ++d) {
        accum[static_cast<size_t>(d)] += vec[d];
      }
    }
    processed += cur;
  }
  const double inv = 1.0 / static_cast<double>(ids.size());
  for (int d = 0; d < dim; ++d) {
    centroid[static_cast<size_t>(d)] =
        static_cast<float>(accum[static_cast<size_t>(d)] * inv);
  }
}

void HbcBuilder::gather_vectors_(const int32_t *ids, size_t count,
                                 std::vector<float> &out) const {
  if (count == 0) {
    out.clear();
    return;
  }
  out.resize(count * static_cast<size_t>(opt_.dim));
  gather_(ids, count, out.data());
}

size_t HbcBuilder::collect_subtree_ids_(const BKTree &tree, int node_index,
                                        const std::vector<int32_t> &ids,
                                        std::vector<int32_t> &out) const {
  if (node_index < 0 || static_cast<size_t>(node_index) >= tree.nodes.size()) {
    return 0;
  }
  const BktNode &node = tree.nodes[static_cast<size_t>(node_index)];
  if (node.end <= node.begin) {
    return 0;
  }
  size_t begin = node.begin;
  size_t end = node.end;
  if (begin >= ids.size() || end > ids.size() || begin >= end) {
    return 0;
  }
  out.insert(out.end(), ids.begin() + static_cast<std::ptrdiff_t>(begin),
             ids.begin() + static_cast<std::ptrdiff_t>(end));
  return end - begin;
}

void HbcBuilder::populate_hbc_tree_flat_(const BKTree &tree,
                                         const std::vector<int32_t> &head_ids,
                                         const std::vector<int32_t> &ids,
                                         HbcResult &out) {
  if (!out.root) {
    return;
  }
  out.leaves.clear();
  out.root->children.clear();
  out.root->is_leaf = false;
  out.root->ids.clear();
  out.root->pop = opt_.total_vectors;
  out.root->centroid.assign(out.root->centroid.size(), 0.0f);

  std::vector<int32_t> node_depth(tree.nodes.size(), 0);
  if (!tree.nodes.empty()) {
    int root_idx = tree.tree_start.empty() ? 0 : tree.tree_start.front();
    std::vector<int32_t> stack;
    stack.push_back(root_idx);
    while (!stack.empty()) {
      int node_idx = stack.back();
      stack.pop_back();
      if (node_idx < 0 || static_cast<size_t>(node_idx) >= tree.nodes.size()) {
        continue;
      }
      const BktNode &node = tree.nodes[static_cast<size_t>(node_idx)];
      if (node.child_start < 0 || node.child_start >= node.child_end) {
        continue;
      }
      int child_depth = node_depth[static_cast<size_t>(node_idx)] + 1;
      for (int child = node.child_start; child < node.child_end; ++child) {
        if (child >= 0 && static_cast<size_t>(child) < tree.nodes.size()) {
          node_depth[static_cast<size_t>(child)] = child_depth;
          stack.push_back(child);
        }
      }
    }
  }

  std::vector<int32_t> ordered_heads = head_ids;
  std::sort(ordered_heads.begin(), ordered_heads.end(),
            [&](int32_t a, int32_t b) {
              int da = (a >= 0 && static_cast<size_t>(a) < node_depth.size())
                           ? node_depth[static_cast<size_t>(a)]
                           : 0;
              int db = (b >= 0 && static_cast<size_t>(b) < node_depth.size())
                           ? node_depth[static_cast<size_t>(b)]
                           : 0;
              if (da != db) {
                return da > db;
              }
              return a < b;
            });

  out.vector_to_leaf.assign(opt_.total_vectors, -1);

  std::vector<int32_t> collector;
  collector.reserve(opt_.bkt_leaf_size * 4);

  size_t heads_processed = 0;
  const size_t heads_total = ordered_heads.size();
  const size_t heads_log_interval = std::max<size_t>(1, heads_total / 10);

  for (int32_t head_node_idx : ordered_heads) {
    if (heads_processed % heads_log_interval == 0 && heads_processed > 0) {
      HBC_LOG("    [populate] " << heads_processed << "/" << heads_total
                                << " heads processed" << std::endl);
    }
    heads_processed++;

    if (head_node_idx < 0 ||
        static_cast<size_t>(head_node_idx) >= tree.nodes.size()) {
      continue;
    }
    const BktNode &bnode = tree.nodes[static_cast<size_t>(head_node_idx)];

    collector.clear();
    collect_subtree_ids_(tree, head_node_idx, ids, collector);
    if (collector.empty()) {
      continue;
    }

    std::vector<int32_t> filtered;
    filtered.reserve(collector.size());
    for (int32_t vid : collector) {
      if (vid < 0 || static_cast<size_t>(vid) >= out.vector_to_leaf.size()) {
        continue;
      }
      if (out.vector_to_leaf[static_cast<size_t>(vid)] >= 0) {
        continue;
      }
      filtered.push_back(vid);
    }
    if (filtered.empty()) {
      continue;
    }

    int leaf_index = static_cast<int>(out.leaves.size());
    for (int32_t vid : filtered) {
      out.vector_to_leaf[static_cast<size_t>(vid)] = leaf_index;
    }

    auto leaf = std::make_unique<HbcNode>();
    leaf->center_id = bnode.centerid;
    leaf->depth = (head_node_idx >= 0 &&
                   static_cast<size_t>(head_node_idx) < node_depth.size())
                      ? node_depth[static_cast<size_t>(head_node_idx)]
                      : out.root->depth + 1;
    leaf->is_leaf = true;
    leaf->ids = std::move(filtered);
    leaf->pop = leaf->ids.size();
    leaf->leaf_id = leaf_index;
    leaf->centroid.assign(static_cast<size_t>(opt_.dim), 0.0f);

    if (!bnode.centroid.empty()) {
      leaf->centroid = bnode.centroid;
    } else if (bnode.centerid >= 0 &&
               static_cast<size_t>(bnode.centerid) < opt_.total_vectors) {
      gather_vectors_(&bnode.centerid, 1, leaf->centroid);
    } else {
      compute_centroid_for_ids_(leaf->ids, leaf->centroid);
    }

    out.root->children.push_back(std::move(leaf));
    out.leaves.push_back(out.root->children.back().get());
  }

  size_t unassigned = 0;
  for (int32_t assigned : out.vector_to_leaf) {
    if (assigned < 0) {
      ++unassigned;
    }
  }
  if (unassigned > 0) {
    HBC_LOG("    [bkt] warning: " << unassigned
                                  << " vectors remain unassigned to heads"
                                  << std::endl);
  }
}
