#include "index/builder.h"

#include "common/io.h"
#include "common/types.h"
#include "index/centroid_navigator_sptag.h"
#include "index/hbc_progress.h"
#include "index/replicator.h"
#include "index/residual_sq8_encoder.h"

#ifdef FUSIONANNS_USE_CUVS_CAGRA
#include "index/centroid_navigator_cagra.h"
#endif

#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using std::size_t;
namespace fs = std::filesystem;

#ifdef _OPENMP
#define OMP_COUT(expr)                                                         \
  do {                                                                         \
    _Pragma("omp critical(log_stream)") { std::cout << expr; }                 \
  } while (0)
#else
#define OMP_COUT(expr)                                                         \
  do {                                                                         \
    std::cout << expr;                                                         \
  } while (0)
#endif

namespace {

constexpr uint32_t kDefaultPageBytes = 4096;

template <typename T> void write_value(std::ofstream &ofs, const T &value) {
  ofs.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

size_t clamp_positive(size_t value, size_t fallback) {
  return value == 0 ? fallback : value;
}

} // namespace

FusionAnnsBuilder::FusionAnnsBuilder(const BuilderOptions &opt) : opt_(opt) {
#ifdef _OPENMP
  default_threads_ = std::max(1, omp_get_max_threads());
#endif
  if (opt_.base_path.empty()) {
    throw std::invalid_argument("base dataset path must not be empty");
  }
  if (opt_.chunk == 0) {
    opt_.chunk = 1;
  }
  if (opt_.page == 0) {
    opt_.page = kDefaultPageBytes;
  }
  if (opt_.max_replications <= 0) {
    opt_.max_replications = 1;
  }
  if (opt_.beam <= 0) {
    opt_.beam = 256;
  }
  if (opt_.output_dir.empty()) {
    throw std::invalid_argument("output directory must not be empty");
  }
  base_ = make_source(opt_.base_fmt, opt_.base_path, 0, 128, opt_.use_mmap);
  // LRU cache disabled per user request
  // if (opt_.cache_size_mb > 0) {
  //   base_ =
  //       std::make_unique<LRUDataSource>(std::move(base_),
  //       opt_.cache_size_mb);
  //   opt_.preload = false; // Disable full preload if LRU is used
  // }
  base_num_ = base_->size();
  dim_ = base_->dim();
  if (base_num_ == 0) {
    throw std::runtime_error("base dataset is empty");
  }
  if (dim_ <= 0) {
    throw std::runtime_error("invalid base dimension");
  }

  if (opt_.has_learn && !opt_.learn_path.empty()) {
    learn_ = make_source(opt_.learn_fmt, opt_.learn_path, opt_.learn_n_hint,
                         dim_, opt_.use_mmap);
  }

  if (opt_.preload) {
    preload_base_dataset_();
  }
}

void FusionAnnsBuilder::build() {
#ifdef _OPENMP
  const char *bind = std::getenv("OMP_PROC_BIND");
  const char *places = std::getenv("OMP_PLACES");
  OMP_COUT("[OMP] enabled, max_threads="
           << omp_get_max_threads() << " proc_bind=" << (bind ? bind : "unset")
           << " places=" << (places ? places : "unset") << std::endl);
#else
  OMP_COUT("[OMP] NOT enabled at compile time" << std::endl);
#endif

  ensure_dirs_();
  OMP_COUT("[1/6] Base dataset: N=" << base_num_ << ", d=" << dim_
                                    << std::endl);

  OMP_COUT("[2/6] Running hierarchical balanced clustering (HBC)..."
           << std::endl);
  HbcResult hbc = run_hbc_();
  vector_to_leaf_assignment_ = hbc.vector_to_leaf;
  primary_lists_.clear();
  primary_lists_.reserve(hbc.leaves.size());
  size_t min_leaf = std::numeric_limits<size_t>::max();
  size_t max_leaf = 0;
  double sum_leaf = 0.0;
  double sum_sq_leaf = 0.0;
  for (HbcNode *leaf : hbc.leaves) {
    if (!leaf) {
      continue;
    }
    std::vector<int32_t> ids = leaf->ids;
    primary_lists_.push_back(std::move(ids));
    size_t sz = primary_lists_.back().size();
    min_leaf = std::min(min_leaf, sz);
    max_leaf = std::max(max_leaf, sz);
    sum_leaf += static_cast<double>(sz);
    sum_sq_leaf += static_cast<double>(sz) * static_cast<double>(sz);
  }
  const size_t nlist = primary_lists_.size();
  double avg_leaf = nlist > 0 ? sum_leaf / static_cast<double>(nlist) : 0.0;
  double variance = nlist > 0 ? (sum_sq_leaf / static_cast<double>(nlist)) -
                                    avg_leaf * avg_leaf
                              : 0.0;
  double stddev_leaf = variance > 0.0 ? std::sqrt(variance) : 0.0;
  OMP_COUT("  > HBC produced " << nlist << " leaves (avg=" << avg_leaf
                               << ", min=" << (nlist ? min_leaf : 0)
                               << ", max=" << max_leaf << ", std=" << std::fixed
                               << std::setprecision(2) << stddev_leaf << ")"
                               << std::endl);

  OMP_COUT("[3/6] Building centroid navigator..." << std::endl);
  std::unique_ptr<ICentroidNavigator> navigator =
      build_centroid_navigator_(hbc.leaves);
  if (!navigator) {
    throw std::runtime_error("failed to construct centroid navigator");
  }
  const int tune_k = std::max({opt_.beam, opt_.max_replications * 4, 32});
  if (auto *sptag_nav =
          dynamic_cast<SPTAGCentroidNavigator *>(navigator.get())) {
    sptag_nav->tune_for_nprobe(tune_k);
  }

  OMP_COUT("[4/6] Applying boundary replication..." << std::endl);
  ReplicationResult replication = run_replication_(hbc, *navigator);
  replication_lists_ = std::move(replication.posting_lists);
  OMP_COUT("  > Replication stats (avg="
           << replication.metrics.avg_replicas
           << ", min=" << replication.metrics.min_replicas
           << ", max=" << replication.metrics.max_replicas << ")" << std::endl);

  if (opt_.use_residual_sq8) {
    OMP_COUT("[5/6] Encoding vectors with Residual SQ8..." << std::endl);
    encode_residual_sq8_();
  } else {
    OMP_COUT("[5/6] Training PQ and encoding vectors..." << std::endl);
    train_pq_and_encode_();
  }

  OMP_COUT("[6/6] Packing raw vectors and writing metadata..." << std::endl);
  create_packed_layout_();
  save_posting_list_metadata_();

  OMP_COUT("\n--- Offline index build completed successfully ---\n");
}

void FusionAnnsBuilder::ensure_dirs_() const {
  const fs::path root = output_dir_path_();
  fs::create_directories(root);
  fs::create_directories(root / "proximity_graph_index");
}

void FusionAnnsBuilder::preload_base_dataset_() {
  if (has_preload_) {
    return;
  }
  const size_t stride = static_cast<size_t>(dim_);
  OMP_COUT("  > Preloading base dataset into memory ("
           << base_num_ << " vectors)..." << std::endl);
  base_cache_.resize(base_num_ * stride);
  const size_t block = std::max<size_t>(1, std::min(opt_.chunk, base_num_));
  size_t offset = 0;
  while (offset < base_num_) {
    size_t current = std::min(block, base_num_ - offset);
    base_->read_block(offset, current, base_cache_.data() + offset * stride);
    offset += current;
    if (offset % (block * 4) == 0 || offset == base_num_) {
      OMP_COUT("    [preload] " << offset << "/" << base_num_
                                << " vectors cached" << std::endl);
    }
  }
  has_preload_ = true;
  double bytes = static_cast<double>(base_cache_.size()) * sizeof(float);
  double gib = bytes / (1024.0 * 1024.0 * 1024.0);
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << gib;
  OMP_COUT("  > Preload complete (" << oss.str() << " GiB cached)"
                                    << std::endl);
}

void FusionAnnsBuilder::read_block_cached_(size_t start, size_t count,
                                           float *out) const {
  if (count == 0) {
    return;
  }
  if (has_preload_) {
    const size_t stride = static_cast<size_t>(dim_);
    std::memcpy(out, base_cache_.data() + start * stride,
                count * stride * sizeof(float));
    return;
  }
  base_->read_block(start, count, out);
}

void FusionAnnsBuilder::read_gather_cached_(const int64_t *ids, size_t count,
                                            float *out) const {
  if (count == 0) {
    return;
  }
  if (has_preload_) {
    const size_t stride = static_cast<size_t>(dim_);
    const float *cache_ptr = base_cache_.data();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      size_t idx = static_cast<size_t>(ids[i]);
      std::memcpy(out + i * stride, cache_ptr + idx * stride,
                  stride * sizeof(float));
    }
    return;
  }
  base_->read_gather(ids, count, out);
}

void FusionAnnsBuilder::read_gather_cached_(const int32_t *ids, size_t count,
                                            float *out) const {
  if (count == 0) {
    return;
  }
  if (has_preload_) {
    const size_t stride = static_cast<size_t>(dim_);
    const float *cache_ptr = base_cache_.data();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      size_t idx = static_cast<size_t>(ids[i]);
      std::memcpy(out + i * stride, cache_ptr + idx * stride,
                  stride * sizeof(float));
    }
    return;
  }

  // Check for contiguous IDs to optimize I/O
  bool contiguous = true;
  if (count > 1) {
    for (size_t i = 1; i < count; ++i) {
      if (ids[i] != ids[i - 1] + 1) {
        contiguous = false;
        break;
      }
    }
  }

  if (contiguous) {
    base_->read_block(static_cast<size_t>(ids[0]), count, out);
    return;
  }

  std::vector<int64_t> gather(count);
  for (size_t i = 0; i < count; ++i) {
    gather[i] = static_cast<int64_t>(ids[i]);
  }
  base_->read_gather(gather.data(), count, out);
}

std::vector<float> FusionAnnsBuilder::sample_train_(size_t n_train) {
  if (n_train == 0) {
    return {};
  }
  if (learn_) {
    size_t take = std::min(n_train, learn_->size());
    std::vector<float> out(take * static_cast<size_t>(dim_));
    const size_t block = std::max<size_t>(1, std::min(opt_.chunk, take));
    std::vector<float> buffer(block * static_cast<size_t>(dim_));
    size_t written = 0;
    for (size_t offset = 0; offset < take; offset += block) {
      size_t cur = std::min(block, take - offset);
      learn_->read_block(offset, cur, buffer.data());
      std::memcpy(out.data() + written * static_cast<size_t>(dim_),
                  buffer.data(),
                  cur * static_cast<size_t>(dim_) * sizeof(float));
      written += cur;
    }
    return out;
  }

  size_t desired = std::min(n_train, base_num_);
  std::vector<float> reservoir(desired * static_cast<size_t>(dim_));
  if (desired == 0) {
    return reservoir;
  }

  const size_t block = std::max<size_t>(1, std::min(opt_.chunk, base_num_));
  std::vector<float> buffer(block * static_cast<size_t>(dim_));

  size_t filled = std::min(block, desired);
  read_block_cached_(0, filled, reservoir.data());
  size_t seen = filled;

  std::mt19937_64 rng(42);
  for (size_t offset = filled; offset < base_num_; offset += block) {
    size_t cur = std::min(block, base_num_ - offset);
    read_block_cached_(offset, cur, buffer.data());
    for (size_t i = 0; i < cur; ++i) {
      uint64_t j = std::uniform_int_distribution<uint64_t>(0, seen + i)(rng);
      if (j < desired) {
        std::memcpy(reservoir.data() + j * static_cast<size_t>(dim_),
                    buffer.data() + i * static_cast<size_t>(dim_),
                    static_cast<size_t>(dim_) * sizeof(float));
      }
    }
    seen += cur;
  }

  return reservoir;
}

HbcResult FusionAnnsBuilder::run_hbc_() {
  HbcBuilderOptions hopt;
  hopt.dim = dim_;
  hopt.total_vectors = base_num_;
  hopt.bkt_kmeans_k = std::max(2, opt_.branch);
  const size_t requested_nlist =
      opt_.nlist > 0 ? static_cast<size_t>(opt_.nlist) : 0;
  if (requested_nlist > 0) {
    hopt.target_leaf_count = requested_nlist;
  }
  size_t target_leaf_size =
      requested_nlist > 0
          ? std::max<size_t>(1, (base_num_ + requested_nlist - 1) /
                                    requested_nlist)
          : 0;
  if (target_leaf_size == 0) {
    target_leaf_size = 4096;
  }
  hopt.target_leaf_count = std::max<size_t>(
      1, hopt.target_leaf_count > 0 ? hopt.target_leaf_count
                                    : base_num_ / target_leaf_size);
  hopt.bkt_leaf_size = std::max<int>(
      8, static_cast<int>(std::min<size_t>(target_leaf_size, 1024)));
  hopt.bkt_sample_size = static_cast<int>(
      std::max<size_t>(1000, std::min<size_t>(100000, opt_.chunk)));
  if (opt_.imbalance_delta > 0.0f) {
    hopt.lambda_factor = opt_.imbalance_delta;
  }
  hopt.chunk_size = std::max<size_t>(opt_.chunk, 100000);
  constexpr size_t kMiB = 1024ull * 1024ull;
  constexpr size_t kGiB = 1024ull * 1024ull * 1024ull;

  // If the outer FusionAnnsBuilder has preloaded the dataset, or if user
  // explicitly set hbc_preload_gib, configure HBC accordingly.
  // When data is preloaded externally, the gather callback reads from memory,
  // so we should use large blocks for better batching performance.
  if (has_preload_) {
    // Data is already in memory via FusionAnnsBuilder::preload_base_dataset_
    // Set a large threshold so HBC knows data access is fast, but it won't
    // do its own redundant preload because the gather callback handles it.
    // We set threshold to 0 to skip HBC's internal preload (it would be
    // redundant), but we need a reasonable stream_block_bytes for batching.
    hopt.dataset_preload_threshold_bytes = 0;
    // Use a large block size since data is already in memory
    hopt.stream_block_bytes = 128ull * kMiB; // 128 MiB per thread block
  } else if (opt_.hbc_preload_gib == 0) {
    hopt.dataset_preload_threshold_bytes = 0;
    // Streaming from disk: use user-specified or default block size
    size_t stream_mb = opt_.hbc_stream_block_mb;
    if (opt_.use_mmap || stream_mb == 0) {
      stream_mb = 32; // Default 32 MiB for streaming
    }
    size_t stream_bytes =
        stream_mb > (std::numeric_limits<size_t>::max() / kMiB)
            ? std::numeric_limits<size_t>::max()
            : stream_mb * kMiB;
    hopt.stream_block_bytes = stream_bytes;
  } else {
    size_t clamped_gib = std::min<size_t>(
        opt_.hbc_preload_gib, std::numeric_limits<size_t>::max() / kGiB);
    hopt.dataset_preload_threshold_bytes = clamped_gib * kGiB;
    size_t stream_mb = opt_.hbc_stream_block_mb;
    if (stream_mb == 0) {
      stream_mb = 32;
    }
    size_t stream_bytes =
        stream_mb > (std::numeric_limits<size_t>::max() / kMiB)
            ? std::numeric_limits<size_t>::max()
            : stream_mb * kMiB;
    hopt.stream_block_bytes = stream_bytes;
  }

  hopt.seed = 42;
#ifdef _OPENMP
  hopt.num_threads = default_threads_;
#endif

  HbcBuilder builder(hopt,
                     [this](const int32_t *ids, size_t count, float *out) {
                       read_gather_cached_(ids, count, out);
                     });

  HbcResult result = builder.build();
  for (HbcNode *leaf : result.leaves) {
    if (leaf) {
      std::sort(leaf->ids.begin(), leaf->ids.end());
    }
  }
  return result;
}

std::unique_ptr<ICentroidNavigator>
FusionAnnsBuilder::build_centroid_navigator_(
    const std::vector<HbcNode *> &leaves) {
  if (leaves.empty()) {
    throw std::runtime_error(
        "cannot build centroid navigator: no leaf centroids available");
  }
  // 存储 centroids 以便 RSQ8 编码使用
  leaf_centroids_.resize(leaves.size() * static_cast<size_t>(dim_));
  for (size_t i = 0; i < leaves.size(); ++i) {
    const HbcNode *leaf = leaves[i];
    if (!leaf || leaf->centroid.size() != static_cast<size_t>(dim_)) {
      throw std::runtime_error(
          "HBC leaf centroid dimension mismatch during navigator build");
    }
    std::copy(leaf->centroid.begin(), leaf->centroid.end(),
              leaf_centroids_.begin() + i * static_cast<size_t>(dim_));
  }
  auto navigator = std::make_unique<SPTAGCentroidNavigator>();
  navigator->build(leaf_centroids_.data(), static_cast<int>(leaves.size()),
                   dim_, default_threads_);
  const std::string nav_dir =
      (output_dir_path_() / "proximity_graph_index").string();
  navigator->save(nav_dir);

#ifdef FUSIONANNS_USE_CUVS_CAGRA
  {
    CAGRACentroidNavigator cagra_nav;
    cagra_nav.build(leaf_centroids_.data(), static_cast<int>(leaves.size()),
                    dim_, 0);
    cagra_nav.save((output_dir_path_() / "centroid_graph_cagra.bin").string());
    OMP_COUT("  > CAGRA centroid graph saved to centroid_graph_cagra.bin"
             << std::endl);
  }
#endif

  return navigator;
}

ReplicationResult
FusionAnnsBuilder::run_replication_(const HbcResult &hbc,
                                    const ICentroidNavigator &navigator) {
  ReplicationOptions ropt;
  ropt.epsilon = opt_.epsilon;
  ropt.max_replicas = static_cast<int>(opt_.max_replications);
  ropt.beam = opt_.beam;
  ropt.block = opt_.chunk;

  BoundaryReplicator replicator(ropt, dim_, base_num_, navigator, hbc.leaves,
                                vector_to_leaf_assignment_,
                                [this](size_t start, size_t count, float *out) {
                                  read_block_cached_(start, count, out);
                                });

  return replicator.run();
}

void FusionAnnsBuilder::build_packed_order_() {
  if (has_packed_order_) {
    return;
  }

  const uint32_t page_bytes = opt_.page;
  const uint32_t vec_bytes =
      static_cast<uint32_t>(dim_) * static_cast<uint32_t>(sizeof(float));
  if (vec_bytes == 0 || vec_bytes > page_bytes) {
    throw std::runtime_error(
        "vector size exceeds page size; increase --page or adjust layout");
  }
  const uint32_t per_page = page_bytes / vec_bytes;
  if (per_page == 0) {
    throw std::runtime_error("page size too small for a single vector");
  }

  packed_order_.clear();
  packed_order_.reserve(base_num_);

  struct TailSlice {
    std::vector<int64_t> ids;
    size_t cursor = 0;
  };
  std::vector<TailSlice> tail_slices;
  tail_slices.reserve(primary_lists_.size());

  for (size_t cid = 0; cid < primary_lists_.size(); ++cid) {
    const auto &list = primary_lists_[cid];
    if (list.empty()) {
      continue;
    }
    std::vector<int32_t> ids = list;
    std::sort(ids.begin(), ids.end());

    size_t consumed = 0;
    const size_t total = ids.size();
    const size_t full_pages = per_page > 0 ? total / per_page : 0;

    for (size_t page_idx = 0; page_idx < full_pages; ++page_idx) {
      for (size_t i = 0; i < per_page; ++i) {
        packed_order_.push_back(static_cast<int64_t>(ids[consumed + i]));
      }
      consumed += per_page;
    }

    size_t tail = total - consumed;
    if (tail > 0) {
      TailSlice slice;
      slice.ids.reserve(tail);
      for (size_t i = 0; i < tail; ++i) {
        slice.ids.push_back(static_cast<int64_t>(ids[consumed + i]));
      }
      tail_slices.push_back(std::move(slice));
    }
  }

  if (!tail_slices.empty()) {
    struct TailRef {
      size_t index;
      size_t remaining;
    };
    struct TailRefLess {
      bool operator()(const TailRef &a, const TailRef &b) const {
        if (a.remaining != b.remaining) {
          return a.remaining < b.remaining;
        }
        return a.index < b.index;
      }
    };

    std::multiset<TailRef, TailRefLess> pool;
    for (size_t idx = 0; idx < tail_slices.size(); ++idx) {
      size_t remaining = tail_slices[idx].ids.size();
      if (remaining > 0) {
        pool.insert(TailRef{idx, remaining});
      }
    }

    std::vector<int64_t> page_ids;
    page_ids.reserve(per_page);

    while (!pool.empty()) {
      page_ids.clear();

      auto max_it = std::prev(pool.end());
      TailRef largest = *max_it;
      pool.erase(max_it);

      TailSlice &base_slice = tail_slices[largest.index];
      while (base_slice.cursor < base_slice.ids.size()) {
        page_ids.push_back(base_slice.ids[base_slice.cursor++]);
      }

      size_t remaining_slots =
          page_ids.size() < per_page ? per_page - page_ids.size() : 0;
      while (remaining_slots > 0 && !pool.empty()) {
        auto min_it = pool.begin();
        TailRef smallest = *min_it;
        pool.erase(min_it);

        TailSlice &slice = tail_slices[smallest.index];
        size_t avail = slice.ids.size() - slice.cursor;
        size_t take = std::min(avail, remaining_slots);
        for (size_t i = 0; i < take; ++i) {
          page_ids.push_back(slice.ids[slice.cursor++]);
        }
        remaining_slots -= take;
        size_t rem = slice.ids.size() - slice.cursor;
        if (rem > 0) {
          pool.insert(TailRef{smallest.index, rem});
        }
      }

      for (int64_t id : page_ids) {
        packed_order_.push_back(id);
      }
    }
  }

  if (packed_order_.size() != base_num_) {
    throw std::runtime_error(
        "packed order size mismatch with base dataset size");
  }

  has_packed_order_ = true;
  OMP_COUT("  > Packed order prepared (" << packed_order_.size() << " entries)"
                                         << std::endl);
}

void FusionAnnsBuilder::train_pq_and_encode_() {
  build_packed_order_();
  OMP_COUT("  > Training PQ (m=" << opt_.m << ", nbits=" << opt_.nbits
                                 << ") and streaming codes..." << std::endl);

  size_t sample_size = std::min<size_t>(1'000'000, base_num_ / 50 + 100'000);
  auto train = sample_train_(sample_size);
  if (train.empty()) {
    throw std::runtime_error("no training data available for pq");
  }

  faiss::ProductQuantizer pq(dim_, opt_.m, opt_.nbits);
  pq.train(train.size() / static_cast<size_t>(dim_), train.data());
  const std::string pq_codec_file = output_path_("pq_codec").string();
  faiss::write_ProductQuantizer(&pq, pq_codec_file.c_str());

  encode_pq_codes_parallel_(pq);
  OMP_COUT("  > PQ codec and codes written under " << opt_.output_dir
                                                   << std::endl);
}

void FusionAnnsBuilder::encode_residual_sq8_() {
  if (replication_lists_.empty()) {
    throw std::runtime_error("No posting lists available for RSQ8 encoding");
  }
  if (leaf_centroids_.empty()) {
    throw std::runtime_error("No centroids available for RSQ8 encoding");
  }

  const size_t nlist = replication_lists_.size();
  OMP_COUT("  > RSQ8 encoding "
           << nlist << " clusters (dim=" << dim_ << ", scale_percentile="
           << opt_.rsq8_scale_percentile << ")..." << std::endl);

  // 配置编码器
  fusionann::RSQ8EncoderOptions enc_opt;
  enc_opt.dim = dim_;
  enc_opt.scale_percentile = opt_.rsq8_scale_percentile;
  enc_opt.use_max_abs = opt_.rsq8_use_max_abs;
  enc_opt.num_threads = default_threads_;

  fusionann::RSQ8Encoder encoder(enc_opt);

  // Gather 回调函数
  auto gather_fn = [this](const int32_t *ids, size_t count, float *out) {
    read_gather_cached_(ids, count, out);
  };

  // 编码
  fusionann::RSQ8EncodedIndex encoded =
      encoder.encode(leaf_centroids_.data(), replication_lists_, gather_fn);

  // 保存索引文件
  const std::string rsq8_path = output_path_("residual_sq8_index.bin").string();
  encoded.save(rsq8_path);

  OMP_COUT("  > RSQ8 index saved to " << rsq8_path << std::endl);
}

void FusionAnnsBuilder::encode_pq_codes_parallel_(
    const faiss::ProductQuantizer &pq) {
  if (!has_packed_order_) {
    build_packed_order_();
  }

  const size_t block_size =
      std::max<size_t>(1, std::min(opt_.chunk, base_num_));
  const size_t code_size = static_cast<size_t>(pq.code_size);
  if (code_size == 0) {
    throw std::runtime_error("PQ code size must be > 0");
  }
  const size_t total_bytes = base_num_ * code_size;

  const fs::path pq_codes_path = output_path_("pq_codes");
  const std::string pq_codes_file = pq_codes_path.string();
  int fd = ::open(pq_codes_file.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC,
                  0644);
  if (fd < 0) {
    throw std::system_error(errno, std::system_category(),
                            "open " + pq_codes_file);
  }
  if (::ftruncate(fd, static_cast<off_t>(total_bytes)) != 0) {
    int err = errno;
    ::close(fd);
    throw std::system_error(err, std::system_category(),
                            "ftruncate " + pq_codes_file);
  }

#ifdef _OPENMP
  omp_set_num_threads(default_threads_);
#endif

#pragma omp parallel
  {
    std::vector<uint8_t> local_codes(block_size * code_size);
    std::vector<int64_t> gather_ids(block_size);
    std::vector<float> buffer(block_size * static_cast<size_t>(dim_));

#pragma omp for schedule(static)
    for (size_t pack_offset = 0; pack_offset < packed_order_.size();
         pack_offset += block_size) {
      size_t current = std::min(block_size, packed_order_.size() - pack_offset);
      for (size_t i = 0; i < current; ++i) {
        gather_ids[i] = packed_order_[pack_offset + i];
      }

      read_gather_cached_(gather_ids.data(), current, buffer.data());
      pq.compute_codes(buffer.data(), local_codes.data(),
                       static_cast<int>(current));

      for (size_t i = 0; i < current; ++i) {
        size_t gid = static_cast<size_t>(gather_ids[i]);
        off_t byte_offset =
            static_cast<off_t>(gid) * static_cast<off_t>(code_size);
        ssize_t written = ::pwrite(fd, local_codes.data() + i * code_size,
                                   static_cast<size_t>(code_size), byte_offset);
        if (written != static_cast<ssize_t>(code_size)) {
#pragma omp critical(builder_exception)
          {
            throw std::system_error(errno, std::system_category(),
                                    "pwrite " + pq_codes_file);
          }
        }
      }
    }
  }

  ::close(fd);
}

void FusionAnnsBuilder::create_packed_layout_() {
  if (!has_packed_order_) {
    build_packed_order_();
  }

  const uint32_t page_bytes = opt_.page;
  const uint32_t vec_bytes =
      static_cast<uint32_t>(dim_) * static_cast<uint32_t>(sizeof(float));
  if (vec_bytes == 0 || vec_bytes > page_bytes) {
    throw std::runtime_error(
        "vector size exceeds page size; increase --page or adjust layout");
  }

  const uint32_t per_page = page_bytes / vec_bytes;
  if (per_page == 0) {
    throw std::runtime_error("page size too small for a single vector");
  }

  const fs::path packed_path = output_path_("packed_raw_vectors.bin");
  std::ofstream ofs(packed_path, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("failed to open " + packed_path.string() +
                             " for write");
  }

  std::vector<VectorLocation> locations(base_num_);
  std::vector<char> page(page_bytes, 0);
  std::vector<float> gather_buffer(per_page * static_cast<size_t>(dim_));
  std::vector<int64_t> gather_ids(per_page);

  uint64_t bytes_written = 0;

  for (size_t start = 0; start < packed_order_.size(); start += per_page) {
    size_t count = std::min<size_t>(per_page, packed_order_.size() - start);
    for (size_t i = 0; i < count; ++i) {
      gather_ids[i] = packed_order_[start + i];
    }
    if (count > 0) {
      read_gather_cached_(gather_ids.data(), count, gather_buffer.data());
    }

    std::fill(page.begin(), page.end(), 0);
    uint32_t page_id = static_cast<uint32_t>(bytes_written / page_bytes);
    for (size_t i = 0; i < count; ++i) {
      const float *vec = gather_buffer.data() + i * static_cast<size_t>(dim_);
      std::memcpy(page.data() + i * vec_bytes, vec, vec_bytes);
      uint16_t offset_in_page = static_cast<uint16_t>(i * vec_bytes);
      locations[static_cast<size_t>(gather_ids[i])] =
          VectorLocation{page_id, offset_in_page};
    }

    ofs.write(page.data(), static_cast<std::streamsize>(page_bytes));
    bytes_written += page_bytes;
  }

  ofs.flush();
  if (!ofs) {
    throw std::runtime_error("failed to finish writing packed_raw_vectors.bin");
  }

  const fs::path loc_path = output_path_("vector_location.map");
  save_binary<VectorLocation>(loc_path.string(), locations);
  OMP_COUT("  > Packed layout written to " << packed_path.string()
                                           << std::endl);
}

void FusionAnnsBuilder::save_posting_list_metadata_() const {
  const fs::path path = output_path_("posting_lists_metadata");
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("failed to open " + path.string() + " for write");
  }

  const uint32_t nlist = static_cast<uint32_t>(replication_lists_.size());
  std::vector<uint64_t> offsets(nlist, 0);
  std::vector<uint32_t> counts(nlist, 0);
  uint64_t total_ids = 0;
  for (uint32_t cid = 0; cid < nlist; ++cid) {
    const auto &vec = replication_lists_[cid];
    if (vec.size() >
        static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("replication list too large for uint32_t: " +
                               std::to_string(cid));
    }
    counts[cid] = static_cast<uint32_t>(vec.size());
    offsets[cid] = total_ids;
    total_ids += static_cast<uint64_t>(vec.size());
  }

  struct ListEntry {
    uint64_t offset;
    uint32_t count;
    uint32_t reserved;
  };

  write_value(ofs, nlist);
  for (uint32_t cid = 0; cid < nlist; ++cid) {
    ListEntry entry{offsets[cid], counts[cid], 0};
    ofs.write(reinterpret_cast<const char *>(&entry), sizeof(ListEntry));
  }

  for (uint32_t cid = 0; cid < nlist; ++cid) {
    const auto &vec = replication_lists_[cid];
    if (vec.empty()) {
      continue;
    }
    ofs.write(reinterpret_cast<const char *>(vec.data()),
              static_cast<std::streamsize>(vec.size() * sizeof(int32_t)));
  }
  ofs.flush();
  if (!ofs) {
    throw std::runtime_error("failed to finish writing posting list metadata");
  }
  OMP_COUT("  > Posting list metadata written to " << path.string()
                                                   << std::endl);
}

fs::path FusionAnnsBuilder::output_dir_path_() const {
  return fs::path(opt_.output_dir);
}

fs::path FusionAnnsBuilder::output_path_(const std::string &relative) const {
  return output_dir_path_() / relative;
}
