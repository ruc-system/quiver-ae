#include "index/replicator.h"

#include "index/hbc_progress.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/utils/distances.h>

namespace {

struct Candidate {
  float dist_sq = 0.0f;
  int leaf_id = -1;
  const HbcNode *node = nullptr;
};

} // namespace

BoundaryReplicator::BoundaryReplicator(const ReplicationOptions &opt, int dim,
                                       size_t total_vectors,
                                       const ICentroidNavigator &navigator,
                                       const std::vector<HbcNode *> &leaves,
                                       const std::vector<int32_t> &assignment,
                                       ReadBlockFn reader)
    : opt_(opt), dim_(dim), total_vectors_(total_vectors),
      navigator_(navigator), leaves_(leaves), primary_assignment_(assignment),
      read_block_(std::move(reader)) {
  if (dim_ <= 0 || total_vectors_ == 0) {
    throw std::invalid_argument("replicator requires valid dataset info");
  }
  if (leaves_.empty()) {
    throw std::invalid_argument(
        "replicator requires non-empty HBC tree leaves");
  }
  if (!read_block_) {
    throw std::invalid_argument("replicator requires dataset reader");
  }
  if (primary_assignment_.size() != total_vectors_) {
    throw std::invalid_argument(
        "primary assignment size must match dataset size");
  }
  if (opt_.max_replicas <= 0) {
    opt_.max_replicas = 1;
  }
  if (opt_.beam <= 0) {
    opt_.beam = 64;
  }
}

BoundaryReplicator::~BoundaryReplicator() = default;

ReplicationResult BoundaryReplicator::run() {
  ReplicationResult result;
  result.posting_lists.resize(leaves_.size());

  const float epsilon_scale = 1.0f + opt_.epsilon;
  const float epsilon_sq = epsilon_scale * epsilon_scale;
  const int search_k = std::max({opt_.beam, opt_.max_replicas * 4, 32});
  const size_t block = std::max<size_t>(1, opt_.block);
  const size_t max_replicas =
      static_cast<size_t>(std::max(1, opt_.max_replicas));

  HbcProgressStart(total_vectors_);

  size_t total_copies = 0;
  size_t min_copies = std::numeric_limits<size_t>::max();
  size_t max_copies = 0;

  struct ThreadRepData {
    std::vector<std::pair<int32_t, int32_t>> pairs;
    // Padding to avoid false sharing
    char padding[64];
  };

  int num_threads = 1;
#ifdef _OPENMP
  num_threads = std::max(1, omp_get_max_threads());
#endif
  std::vector<ThreadRepData> thread_data(num_threads);
  // Pre-reserve to avoid frequent reallocations
  // Estimate: avg_replicas * (total / threads) * safety_factor
  size_t est_per_thread =
      (total_vectors_ * std::max<size_t>(1, opt_.max_replicas)) / num_threads;
  for (auto &td : thread_data) {
    td.pairs.reserve(est_per_thread);
  }

  std::vector<float> buffer(block * static_cast<size_t>(dim_));

  auto compute_selection = [&](size_t vid, const float *vec,
                               std::vector<int> &out_ids,
                               std::vector<Candidate> &candidate_buf,
                               std::vector<Candidate> &filtered_buf,
                               std::vector<Candidate> &selected_buf) -> size_t {
    out_ids.clear();
    candidate_buf.clear();
    filtered_buf.clear();
    selected_buf.clear();

    const int primary =
        (vid < primary_assignment_.size()) ? primary_assignment_[vid] : -1;
    const bool primary_valid =
        (primary >= 0 && static_cast<size_t>(primary) < leaves_.size());

    ProbeResult probe = navigator_.search_one(vec, search_k);
    candidate_buf.reserve(probe.ids.size());
    for (size_t idx = 0; idx < probe.ids.size(); ++idx) {
      long raw_id = probe.ids[idx];
      if (raw_id < 0 || static_cast<size_t>(raw_id) >= leaves_.size()) {
        continue;
      }
      const HbcNode *node = leaves_[static_cast<size_t>(raw_id)];
      if (!node || node->centroid.size() != static_cast<size_t>(dim_)) {
        continue;
      }
      int leaf_id = static_cast<int>(raw_id);
      bool duplicate = false;
      for (const Candidate &existing : candidate_buf) {
        if (existing.leaf_id == leaf_id) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        continue;
      }
      float dist_sq = faiss::fvec_L2sqr(vec, node->centroid.data(), dim_);
      candidate_buf.push_back(Candidate{dist_sq, leaf_id, node});
    }

    if (candidate_buf.empty()) {
      if (primary_valid) {
        out_ids.push_back(primary);
        return out_ids.size();
      }
      return 0;
    }

    std::sort(candidate_buf.begin(), candidate_buf.end(),
              [](const Candidate &a, const Candidate &b) {
                return a.dist_sq < b.dist_sq;
              });

    const float base_dist = candidate_buf.front().dist_sq;
    const float threshold = base_dist * epsilon_sq;
    for (const Candidate &cand : candidate_buf) {
      if (cand.dist_sq <= threshold) {
        filtered_buf.push_back(cand);
      } else {
        break;
      }
    }

    if (filtered_buf.empty()) {
      if (primary_valid) {
        out_ids.push_back(primary);
        return out_ids.size();
      }
      return 0;
    }

    selected_buf.push_back(filtered_buf.front());
    for (size_t i = 1; i < filtered_buf.size(); ++i) {
      const Candidate &cand = filtered_buf[i];
      bool dominated = false;
      for (const Candidate &chosen : selected_buf) {
        if (!chosen.node || !cand.node) {
          continue;
        }
        float inter_dist = faiss::fvec_L2sqr(chosen.node->centroid.data(),
                                             cand.node->centroid.data(), dim_);
        if (cand.dist_sq > inter_dist) {
          dominated = true;
          break;
        }
      }
      if (!dominated) {
        selected_buf.push_back(cand);
      }
    }

    if (primary_valid) {
      out_ids.push_back(primary);
    }

    for (const Candidate &cand : selected_buf) {
      if (out_ids.size() >= max_replicas) {
        break;
      }
      if (cand.leaf_id == primary) {
        continue;
      }
      out_ids.push_back(cand.leaf_id);
    }

    if (out_ids.empty() && !selected_buf.empty()) {
      for (const Candidate &cand : selected_buf) {
        if (out_ids.size() >= max_replicas) {
          break;
        }
        out_ids.push_back(cand.leaf_id);
      }
    }

    if (out_ids.empty() && primary_valid) {
      out_ids.push_back(primary);
    }

    if (out_ids.size() > max_replicas) {
      out_ids.resize(max_replicas);
    }

    return out_ids.size();
  };

  size_t processed = 0;
  while (processed < total_vectors_) {
    const size_t current = std::min(block, total_vectors_ - processed);
    read_block_(processed, current, buffer.data());

#if defined(_OPENMP)
    size_t block_total = 0;
    size_t block_min = std::numeric_limits<size_t>::max();
    size_t block_max = 0;

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto &local_pairs = thread_data[tid].pairs;

      std::vector<int> selected_ids;
      selected_ids.reserve(max_replicas);
      std::vector<Candidate> candidate_buf;
      candidate_buf.reserve(static_cast<size_t>(search_k));
      std::vector<Candidate> filtered_buf;
      filtered_buf.reserve(static_cast<size_t>(search_k));
      std::vector<Candidate> selected_buf;
      selected_buf.reserve(max_replicas);

      size_t thread_total = 0;
      size_t thread_min = std::numeric_limits<size_t>::max();
      size_t thread_max = 0;

#pragma omp for schedule(static)
      for (size_t i = 0; i < current; ++i) {
        const float *vec = buffer.data() + i * static_cast<size_t>(dim_);
        const size_t vid = processed + i;
        size_t copies = compute_selection(vid, vec, selected_ids, candidate_buf,
                                          filtered_buf, selected_buf);
        if (copies == 0) {
          continue;
        }
        for (int list_id : selected_ids) {
          if (list_id < 0 ||
              static_cast<size_t>(list_id) >= result.posting_lists.size()) {
            continue;
          }
          local_pairs.emplace_back(static_cast<int32_t>(list_id),
                                   static_cast<int32_t>(vid));
        }
        thread_total += copies;
        thread_min = std::min(thread_min, copies);
        thread_max = std::max(thread_max, copies);
      }

#pragma omp critical
      {
        block_total += thread_total;
        if (thread_min != std::numeric_limits<size_t>::max()) {
          block_min = std::min(block_min, thread_min);
        }
        block_max = std::max(block_max, thread_max);
      }
    }

    if (block_min != std::numeric_limits<size_t>::max()) {
      min_copies = std::min(min_copies, block_min);
    }
    max_copies = std::max(max_copies, block_max);
    total_copies += block_total;
#else
    std::vector<int> selected_ids;
    selected_ids.reserve(max_replicas);
    std::vector<Candidate> candidate_buf;
    candidate_buf.reserve(static_cast<size_t>(search_k));
    std::vector<Candidate> filtered_buf;
    filtered_buf.reserve(static_cast<size_t>(search_k));
    std::vector<Candidate> selected_buf;
    selected_buf.reserve(max_replicas);

    for (size_t i = 0; i < current; ++i) {
      const float *vec = buffer.data() + i * static_cast<size_t>(dim_);
      const size_t vid = processed + i;
      size_t copies = compute_selection(vid, vec, selected_ids, candidate_buf,
                                        filtered_buf, selected_buf);
      if (copies == 0) {
        continue;
      }
      for (int list_id : selected_ids) {
        if (list_id < 0 ||
            static_cast<size_t>(list_id) >= result.posting_lists.size()) {
          continue;
        }
        // In non-parallel mode, we can just push directly or use thread_data[0]
        // Consistency: use thread_data[0] to reuse scatter logic
        thread_data[0].pairs.emplace_back(static_cast<int32_t>(list_id),
                                          static_cast<int32_t>(vid));
      }
      min_copies = std::min(min_copies, copies);
      max_copies = std::max(max_copies, copies);
      total_copies += copies;
    }
#endif

    processed += current;
    HbcProgressAdvance(current);
  }

  HbcProgressFinish();

  // Parallel Scatter
  // 1. Compute histogram per thread
  const size_t num_lists = result.posting_lists.size();
  std::vector<std::vector<size_t>> thread_histograms(
      num_threads, std::vector<size_t>(num_lists, 0));

#pragma omp parallel for schedule(static)
  for (int t = 0; t < num_threads; ++t) {
    for (const auto &p : thread_data[t].pairs) {
      if (p.first >= 0 && static_cast<size_t>(p.first) < num_lists) {
        thread_histograms[t][static_cast<size_t>(p.first)]++;
      }
    }
  }

  // 2. Compute global offsets (prefix sum)
  // list_offsets[list_id] = total count for list
  // thread_write_offsets[t][list_id] = where thread t starts writing for
  // list_id
  std::vector<size_t> list_sizes(num_lists, 0);
  std::vector<std::vector<size_t>> thread_write_offsets(
      num_threads, std::vector<size_t>(num_lists, 0));

  for (size_t list_id = 0; list_id < num_lists; ++list_id) {
    size_t current_offset = 0;
    for (int t = 0; t < num_threads; ++t) {
      thread_write_offsets[t][list_id] = current_offset;
      current_offset += thread_histograms[t][list_id];
    }
    list_sizes[list_id] = current_offset;
    result.posting_lists[list_id].resize(current_offset);
  }

  // 3. Scatter data
#pragma omp parallel for schedule(static)
  for (int t = 0; t < num_threads; ++t) {
    const auto &pairs = thread_data[t].pairs;
    // Make a local mutable copy of the row for this thread
    std::vector<size_t> local_offsets = thread_write_offsets[t];

    for (const auto &p : pairs) {
      int list_id = p.first;
      int32_t vid = p.second;
      if (list_id >= 0 && static_cast<size_t>(list_id) < num_lists) {
        size_t write_pos = local_offsets[static_cast<size_t>(list_id)]++;
        result.posting_lists[static_cast<size_t>(list_id)][write_pos] = vid;
      }
    }
  }

  // 4. Sort each posting list to ensure vector IDs are ordered
  //    This improves cache locality during search and is required for
  //    certain optimizations (e.g., binary search, merge operations)
#pragma omp parallel for schedule(dynamic)
  for (size_t list_id = 0; list_id < num_lists; ++list_id) {
    auto &list = result.posting_lists[list_id];
    if (!list.empty()) {
      std::sort(list.begin(), list.end());
    }
  }

  if (total_vectors_ > 0) {
    result.metrics.avg_replicas =
        static_cast<double>(total_copies) / static_cast<double>(total_vectors_);
  }
  result.metrics.min_replicas =
      (min_copies == std::numeric_limits<size_t>::max()) ? 0 : min_copies;
  result.metrics.max_replicas = max_copies;

  return result;
}
