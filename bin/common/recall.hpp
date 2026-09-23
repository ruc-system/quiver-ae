#pragma once

#include <cassert>
#include <cstddef>
#include <set>
#include <vector>

namespace bin_common {

struct PerQueryRecall {
  int hits = 0;
  double recall = 0.0;
};

inline PerQueryRecall calc_one_query_recall(const int* results,
                                            const int* ground_truth,
                                            int topk) {
  PerQueryRecall out;
  std::set<int> expected(ground_truth, ground_truth + topk);
  for (int j = 0; j < topk; ++j) {
    if (expected.find(results[j]) != expected.end()) {
      ++out.hits;
    }
  }
  out.recall = static_cast<double>(out.hits) / static_cast<double>(topk);
  return out;
}

inline std::vector<PerQueryRecall> calc_per_query_recall(
    const int* results, const int* ground_truth, size_t num_queries,
    size_t ground_truth_width, int topk) {
  assert(topk <= static_cast<int>(ground_truth_width));

  std::vector<PerQueryRecall> per_query;
  per_query.reserve(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    per_query.push_back(calc_one_query_recall(
        results + i * static_cast<size_t>(topk),
        ground_truth + i * ground_truth_width, topk));
  }
  return per_query;
}

inline double calc_recall(const int* results, const int* ground_truth,
                          size_t num_queries, size_t ground_truth_width,
                          int topk) {
  size_t hit_count = 0;
  assert(topk <= static_cast<int>(ground_truth_width));

  for (size_t i = 0; i < num_queries; ++i) {
    std::set<int> expected(ground_truth + i * ground_truth_width,
                           ground_truth + i * ground_truth_width + topk);
    for (int j = 0; j < topk; ++j) {
      if (expected.find(results[i * topk + j]) != expected.end()) {
        ++hit_count;
      }
    }
  }

  return static_cast<double>(hit_count) /
         static_cast<double>(num_queries * static_cast<size_t>(topk));
}

}  // namespace bin_common
