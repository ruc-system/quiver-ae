#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

struct HbcNode {
  std::vector<int32_t> ids;
  std::vector<float> centroid;
  std::vector<std::unique_ptr<HbcNode>> children;
  int32_t center_id = -1;
  size_t pop = 0;
  bool is_leaf = false;
  int leaf_id = -1;
  int depth = 0;
};

struct BktNode {
  int32_t centerid = -1;
  int32_t child_start = -1;
  int32_t child_end = -1;
  size_t pop = 0;
  bool star = false;
  size_t begin = 0;
  size_t end = 0;
  std::vector<float> centroid;
};

struct BKTree {
  std::vector<int32_t> tree_start;
  std::vector<BktNode> nodes;
  std::vector<int32_t> parent;
  std::unordered_map<int32_t, int32_t> sample_center_map;
};

struct HbcResult {
  std::unique_ptr<HbcNode> root;
  std::vector<HbcNode *> leaves;
  std::vector<int32_t> vector_to_leaf;
  BKTree tree;
  std::vector<int32_t> head_nodes;
};

struct HbcBuilderOptions {
  int dim = 0;
  size_t total_vectors = 0;

  int bkt_kmeans_k = 32;
  int bkt_leaf_size = 8;
  int bkt_sample_size = 1000;
  bool bkt_dynamic_k = true;

  int select_threshold_hi = 6;
  int split_threshold_hi = 25;
  int split_factor = 5;

  float lambda_factor = 100.0f;
  size_t target_leaf_size = 0;
  size_t target_leaf_count = 0;
  size_t chunk_size = 8192;
  size_t dataset_preload_threshold_bytes = 16ull * 1024 * 1024 * 1024;
  size_t stream_block_bytes = 64ull * 1024 * 1024;

  uint64_t seed = 42;
  int num_threads = 0;
};

class HbcBuilder {
public:
  using GatherFn =
      std::function<void(const int32_t *ids, size_t count, float *out)>;

  HbcBuilder(const HbcBuilderOptions &opt, GatherFn gather);

  HbcResult build();

private:
  void init_root_(HbcNode &node);
  void compute_centroid_for_ids_(const std::vector<int32_t> &ids,
                                 std::vector<float> &centroid) const;
  void gather_vectors_(const int32_t *ids, size_t count,
                       std::vector<float> &out) const;
  void populate_hbc_tree_flat_(const BKTree &tree,
                               const std::vector<int32_t> &head_ids,
                               const std::vector<int32_t> &ids, HbcResult &out);
  size_t collect_subtree_ids_(const BKTree &tree, int node_index,
                              const std::vector<int32_t> &ids,
                              std::vector<int32_t> &out) const;

  HbcBuilderOptions opt_;
  GatherFn gather_;
  size_t target_leaf_count_ = 0;
  std::mt19937_64 rng_;
};
