#pragma once

#include "index/centroid_navigator.h"
#include "index/hbc_builder.h"
#include "index/replication_types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

struct ReplicationOptions {
  float epsilon = 0.1f;
  int max_replicas = 8;
  int beam = 64;
  size_t block = 8192;
};

class BoundaryReplicator {
public:
  using ReadBlockFn =
      std::function<void(size_t start, size_t count, float *out)>;

  BoundaryReplicator(const ReplicationOptions &opt, int dim,
                     size_t total_vectors, const ICentroidNavigator &navigator,
                     const std::vector<HbcNode *> &leaves,
                     const std::vector<int32_t> &primary_assignment,
                     ReadBlockFn reader);

  ~BoundaryReplicator();

  ReplicationResult run();

private:
  ReplicationOptions opt_;
  int dim_ = 0;
  size_t total_vectors_ = 0;
  const ICentroidNavigator &navigator_;
  std::vector<HbcNode *> leaves_;
  const std::vector<int32_t> &primary_assignment_;
  ReadBlockFn read_block_;
};
