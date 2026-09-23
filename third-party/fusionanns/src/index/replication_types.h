#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct ReplicationMetrics {
  size_t min_replicas = 0;
  size_t max_replicas = 0;
  double avg_replicas = 0.0;
};

struct ReplicationResult {
  std::vector<std::vector<int32_t>> posting_lists;
  ReplicationMetrics metrics;
};

