#pragma once

#include <cstddef>
#include <cstdint>

#include "runtime_stats.hpp"

namespace shared {

inline void io_stats_record_enqueued(LoaderStatsAccumulator *stats,
                                     uint64_t count = 1) {
  if (stats != nullptr) stats->record_enqueued(count);
}

inline void io_stats_record_submitted(LoaderStatsAccumulator *stats,
                                      size_t ssd_id, uint64_t count = 1) {
  if (stats != nullptr) stats->record_submitted(ssd_id, count);
}

inline void io_stats_record_completed(LoaderStatsAccumulator *stats,
                                      size_t ssd_id, uint64_t count = 1) {
  if (stats != nullptr) stats->record_completed(ssd_id, count);
}

inline void io_stats_record_submit_backpressure(LoaderStatsAccumulator *stats,
                                                uint64_t count = 1) {
  if (stats != nullptr) stats->record_submit_backpressure(count);
}

inline void io_stats_record_nvme_error(LoaderStatsAccumulator *stats,
                                       uint64_t count = 1) {
  if (stats != nullptr) stats->record_nvme_error(count);
}

inline void io_stats_record_hot_sample(LoaderStatsAccumulator *stats,
                                       uint64_t submitted_pages,
                                       double sample_seconds) {
  if (stats != nullptr) stats->record_hot_sample(submitted_pages, sample_seconds);
}

}  // namespace shared
