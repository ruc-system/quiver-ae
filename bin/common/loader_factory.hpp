#pragma once

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "shared/common/logging.hpp"
#include "shared/io/loader.hpp"
#include "shared/io/spdk_loader_config.hpp"

namespace bin_common {

// Entry-layer helper only: this translates CLI/backend choice into calls to the
// shared IO constructors during the Task 6 -> Task 7 transition. It is not the
// canonical backend-policy layer for GustANN, FlashANNS, or Quiver internals,
// and system code should not treat it as a shared strategy abstraction.
//
// SPDK queue/task/runner context capacities are host-side resources. They use
// fixed defaults rather than being derived from GPU batch or lane counts.
// runners_per_ssd: number of independent SPDK runner threads per SSD. Quiver
// fans them out across NUMA-0 even cores; flashanns / gustann silently ignore
// this argument (single runner) for baseline parity.
inline std::shared_ptr<shared::IndexLoader> create_index_loader(
    const std::string& index_file, int64_t num_pages,
    const std::vector<std::string>& ssd_lists, int batch, int thread,
    int pipe_w, int queries_per_block, int runners_per_ssd) {
  if (!ssd_lists.empty()) {
#if QUIVER_ENABLE_SPDK
    const auto spdk_options = shared::default_spdk_loader_options();
    INFO("Using SPDK backend (submit_queue_capacity={} task_context_count={} "
         "runner_io_context_count={} Q={} runners_per_ssd={})",
         spdk_options.submit_queue_capacity, spdk_options.task_context_count,
         spdk_options.runner_io_context_count, queries_per_block,
         runners_per_ssd);
    return shared::create_spdk_loader(
        ssd_lists, spdk_options.submit_queue_capacity, thread,
        spdk_options.task_context_count, spdk_options.runner_io_context_count,
        runners_per_ssd);
#else
    ERROR("This binary was built without SPDK support");
    std::exit(1);
#endif
  }

  INFO("Using memory backend");
  return shared::create_mem_loader_sync(index_file.c_str(), num_pages);
}

inline std::shared_ptr<shared::IndexLoader> create_index_loader(
    const std::string& index_file, int64_t num_pages,
    const std::vector<std::string>& ssd_lists, int batch, int thread,
    int pipe_w, int queries_per_block) {
  return create_index_loader(index_file, num_pages, ssd_lists, batch, thread,
                             pipe_w, queries_per_block, 1);
}

inline std::shared_ptr<shared::IndexLoader> create_index_loader(
    const std::string& index_file, int64_t num_pages,
    const std::vector<std::string>& ssd_lists, int batch, int thread,
    int pipe_w) {
  return create_index_loader(index_file, num_pages, ssd_lists, batch, thread,
                             pipe_w, 1, 1);
}

}  // namespace bin_common
