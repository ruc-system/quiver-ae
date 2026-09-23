#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "memory_backend.hpp"
#include "runtime_stats.hpp"

namespace shared {

using IoRequest = std::pair<int, void *>;  // (blockId, dest_buffer)

struct TracedIoRequest {
  int block_id;
  void *dest;
  int trace_index;
  int64_t *enqueue_ns;
  int64_t *submit_ns;
  int64_t *complete_ns;
};

struct DirectIoRequest {
    int block_id;
    void* dest;
    volatile int32_t* status_ptr;
    int32_t done_value;
};

inline constexpr int64_t PAGE_SIZE = 4096;

class IndexLoader {
 public:
  virtual void submit_task(const std::vector<IoRequest> &requests, int thread_id,
                           int ctx_id) = 0;
  virtual void submit_traced_task(const std::vector<TracedIoRequest> &requests,
                                  int thread_id, int ctx_id) {
    std::vector<IoRequest> untraced;
    untraced.reserve(requests.size());
    for (const auto &request : requests) {
      untraced.emplace_back(request.block_id, request.dest);
    }
    submit_task(untraced, thread_id, ctx_id);
  }
  virtual bool poll_task(int ctx_id) = 0;
  virtual void submit_direct(const std::vector<DirectIoRequest>& reqs, int tid) {
      (void)reqs; (void)tid;
      throw std::runtime_error("submit_direct not implemented");
  }
  virtual LoaderStatsSnapshot snapshot_stats() const { return {}; }
  virtual void clear_stats() {}
  virtual uint8_t *create_buffer(int64_t size) = 0;
  virtual void destroy_buffer(uint8_t *) = 0;
  virtual ~IndexLoader() {}
};

std::shared_ptr<IndexLoader> create_mem_loader_sync(
    const char *filename, int64_t num_pages,
    MemoryBackend backend = MemoryBackend::kHeap);

// Note: runners_per_ssd is accepted for source compatibility with the
// quiver-side multi-runner refactor, but in this baseline implementation it
// is ignored — behavior is unchanged from the single-runner-per-SSD model.
std::shared_ptr<IndexLoader> create_spdk_loader(
    const std::vector<std::string> &ssds, int queue_cap, int thread_cnt,
    int ctx_cnt, int runners_per_ssd = 1);

}  // namespace shared
