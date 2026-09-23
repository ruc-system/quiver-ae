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

#ifdef FLASHANNS_LATENCY_PROBE
struct TracedIoRequest {
    int block_id;
    void* dest;
    int64_t submit_ns;
    volatile int64_t* complete_ns;
};
#endif

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
#ifdef FLASHANNS_LATENCY_PROBE
  virtual void submit_traced_task(
      const std::vector<TracedIoRequest>& requests, int thread_id, int ctx_id) {
      std::vector<IoRequest> untraced;
      untraced.reserve(requests.size());
      for (const auto& request : requests) {
          untraced.emplace_back(request.block_id, request.dest);
      }
      submit_task(untraced, thread_id, ctx_id);
  }
#endif
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

// FlashANNS baseline uses one SPDK runner per SSD. submit_queue_capacity is
// the per-producer SPSC inbox depth; task_context_count tracks batches in
// submit_task/poll_task; runner_io_context_count is the per-runner IO ctx pool.
std::shared_ptr<IndexLoader> create_spdk_loader(
    const std::vector<std::string> &ssds, int submit_queue_capacity,
    int thread_cnt, int task_context_count, int runner_io_context_count);

}  // namespace shared
