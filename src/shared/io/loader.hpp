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

struct DirectIoRequest {
    int block_id;
    void* dest;
    volatile int32_t* status_ptr;
    int32_t done_value;
    volatile int64_t* complete_ns = nullptr;
};

inline constexpr int64_t PAGE_SIZE = 4096;

class IndexLoader {
 public:
  virtual void submit_task(const std::vector<IoRequest> &requests, int thread_id,
                           int ctx_id) = 0;
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

// submit_queue_capacity: per-producer SPSC inbox depth for each SSD runner.
// task_context_count: compatibility slots for the legacy submit_task/poll_task
// path. runner_io_context_count: per-runner IO ctx pool size; one ctx tracks
// one SSD IO. runners_per_ssd: independent SPDK runner threads per SSD.
std::shared_ptr<IndexLoader> create_spdk_loader(
    const std::vector<std::string> &ssds, int submit_queue_capacity,
    int thread_cnt, int task_context_count, int runner_io_context_count,
    int runners_per_ssd = 1);

}  // namespace shared
