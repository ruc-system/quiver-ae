#pragma once

namespace shared {

struct SpdkLoaderOptions {
  int submit_queue_capacity;
  int task_context_count;
  int runner_io_context_count;
};

inline SpdkLoaderOptions default_spdk_loader_options() {
  return {
      4096,  // Per-producer SPSC inbox capacity for each SSD runner.
      1024,  // Compatibility slots for the legacy submit_task/poll_task path.
      1024,  // Per-runner IO callback contexts; one context tracks one IO.
  };
}

}  // namespace shared
