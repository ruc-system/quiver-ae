#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/lockfree/queue.hpp>

inline size_t align_up_h(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return value + (alignment - remainder);
}

struct GpuBlock {
  void *base_ptr = nullptr;
  size_t total_bytes = 0;
  size_t current_offset = 0;

  void reset() { current_offset = 0; }

  template <typename T>
  T *allocate(size_t count, size_t alignment = 256) {
    if (!base_ptr) {
      throw std::runtime_error("GpuBlock base pointer is null");
    }
    if (total_bytes == 0) {
      throw std::runtime_error("GpuBlock total_bytes is zero");
    }
    if (alignment == 0) {
      alignment = 1;
    }
    size_t bytes_needed = count * sizeof(T);
    size_t aligned_offset = align_up_h(current_offset, alignment);
    if (aligned_offset + bytes_needed > total_bytes) {
      throw std::runtime_error(
          "GpuBlock OOM: allocation exceeds block capacity");
    }
    current_offset = aligned_offset + bytes_needed;
    return reinterpret_cast<T *>(static_cast<char *>(base_ptr) +
                                 aligned_offset);
  }

  size_t remaining_bytes() const { return total_bytes - current_offset; }
};

class GpuBlockPool {
public:
  GpuBlockPool() = default;
  ~GpuBlockPool();
  GpuBlockPool(const GpuBlockPool &) = delete;
  GpuBlockPool &operator=(const GpuBlockPool &) = delete;

  bool init(size_t bytes_per_block, float reserve_ratio = 0.8f);
  GpuBlock *acquire_block();
  void release_block(GpuBlock *block);
  int capacity() const { return static_cast<int>(blocks_.size()); }
  size_t block_bytes() const { return block_bytes_; }

private:
  bool initialized_ = false;
  void *pool_base_ = nullptr;
  size_t pool_bytes_ = 0;
  size_t block_bytes_ = 0;

  std::vector<std::unique_ptr<GpuBlock>> blocks_;
  std::unique_ptr<boost::lockfree::queue<GpuBlock *>> free_blocks_;
};
