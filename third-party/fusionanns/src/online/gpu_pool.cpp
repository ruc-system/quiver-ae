#include "gpu_pool.h"

#include <iostream>
#include <stdexcept>
#include <thread>

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    cudaError_t e = (err);                                                     \
    if (e != cudaSuccess) {                                                    \
      throw std::runtime_error(cudaGetErrorString(e));                         \
    }                                                                          \
  }

GpuBlockPool::~GpuBlockPool() {
  if (pool_base_) {
    cudaFree(pool_base_);
    pool_base_ = nullptr;
  }
  blocks_.clear();
  free_blocks_.reset();
  pool_bytes_ = 0;
  block_bytes_ = 0;
  initialized_ = false;
}

bool GpuBlockPool::init(size_t bytes_per_block, float reserve_ratio) {
  if (initialized_) {
    return true;
  }
  if (reserve_ratio <= 0.f || reserve_ratio > 0.95f) {
    throw std::invalid_argument("reserve_ratio should be within (0,0.95]");
  }
  if (bytes_per_block == 0) {
    throw std::invalid_argument("bytes_per_block must be positive");
  }

  block_bytes_ = align_up_h(bytes_per_block, 256);

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  pool_bytes_ = static_cast<size_t>(reserve_ratio * free_bytes);
  if (pool_bytes_ < block_bytes_) {
    throw std::runtime_error("Insufficient free GPU memory for one block");
  }
  size_t block_count = pool_bytes_ / block_bytes_;
  if (block_count == 0) {
    throw std::runtime_error("Computed block count is zero");
  }

  free_blocks_ =
      std::make_unique<boost::lockfree::queue<GpuBlock *>>(block_count);

  std::cout << "  > Allocating GPU memory for block pool..." << std::endl
            << "    - Block size: " << (block_bytes_ / (1024.0 * 1024))
            << " MiB" << std::endl
            << "    - Block count: " << block_count << std::endl
            << "    - Total pool size: "
            << (block_count * block_bytes_ / (1024.0 * 1024 * 1024)) << " GiB"
            << std::endl;

  CUDA_CHECK(cudaMalloc(&pool_base_, block_count * block_bytes_));
  pool_bytes_ = block_count * block_bytes_;

  std::cout << "  > GPU memory allocated successfully." << std::endl;

  blocks_.reserve(block_count);
  char *cursor = static_cast<char *>(pool_base_);
  for (size_t i = 0; i < block_count; ++i) {
    auto block = std::make_unique<GpuBlock>();
    block->base_ptr = cursor;
    block->total_bytes = block_bytes_;
    block->reset();
    free_blocks_->push(block.get());
    blocks_.push_back(std::move(block));
    cursor += block_bytes_;
  }

  initialized_ = true;
  return true;
}

GpuBlock *GpuBlockPool::acquire_block() {
  GpuBlock *block = nullptr;
  while (!free_blocks_->pop(block)) {
    std::this_thread::yield();
  }
  return block;
}

void GpuBlockPool::release_block(GpuBlock *block) {
  if (!block)
    return;

  block->reset();
  while (!free_blocks_->push(block)) {
    std::this_thread::yield();
  }
}
