#include "io_stats.hpp"
#include "loader.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../common/logging.hpp"

namespace shared {

class MemLoaderSync : public IndexLoader {
  MemoryBackend backend_;
  uint8_t *index = nullptr;
  int fd_ = -1;
  size_t map_len_ = 0;
  uint8_t *map_base_ = nullptr;
  LoaderStatsAccumulator stats_{1};

  LoaderStatsAccumulator *stats_sink() { return &stats_; }

  void load_heap(const char *filename, int64_t num_pages) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
      ERROR("Failed to open Mem Index: {}", filename);
      exit(-1);
    }
    index = new uint8_t[num_pages * PAGE_SIZE];
    fseek(file, PAGE_SIZE, SEEK_SET);
    int64_t ret = fread(index, PAGE_SIZE, num_pages, file);
    if (ret != num_pages) {
      ERROR("Mem Index Load FAILED: {} pages read from {}", ret, filename);
      fclose(file);
      exit(-1);
    }
    fclose(file);
    INFO("Mem Index Loaded with heap backend: {}", filename);
  }

  void load_mmap(const char *filename, int64_t num_pages) {
    fd_ = open(filename, O_RDONLY);
    if (fd_ < 0) {
      ERROR("Failed to open mmap Mem Index {}: {}", filename,
            std::strerror(errno));
      exit(-1);
    }

    struct stat st {};
    if (fstat(fd_, &st) != 0) {
      ERROR("Failed to stat mmap Mem Index {}: {}", filename,
            std::strerror(errno));
      close(fd_);
      fd_ = -1;
      exit(-1);
    }

    const int64_t required = (num_pages + 1) * PAGE_SIZE;
    if (st.st_size < required) {
      ERROR("mmap Mem Index {} is too small: size={} required={}", filename,
            static_cast<long long>(st.st_size),
            static_cast<long long>(required));
      close(fd_);
      fd_ = -1;
      exit(-1);
    }

    map_len_ = static_cast<size_t>(st.st_size);
    void *mapped = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
      ERROR("Failed to mmap Mem Index {}: {}", filename, std::strerror(errno));
      close(fd_);
      fd_ = -1;
      map_len_ = 0;
      exit(-1);
    }
    map_base_ = static_cast<uint8_t *>(mapped);
    index = map_base_ + PAGE_SIZE;
    INFO("Mem Index Loaded with mmap backend: {}", filename);
  }

 public:
  MemLoaderSync(const char *filename, int64_t num_pages,
                MemoryBackend backend = MemoryBackend::kHeap)
      : backend_(backend) {
    if (backend_ == MemoryBackend::kHeap) {
      load_heap(filename, num_pages);
    } else {
      load_mmap(filename, num_pages);
    }
  }

  void submit_task(const std::vector<IoRequest> &pages, int, int) override {
    for (auto [blk, dst] : pages) {
      memcpy(dst, index + (int64_t)blk * PAGE_SIZE, PAGE_SIZE);
    }
  }

  void submit_direct(const std::vector<DirectIoRequest>& reqs, int) override {
    for (const auto& req : reqs) {
      memcpy(req.dest, index + (int64_t)req.block_id * PAGE_SIZE, PAGE_SIZE);
      if (req.complete_ns != nullptr) {
        const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
        __atomic_store_n(req.complete_ns, now_ns, __ATOMIC_RELEASE);
      }
      if (req.status_ptr != nullptr) {
        __atomic_store_n(req.status_ptr, req.done_value, __ATOMIC_RELEASE);
      }
    }
  }

  LoaderStatsSnapshot snapshot_stats() const override {
    return stats_.snapshot();
  }

  void clear_stats() override {
    stats_.clear();
  }

  uint8_t *create_buffer(int64_t size) override { return new uint8_t[size]; }

  void destroy_buffer(uint8_t *buf) override { delete[] buf; }

  bool poll_task(int) override { return true; }

  ~MemLoaderSync() override {
    if (backend_ == MemoryBackend::kMmap) {
      if (map_base_ != nullptr && map_len_ != 0) {
        munmap(map_base_, map_len_);
      }
      if (fd_ >= 0) {
        close(fd_);
      }
    } else {
      delete[] index;
    }
  }
};

std::shared_ptr<IndexLoader> create_mem_loader_sync(
    const char *filename, int64_t num_pages, MemoryBackend backend) {
  return std::make_shared<MemLoaderSync>(filename, num_pages, backend);
}

}  // namespace shared
