#pragma once

#include "../common/types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// IO 管理器采用策略模式，允许在不同实现之间切换。
// Strategy-based IO manager that lets us swap storage backends.
class IOManager {
public:
  struct IOGlobal {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t io_count = 0;
    uint64_t evictions = 0;
  };

  enum class BackendKind { Pread, MMap, IoUring };
  enum class VectorStorage { Float32, Uint8 };

  static constexpr size_t kDefaultCachePages = 1048576; // 4GB for 4KB pages

  IOManager(const std::string &map_file, const std::string &packed_vectors_file,
            int dim, BackendKind kind = BackendKind::Pread,
            size_t cache_pages = kDefaultCachePages,
            VectorStorage storage = VectorStorage::Float32);
  ~IOManager();

  IOManager(const IOManager &) = delete;
  IOManager &operator=(const IOManager &) = delete;

  void get_vectors(const std::vector<long> &ids, std::vector<float> &out_data,
                   uint32_t *out_unique_pages = nullptr);
  IOGlobal snapshot() const;
  size_t vector_count() const { return location_map_.size(); }

public:
  class Backend;

private:
  int dim_ = 128;
  std::vector<VectorLocation> location_map_;
  std::unique_ptr<Backend> backend_;
};
