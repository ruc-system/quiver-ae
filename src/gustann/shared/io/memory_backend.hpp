#pragma once

#include <optional>
#include <string>

namespace shared {

enum class MemoryBackend {
  kHeap,
  kMmap,
};

inline std::optional<MemoryBackend> parse_memory_backend(
    const std::string& value) {
  if (value == "heap") {
    return MemoryBackend::kHeap;
  }
  if (value == "mmap") {
    return MemoryBackend::kMmap;
  }
  return std::nullopt;
}

inline const char* memory_backend_name(MemoryBackend backend) {
  switch (backend) {
    case MemoryBackend::kHeap:
      return "heap";
    case MemoryBackend::kMmap:
      return "mmap";
  }
  return "unknown";
}

}  // namespace shared
