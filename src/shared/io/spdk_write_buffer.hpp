#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>

namespace shared {

inline std::size_t spdk_write_read_buffer_bytes_from_env(
    const char* env_name = "SPDK_WRITE_READ_BUFFER_MB",
    std::size_t default_mib = 64, std::size_t page_size = 4096) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_mib * 1024 * 1024;
  }

  char* end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || *end != '\0' || value == 0 ||
      value > std::numeric_limits<std::size_t>::max() / (1024ULL * 1024ULL)) {
    return default_mib * 1024 * 1024;
  }

  std::size_t bytes = static_cast<std::size_t>(value) * 1024 * 1024;
  if (page_size == 0) {
    return bytes;
  }

  const std::size_t remainder = bytes % page_size;
  if (remainder == 0) {
    return bytes;
  }
  return bytes + (page_size - remainder);
}

}  // namespace shared
