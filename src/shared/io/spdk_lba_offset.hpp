#pragma once

#include <cerrno>
#include <cstdlib>

namespace shared {

inline long spdk_base_lba_from_env() {
  const char* env = std::getenv("SPDK_BASE_LBA");
  if (env == nullptr || *env == '\0') {
    return 0;
  }

  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  if (errno != 0 || end == env || *end != '\0' || value < 0) {
    return 0;
  }
  return value;
}

inline long spdk_block_offset_from_env() {
  const char* env = std::getenv("QUIVER_SPDK_BLOCK_OFFSET");
  if (env == nullptr || *env == '\0') {
    return 0;
  }

  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  if (errno != 0 || end == env || *end != '\0' || value < 0) {
    return 0;
  }
  return value;
}

}  // namespace shared
