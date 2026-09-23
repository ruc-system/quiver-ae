#include "shared/io/spdk_write_buffer.hpp"

#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "spdk_write_buffer_test failed: " << message << std::endl;
    std::exit(1);
  }
}

}  // namespace

int main() {
  unsetenv("SPDK_WRITE_READ_BUFFER_MB");
  require(shared::spdk_write_read_buffer_bytes_from_env() == 64ULL * 1024 * 1024,
          "default buffer should be 64MiB");

  setenv("SPDK_WRITE_READ_BUFFER_MB", "128", 1);
  require(shared::spdk_write_read_buffer_bytes_from_env() ==
              128ULL * 1024 * 1024,
          "env buffer should be parsed as MiB");

  setenv("SPDK_WRITE_READ_BUFFER_MB", "0", 1);
  require(shared::spdk_write_read_buffer_bytes_from_env() == 64ULL * 1024 * 1024,
          "zero buffer should fall back to default");

  setenv("SPDK_WRITE_READ_BUFFER_MB", "abc", 1);
  require(shared::spdk_write_read_buffer_bytes_from_env() == 64ULL * 1024 * 1024,
          "invalid buffer should fall back to default");

  unsetenv("SPDK_WRITE_READ_BUFFER_MB");
  return 0;
}
