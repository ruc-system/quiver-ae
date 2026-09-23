#include "shared/io/spdk_lba_offset.hpp"

#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "spdk_lba_offset_test failed: " << message << std::endl;
    std::exit(1);
  }
}

}  // namespace

int main() {
  unsetenv("SPDK_BASE_LBA");
  require(shared::spdk_base_lba_from_env() == 0,
          "SPDK_BASE_LBA should default to zero");

  setenv("SPDK_BASE_LBA", "671088640", 1);
  require(shared::spdk_base_lba_from_env() == 671088640,
          "SPDK_BASE_LBA should parse the 2.5TiB 4KB-block offset");

  setenv("SPDK_BASE_LBA", "", 1);
  require(shared::spdk_base_lba_from_env() == 0,
          "empty SPDK_BASE_LBA should behave like unset");

  setenv("SPDK_BASE_LBA", "abc", 1);
  require(shared::spdk_base_lba_from_env() == 0,
          "invalid SPDK_BASE_LBA should fall back to zero");

  unsetenv("SPDK_BASE_LBA");
  unsetenv("QUIVER_SPDK_BLOCK_OFFSET");
  require(shared::spdk_block_offset_from_env() == 0,
          "QUIVER_SPDK_BLOCK_OFFSET should default to zero");

  setenv("QUIVER_SPDK_BLOCK_OFFSET", "1", 1);
  require(shared::spdk_block_offset_from_env() == 1,
          "QUIVER_SPDK_BLOCK_OFFSET should parse a one-page offset");

  setenv("QUIVER_SPDK_BLOCK_OFFSET", "", 1);
  require(shared::spdk_block_offset_from_env() == 0,
          "empty QUIVER_SPDK_BLOCK_OFFSET should behave like unset");

  setenv("QUIVER_SPDK_BLOCK_OFFSET", "-1", 1);
  require(shared::spdk_block_offset_from_env() == 0,
          "negative QUIVER_SPDK_BLOCK_OFFSET should fall back to zero");

  unsetenv("QUIVER_SPDK_BLOCK_OFFSET");
  return 0;
}
