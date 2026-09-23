#include "shared/io/loader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "flashanns_mem_loader_test failed: " << message << std::endl;
    std::exit(1);
  }
}

std::filesystem::path make_test_index() {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("flashanns_mem_loader_test_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  const auto path = dir / "disk.index";
  std::ofstream out(path, std::ios::binary);
  std::vector<uint8_t> page(shared::PAGE_SIZE, 0);
  out.write(reinterpret_cast<const char*>(page.data()), page.size());
  for (int page_id = 0; page_id < 3; ++page_id) {
    std::fill(page.begin(), page.end(), static_cast<uint8_t>(0x70 + page_id));
    out.write(reinterpret_cast<const char*>(page.data()), page.size());
  }
  require(out.good(), "failed to write fixture index");
  return path;
}

void expect_page(uint8_t* page, uint8_t expected, const char* label) {
  for (int i = 0; i < shared::PAGE_SIZE; ++i) {
    if (page[i] != expected) {
      std::cerr << label << " byte " << i << " expected "
                << static_cast<int>(expected) << " got "
                << static_cast<int>(page[i]) << std::endl;
      std::exit(1);
    }
  }
}

}  // namespace

int main() {
  const auto path = make_test_index();
  auto heap_loader = shared::create_mem_loader_sync(path.c_str(), 3);
  auto mmap_loader = shared::create_mem_loader_sync(
      path.c_str(), 3, shared::MemoryBackend::kMmap);

  std::vector<uint8_t> heap_page(shared::PAGE_SIZE);
  std::vector<uint8_t> mmap_page(shared::PAGE_SIZE);
  heap_loader->submit_task({{0, heap_page.data()}}, 0, 0);
  mmap_loader->submit_task({{0, mmap_page.data()}}, 0, 0);
  require(std::memcmp(heap_page.data(), mmap_page.data(), shared::PAGE_SIZE) ==
              0,
          "heap and mmap submit_task should match");

  std::vector<uint8_t> heap_direct_page(shared::PAGE_SIZE);
  volatile int32_t heap_status = 0;
  shared::DirectIoRequest heap_direct{2, heap_direct_page.data(), &heap_status,
                                      7};
  heap_loader->submit_direct({heap_direct}, 0);
  require(__atomic_load_n(&heap_status, __ATOMIC_ACQUIRE) == 7,
          "heap submit_direct should set completion status");
  expect_page(heap_direct_page.data(), 0x72, "heap submit_direct page 2");

  std::vector<uint8_t> mmap_direct_page(shared::PAGE_SIZE);
  volatile int32_t mmap_status = 0;
  shared::DirectIoRequest mmap_direct{2, mmap_direct_page.data(), &mmap_status,
                                      9};
  mmap_loader->submit_direct({mmap_direct}, 0);
  require(__atomic_load_n(&mmap_status, __ATOMIC_ACQUIRE) == 9,
          "mmap submit_direct should set completion status");
  expect_page(mmap_direct_page.data(), 0x72, "mmap submit_direct page 2");

  std::filesystem::remove_all(path.parent_path());
  return 0;
}
