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
    std::cerr << "mem_loader_test failed: " << message << std::endl;
    std::exit(1);
  }
}

std::filesystem::path make_test_index() {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("quiver_mem_loader_test_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  const auto path = dir / "disk.index";
  std::ofstream out(path, std::ios::binary);
  std::vector<uint8_t> page(shared::PAGE_SIZE, 0);
  out.write(reinterpret_cast<const char*>(page.data()), page.size());
  for (int page_id = 0; page_id < 3; ++page_id) {
    std::fill(page.begin(), page.end(), static_cast<uint8_t>(0x40 + page_id));
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

void exercise_loader(shared::MemoryBackend backend, const std::string& path) {
  auto loader = shared::create_mem_loader_sync(path.c_str(), 3, backend);

  std::vector<uint8_t> task_page(shared::PAGE_SIZE);
  loader->submit_task({{1, task_page.data()}}, 0, 0);
  require(loader->poll_task(0), "poll_task should complete synchronously");
  expect_page(task_page.data(), 0x41, "submit_task page 1");

  std::vector<uint8_t> direct_page(shared::PAGE_SIZE);
  volatile int32_t status = 0;
  shared::DirectIoRequest direct{2, direct_page.data(), &status, 7};
  loader->submit_direct({direct}, 0);
  require(__atomic_load_n(&status, __ATOMIC_ACQUIRE) == 7,
          "submit_direct should set completion status");
  expect_page(direct_page.data(), 0x42, "submit_direct page 2");

  uint8_t* staging = loader->create_buffer(shared::PAGE_SIZE);
  require(staging != nullptr, "create_buffer should allocate staging memory");
  std::memset(staging, 0x5a, shared::PAGE_SIZE);
  expect_page(staging, 0x5a, "staging buffer");
  loader->destroy_buffer(staging);
}

}  // namespace

int main() {
  const auto path = make_test_index();
  exercise_loader(shared::MemoryBackend::kHeap, path.string());
  exercise_loader(shared::MemoryBackend::kMmap, path.string());
  std::filesystem::remove_all(path.parent_path());
  return 0;
}
