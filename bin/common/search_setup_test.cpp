#include "common/search_setup.hpp"
#include "common/fixed_params.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "search_setup_test failed: " << message << std::endl;
    std::exit(1);
  }
}

std::filesystem::path write_index(const std::filesystem::path& path,
                                  uint8_t starter_value) {
  std::ofstream out(path, std::ios::binary);
  std::vector<uint8_t> page(shared::PAGE_SIZE, 0);
  out.write(reinterpret_cast<const char*>(page.data()), page.size());
  std::fill(page.begin(), page.end(), starter_value);
  out.write(reinterpret_cast<const char*>(page.data()), page.size());
  require(out.good(), "failed to write fixture index");
  return path;
}

void expect_page_value(const uint8_t* page, uint8_t expected,
                       const char* label) {
  for (int i = 0; i < shared::PAGE_SIZE; ++i) {
    if (page[i] != expected) {
      std::cerr << label << " byte " << i << " expected "
                << static_cast<int>(expected) << " got "
                << static_cast<int>(page[i]) << std::endl;
      std::exit(1);
    }
  }
}

bin_common::SearchSetup make_setup(const std::string& disk_index,
                                   const std::string& memory_index) {
  bin_common::SearchSetup setup;
  setup.index_file = disk_index;
  setup.memory_index_file = memory_index;
  setup.layout.enter_point = 0;
  setup.layout.nodes_per_page = 1;
  return setup;
}

}  // namespace

int main() {
  require(bin_common::resolve_fixed_value(7, -1) == 7,
          "unset fixed value should keep CLI value");
  require(bin_common::resolve_fixed_value(7, 2) == 2,
          "set fixed value should override CLI value");
  require(bin_common::has_fixed_value(2),
          "positive fixed value should be detected");
  require(!bin_common::has_fixed_value(-1),
          "negative fixed value should mean unset");

  const int results[] = {
      1, 2, 3,
      8, 9, 10,
  };
  const int gt[] = {
      3, 2, 1, 4,
      7, 8, 11, 12,
  };
  auto per_query = bin_common::calc_per_query_recall(results, gt, 2, 4, 3);
  require(per_query.size() == 2, "per-query recall should return one row per query");
  require(per_query[0].hits == 3, "query 0 should have three hits");
  require(per_query[0].recall == 1.0, "query 0 should have recall 1.0");
  require(per_query[1].hits == 1, "query 1 should have one hit");
  require(per_query[1].recall > 0.333 && per_query[1].recall < 0.334,
          "query 1 should have recall 1/3");

  const auto dir = std::filesystem::temp_directory_path() /
                   ("quiver_search_setup_test_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);

  const auto disk_index = write_index(dir / "disk.index", 0x11);
  const auto memory_index = write_index(dir / "memory.index", 0x22);
  const auto symlink_dir = dir / "symlink_index";
  std::filesystem::create_directories(symlink_dir);
  const auto symlink_target = write_index(symlink_dir / "ann.index", 0x33);
  const auto symlink_index = symlink_dir / "deep1b_disk.index";
  std::filesystem::create_symlink(symlink_target, symlink_index);
  require(bin_common::find_index_file(symlink_dir.string()) ==
              symlink_index.string(),
          "index discovery should accept symlinked *_disk.index files");

  auto memory_setup = make_setup(disk_index.string(), memory_index.string());
  auto memory_starter = bin_common::load_starter_page(memory_setup);
  expect_page_value(memory_starter.get(), 0x22,
                    "memory backend starter page");

  auto spdk_setup = make_setup(disk_index.string(), memory_index.string());
  spdk_setup.ssd_lists.push_back("0000:00:00.0");
  auto spdk_starter = bin_common::load_starter_page(spdk_setup);
  expect_page_value(spdk_starter.get(), 0x11, "SPDK starter page");

  std::filesystem::remove_all(dir);
  return 0;
}
