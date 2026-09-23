#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace shared {

struct NavGraph {
  std::vector<int> mapping;
  uint8_t *data = nullptr;
  uint8_t *data_dev = nullptr;
  int *graph = nullptr;
  int *graph_dev = nullptr;
  int *mapping_dev = nullptr;
  int data_len = 0;
  int num_node = 0;
  int start = 0;
  int max_m = 0;

  void init(std::string index_file, std::string data_file, std::string map_file,
            int data_size);

  void translate(int *entry, int qcnt);

  ~NavGraph();
};

}  // namespace shared
