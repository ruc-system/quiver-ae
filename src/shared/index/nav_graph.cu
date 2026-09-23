#include "nav_graph.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>

#include "../common/cuda_utils.cuh"
#include "../common/logging.hpp"

namespace shared {

void NavGraph::init(std::string index_file, std::string data_file,
                    std::string map_file, int data_size) {
  FILE *f = fopen(data_file.c_str(), "r");
  if (!f) {
    ERROR("Failed to open data file: {}", data_file);
    exit(-1);
  }
  fread(&num_node, sizeof(int), 1, f);
  fread(&data_len, sizeof(int), 1, f);
  data = new uint8_t[1ll * num_node * data_len * data_size];

  fread(data, 1, 1ll * num_node * data_len * data_size, f);

  fclose(f);

  std::ifstream map_input(map_file);
  if (!map_input.is_open()) {
    ERROR("Failed to open map file: {}", map_file);
    exit(-1);
  }

  mapping.clear();
  mapping.reserve(num_node);
  for (int i = 0; i < num_node; i++) {
    int x = -1;
    if (!(map_input >> x)) {
      ERROR("Failed to read map entry {} from {}", i, map_file);
      exit(-1);
    }
    mapping.push_back(x);
  }

  f = fopen(index_file.c_str(), "r");
  if (!f) {
    ERROR("Failed to open index file: {}", index_file);
    exit(-1);
  }
  int dummy;
  fread(&dummy, sizeof(int), 1, f);
  fread(&dummy, sizeof(int), 1, f);
  fread(&max_m, sizeof(int), 1, f);
  fread(&start, sizeof(int), 1, f);
  fread(&dummy, sizeof(int), 1, f);
  fread(&dummy, sizeof(int), 1, f);
  graph = new int[1ll * num_node * max_m];

  assert(0 < max_m && max_m <= 32);
  for (int i = 0; i < num_node; i++) {
    int d;
    fread(&d, 1, sizeof(int), f);
    assert(0 < d && d <= max_m);
    fread(graph + max_m * i, sizeof(int), d, f);
    for (int j = d; j < max_m; j++) {
      graph[max_m * i + j] = -1;
    }
  }

  fclose(f);
  copy_to_dev(data, data_dev, 1l * num_node * data_len * data_size);
  copy_to_dev(graph, graph_dev, 1l * max_m * num_node);
  copy_to_dev(mapping.data(), mapping_dev, num_node);
}

void NavGraph::translate(int *entry, int qcnt) {
  for (int i = 0; i < qcnt; i++) {
    entry[i] = mapping[entry[i]];
  }
}

NavGraph::~NavGraph() {
  delete[] data;
  delete[] graph;
  if (data_dev) {
    cudaFree(data_dev);
  }
  if (graph_dev) {
    cudaFree(graph_dev);
  }
  if (mapping_dev) {
    cudaFree(mapping_dev);
  }
}

}  // namespace shared
