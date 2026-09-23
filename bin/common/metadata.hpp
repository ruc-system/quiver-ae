#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>

#include "shared/common/data_type.hpp"
#include "shared/common/layout.hpp"
#include "shared/common/logging.hpp"

namespace bin_common {

inline void parse_diskann_metadata(const std::string& fpath,
                                   shared::Layout& layout,
                                   shared::DataType data_type) {
  const size_t data_size =
      data_type == shared::FLOAT ? sizeof(float) : sizeof(int8_t);

  std::ifstream input(fpath, std::ios::binary);
  if (!input.is_open()) {
    ERROR("Failed to open file {}", fpath);
    std::exit(-1);
  }
  INFO("load DiskANN index from {}", fpath);

  uint32_t nr = 0;
  uint32_t nc = 0;
  input.read(reinterpret_cast<char*>(&nr), sizeof(uint32_t));
  input.read(reinterpret_cast<char*>(&nc), sizeof(uint32_t));
  (void)nr;
  (void)nc;

  uint64_t disk_nnodes = 0;
  uint64_t disk_ndims = 0;
  input.read(reinterpret_cast<char*>(&disk_nnodes), sizeof(uint64_t));
  input.read(reinterpret_cast<char*>(&disk_ndims), sizeof(uint64_t));
  layout.num_data = disk_nnodes;
  layout.num_dims = disk_ndims;

  uint64_t medoid_id = 0;
  uint64_t max_node_len = 0;
  uint64_t nnodes_per_sector = 0;
  input.read(reinterpret_cast<char*>(&medoid_id), sizeof(uint64_t));
  input.read(reinterpret_cast<char*>(&max_node_len), sizeof(uint64_t));
  input.read(reinterpret_cast<char*>(&nnodes_per_sector), sizeof(uint64_t));
  layout.enter_point = medoid_id;
  layout.max_m0 =
      ((max_node_len - disk_ndims * data_size) / sizeof(uint32_t)) - 1;

  uint64_t num_frozen = 0;
  uint64_t file_frozen_id = 0;
  uint64_t reorder_exists = 0;
  input.read(reinterpret_cast<char*>(&num_frozen), sizeof(uint64_t));
  input.read(reinterpret_cast<char*>(&file_frozen_id), sizeof(uint64_t));
  input.read(reinterpret_cast<char*>(&reorder_exists), sizeof(uint64_t));
  (void)num_frozen;
  (void)file_frozen_id;
  (void)reorder_exists;

  layout.nodes_per_page = nnodes_per_sector;
  layout.num_pages =
      (layout.num_data + layout.nodes_per_page - 1) / layout.nodes_per_page;
  layout.node_size = max_node_len;
  layout.data_size = disk_ndims * data_size;

  INFO("num_data={}, num_dims={}, max_m0={}, enter_point={}, node_size={}, "
       "nodes_per_page={}",
       layout.num_data, layout.num_dims, layout.max_m0, layout.enter_point,
       layout.node_size, layout.nodes_per_page);
}

}  // namespace bin_common
