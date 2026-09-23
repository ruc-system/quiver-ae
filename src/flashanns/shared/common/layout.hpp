#pragma once

#include <cstdint>

namespace shared {

struct Layout {
  int64_t num_dims;
  int64_t max_m0;
  int64_t enter_point;

  int64_t data_size;
  int64_t node_size;

  int64_t num_pages;
  int64_t num_data;

  int64_t nodes_per_page;
};

}  // namespace shared
