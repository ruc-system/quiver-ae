#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>

namespace bin_common {

inline constexpr int kMaxPipeWidth = 16;
inline constexpr int kMaxQueriesPerBlock = 8;

inline bool validate_positive_arg(int value, const char* arg_name) {
  if (value > 0) {
    return true;
  }
  std::cerr << arg_name << " must be positive" << std::endl;
  return false;
}

inline bool validate_int_range_arg(int value, const char* arg_name,
                                   int min_value, int max_value) {
  if (value >= min_value && value <= max_value) {
    return true;
  }
  std::cerr << arg_name << " must be in [" << min_value << ", " << max_value
            << "]" << std::endl;
  return false;
}

inline void validate_positive_list_or_die(const std::vector<int>& values,
                                          const char* arg_name) {
  for (const int value : values) {
    if (value <= 0) {
      std::cerr << arg_name << " values must be positive: " << value
                << std::endl;
      std::exit(1);
    }
  }
}

}  // namespace bin_common
