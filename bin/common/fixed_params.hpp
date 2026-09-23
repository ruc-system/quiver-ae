#pragma once

namespace bin_common {

inline bool has_fixed_value(int fixed_value) { return fixed_value > 0; }

inline int resolve_fixed_value(int cli_value, int fixed_value) {
  return has_fixed_value(fixed_value) ? fixed_value : cli_value;
}

}  // namespace bin_common
