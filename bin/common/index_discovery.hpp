#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "shared/common/logging.hpp"

namespace bin_common {
namespace fs = std::filesystem;

namespace detail {

inline bool has_suffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

inline void fail_ambiguous_candidates(const char* label,
                                      const std::string& index_dir,
                                      const std::vector<std::string>& matches) {
  std::string joined;
  for (size_t i = 0; i < matches.size(); ++i) {
    if (i > 0) {
      joined += ", ";
    }
    joined += matches[i];
  }

  ERROR("Multiple {} candidates found in {}: {}", label, index_dir, joined);
  std::exit(1);
}

inline std::vector<std::string> list_suffix_matches(
    const std::string& index_dir, const std::string& suffix) {
  std::vector<std::string> matches;
  for (const auto& entry : fs::directory_iterator(index_dir)) {
    if (!fs::is_regular_file(entry.path())) {
      continue;
    }

    const auto name = entry.path().filename().string();
    if (has_suffix(name, suffix)) {
      matches.push_back(entry.path().string());
    }
  }

  std::sort(matches.begin(), matches.end());
  return matches;
}

}  // namespace detail

inline std::string find_index_file(const std::string& index_dir) {
  if (!fs::is_directory(index_dir)) {
    ERROR("--index-dir '{}' does not exist or is not a directory", index_dir);
    std::exit(1);
  }

  const auto matches = detail::list_suffix_matches(index_dir, "_disk.index");
  if (matches.size() == 1) {
    return matches.front();
  }
  if (matches.size() > 1) {
    detail::fail_ambiguous_candidates("*_disk.index", index_dir, matches);
  }

  const auto fallback = fs::path(index_dir) / "disk.index";
  if (fs::exists(fallback) && fs::is_regular_file(fallback)) {
    return fallback.string();
  }

  ERROR("No *_disk.index or disk.index found in {}", index_dir);
  std::exit(1);
}

inline std::string find_pq_prefix(const std::string& index_dir) {
  if (!fs::is_directory(index_dir)) {
    ERROR("--index-dir '{}' does not exist or is not a directory", index_dir);
    std::exit(1);
  }

  const auto matches =
      detail::list_suffix_matches(index_dir, "_pq_pivots.bin");
  if (matches.size() == 1) {
    const std::string& path = matches.front();
    return path.substr(0, path.size() - std::string("_pivots.bin").size());
  }
  if (matches.size() > 1) {
    detail::fail_ambiguous_candidates("*_pq_pivots.bin", index_dir, matches);
  }

  const auto fallback = fs::path(index_dir) / "pq";
  if (fs::exists(fallback.string() + "_pivots.bin")) {
    return fallback.string();
  }

  ERROR("No PQ pivot files found in {}", index_dir);
  std::exit(1);
}

inline std::string find_nav_dir(const std::string& index_dir) {
  if (fs::exists(fs::path(index_dir) / "nav_index")) {
    return index_dir;
  }

  if (fs::exists(fs::path(index_dir) / "nav" / "nav_index")) {
    return (fs::path(index_dir) / "nav").string();
  }

  return "";
}

}  // namespace bin_common
