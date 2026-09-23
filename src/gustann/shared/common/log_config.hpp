#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

#include "spdlog/spdlog.h"

namespace shared {

inline spdlog::level::level_enum parse_log_level(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return spdlog::level::warn;
  }
  if (std::strcmp(value, "trace") == 0) return spdlog::level::trace;
  if (std::strcmp(value, "debug") == 0) return spdlog::level::debug;
  if (std::strcmp(value, "info") == 0) return spdlog::level::info;
  if (std::strcmp(value, "warn") == 0) return spdlog::level::warn;
  if (std::strcmp(value, "warning") == 0) return spdlog::level::warn;
  if (std::strcmp(value, "error") == 0) return spdlog::level::err;
  if (std::strcmp(value, "err") == 0) return spdlog::level::err;
  if (std::strcmp(value, "off") == 0) return spdlog::level::off;
  return spdlog::level::warn;
}

inline spdlog::level::level_enum resolve_log_level_from_env() {
  return parse_log_level(std::getenv("QUIVER_LOG_LEVEL"));
}

inline void configure_logging_from_env() {
  spdlog::set_pattern("[%^%-8l%$] %Y-%m-%d %H:%M:%S [%s:%#] %v");
  spdlog::set_level(resolve_log_level_from_env());
}

inline bool spdk_diag_enabled() {
  const char* env = std::getenv("QUIVER_SPDK_DIAG");
  return env != nullptr && std::atoi(env) != 0;
}

inline bool terminal_color_enabled() {
  const char* no_color = std::getenv("NO_COLOR");
  if (no_color != nullptr && no_color[0] != '\0') {
    return false;
  }
  const char* color = std::getenv("QUIVER_COLOR");
  if (color != nullptr && std::strcmp(color, "0") == 0) {
    return false;
  }
  return true;
}

inline const char* ansi_bold() { return terminal_color_enabled() ? "\033[1m" : ""; }
inline const char* ansi_cyan() { return terminal_color_enabled() ? "\033[36m" : ""; }
inline const char* ansi_yellow() { return terminal_color_enabled() ? "\033[33m" : ""; }
inline const char* ansi_gray() { return terminal_color_enabled() ? "\033[90m" : ""; }
inline const char* ansi_reset() { return terminal_color_enabled() ? "\033[0m" : ""; }

inline std::string colorize(const std::string& text, const char* color,
                            bool bold = false) {
  if (!terminal_color_enabled()) {
    return text;
  }
  std::string out;
  if (bold) out += ansi_bold();
  out += color;
  out += text;
  out += ansi_reset();
  return out;
}

}  // namespace shared
