#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>

#define SPDLOG_EOL ""
#define SPDLOG_TRACE_ON
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#define __FILENAME__                                                        \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define INFO(x, ...)                                                        \
  logger_->info("[{}:{}] " x, __FILENAME__, __LINE__, __VA_ARGS__);
#define DEBUG(x, ...)                                                       \
  logger_->debug("[{}:{}] " x, __FILENAME__, __LINE__, __VA_ARGS__);
#define WARN(x, ...)                                                        \
  logger_->warn("[{}:{}] " x, __FILENAME__, __LINE__, __VA_ARGS__);
#define TRACE(x, ...)                                                       \
  logger_->trace("[{}:{}] " x, __FILENAME__, __LINE__, __VA_ARGS__);
#define CRITICAL(x, ...)                                                    \
  logger_->critical("[{}:{}] " x, __FILENAME__, __LINE__, __VA_ARGS__);

#define INFO0(x) logger_->info("[{}:{}] " x, __FILENAME__, __LINE__);
#define DEBUG0(x) logger_->debug("[{}:{}] " x, __FILENAME__, __LINE__);
#define WARN0(x) logger_->warn("[{}:{}] " x, __FILENAME__, __LINE__);
#define TRACE0(x) logger_->trace("[{}:{}] " x, __FILENAME__, __LINE__);
#define CRITICAL0(x) logger_->critical("[{}:{}] " x, __FILENAME__, __LINE__);

#define ASSERT(x)                                                           \
  do {                                                                      \
    if (!(x)) {                                                             \
      CRITICAL("Assertion failed {}", #x)                                   \
      abort();                                                              \
    }                                                                       \
  } while (0)

#define CHECK_(...) std::fstream("/dev/null")
#define FB_LOG_EVERY_MS(...) std::fstream("/dev/null")

#define FOLLY_ALWAYS_INLINE
#define FOLLY_UNLIKELY(a) (a)

static std::shared_ptr<spdlog::logger> logger_;
