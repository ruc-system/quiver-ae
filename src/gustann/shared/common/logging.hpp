#pragma once

#include <cstdlib>

#ifndef SPDLOG_EOL
#define SPDLOG_EOL ""
#endif

#ifndef SPDLOG_TRACE_ON
#define SPDLOG_TRACE_ON
#endif

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#define INFO SPDLOG_INFO
#define DEBUG SPDLOG_DEBUG
#define WARN SPDLOG_WARN
#define ERROR SPDLOG_ERROR

#ifndef ASSERT
#define ASSERT(x)                                                          \
  do {                                                                     \
    if (!(x)) {                                                            \
      ERROR("Assertion failed {}", #x);                                    \
      std::abort();                                                        \
    }                                                                      \
  } while (0)
#endif
