#pragma once

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

#include "../../../shared/common/ae_csv.hpp"
#include "shared/common/data_type.hpp"
#include "shared/common/log_config.hpp"
#include "shared/io/loader.hpp"

namespace shared {

struct QueryReportParams {
  std::string data_type;
  int topk = 0;
  int ef_search = 0;
  int repeat = 1;
};

struct LatencyReportStats {
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p99_ms = 0.0;
  double p999_ms = 0.0;
  double max_ms = 0.0;
};

struct RunReport {
  std::string program;
  QueryReportParams query;
  std::string runtime_params;
  double qps = 0.0;
  double recall_at_k = -1.0;
  std::string latency_label = "Latency(ms)";
  LatencyReportStats latency;
  uint64_t io_completed = 0;
  double search_time_s = 0.0;
  uint64_t hot_max_submitted_pages_per_sec = 0;
  uint64_t hot_sample_count = 0;
  int num_blocks = 0;
  int mini_batch = 0;
  int queries_per_block = 0;
  int pipe_width = 0;
};

inline std::string format_iops(uint64_t value) {
  const std::string raw = std::to_string(value);
  std::string out;
  out.reserve(raw.size() + raw.size() / 3);
  for (size_t i = 0; i < raw.size(); ++i) {
    if (i != 0 && (raw.size() - i) % 3 == 0) {
      out.push_back(',');
    }
    out.push_back(raw[i]);
  }
  return out;
}

inline const char* data_type_name(DataType data_type) {
  switch (data_type) {
    case UINT8:
      return "uint8";
    case INT8:
      return "int8";
    case FLOAT:
      return "float";
  }
  return "unknown";
}

inline double avg_bandwidth_gbps(uint64_t io_completed, double search_time_s) {
  if (search_time_s <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(io_completed) * static_cast<double>(PAGE_SIZE) /
         search_time_s / 1024.0 / 1024.0 / 1024.0;
}

inline uint64_t avg_iops(uint64_t io_completed, double search_time_s) {
  if (search_time_s <= 0.0) {
    return 0;
  }
  return static_cast<uint64_t>(
      static_cast<double>(io_completed) / search_time_s + 0.5);
}

inline double bandwidth_from_pages_per_sec(uint64_t pages_per_sec) {
  return static_cast<double>(pages_per_sec) * static_cast<double>(PAGE_SIZE) /
         1024.0 / 1024.0 / 1024.0;
}

inline std::string format_run_report(const RunReport& report) {
  std::ostringstream os;
  os << std::fixed;
  os << colorize("========== " + report.program + " Result ==========",
                 ansi_cyan(), true)
     << "\n";
  os << colorize("Query Params:", ansi_cyan(), true) << "\n";
  os << "  topk=" << report.query.topk << " ef_search="
     << report.query.ef_search << " repeat=" << report.query.repeat
     << " data_type=" << report.query.data_type << "\n";
  os << colorize("Runtime Params:", ansi_cyan(), true) << "\n";
  os << "  " << report.runtime_params << "\n\n";
  os << colorize("Performance:", ansi_cyan(), true) << "\n";
  os << "  " << colorize("QPS:", ansi_yellow(), true) << " "
     << std::setprecision(2) << report.qps << "\n";
  if (report.recall_at_k >= 0.0) {
    os << "  "
       << colorize("Recall@" + std::to_string(report.query.topk) + ":",
                   ansi_yellow(), true)
       << " " << std::setprecision(6) << report.recall_at_k << "\n";
  }
  os << "  " << colorize(report.latency_label + ":", ansi_yellow(), true)
     << " avg=" << std::setprecision(3)
     << report.latency.avg_ms << " p50=" << report.latency.p50_ms
     << " p90=" << report.latency.p90_ms << " p99="
     << report.latency.p99_ms << " p99.9=" << report.latency.p999_ms
     << " max=" << report.latency.max_ms << "\n";
  os << "  " << colorize("SSD Avg Bandwidth:", ansi_yellow(), true) << " "
     << std::setprecision(2)
     << avg_bandwidth_gbps(report.io_completed, report.search_time_s)
     << " GB/s\n";
  os << "  " << colorize("SSD Avg IOPS:", ansi_yellow(), true) << " "
     << format_iops(avg_iops(report.io_completed, report.search_time_s))
     << "\n";
  if (report.hot_sample_count > 0) {
    os << "  " << colorize("SSD Hot Avg Bandwidth:", ansi_yellow(), true)
       << " " << std::setprecision(2)
       << bandwidth_from_pages_per_sec(report.hot_max_submitted_pages_per_sec)
       << " GB/s\n";
    os << "  " << colorize("SSD Hot Avg IOPS:", ansi_yellow(), true) << " "
       << format_iops(report.hot_max_submitted_pages_per_sec) << "\n";
  }
  os << colorize("===================================", ansi_cyan(), true) << "\n";
  return os.str();
}

inline void print_run_report(const RunReport& report) {
  const std::string text = format_run_report(report);
  std::printf("%s", text.c_str());
  ae::append_metrics({report.program.c_str(), report.query.ef_search,
                      report.num_blocks, report.mini_batch,
                      report.queries_per_block, report.pipe_width, report.qps,
                      report.latency.avg_ms, report.latency.p50_ms,
                      report.latency.p90_ms, report.latency.p99_ms,
                      report.latency.p999_ms, report.latency.max_ms,
                      report.latency_label.c_str()});
}

}  // namespace shared
