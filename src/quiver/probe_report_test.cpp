#define QUIVER_LATENCY_PROBE

#include "probe_report.cuh"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "quiver_probe_report_test failed: " << message << std::endl;
    std::exit(1);
  }
}

void require_near(double actual, double expected, double tolerance,
                  const char* message) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "quiver_probe_report_test failed: " << message
              << " actual=" << actual << " expected=" << expected
              << std::endl;
    std::exit(1);
  }
}

}  // namespace

int main() {
  std::vector<int64_t> submit_ns{1000, 2000, 3000, 4000};
  std::vector<int64_t> complete_ns{1500, 0, 3250, 3999};
  std::vector<double> latencies;

  quiver::append_completed_io_latencies(latencies, submit_ns, complete_ns);

  require(latencies.size() == 2, "only completed requests should be recorded");
  require(latencies[0] == 0.5, "first completed request latency should be 0.5 us");
  require(latencies[1] == 0.25, "second completed request latency should be 0.25 us");

  setenv("QUIVER_PROBE_CDF_POINTS", "5", 1);
  const std::string csv_path =
      "/tmp/quiver_probe_report_test_" + std::to_string(::getpid()) + ".csv";
  quiver::write_probe_cdf_csv(csv_path.c_str(),
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  std::ifstream csv(csv_path);
  require(csv.good(), "CDF CSV should be written");
  std::string line;
  std::getline(csv, line);
  require(line == "metric,value_us,cdf,total_count",
          "CDF CSV should use the shared probe header");

  std::vector<double> values;
  while (std::getline(csv, line)) {
    const size_t first = line.find(',');
    const size_t second = line.find(',', first + 1);
    require(first != std::string::npos && second != std::string::npos,
            "CDF row should contain metric,value_us,cdf,total_count");
    require(line.substr(0, first) == "EndToEndIO",
            "CDF metric should be EndToEndIO");
    values.push_back(std::stod(line.substr(first + 1, second - first - 1)));
  }
  std::remove(csv_path.c_str());

  require(values.size() == 5, "CDF should emit requested point count");
  require(values[0] == 0, "first interval minimum should be 0");
  require(values[1] == 2, "second interval minimum should be 2");
  require(values[2] == 4, "third interval minimum should be 4");
  require(values[3] == 6, "fourth interval minimum should be 6");
  require(values[4] == 8, "last interval minimum should be 8, not global max");

  quiver::QuiverProbeBlockTrace block_trace{};
  block_trace.block_id = 7;
  block_trace.query_count = 12;
  block_trace.t_block_start = 1000;
  block_trace.t_block_done = 111000;
  block_trace.no_ready_wait_ns = 20000;
  block_trace.no_ready_spins = 42;

  const std::vector<quiver::QuiverBlockTimelineBreak> timelines =
      quiver::collect_native_block_timeline_breaks(&block_trace, 1);
  require(timelines.size() == 1, "one native block trace should produce one timeline");
  require(timelines[0].block_id == 7, "block id should be preserved");
  require(timelines[0].query_count == 12, "query count should be preserved");
  require_near(timelines[0].total, 110.0, 1e-9,
               "native block total should use block start and done");
  require_near(timelines[0].no_compute_idle, 20.0, 1e-9,
               "native block idle should use no-ready wait measured in kernel");
  require_near(timelines[0].active_compute, 90.0, 1e-9,
               "native block busy should be total minus no-ready wait");
  require_near(timelines[0].idle_ratio_pct, 100.0 * 20.0 / 110.0, 1e-9,
               "native block idle ratio should use measured no-ready wait");
  return 0;
}
