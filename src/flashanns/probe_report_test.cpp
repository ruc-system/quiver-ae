#define FLASHANNS_LATENCY_PROBE

#include "probe_report.cuh"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "flashanns_probe_report_test failed: " << message << std::endl;
    std::exit(1);
  }
}

void require_near(double actual, double expected, double tolerance,
                  const char* message) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "flashanns_probe_report_test failed: " << message
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

  flashanns::append_completed_io_latencies(latencies, submit_ns, complete_ns);

  require(latencies.size() == 2, "only completed requests should be recorded");
  require(latencies[0] == 0.5, "first completed request latency should be 0.5 us");
  require(latencies[1] == 0.25, "second completed request latency should be 0.25 us");

  setenv("FLASHANNS_PROBE_CDF_POINTS", "5", 1);
  const std::string csv_path =
      "/tmp/flashanns_probe_report_test_" + std::to_string(::getpid()) + ".csv";
  flashanns::write_end_to_end_io_cdf_csv(
      csv_path.c_str(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  std::ifstream csv(csv_path);
  require(csv.good(), "CDF CSV should be written");
  std::string line;
  std::getline(csv, line);
  std::vector<double> values;
  while (std::getline(csv, line)) {
    const size_t first = line.find(',');
    const size_t second = line.find(',', first + 1);
    values.push_back(std::stod(line.substr(first + 1, second - first - 1)));
  }
  std::remove(csv_path.c_str());

  require(values.size() == 5, "CDF should emit requested point count");
  require(values[0] == 0, "first interval minimum should be 0");
  require(values[1] == 2, "second interval minimum should be 2");
  require(values[2] == 4, "third interval minimum should be 4");
  require(values[3] == 6, "fourth interval minimum should be 6");
  require(values[4] == 8, "last interval minimum should be 8, not global max");

  flashanns::FlashProbeQueryTrace tr{};
  tr.t_query_start = 1000;
  tr.t_init_done = 3000;
  tr.t_boot_io_done = 13000;
  tr.t_boot_pq_done = 16000;
  tr.t_boot_merge_done = 20000;
  tr.t_boot_sort_done = 26000;
  tr.t_boot_fill_done = 31000;
  tr.t_pipe_done = 90000;
  tr.t_query_done = 101000;
  tr.pipe_io_ns = 20000;
  tr.pipe_pq_ns = 7000;
  tr.pipe_merge_ns = 9000;
  tr.pipe_sort_ns = 11000;
  tr.pipe_issue_ns = 12000;

  const flashanns::FlashBlockTimelineBreak timeline =
      flashanns::compute_block_timeline_break(tr);
  require_near(timeline.total, 100.0, 1e-9,
               "block total should cover query start to done");
  require_near(timeline.no_compute_idle, 30.0, 1e-9,
               "block idle should include boot wait plus pipe no-ready wait");
  require_near(timeline.active_compute, 70.0, 1e-9,
               "block active time should be total minus no-compute idle");
  require_near(timeline.idle_ratio_pct, 30.0, 1e-9,
               "block idle ratio should be based on block total");
  return 0;
}
