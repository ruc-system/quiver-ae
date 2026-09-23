// Ablation binary: Q=1 Quiver early-exit strawmen.
//
// This intentionally keeps early-exit knobs out of quiver_search.  The binary
// runs fixed policies for each num_blocks setting:
//   1. none              : baseline natural termination
//   2. darth_cpu_online  : direct DARTH-style CPU predictor communication
//   3. gpruning: rule-based PiP-style top-k update saturation
//   4. darth_trace       : full-search trace collection for DARTH training
//
// The constants below are placeholders for paper strawmen.  They are not user
// tuning knobs; future Quiver-specific policy work should replace the kernel
// policy itself rather than adding more CLI surface to quiver_search.

#include "spdlog/spdlog.h"
#include <argparse/argparse.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/quiver/search.cuh"
#include "shared/common/log_config.hpp"

#include "common/cli_shared.hpp"
#include "common/cli_validation.hpp"
#include "common/loader_factory.hpp"
#include "common/search_setup.hpp"

namespace {

constexpr int kPipeWidth = 1;
constexpr int kQueriesPerBlock = 1;
constexpr int kDarthInterval = 2;
constexpr float kDarthThreshold = 0.9f;
constexpr int kGPruningWarmupSteps = 10;
constexpr int kGPruningPatience = 8;

struct PolicyRun {
  const char* name;
  quiver::EarlyExitOptions options;
};

std::vector<int> parse_int_list_arg(const std::string& raw,
                                    const std::string& arg_name) {
  std::string normalized = raw;
  for (char& c : normalized) {
    if (c == ',') {
      c = ' ';
    }
  }

  std::vector<int> values;
  std::istringstream iss(normalized);
  std::string token;
  while (iss >> token) {
    try {
      values.push_back(std::stoi(token));
    } catch (const std::exception&) {
      std::cerr << "invalid integer in " << arg_name << ": " << token
                << std::endl;
      std::exit(1);
    }
  }
  if (values.empty()) {
    std::cerr << "empty list for " << arg_name << std::endl;
    std::exit(1);
  }
  return values;
}

std::vector<PolicyRun> all_policy_runs() {
  quiver::EarlyExitOptions none;

  quiver::EarlyExitOptions darth;
  darth.policy = quiver::EarlyExitPolicy::kDarth;
  darth.darth_interval = kDarthInterval;
  darth.darth_threshold = kDarthThreshold;

  quiver::EarlyExitOptions gpruning;
  gpruning.policy = quiver::EarlyExitPolicy::kGPruning;
  gpruning.gpruning_warmup_steps = kGPruningWarmupSteps;
  gpruning.gpruning_patience = kGPruningPatience;

  quiver::EarlyExitOptions trace;
  trace.policy = quiver::EarlyExitPolicy::kDarthTrace;
  trace.darth_interval = kDarthInterval;
  trace.darth_threshold = kDarthThreshold;

  return {
      {"none", none},
      {"darth_cpu_online", darth},
      {"gpruning", gpruning},
      {"darth_trace", trace},
  };
}

std::vector<PolicyRun> select_policy_runs(const std::string& type) {
  const auto all = all_policy_runs();
  if (type == "all") return {all[0], all[1], all[2]};
  if (type == "none") return {all[0]};
  if (type == "darth") return {all[1]};
  if (type == "gpruning") return {all[2]};
  if (type == "trace") return {all[3]};
  std::cerr << "invalid --early-exit-type: " << type
            << " (expected all, none, darth, or gpruning)" << std::endl;
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  shared::configure_logging_from_env();

  argparse::ArgumentParser program("strawmen_early_exit", "",
                                   argparse::default_arguments::none);

  bin_common::SharedCliArgs shared_args;
  int num_blocks = 648;
  int poll_threads = 6;
  std::string num_blocks_list;
  std::string early_exit_type = "all";

  bin_common::register_long_only_help(program);
  bin_common::register_shared_base_args(program, shared_args);
  program.add_argument("--num-blocks")
      .default_value(648)
      .store_into(num_blocks);
  program.add_argument("--poll-threads").store_into(poll_threads);
  program.add_argument("--num-blocks-list").store_into(num_blocks_list);
  program.add_argument("--early-exit-type")
      .default_value(std::string("all"))
      .store_into(early_exit_type)
      .help("Strawman policy to run: all, none, darth, or gpruning.");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  if (!bin_common::validate_positive_arg(num_blocks, "--num-blocks") ||
      !bin_common::validate_positive_arg(poll_threads, "--poll-threads")) {
    return 1;
  }
  const std::vector<int> run_num_blocks =
      num_blocks_list.empty()
          ? std::vector<int>{num_blocks}
          : parse_int_list_arg(num_blocks_list, "--num-blocks-list");
  bin_common::validate_positive_list_or_die(
      run_num_blocks, "--num-blocks-list");

  auto setup = bin_common::prepare_search_setup(shared_args);
  bin_common::GroundTruthData gt;
  if (!shared_args.ground_truth_file.empty()) {
    gt = bin_common::load_ground_truth(shared_args.ground_truth_file);
  }
  auto starter = bin_common::load_starter_page(setup);
  const auto policies = select_policy_runs(early_exit_type);
  for (const auto& run_nb : run_num_blocks) {
    for (const auto& policy : policies) {
      std::cout << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << shared::colorize("EarlyExit Strawman:", shared::ansi_cyan(), true)
                << " policy=" << policy.name
                << " num_blocks=" << run_nb
                << " pipe_width=" << kPipeWidth
                << " queries_per_block=" << kQueriesPerBlock << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << std::endl;

      auto loader = bin_common::create_index_loader(
          setup.index_file, setup.layout.num_pages, setup.ssd_lists, run_nb,
          poll_threads, kPipeWidth, kQueriesPerBlock);
      if (!loader) {
        return 1;
      }

      quiver::EarlyExitOptions options = policy.options;
      std::string darth_trace_csv;
      if (options.policy == quiver::EarlyExitPolicy::kDarth ||
          options.policy == quiver::EarlyExitPolicy::kDarthTrace) {
        options.darth_ground_truth = gt.values.get();
        options.darth_ground_truth_width = static_cast<int>(gt.width);
        options.darth_ground_truth_count = static_cast<int>(gt.count);
        darth_trace_csv = "darth_trace_nb" + std::to_string(run_nb) + ".csv";
        options.darth_trace_csv_path = darth_trace_csv.c_str();
      }

      quiver::run_static_batch_search(
          setup.layout, setup.data_type, kPipeWidth, poll_threads, run_nb,
          loader, setup.pq.get(), setup.nav.get(),
          static_cast<int>(setup.layout.enter_point), starter.get(),
          setup.queries.values.get(), static_cast<int>(setup.queries.count),
          shared_args.topk, shared_args.ef_search, setup.nns.get(),
          setup.distances.get(), setup.found_counts.get(), kQueriesPerBlock,
          shared_args.repeat, options);

      bin_common::print_recall_report(shared_args, setup);
    }
  }
  return 0;
}
