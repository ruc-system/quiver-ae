#include "spdlog/spdlog.h"
#include <argparse/argparse.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/gustann/executor.hpp"
#include "shared/common/log_config.hpp"

#include "common/cli_shared.hpp"
#include "common/cli_validation.hpp"
#include "common/fixed_params.hpp"
#include "common/search_setup.hpp"

namespace {

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

}  // namespace

int main(int argc, char** argv) {
  shared::configure_logging_from_env();

  argparse::ArgumentParser program("gustann_search", "",
                                   argparse::default_arguments::none);

  bin_common::SharedCliArgs shared_args;
  int mini_batch = 1120;
  int worker_threads = 2;
  int runner_contexts = 20;
  int pipe_width = 1;
  std::string mini_batch_list;

  bin_common::register_long_only_help(program);
  bin_common::register_shared_base_args(program, shared_args);
  program.add_argument("--mini-batch").store_into(mini_batch);
  program.add_argument("--worker-threads").store_into(worker_threads);
  program.add_argument("--runner-contexts").store_into(runner_contexts);
#ifndef GUSTANN_FIXED_PIPE_WIDTH
  program.add_argument("--pipe-width").store_into(pipe_width);
#endif
  program.add_argument("--mini-batch-list").store_into(mini_batch_list);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  if (const int rc = bin_common::fail_if_spdk_requested_without_support(shared_args)) {
    return rc;
  }
#ifdef GUSTANN_FIXED_PIPE_WIDTH
  pipe_width = bin_common::resolve_fixed_value(pipe_width, GUSTANN_FIXED_PIPE_WIDTH);
#endif
  if (!bin_common::validate_positive_arg(mini_batch, "--mini-batch") ||
      !bin_common::validate_positive_arg(worker_threads, "--worker-threads") ||
      !bin_common::validate_positive_arg(runner_contexts, "--runner-contexts") ||
      !bin_common::validate_int_range_arg(
          pipe_width, "--pipe-width", 1, bin_common::kMaxPipeWidth)) {
    return 1;
  }

  const std::vector<int> run_mini_batches =
      mini_batch_list.empty()
          ? std::vector<int>{mini_batch}
          : parse_int_list_arg(mini_batch_list, "--mini-batch-list");
  bin_common::validate_positive_list_or_die(
      run_mini_batches, "--mini-batch-list");

  auto setup = bin_common::prepare_search_setup(shared_args);
#if !QUIVER_ENABLE_SPDK
  if (!setup.ssd_lists.empty()) {
    return bin_common::fail_without_spdk_support();
  }
#endif
  for (const auto& run_mini_batch : run_mini_batches) {
    if (run_mini_batches.size() > 1) {
      std::cout << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << shared::colorize("GustANN Sweep:", shared::ansi_cyan(), true)
                << " mini_batch=" << run_mini_batch
                << " pipe_width=" << pipe_width << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << std::endl;
    }

    gustann::HybridExecutorConfig config;
    config.mini_batch = run_mini_batch;
    config.thread_cnt = worker_threads;
    config.ctx_per_thread = runner_contexts;
    config.pipe_w = pipe_width;
    config.use_backend = setup.ssd_lists.empty()
                             ? gustann::HybridExecutorConfig::MEMORY
                             : gustann::HybridExecutorConfig::SPDK;
    config.ssd_lists = setup.ssd_lists;
    config.memory_backend = setup.memory_backend;
    config.memory_index_file = setup.memory_index_file;

    gustann::HybridExecutor executor(setup.layout, setup.data_type,
                                     setup.index_file, config);
    executor.search(setup.queries.values.get(),
                    static_cast<int>(setup.queries.count), shared_args.topk,
                    shared_args.ef_search, setup.nns.get(),
                    setup.distances.get(), setup.found_counts.get(),
                    setup.pq.get(), setup.nav.get(), shared_args.repeat);

    bin_common::print_recall_report(shared_args, setup);
  }
  return 0;
}
