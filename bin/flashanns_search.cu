#include "spdlog/spdlog.h"
#include <argparse/argparse.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/flashanns/search.cuh"
#include "shared/common/log_config.hpp"
#include "shared/io/spdk_loader_config.hpp"

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

  argparse::ArgumentParser program("flashanns_search", "",
                                   argparse::default_arguments::none);

  bin_common::SharedCliArgs shared_args;
  shared::SpdkLoaderOptions spdk_options =
      shared::default_spdk_loader_options();
  int num_blocks = 756;
  int poll_threads = 6;
  int pipe_width = 2;  // FlashANNS paper default (was 4 in earlier revisions)
  std::string num_blocks_list;

  bin_common::register_long_only_help(program);
  bin_common::register_shared_base_args(program, shared_args);
  program.add_argument("--num-blocks")
      .default_value(756)
      .store_into(num_blocks);
  program.add_argument("--poll-threads").store_into(poll_threads);
  program.add_argument("--spdk-submit-queue-cap")
      .default_value(spdk_options.submit_queue_capacity)
      .store_into(spdk_options.submit_queue_capacity)
      .help("Per-producer SPSC submit queue capacity for each SSD runner.");
  program.add_argument("--spdk-task-contexts")
      .default_value(spdk_options.task_context_count)
      .store_into(spdk_options.task_context_count)
      .help("Batch/task context count for submit_task/poll_task completion tracking.");
  program.add_argument("--spdk-runner-io-contexts")
      .default_value(spdk_options.runner_io_context_count)
      .store_into(spdk_options.runner_io_context_count)
      .help("Per-runner IO callback context pool size; one context tracks one SSD IO.");
#ifndef FLASHANNS_FIXED_PIPE_WIDTH
  program.add_argument("--pipe-width")
      .default_value(2)
      .store_into(pipe_width);
#endif
  program.add_argument("--num-blocks-list").store_into(num_blocks_list);

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
#ifdef FLASHANNS_FIXED_PIPE_WIDTH
  pipe_width = bin_common::resolve_fixed_value(pipe_width, FLASHANNS_FIXED_PIPE_WIDTH);
#endif
  if (spdk_options.submit_queue_capacity <= 0) {
    std::cerr << "--spdk-submit-queue-cap must be positive" << std::endl;
    return 1;
  }
  if (spdk_options.task_context_count < poll_threads) {
    std::cerr << "--spdk-task-contexts must be >= --poll-threads" << std::endl;
    return 1;
  }
  if (spdk_options.runner_io_context_count <= 0) {
    std::cerr << "--spdk-runner-io-contexts must be positive" << std::endl;
    return 1;
  }
  if (!bin_common::validate_positive_arg(num_blocks, "--num-blocks") ||
      !bin_common::validate_positive_arg(poll_threads, "--poll-threads") ||
      !bin_common::validate_int_range_arg(
          pipe_width, "--pipe-width", 1, bin_common::kMaxPipeWidth)) {
    return 1;
  }

  const std::vector<int> run_num_blocks =
      num_blocks_list.empty()
          ? std::vector<int>{num_blocks}
          : parse_int_list_arg(num_blocks_list, "--num-blocks-list");
  bin_common::validate_positive_list_or_die(
      run_num_blocks, "--num-blocks-list");

  auto setup = bin_common::prepare_search_setup(shared_args);
  auto starter = bin_common::load_starter_page(setup);
  for (const auto& run_nb : run_num_blocks) {
    if (run_num_blocks.size() > 1) {
      std::cout << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << shared::colorize("FlashANNS Sweep:", shared::ansi_cyan(), true)
                << " num_blocks=" << run_nb
                << " pipe_width=" << pipe_width << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << std::endl;
    }

    std::shared_ptr<shared::IndexLoader> loader;
    if (!setup.ssd_lists.empty()) {
#if QUIVER_ENABLE_SPDK
      spdlog::info("Using SPDK backend (submit_queue_capacity={} "
                   "task_context_count={} runner_io_context_count={})",
                   spdk_options.submit_queue_capacity,
                   spdk_options.task_context_count,
                   spdk_options.runner_io_context_count);
      loader = shared::create_spdk_loader(
          setup.ssd_lists, spdk_options.submit_queue_capacity, poll_threads,
          spdk_options.task_context_count,
          spdk_options.runner_io_context_count);
#else
      return bin_common::fail_without_spdk_support();
#endif
    } else {
      spdlog::info("Using memory backend: {} index_file={}",
                   shared::memory_backend_name(setup.memory_backend),
                   setup.memory_index_file);
      loader = shared::create_mem_loader_sync(
          setup.memory_index_file.c_str(), setup.layout.num_pages,
          setup.memory_backend);
    }
    if (!loader) {
      return 1;
    }

    flashanns::run_persistent_search(
        setup.layout, setup.data_type, pipe_width, poll_threads,
        spdk_options.task_context_count, run_nb, loader, setup.pq.get(), setup.nav.get(),
        static_cast<int>(setup.layout.enter_point), starter.get(),
        setup.queries.values.get(), static_cast<int>(setup.queries.count),
        shared_args.topk, shared_args.ef_search, setup.nns.get(),
        setup.distances.get(), setup.found_counts.get(), shared_args.repeat);

    bin_common::print_recall_report(shared_args, setup);
  }
  return 0;
}
