// Strawman PQ-in-global-memory: explicit context spill-load on every switch.
//
// Built only when -DQUIVER_BUILD_STRAWMEN=ON.
// CLI mirrors strawmen_pq_in_smem so the ablation can isolate context
// placement: per-slot search context lives in global memory and is explicitly
// loaded into shared memory when a slot is scheduled, then written back before
// switching away.

#include "spdlog/spdlog.h"
#include <argparse/argparse.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/quiver/search.cuh"
#include "shared/common/log_config.hpp"
#include "shared/io/spdk_loader_config.hpp"

#include "common/cli_shared.hpp"
#include "common/cli_validation.hpp"
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

  argparse::ArgumentParser program("strawmen_pq_in_gmem", "",
                                   argparse::default_arguments::none);

  bin_common::SharedCliArgs shared_args;
  int num_blocks = 648;
  int poll_threads = 6;
  int pipe_width = 1;
  int queries_per_block = 2;
  std::string num_blocks_list;

  bin_common::register_long_only_help(program);
  bin_common::register_shared_base_args(program, shared_args);
  program.add_argument("--num-blocks").default_value(648).store_into(num_blocks);
  program.add_argument("--poll-threads").store_into(poll_threads);
  program.add_argument("--pipe-width").store_into(pipe_width);
  program.add_argument("--queries-per-block")
      .default_value(2).store_into(queries_per_block);
  program.add_argument("--num-blocks-list").store_into(num_blocks_list);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  if (!bin_common::validate_positive_arg(num_blocks, "--num-blocks") ||
      !bin_common::validate_positive_arg(poll_threads, "--poll-threads") ||
      !bin_common::validate_int_range_arg(
          pipe_width, "--pipe-width", 1, bin_common::kMaxPipeWidth) ||
      !bin_common::validate_int_range_arg(
          queries_per_block, "--queries-per-block", 1,
          bin_common::kMaxQueriesPerBlock)) {
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
                << shared::colorize("PQInGmem Sweep:", shared::ansi_cyan(), true)
                << " num_blocks=" << run_nb
                << " pipe_width=" << pipe_width
                << " queries_per_block=" << queries_per_block << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << std::endl;
    }

    std::shared_ptr<shared::IndexLoader> loader;
    if (!setup.ssd_lists.empty()) {
#if QUIVER_ENABLE_SPDK
      const auto spdk_options = shared::default_spdk_loader_options();
      spdlog::info("Using SPDK backend (submit_queue_capacity={} "
                   "task_context_count={} runner_io_context_count={} Q={})",
                   spdk_options.submit_queue_capacity,
                   spdk_options.task_context_count,
                   spdk_options.runner_io_context_count, queries_per_block);
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

    quiver::run_pq_in_gmem_search(
        setup.layout, setup.data_type, pipe_width, poll_threads, run_nb,
        loader, setup.pq.get(), setup.nav.get(),
        static_cast<int>(setup.layout.enter_point), starter.get(),
        setup.queries.values.get(), static_cast<int>(setup.queries.count),
        shared_args.topk, shared_args.ef_search, setup.nns.get(),
        setup.distances.get(), setup.found_counts.get(),
        queries_per_block, shared_args.repeat);

    bin_common::print_recall_report(shared_args, setup);
  }
  return 0;
}
