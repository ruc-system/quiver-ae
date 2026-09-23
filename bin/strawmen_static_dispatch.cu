// Ablation binary: StaticDispatch with configurable queries per block.
//
// This binary runs Quiver's static batch scheduler:
//   - Q=1: one query per block, FlashANNS-style batch lockstep.
//   - Q>1: each batch pre-admits a fixed query range to Quiver's Q-interleave
//     streaming kernel, preserving the static batch boundary.
//
// Unlike quiver_search, host admission happens once per static batch. The host
// launches ceil(N / (num_blocks * Q)) batches and synchronizes between batches.

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
#include "common/fixed_params.hpp"
#include "common/loader_factory.hpp"
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

  argparse::ArgumentParser program("strawmen_static_dispatch", "",
                                   argparse::default_arguments::none);

  bin_common::SharedCliArgs shared_args;
  int num_blocks = 648;
  int poll_threads = 6;
  int pipe_width = 1;
  int queries_per_block = 1;
  std::string num_blocks_list;

  bin_common::register_long_only_help(program);
  bin_common::register_shared_base_args(program, shared_args);
  program.add_argument("--num-blocks")
      .default_value(648)
      .store_into(num_blocks);
  program.add_argument("--poll-threads").store_into(poll_threads);
#ifndef QUIVER_FIXED_PIPE_WIDTH
  program.add_argument("--pipe-width")
      .default_value(1)
      .store_into(pipe_width);
#endif
#ifndef QUIVER_FIXED_QUERIES_PER_BLOCK
  program.add_argument("--queries-per-block")
      .default_value(1)
      .store_into(queries_per_block);
#endif
  program.add_argument("--num-blocks-list").store_into(num_blocks_list);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

#ifdef QUIVER_FIXED_PIPE_WIDTH
  pipe_width = bin_common::resolve_fixed_value(pipe_width, QUIVER_FIXED_PIPE_WIDTH);
#endif
#ifdef QUIVER_FIXED_QUERIES_PER_BLOCK
  queries_per_block = bin_common::resolve_fixed_value(
      queries_per_block, QUIVER_FIXED_QUERIES_PER_BLOCK);
#endif

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
                << shared::colorize("StaticDispatch Sweep:", shared::ansi_cyan(), true)
                << " num_blocks=" << run_nb
                << " pipe_width=" << pipe_width
                << " queries_per_block=" << queries_per_block << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << std::endl;
    }

    auto loader = bin_common::create_index_loader(
        setup.index_file, setup.layout.num_pages, setup.ssd_lists, run_nb,
        poll_threads, pipe_width, queries_per_block);
    if (!loader) {
      return 1;
    }

    quiver::run_static_batch_search(
        setup.layout, setup.data_type, pipe_width, poll_threads, run_nb,
        loader, setup.pq.get(), setup.nav.get(),
        static_cast<int>(setup.layout.enter_point), starter.get(),
        setup.queries.values.get(), static_cast<int>(setup.queries.count),
        shared_args.topk, shared_args.ef_search, setup.nns.get(),
        setup.distances.get(), setup.found_counts.get(), queries_per_block,
        shared_args.repeat);

    bin_common::print_recall_report(shared_args, setup);
  }
  return 0;
}
