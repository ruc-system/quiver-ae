#include "spdlog/spdlog.h"
#include <argparse/argparse.hpp>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/quiver/search.cuh"
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

quiver::EarlyExitPolicy parse_early_exit_policy(const std::string& raw) {
  if (raw == "none") return quiver::EarlyExitPolicy::kDefault;
  if (raw == "gpruning") return quiver::EarlyExitPolicy::kGPruning;
  std::cerr << "invalid --early-exit-policy: " << raw
            << " (expected gpruning or none)" << std::endl;
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  shared::configure_logging_from_env();

  argparse::ArgumentParser program("quiver_search", "",
                                   argparse::default_arguments::none);

  bin_common::SharedCliArgs shared_args;
  shared::SpdkLoaderOptions spdk_options =
      shared::default_spdk_loader_options();
  int num_blocks = 756;
  int poll_threads = 6;
  int pipe_width = 2;
  int queries_per_block = 1;
  int runners_per_ssd = 1;
  std::string num_blocks_list;
  bool skip_nav = false;
  bool per_step_recall = false;
  float recall_threshold = 0.9f;
  std::string step_recall_csv = "quiver_step_recall.csv";
  std::string final_recall_csv;
  std::string result_prefix;
  std::string early_exit_policy_name = "gpruning";

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
      .help("Compatibility context count for the legacy submit_task/poll_task path.");
  program.add_argument("--spdk-runner-io-contexts")
      .default_value(spdk_options.runner_io_context_count)
      .store_into(spdk_options.runner_io_context_count)
      .help("Per-runner IO callback context pool size; one context tracks one SSD IO.");
#ifndef QUIVER_FIXED_PIPE_WIDTH
  program.add_argument("--pipe-width")
      .default_value(2)
      .store_into(pipe_width);
#endif
#ifndef QUIVER_FIXED_QUERIES_PER_BLOCK
  program.add_argument("--queries-per-block")
      .default_value(1)
      .store_into(queries_per_block);
#endif
  program.add_argument("--runners-per-ssd")
      .default_value(1)
      .store_into(runners_per_ssd)
      .help("Number of independent SPDK runner threads per SSD. Higher values "
            "raise per-SSD IOPS ceiling at the cost of more cores. "
            "Recommended sweep: 1, 3, 6, 9, 12. In compatibility mode "
            "(N=1) the runner uses the legacy core mapping; N>1 fans out "
            "from core (4 + global_runner_id*2) on NUMA 0.");
  program.add_argument("--num-blocks-list").store_into(num_blocks_list);
  program.add_argument("--skip-nav")
      .default_value(false)
      .implicit_value(true)
      .store_into(skip_nav)
      .help("Skip nav graph entry-point search; use global enter_point for all queries.");
  program.add_argument("--per-step-recall")
      .default_value(false)
      .implicit_value(true)
      .store_into(per_step_recall)
      .help("Enable experimental in-kernel per-query recall threshold tracking.");
  program.add_argument("--recall-threshold")
      .default_value(0.9f)
      .store_into(recall_threshold)
      .help("Per-query recall threshold for --per-step-recall.");
  program.add_argument("--step-recall-csv")
      .default_value(std::string("quiver_step_recall.csv"))
      .store_into(step_recall_csv)
      .help("CSV output path for --per-step-recall.");
  program.add_argument("--final-recall-csv")
      .store_into(final_recall_csv)
      .help("CSV output path for final per-query recall after normal search exits.");
  program.add_argument("--result-prefix")
      .store_into(result_prefix)
      .help("Write search output to <prefix>_ids.bin and "
            "<prefix>_distances.bin. Each file starts with int32 count and "
            "int32 topk, followed by row-major values.");
  program.add_argument("--early-exit-policy")
      .default_value(std::string("gpruning"))
      .store_into(early_exit_policy_name)
      .help("Quiver early-exit policy: gpruning or none. Default: gpruning.");

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
#ifdef QUIVER_FIXED_PIPE_WIDTH
  pipe_width = bin_common::resolve_fixed_value(pipe_width, QUIVER_FIXED_PIPE_WIDTH);
#endif
#ifdef QUIVER_FIXED_QUERIES_PER_BLOCK
  queries_per_block = bin_common::resolve_fixed_value(
      queries_per_block, QUIVER_FIXED_QUERIES_PER_BLOCK);
#endif
  if (spdk_options.submit_queue_capacity <= 0) {
    std::cerr << "--spdk-submit-queue-cap must be positive" << std::endl;
    return 1;
  }
  if (spdk_options.task_context_count <= 0) {
    std::cerr << "--spdk-task-contexts must be positive" << std::endl;
    return 1;
  }
  if (spdk_options.runner_io_context_count <= 0) {
    std::cerr << "--spdk-runner-io-contexts must be positive" << std::endl;
    return 1;
  }
  if (!bin_common::validate_positive_arg(num_blocks, "--num-blocks") ||
      !bin_common::validate_positive_arg(poll_threads, "--poll-threads") ||
      !bin_common::validate_int_range_arg(
          pipe_width, "--pipe-width", 1, bin_common::kMaxPipeWidth) ||
      !bin_common::validate_int_range_arg(queries_per_block,
          "--queries-per-block", 1, bin_common::kMaxQueriesPerBlock)) {
    return 1;
  }

  const bool runners_per_ssd_cli_used = program.is_used("--runners-per-ssd");
  if (!runners_per_ssd_cli_used) {
    if (const char* env = std::getenv("SPDK_RUNNERS_PER_SSD")) {
      int v = std::atoi(env);
      if (v >= 1 && v <= 32) {
        runners_per_ssd = v;
      } else {
        std::cerr << "[Quiver] SPDK_RUNNERS_PER_SSD out of range [1,32]: "
                  << env << " (ignored)" << std::endl;
      }
    }
  }
  std::cout << "[Quiver] runners_per_ssd=" << runners_per_ssd
            << (runners_per_ssd_cli_used ? " (cli)" : "") << std::endl;

  quiver::EarlyExitOptions early_exit;
  early_exit.policy = parse_early_exit_policy(early_exit_policy_name);
  early_exit.gpruning_warmup_steps = 10;
  early_exit.gpruning_patience = 8;
  std::cout << "[Quiver] early-exit policy="
            << (early_exit.enabled() ? "gpruning" : "none") << std::endl;

  const std::vector<int> run_num_blocks =
      num_blocks_list.empty()
          ? std::vector<int>{num_blocks}
          : parse_int_list_arg(num_blocks_list, "--num-blocks-list");
  bin_common::validate_positive_list_or_die(
      run_num_blocks, "--num-blocks-list");
  if (!result_prefix.empty() && run_num_blocks.size() != 1) {
    std::cerr << "--result-prefix cannot be combined with a multi-value "
                 "--num-blocks-list"
              << std::endl;
    return 1;
  }

  auto setup = bin_common::prepare_search_setup(shared_args);
  bin_common::GroundTruthData step_recall_gt;
  if (per_step_recall) {
    if (shared_args.ground_truth_file.empty()) {
      std::cerr << "--per-step-recall requires --ground-truth" << std::endl;
      return 1;
    }
    if (pipe_width != 1 || queries_per_block != 1) {
      std::cerr << "--per-step-recall currently requires pipe_width=1 "
                << "and queries_per_block=1" << std::endl;
      return 1;
    }
    step_recall_gt = bin_common::load_ground_truth(shared_args.ground_truth_file);
    std::cout << "[Quiver] per-step recall enabled: threshold="
              << recall_threshold << " csv=" << step_recall_csv
              << " gt_count=" << step_recall_gt.count
              << " gt_width=" << step_recall_gt.width << std::endl;
  }
  auto starter = bin_common::load_starter_page(setup);
  if (skip_nav) {
    std::cout << "[Quiver] --skip-nav enabled: nav graph disabled; "
              << "all queries start from global enter_point="
              << setup.layout.enter_point << std::endl;
  }
  for (const auto& run_nb : run_num_blocks) {
    if (run_num_blocks.size() > 1) {
      std::cout << "\n"
                << shared::colorize("============================================================",
                                    shared::ansi_gray())
                << "\n"
                << shared::colorize("Quiver Sweep:", shared::ansi_cyan(), true)
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
      spdlog::info("Using SPDK backend (submit_queue_capacity={} "
                   "task_context_count={} runner_io_context_count={} "
                   "runners_per_ssd={})",
                   spdk_options.submit_queue_capacity,
                   spdk_options.task_context_count,
                   spdk_options.runner_io_context_count, runners_per_ssd);
      loader = shared::create_spdk_loader(
          setup.ssd_lists, spdk_options.submit_queue_capacity, poll_threads,
          spdk_options.task_context_count,
          spdk_options.runner_io_context_count, runners_per_ssd);
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

    quiver::run_persistent_search(
        setup.layout, setup.data_type, pipe_width, poll_threads, run_nb, loader,
        setup.pq.get(),
        skip_nav ? nullptr : setup.nav.get(),
        static_cast<int>(setup.layout.enter_point), starter.get(),
        setup.queries.values.get(), static_cast<int>(setup.queries.count),
        shared_args.topk, shared_args.ef_search, setup.nns.get(),
        setup.distances.get(), setup.found_counts.get(),
        queries_per_block, shared_args.repeat,
        per_step_recall
            ? quiver::PerStepRecallOptions{
                  step_recall_gt.values.get(),
                  static_cast<int>(step_recall_gt.width),
                  static_cast<int>(step_recall_gt.count),
                  recall_threshold,
                  step_recall_csv.c_str()}
            : quiver::PerStepRecallOptions{},
        early_exit);

    bin_common::print_recall_report(shared_args, setup);
    bin_common::write_per_query_recall_csv(
        shared_args, setup, final_recall_csv);
    try {
      bin_common::write_search_results(shared_args, setup, result_prefix);
    } catch (const std::exception& err) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
  }
  return 0;
}
