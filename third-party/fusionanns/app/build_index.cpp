#include "index/builder.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <CLI/CLI.hpp>

namespace {

BaseFormat parse_format(const std::string &value) {
  if (value == "fvecs") {
    return BaseFormat::FVECs;
  }
  if (value == "u8bin") {
    return BaseFormat::U8BIN;
  }
  if (value == "bvecs") {
    return BaseFormat::BVECS;
  }
  if (value == "fbin") {
    return BaseFormat::FBIN;
  }
  throw std::runtime_error("invalid dataset format: " + value);
}

} // namespace

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;

  BuilderOptions opt;
  opt.base_path = "data/sift/sift_base.fvecs";

  std::string base_fmt = "fvecs";
  std::string learn_fmt = "fvecs";

  CLI::App app{"FusionANNS Index Builder"};

  app.add_option(
      "--base", opt.base_path,
      "Path to the base dataset (default: data/sift/sift_base.fvecs)");
  app.add_option("--base-format", base_fmt,
                 "Format of the base dataset (fvecs|u8bin|bvecs|fbin)");
  app.add_option("--learn", opt.learn_path,
                 "Optional path to the learn dataset");
  app.add_option("--learn-format", learn_fmt,
                 "Format of the learn dataset (fvecs|u8bin|bvecs|fbin)");
  app.add_option("--learn-n", opt.learn_n_hint,
                 "Hint size for the learn dataset");
  app.add_option("--nlist", opt.nlist,
                 "Number of coarse centroids (posting lists)");
  app.add_option("--branch", opt.branch,
                 "Branch factor for SPTAG BKT splits (default: 32)");
  app.add_option("--beam", opt.beam,
                 "Centroid candidate count for replication search");
  app.add_option("--m", opt.m, "PQ sub-vector count (M)");
  app.add_option("--nbits", opt.nbits, "PQ bits per sub-vector");
  app.add_option("--chunk", opt.chunk,
                 "Streaming chunk size measured in vectors");
  app.add_option("--page", opt.page, "Page size for packed raw layout (bytes)");
  app.add_option("--output-dir", opt.output_dir,
                 "Directory for built index artifacts (default: indices)");
  app.add_option("--epsilon", opt.epsilon,
                 "Replication epsilon (threshold uses (1+epsilon)^2)");
  app.add_option("--max-repl", opt.max_replications,
                 "Maximum replication copies per vector");
  app.add_option("--imbalance-delta", opt.imbalance_delta,
                 "Legacy imbalance tolerance (unused)");
  app.add_flag("--preload,!--no-preload", opt.preload,
               "Preload base dataset into memory for faster clustering");
  app.add_option(
      "--hbc-preload-gib", opt.hbc_preload_gib,
      "Max dataset size (GiB) to fully preload for HBC (0 disables)");
  app.add_option("--hbc-stream-block-mb", opt.hbc_stream_block_mb,
                 "Per-thread streaming block size for HBC (MiB, 0=auto)");
  app.add_option("--cache-size", opt.cache_size_mb,
                 "LRU cache size for base dataset (MiB, 0=disabled)");
  app.add_flag("--mmap", opt.use_mmap, "Use mmap for reading datasets");

  // Residual SQ8 选项
  app.add_flag(
      "--rsq8", opt.use_residual_sq8,
      "Use Residual SQ8 encoding instead of PQ (optimized for Tensor Core)");
  app.add_option("--rsq8-percentile", opt.rsq8_scale_percentile,
                 "Scale calculation percentile for RSQ8 (default: 0.99)");
  app.add_flag("--rsq8-max-abs", opt.rsq8_use_max_abs,
               "Use max absolute value for RSQ8 scale calculation");

  CLI11_PARSE(app, argc, argv);

  try {
    opt.base_fmt = parse_format(base_fmt);
    if (!opt.learn_path.empty()) {
      opt.has_learn = true;
      opt.learn_fmt = parse_format(learn_fmt);
    } else {
      opt.has_learn = false;
    }
    if (opt.output_dir.empty()) {
      throw std::invalid_argument("--output-dir must not be empty");
    }

    FusionAnnsBuilder builder(opt);
    builder.build();
  } catch (const std::exception &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
