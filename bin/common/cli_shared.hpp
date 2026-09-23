#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include <argparse/argparse.hpp>

#include "shared/common/logging.hpp"
#include "shared/common/data_type.hpp"

namespace bin_common {

struct SharedCliArgs {
  std::string index_dir;
  std::string query_file;
  std::string ground_truth_file;
  std::string data_type;
  std::string ssd_list_file;
  std::string memory_backend = "heap";
  std::string memory_index_file;
  int topk = 10;
  int ef_search = 30;
  int repeat = 20;
};

inline void register_long_only_help(argparse::ArgumentParser& program) {
  program.add_argument("--help")
      .action([&program](const auto&) {
        std::cout << program;
        std::exit(0);
      })
      .default_value(false)
      .implicit_value(true)
      .help("shows help message and exits")
      .nargs(0);
}

// Shared input/eval arguments reused by the three final entrypoints.
inline void register_shared_base_args(argparse::ArgumentParser& program,
                                      SharedCliArgs& args) {
  program.add_argument("--index-dir").required().store_into(args.index_dir);
  program.add_argument("--query").required().store_into(args.query_file);
  program.add_argument("--topk").store_into(args.topk);
  program.add_argument("--ground-truth").store_into(args.ground_truth_file);
  program.add_argument("--data-type").required().store_into(args.data_type);
  program.add_argument("--ef-search").store_into(args.ef_search);
  program.add_argument("--ssd-list-file").store_into(args.ssd_list_file);
  program.add_argument("--memory-backend")
      .default_value(std::string("heap"))
      .store_into(args.memory_backend)
      .help("Memory backend to use when --ssd-list-file is empty: heap or mmap.");
  program.add_argument("--memory-index-file")
      .store_into(args.memory_index_file)
      .help("Index file used by memory backends; defaults to --index-dir discovery.");
  program.add_argument("--repeat").store_into(args.repeat);
}

inline shared::DataType parse_data_type(const std::string& value) {
  if (value == "uint8") {
    return shared::UINT8;
  }
  if (value == "float") {
    return shared::FLOAT;
  }
  if (value == "int8") {
    return shared::INT8;
  }

  ERROR("Unsupported data type '{}'. Expected one of: uint8, int8, float", value);
  std::exit(1);
}

}  // namespace bin_common
