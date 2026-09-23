#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "shared/common/data_type.hpp"
#include "shared/common/layout.hpp"
#include "shared/common/logging.hpp"
#include "shared/index/nav_graph.hpp"
#include "shared/index/pq_search.hpp"
#include "shared/io/loader.hpp"

#include "cli_shared.hpp"
#include "data_io.hpp"
#include "index_discovery.hpp"
#include "metadata.hpp"
#include "recall.hpp"

namespace bin_common {

inline int fail_without_spdk_support() {
  std::cerr << "This binary was built without SPDK support" << std::endl;
  return 1;
}

inline int fail_if_spdk_requested_without_support(const SharedCliArgs& args) {
#if !QUIVER_ENABLE_SPDK
  if (!args.ssd_list_file.empty() && !read_ssd_list(args.ssd_list_file).empty()) {
    return fail_without_spdk_support();
  }
#endif
  (void)args;
  return 0;
}

struct SearchSetup {
  std::string index_file;
  shared::MemoryBackend memory_backend = shared::MemoryBackend::kHeap;
  std::string memory_index_file;
  shared::DataType data_type = shared::FLOAT;
  shared::Layout layout{};
  QueryData queries;
  std::vector<std::string> ssd_lists;
  std::unique_ptr<shared::PQSearch> pq;
  std::unique_ptr<shared::NavGraph> nav;
  std::unique_ptr<int[]> nns;
  std::unique_ptr<float[]> distances;
  std::unique_ptr<int[]> found_counts;
};

inline void validate_query_dimensions_or_die(const QueryData& queries,
                                             const shared::Layout& layout,
                                             const std::string& query_file,
                                             const std::string& index_file) {
  if (queries.dims == static_cast<size_t>(layout.num_dims)) {
    return;
  }

  ERROR(
      "Query dimension mismatch: query file '{}' has {} dims, but index '{}' "
      "expects {} dims",
      query_file, queries.dims, index_file, layout.num_dims);
  std::exit(1);
}

inline SearchSetup prepare_search_setup(const SharedCliArgs& args) {
  SearchSetup setup;
  setup.data_type = parse_data_type(args.data_type);
  setup.index_file = find_index_file(args.index_dir);
  const auto parsed_backend = shared::parse_memory_backend(args.memory_backend);
  if (!parsed_backend.has_value()) {
    ERROR("Unsupported --memory-backend '{}'. Expected one of: heap, mmap",
          args.memory_backend);
    std::exit(1);
  }
  setup.memory_backend = *parsed_backend;
  setup.memory_index_file =
      args.memory_index_file.empty() ? setup.index_file : args.memory_index_file;
  const std::string pq_prefix = find_pq_prefix(args.index_dir);

  INFO("Index: {}", setup.index_file);
  INFO("PQ prefix: {}", pq_prefix);

  parse_diskann_metadata(setup.index_file, setup.layout, setup.data_type);

  setup.pq = std::make_unique<shared::PQSearch>();
  setup.pq->read_data(pq_prefix + "_pivots.bin", pq_prefix + "_compressed.bin");

  const std::string nav_dir = find_nav_dir(args.index_dir);
  if (!nav_dir.empty()) {
    setup.nav = std::make_unique<shared::NavGraph>();
    const int elem_size =
        (setup.data_type == shared::FLOAT) ? sizeof(float) : sizeof(int8_t);
    setup.nav->init(nav_dir + "/nav_index", nav_dir + "/nav_index.data",
                    nav_dir + "/map.txt", elem_size);
    INFO("NavGraph loaded from {}", nav_dir);
  } else {
    INFO("No nav_index found in {}, skipping NavGraph", args.index_dir);
  }

  setup.ssd_lists = read_ssd_list(args.ssd_list_file);
  setup.queries = load_queries_as_float(args.query_file, args.repeat);
  validate_query_dimensions_or_die(setup.queries, setup.layout,
                                   args.query_file, setup.index_file);

  const size_t result_count =
      setup.queries.count * static_cast<size_t>(args.topk);
  setup.nns = std::make_unique<int[]>(result_count);
  setup.distances = std::make_unique<float[]>(result_count);
  setup.found_counts = std::make_unique<int[]>(setup.queries.count);
  return setup;
}

inline std::unique_ptr<uint8_t[]> load_starter_page(const SearchSetup& setup) {
  auto starter = std::make_unique<uint8_t[]>(shared::PAGE_SIZE);
  const std::string& starter_index_file =
      setup.ssd_lists.empty() ? setup.memory_index_file : setup.index_file;
  FILE* file = std::fopen(starter_index_file.c_str(), "rb");
  if (file == nullptr) {
    ERROR("Failed to open file {}", starter_index_file);
    std::exit(1);
  }

  const long page_offset = static_cast<long>(shared::PAGE_SIZE) *
                           (setup.layout.enter_point /
                                setup.layout.nodes_per_page +
                            1);
  if (std::fseek(file, page_offset, SEEK_SET) != 0) {
    ERROR("Failed to seek starter page in {}", starter_index_file);
    std::fclose(file);
    std::exit(1);
  }

  const size_t bytes_read =
      std::fread(starter.get(), 1, shared::PAGE_SIZE, file);
  std::fclose(file);
  if (bytes_read != static_cast<size_t>(shared::PAGE_SIZE)) {
    ERROR("Failed to read starter page from {}", starter_index_file);
    std::exit(1);
  }

  return starter;
}

inline void print_sample_results(const SearchSetup& setup, int topk) {
  const int query_count = static_cast<int>(setup.queries.count);
  for (int i = 0; i < 5 && i < query_count; ++i) {
    for (int j = 0; j < 5 && j < topk; ++j) {
      std::printf("%lf(%d)\t", setup.distances[i * topk + j],
                  setup.nns[i * topk + j]);
    }
    std::printf("\n");
  }
}

inline void print_recall_report(const SharedCliArgs& args,
                                const SearchSetup& setup) {
  if (args.ground_truth_file.empty()) {
    return;
  }

  const auto gt = load_ground_truth(args.ground_truth_file);
  const double recall =
      calc_recall(setup.nns.get(), gt.values.get(), gt.count, gt.width,
                  args.topk);
  std::printf("Recall @ %d: %lf\n", args.topk, recall);
}

inline void write_per_query_recall_csv(const SharedCliArgs& args,
                                       const SearchSetup& setup,
                                       const std::string& csv_path) {
  if (args.ground_truth_file.empty() || csv_path.empty()) {
    return;
  }

  const auto gt = load_ground_truth(args.ground_truth_file);
  std::ofstream out(csv_path);
  if (!out.is_open()) {
    ERROR("Failed to open per-query recall CSV {}", csv_path);
    std::exit(1);
  }

  out << "query_id,gt_query_id,hits,recall\n";
  for (size_t i = 0; i < setup.queries.count; ++i) {
    const size_t gt_query = i % gt.count;
    const auto recall = calc_one_query_recall(
        setup.nns.get() + i * static_cast<size_t>(args.topk),
        gt.values.get() + gt_query * gt.width, args.topk);
    out << i << "," << gt_query << "," << recall.hits << ","
        << std::fixed << std::setprecision(6) << recall.recall << "\n";
  }
  std::cout << "[Quiver] Per-query recall CSV: " << csv_path << std::endl;
}

}  // namespace bin_common
