#pragma once

#include <memory>
#include <string>
#include <vector>

#include "shared/common/data_type.hpp"
#include "shared/common/layout.hpp"
#include "shared/index/nav_graph.hpp"
#include "shared/index/pq_search.hpp"
#include "shared/io/loader.hpp"

namespace gustann {

struct HybridExecutorConfig {
  int mini_batch;
  int thread_cnt;
  int ctx_per_thread;
  int pipe_w = 1;
  enum {
    SPDK,
    MEMORY,
  } use_backend;
  std::vector<std::string> ssd_lists;
  shared::MemoryBackend memory_backend = shared::MemoryBackend::kHeap;
  std::string memory_index_file;
};

class HybridExecutor {
 public:
  HybridExecutor(const shared::Layout &layout,
                 const shared::DataType &data_type, const std::string &fpath,
                 const HybridExecutorConfig &config);
  ~HybridExecutor() = default;
  HybridExecutor(const HybridExecutor &) = delete;
  HybridExecutor &operator=(const HybridExecutor &) = delete;
  HybridExecutor(HybridExecutor &&) = delete;
  HybridExecutor &operator=(HybridExecutor &&) = delete;

  void search(const float *qdata, int num_queries, int topk, int ef_search,
              int *nns, float *distances, int *found_cnt,
              shared::PQSearch *pq = nullptr,
              shared::NavGraph *nav = nullptr,
              int repeat = 1);

 private:
  shared::Layout layout_;
  shared::DataType data_type_;
  std::unique_ptr<uint8_t[]> starter_;
  std::shared_ptr<shared::IndexLoader> loader_;
  int mini_batch_;
  int thread_cnt_;
  int ctx_per_thread_;
  int pipe_w_;
};

}  // namespace gustann
