#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "shared/common/data_type.hpp"
#include "shared/common/layout.hpp"
#include "shared/index/nav_graph.hpp"
#include "shared/index/pq_search.hpp"
#include "shared/io/loader.hpp"
#include "io_control.cuh"

namespace flashanns {

void run_persistent_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
    int task_context_count,
    int mini_batch,
    std::shared_ptr<shared::IndexLoader> loader,
    shared::PQSearch* pq,
    shared::NavGraph* nav,
    int enter_point,
    uint8_t* starter,
    const float* qdata,
    int num_queries,
    int topk,
    int ef_search,
    int* nns,
    float* distances,
    int* found_cnt,
    int repeat = 1);

} // namespace flashanns
