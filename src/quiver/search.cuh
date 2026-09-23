#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "../shared/common/data_type.hpp"
#include "../shared/common/layout.hpp"
#include "../shared/index/nav_graph.hpp"
#include "../shared/index/pq_search.hpp"
#include "../shared/io/loader.hpp"
#include "io_control.cuh"

namespace quiver {

struct PerStepRecallOptions {
    const int* ground_truth = nullptr;
    int ground_truth_width = 0;
    int ground_truth_count = 0;
    float threshold = 0.9f;
    const char* csv_path = nullptr;

    bool enabled() const {
        return ground_truth != nullptr && ground_truth_width > 0 &&
               ground_truth_count > 0 && csv_path != nullptr;
    }
};

struct EarlyExitOptions {
    EarlyExitPolicy policy = EarlyExitPolicy::kDefault;
    int darth_interval = 0;
    float darth_threshold = 0.9f;
    int gpruning_warmup_steps = 0;
    int gpruning_patience = 0;
    const int* darth_ground_truth = nullptr;
    int darth_ground_truth_width = 0;
    int darth_ground_truth_count = 0;
    const char* darth_trace_csv_path = nullptr;

    bool enabled() const {
        return policy != EarlyExitPolicy::kDefault;
    }
};

void run_persistent_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
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
    int queries_per_block = 1,
    int repeat = 1,
    const PerStepRecallOptions& per_step_recall = {},
    const EarlyExitOptions& early_exit = {});

// StaticDispatch (FlashANNS-style) batch search.
// Launches ceil(num_queries / (num_blocks * Q)) kernel batches sequentially,
// each batch processing (num_blocks * Q) queries in lockstep.
//   - Q=1 (default): kernel_q1 fast path (= strawmen_static_dispatch)
//   - Q≥2: kernel_qi with Q slots per block, using a pre-admitted finite
//     StreamingDispatch range per batch for the static ablation.
void run_static_batch_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
    int num_blocks,
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
    int queries_per_block = 1,
    int repeat = 1,
    const EarlyExitOptions& early_exit = {});

#ifdef QUIVER_BUILD_STRAWMEN

inline constexpr int kNaiveBlockMultiQueryThreadsPerQuery = 128;

// Strawman barrier: per-hop block barrier.
// Same signature/semantics as run_persistent_search but launches
// persistent_search_kernel_barrier instead of
// persistent_search_kernel_qi. Used by bin/strawmen_barrier.cu.
//
// Ablation isolation: scheduler discipline only; everything else (streaming
// admission, per-slot resource layout, IO state machine) is unchanged.
void run_barrier_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
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
    int queries_per_block = 2,
    int repeat = 1);

// Strawman PQ-in-shared-memory: PQ LUT in shared memory.
// Same signature as run_persistent_search but routes the per-slot PQ
// distance table into dynamic shared memory instead of global memory.
// On A100 with SIFT-100M (num_chunks=64) this fits at Q≤2 (with reduced
// occupancy) and is expected to fail cudaFuncSetAttribute at Q≥3 because
// 3 × 64 KB exceeds the 164 KB per-block opt-in cap. The launcher prints
// a clean "LAUNCH_ABORTED (smem overflow)" diagnostic in that case.
//
// Ablation isolation: memory class of pq_dists only; scheduler, IO state
// machine, streaming admission identical to Quiver baseline.
void run_pq_in_smem_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
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
    int queries_per_block = 2,
    int repeat = 1);

// Strawman PQ-in-global-memory / explicit context spill-load.
// Same scheduler and IO state machine as Quiver Q-interleave, but no per-slot
// search context is kept resident in shared memory across slot switches.  Each
// scheduled slot explicitly loads its full context from global memory into one
// block-local shared workspace, processes ready IO, then writes the context
// back to global memory before the scheduler can switch to another slot.
void run_pq_in_gmem_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
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
    int queries_per_block = 2,
    int repeat = 1);

// Strawman naive block multi-query: one 128-thread query executor per query
// slot. Q=1 is the original single-query resource baseline; Q>1 naively
// duplicates that executor and its resources inside one CUDA block.
//
// Ablation isolation: block-thread topology + intra-block scheduler only;
// streaming admission, per-slot resource layout, IO state machine, and
// PQ memory class identical to Quiver.
void run_naive_block_multi_query_search(
    const shared::Layout& layout,
    shared::DataType data_type,
    int pipe_w,
    int thread_cnt,
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
    int queries_per_block = 1,
    int repeat = 1);

#endif  // QUIVER_BUILD_STRAWMEN

} // namespace quiver
