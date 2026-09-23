#pragma once

// ============================================================================
// Dispatch policies: how a query ID is bound to a (block, slot) pair.
//
// Phase 3.1a: StreamingDispatch introduced (kernel template stub).
// Phase 3.1b: StaticDispatch introduced. Host launches one batch at a time,
//             updates args.batch_offset between launches, and synchronizes
//             between batches (FlashANNS-style lockstep).
//
// Design notes:
//  - Policy methods are `__device__ static __forceinline__` so the compiler
//    inlines them into the kernel and dead-code-eliminates any constexpr
//    branches (`if constexpr (Dispatch::is_persistent)`).
//  - The policy does NOT own state; all runtime state lives in
//    PersistentKernelArgs / PersistentKernelComm.
// ============================================================================

#include "../../io_control.cuh"

namespace quiver {

// ----------------------------------------------------------------------------
// StreamingDispatch
//   - Outer `while (true)` loop: each block consumes queries until exit.
//   - Acquires next qid via atomicAdd(args.d_next_query).
//   - Spin-waits on comm->feed_count for CPU to admit the query.
//   - Exits when comm->terminate is set while waiting.
//
// Semantics of acquire_next_qid():
//   - Must be called with __syncthreads() after.  Only tid==0 does real work;
//     other threads see the result via the shared slot written into
//     *s_query_id_out.
//   - Returns the qid on success, or -1 to signal "kernel should exit".
//
// The kernel is expected to write the result to a shared int `s_query_id`
// before calling __syncthreads(), then read it on every thread.
// ----------------------------------------------------------------------------
struct StreamingDispatch {
    // Whether the kernel's outer query loop should run more than once.
    static constexpr bool is_persistent = true;

    // Only thread 0 in the block should call this.
    //
    // Writes the acquired qid (or -1 on terminate) to *s_query_id_out.
    // Caller must then __syncthreads() to broadcast to the whole block.
    __device__ static __forceinline__ void
    acquire_next_qid_tid0(PersistentKernelArgs& args,
                          int /*bid*/, int /*slot*/,
                          int* s_query_id_out)
    {
        int qid = atomicAdd(args.d_next_query, 1);
        *s_query_id_out = qid;

        // Spin-wait until CPU has fed this query, or terminate signal.
        while (qid >= *(volatile int32_t*)&args.comm->feed_count) {
            if (*(volatile int32_t*)&args.comm->terminate) {
                *s_query_id_out = -1;  // signal exit
                break;
            }
        }
        // ---- Cross-stream memory acquire fence ----
        // The CPU produces d_all_qdata[qid] via cudaMemcpyAsync on
        // h2d_stream, then bumps feed_count (zero-copy volatile write from
        // host).  This persistent kernel runs on a DIFFERENT stream, so the
        // CUDA runtime does NOT establish a dependency between the h2d DMA
        // and this kernel's memory accesses.  Without a fence, the L2 / L1
        // caches visible to this thread may still hold stale (or
        // uninitialized) lines for d_all_qdata[qid], causing nav / PQ init
        // to read garbage and the search to wedge (no SSD IO issued,
        // kernel appears alive but makes no progress).
        //
        // __threadfence_system() is the CUDA-documented acquire fence
        // that pairs with the host-side write + zero-copy visibility:
        // once feed_count is observed, this fence guarantees subsequent
        // reads of d_all_qdata / d_entry_nodes see the DMA-written values.
        __threadfence_system();
    }
};

// ----------------------------------------------------------------------------
// StaticDispatch (FlashANNS-style batch launch)
//   - Each kernel launch processes exactly one batch of `num_blocks` blocks.
//   - In Q=1 mode: block's single query is qid = batch_offset + blockIdx.x.
//   - In Q≥2 mode: block holds Q statically-bound qids:
//                  qid[q] = batch_offset + blockIdx.x * Q + q  for q ∈ [0, Q)
//     where Q = args.queries_per_block.
//   - Each qid is bound at most once per kernel launch. Once a slot finishes
//     its qid, it is permanently retired; the slot will NOT pick a new qid
//     within this kernel launch.
//   - Host is responsible for launching ceil(total / (num_blocks*Q)) batches,
//     bumping args.batch_offset by num_blocks*Q each time, and synchronizing
//     between batches.
// ----------------------------------------------------------------------------
struct StaticDispatch {
    // Kernel's outer loop runs only one iteration; a constexpr guard in the
    // kernel body returns after the first query.
    // Q=1 case: the outer while(true) in kernel_q1 exits after one query.
    // Q≥2 case: kernel_qi exits after all Q slots have retired.
    static constexpr bool is_persistent = false;

    // Q=1 entry: keep the original signature (used by kernel_q1).
    // `slot` is unused for Q=1.
    __device__ static __forceinline__ void
    acquire_next_qid_tid0(PersistentKernelArgs& args,
                          int bid, int /*slot*/,
                          int* s_query_id_out)
    {
        const int qid = args.batch_offset + bid;
        // Out-of-range in the final (possibly smaller) batch: exit signal.
        *s_query_id_out = (qid < args.total_queries) ? qid : -1;
    }

    // Q≥2 entry: used by kernel_qi. Returns the qid for (bid, slot_q) or -1
    // if out of range (final batch may have fewer queries than num_blocks*Q).
    // Each (bid, slot_q) pair is valid exactly once per kernel launch; the
    // caller must ensure it doesn't call this twice for the same slot.
    __device__ static __forceinline__ int
    static_qid_for_slot(const PersistentKernelArgs& args, int bid, int slot_q)
    {
        const int Q = args.queries_per_block;
        const int qid = args.batch_offset + bid * Q + slot_q;
        return (qid < args.total_queries) ? qid : -1;
    }
};

} // namespace quiver
