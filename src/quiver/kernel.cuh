#pragma once

// Quiver kernel entry point (Phase 1 refactor aggregator).
//
// This file used to contain the 4 kernel helper functions
// (persistent_merge_data, persistent_visit_select, persistent_select_next,
// issue_request) plus the two persistent kernels (Q=1 fast path and Q-interleave).
// Phase 1 of the refactor split these into three separate files under kernel_core/:
//
//   kernel_core/kernel_helpers.cuh  — 4 helper __device__ functions
//   kernel_core/kernel_q1.cuh       — persistent_search_kernel (Q=1 streaming)
//   kernel_core/kernel_qi.cuh       — persistent_search_kernel_qi (Q-interleave)
//
// This file is kept as a compatibility aggregator so that existing callers
// (currently only quiver/search.cu) do not need to change their #include.

#include "kernel_core/kernel_helpers.cuh"
#include "kernel_core/kernel_q1.cuh"
#include "kernel_core/kernel_qi.cuh"
