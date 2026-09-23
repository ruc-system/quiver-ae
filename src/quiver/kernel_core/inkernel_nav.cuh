#pragma once

// In-kernel nav graph entry-point search.
//
// Replaces the external shared::get_entry_kernel + nav_translate_kernel
// launches.  Each call computes one query's entry point using ALL 128
// threads of the calling block (4 warps), reading qdata BEFORE it is
// modified by PQ init_query (which subtracts centroid in place).
//
// Algorithm = port of
//   src/flashanns/shared/index/nav_kernel.cuh :: get_entry_kernel_inner
// re-laid-out so that the R=28 neighbor distance computations are
// distributed evenly across the block's warps.  The selection step
// (a 32-element warp-shuffle sort restricted to L=4 promotions) stays
// on warp 0: it is already optimal for L+R=32 elements and adding more
// warps to it would only add inter-warp sync.
//
// Speedup vs the previous warp-only version (called under
// `if (tid < 32)`):
//   distance phase: ~4× fewer warp-time per hop (28/4 = 7 distances
//                   per warp instead of 28 sequential on warp 0)
//   selection phase: unchanged (warp 0 does L=4 reductions)
//   The block-wide scope also lets us drop the caller-side
//   `if (tid < 32)` guard and the s_nav_result broadcast: the function
//   itself returns a uniform value to every thread.
//
// Caller contract:
//   - Must be entered by EVERY thread in the block (uniform control flow
//     up to and through this call).  blockDim.x must be a multiple of 32.
//   - Must be called BEFORE PQ init_query mutates qdata in place.
//   - pq_id / pq_dist arrays must be at least L+R = 32 elements each.
//   - s_idx_scratch is a single int in shmem used to broadcast the
//     next-pivot index from warp 0 to all warps each loop iteration.
//
// Shared-memory scratch (caller-provided):
//   pq_id          : uint32_t[L + R]    (256 B)
//   pq_dist        : float   [L + R]    (256 B; can overlap nav_pq_id)
//   s_idx_scratch  : int     [1]        (cross-warp broadcast)
//
// Return value: every thread in the block receives the mapped base-graph
// entry node id for the best nav candidate.

#include <cuda_runtime.h>
#include <cstdint>

#include "../../shared/common/gpu_primitives.cuh"

namespace quiver {

// Matches flashanns/shared/index/nav_kernel.cuh constants.
constexpr int kInkernelNavL = 4;
constexpr int kInkernelNavR = 28;

// data_type: base graph data type (uint8_t / float).  Nav graph stores
// vectors in the same dtype (see NavGraph::init in
// shared/index/nav_graph.cu).
//
// BLOCK-LEVEL: must be entered by all blockDim.x threads of the block
// with uniform control flow.
template <class T>
__device__ __forceinline__ int inkernel_nav_entry(
    const float* qdata,           // [num_dims]
    const T*     nav_data,        // [num_node * num_dims]
    const int*   nav_graph,       // [num_node * max_m]
    const int*   nav_mapping,     // [num_node], nav id -> base graph node id
    int          num_dims,
    int          max_m,
    int          init_ef,         // == min(ef_search, 5); currently unused
    int          entry_root,      // args.nav_start
    uint32_t*    pq_id,           // shmem [L + R]
    float*       pq_dist,         // shmem [L + R]
    int*         s_idx_scratch)   // shmem [1] for cross-warp idx broadcast
{
    (void)init_ef;  // parity with get_entry_kernel_inner; ef implicit in L/R
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane      = tid & 31;
    const int kNumWarps = blockDim.x >> 5;
    const int L = kInkernelNavL;
    const int R = kInkernelNavR;

    // ---- Init: warp 0 seeds the priority queue ----
    if (warp_id == 0) {
        float d0 = square_sum_32<const T>(
            qdata, nav_data + (long)num_dims * entry_root, num_dims);
        if (lane == 0) {
            pq_id[0]   = (uint32_t)entry_root;
            pq_dist[0] = d0;
        } else if (lane < L + R) {
            pq_id[lane]   = 0xffffffffu;
            pq_dist[lane] = INFINITY;
        }
    }
    __syncthreads();

    int idx = 0;
    while (idx != -1) {
        // Read u from the chosen pivot slot.  pq_id[idx] is consistent
        // here because the previous loop iteration's selection (and the
        // init phase for the very first iteration) ended in __syncthreads.
        const int u = (int)(pq_id[idx] & 0x7fffffffu);
        if (tid == 0) {
            pq_id[idx] |= 0x80000000u;  // mark visited
        }
        // No sync needed before distance phase: the visited bit is only
        // read by the *next* selection phase, and the distance phase
        // writes to non-overlapping slots [L, L+R).

        // ---- Distance phase: distribute R neighbors across warps ----
        // For R=28, kNumWarps=4: each warp processes 7 contiguous slots.
        // Each `square_sum_32` is warp-local (uses __shfl_down_sync /
        // __syncwarp), so warps run independently in this phase.
        const int* edge   = nav_graph + (long)max_m * u;
        const int per_warp = (R + kNumWarps - 1) / kNumWarps;
        const int my_start = warp_id * per_warp;
        const int my_end   = (my_start + per_warp < R)
                             ? (my_start + per_warp) : R;
        for (int i = my_start; i < my_end; ++i) {
            int v = edge[i];
            if (v == -1) {
                // Match get_entry_kernel_inner's `break` semantic per warp:
                // leftover slots in [i+L, my_end+L) keep their previous
                // values (either INF from init, or visited from a previous
                // hop; selection's duplicate-id check filters those).
                // Per-warp break is identical to single-warp break for
                // tail-padded adjacency lists (the common case).
                break;
            }
            float d = square_sum_32<const T>(
                qdata, nav_data + (long)num_dims * v, num_dims);
            if (lane == 0) {
                pq_id[i + L]   = (uint32_t)v;
                pq_dist[i + L] = d;
            }
        }
        __syncthreads();

        // ---- Selection phase: warp 0 only ----
        // Promotes the L smallest unvisited candidates to positions [0, L).
        // L=4 reductions over 32 lanes: warp-shuffle is already optimal,
        // adding more warps here would just add inter-warp sync.
        if (warp_id == 0) {
            #pragma unroll
            for (int i = 0; i < L; ++i) {
                float dist = (lane < L + R) ? pq_dist[lane] : INFINITY;
                int   pos  = lane;
                if (lane < i) {
                    dist = INFINITY;
                } else if (i != 0 && lane < L + R &&
                           ((pq_id[lane] ^ pq_id[i - 1]) & 0x7fffffffu) == 0) {
                    // Duplicate id with last-round selected: merge the
                    // visited-bit and drop from consideration this round.
                    pq_id[i - 1]  |= pq_id[lane];
                    pq_dist[lane]  = INFINITY;
                    dist           = INFINITY;
                }
                __syncwarp();

                #pragma unroll
                for (int offset = 32 / 2; offset > 0; offset /= 2) {
                    int   _id    = __shfl_down_sync(0xffffffff, pos, offset);
                    float _value = __shfl_down_sync(0xffffffff, dist, offset);
                    if (_value < dist) {
                        dist = _value;
                        pos  = _id;
                    }
                }
                __syncwarp();

                if (lane == 0 && i != pos) {
                    float t = pq_dist[i];
                    float y = pq_dist[pos];
                    if (t != y) {
                        pq_dist[i]   = y;
                        pq_dist[pos] = t;
                        uint32_t m   = pq_id[i];
                        pq_id[i]     = pq_id[pos];
                        pq_id[pos]   = m;
                    }
                }
                __syncwarp();
            }

            // Find first unvisited slot in [0, L); broadcast to all warps
            // via shmem (shfl_sync would only reach warp 0).
            uint32_t my_id = (lane < L + R) ? pq_id[lane] : 0x80000000u;
            int val = __ballot_sync(0xffffffff, !(my_id & 0x80000000u));
            if (lane == 0) {
                int x = __clz(__brev(val));
                *s_idx_scratch = (x < L) ? x : -1;
            }
        }
        __syncthreads();
        idx = *s_idx_scratch;
    }

    // pq_id[0] is now the best nav entry. Map it to the base-graph node here
    // so callers avoid an extra shared-memory result slot and block sync.
    int raw = (int)(pq_id[0] & 0x7fffffffu);
    return nav_mapping[raw];
}

}  // namespace quiver
