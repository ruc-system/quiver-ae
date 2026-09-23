# Query Pipeline Optimization Design

## Context

Profiling on sift1m (warmup=10000, cache hot, threads=1) shows avg query time 0.284ms:

| Stage | Avg Time | % |
|-------|----------|---|
| Graph Search (CPU) | 91.1 µs | 32.0% |
| Fetch (GPU/Wait) | 72.8 µs | 25.6% |
| Rerank (Mixed) | 57.7 µs | 20.3% |
| Dedup (CPU) | 24.3 µs | 8.6% |
| Gather (CPU) | 18.5 µs | 6.5% |
| PQ Scan (GPU) | 6.6 µs | 2.3% |
| PQ Prep (GPU) | 5.8 µs | 2.0% |
| Sort (CPU) | 3.5 µs | 1.2% |

Three independent optimizations, each on a separate branch for isolated comparison.

---

## Branch 1: `opt/double-buffer-pipeline`

### Goal

Overlap Query N's CPU-heavy tail (FetchResults wait + Rerank) with Query N+1's CPU preparation (GraphSearch + Gather + Dedup) and GPU work (DistTable + PQ Scan).

### Current flow (single query per thread, serial)

```
Q_N: [GPU Prep] [GraphSearch] [Gather] [Dedup] [GPU PQ] [FetchWait] [Rerank] | Q_N+1: ...
      ^^^^^^^^                                   ^^^^^^   ^^^^^^^^^   ^^^^^^
      2.0%                                       2.3%     25.6%       20.3%    = 50.2% could overlap
```

### Proposed flow (double buffer, pipelined)

```
Block_curr: [GPU_N] ─────────── [FetchWait_N] [Rerank_N]         [GPU_N+2] ─── ...
Block_next:         [GraphSearch_N+1] [Gather] [Dedup] [GPU_N+1] ─────────── [FetchWait_N+1] [Rerank_N+1]
```

Each thread holds two GPU blocks (`block_curr`, `block_next`) from the existing pool. While `block_curr` is in FetchWait + Rerank, the CPU prepares the next query and launches GPU work on `block_next`. After Rerank finishes, swap blocks.

### Memory impact

- Pool has 128,969 blocks, 48 threads × 2 = 96 blocks needed. No change to pool sizing.
- Two sets of pinned buffers per thread (h_query, h_candidate) — add second `ThreadPinnedBuffers`.
- Two sets of CUDA events per thread.

### Key changes

**File: `src/online/benchmark_runner.cpp`**

1. `process_task` lambda refactored into a state machine or two-phase function:
   - Phase A (CPU+GPU dispatch): GraphSearch → Gather → Dedup → score_candidates (async GPU)
   - Phase B (CPU blocking): FetchResults (sync) → Rerank → return results
2. Main loop becomes:
   ```
   phase_A(query_0, block_a, pinned_a, events_a)  // prime the pipeline
   for query_1..N:
       phase_A(query_i, block_b, pinned_b, events_b)  // CPU prep + GPU launch
       phase_B(query_i-1, block_a, pinned_a, events_a) // wait + rerank previous
       swap(a, b)
   phase_B(last_query, block_a, pinned_a, events_a)    // drain
   ```
3. Resource allocation: acquire 2 blocks, 2 pinned buffers, 2 event sets per thread.

### Expected improvement

- GraphSearch(91µs) + Gather(18.5µs) + Dedup(24.3µs) = 134µs of CPU work can overlap with FetchWait(72.8µs) + Rerank(57.7µs) = 130.5µs
- Near-perfect overlap: ~130µs hidden → effective query time drops from ~284µs to ~154µs
- Theoretical speedup: ~1.8× single-thread QPS

### Risks

- Complexity in state management (two active queries per thread)
- Timing/profiling stats need to track which block belongs to which query
- Edge cases: first query (no overlap) and last query (drain)

---

## Branch 2: `opt/bitset-dedup`

### Goal

Replace `boost::unordered_flat_set` deduplication with a `thread_local` bitset. Merge Gather and Dedup into a single pass.

### Current implementation

```cpp
// Gather: O(N) push_back to candidate_ids
for (centroid_id : probe.ids) {
    view = metadata.view(centroid_id);
    for (j : view) candidate_ids.push_back(view.data[j]);
}
// Dedup: O(N) hash set insert, write uniques to pinned
boost::unordered_flat_set<long long> seen;
seen.reserve(num_candidates * 2);
for (id : candidate_ids) {
    if (seen.insert(id).second) h_candidate_pinned[unique++] = id;
}
```

Two passes over data, hash set allocation + probing overhead.

### Proposed implementation

```cpp
// Single-pass Gather+Dedup with bitset
static thread_local std::vector<uint64_t> seen_bits;  // ceil(max_id/64) words
// reset: memset 0 (only the used portion tracked by high-water mark)
size_t unique = 0;
for (centroid_id : probe.ids) {
    view = metadata.view(centroid_id);
    for (j = 0; j < view.size; ++j) {
        int32_t id = view.data[j];
        size_t word = id >> 6, bit = id & 63;
        if (!(seen_bits[word] & (1ULL << bit))) {
            seen_bits[word] |= (1ULL << bit);
            h_candidate_pinned[unique++] = id;
        }
    }
}
// cleanup: reset only touched words (track min/max word indices)
```

### Memory

- sift1m: max_id ~1M → 128KB bitset per thread, fits in L2 cache
- random110m: max_id ~110M → ~13MB per thread. Too large for cache; fall back to hash set for large datasets.
- Decision: use bitset when `max_vector_id < threshold` (e.g., 16M → 2MB bitset), else existing hash set.

### Key changes

**File: `src/online/benchmark_runner.cpp`**

1. Add `max_vector_id` to `BenchmarkConfig` or derive from metadata.
2. In `process_task`, replace separate Gather + Dedup with fused single-pass loop.
3. Add threshold-based dispatch (bitset vs hash set).
4. Track min/max word index for efficient reset (avoid memset of entire bitset).

### Expected improvement

- Eliminate intermediate `candidate_ids` vector entirely
- Single pass instead of two
- Bitset check is 1 memory load + bit test vs hash probe (multiple loads + hash compute)
- Estimated: Gather(18.5µs) + Dedup(24.3µs) = 42.8µs → ~15-20µs
- Savings: ~20-25µs per query (~8% of total)

### Risks

- Only beneficial when max_vector_id is bounded and small enough for L2-resident bitset
- Need to expose max_vector_id from PostingListAccessor or metadata

---

## Branch 3: `opt/nprobe-sweep`

### Goal

Find optimal nprobe for sift1m by scanning nprobe=[16, 24, 32, 40, 48] and measuring QPS vs Recall@10.

### Implementation

No code changes. Create a shell script that runs the query server with different nprobe values and collects results.

**File: `scripts/nprobe_sweep.sh`**

```bash
for nprobe in 16 24 32 40 48; do
    run query_server --nprobe $nprobe ... | grep "qps_eff\|Recall"
done
```

### Expected output

Table of nprobe → {QPS_eff, QPS_wall, avg_latency, p99_latency, Recall@10}

Lower nprobe reduces:
- GraphSearch time (fewer centroids to visit)
- Gather time (fewer posting lists)
- Dedup time (fewer candidates)
- GPU PQ time (fewer candidates to score)
- Rerank I/O (fewer approximate results to rerank)

### Risks

- None (parameter-only change, easily reversible)

---

## Branch strategy

```
main ─────────────────────────────────────────────────
  ├── opt/double-buffer-pipeline  (Branch 1)
  ├── opt/bitset-dedup            (Branch 2)
  └── opt/nprobe-sweep            (Branch 3)
```

Each branch starts from current main. Performance comparison runs the same benchmark (sift1m, warmup=10000, threads=48, nprobe=48, rerank_size=512) on each branch + main baseline.

## Success criteria

- Branch 1: measurable QPS improvement with no Recall regression
- Branch 2: reduced Gather+Dedup time visible in NVTX profiling
- Branch 3: nprobe vs Recall@10 trade-off table for informed parameter selection
