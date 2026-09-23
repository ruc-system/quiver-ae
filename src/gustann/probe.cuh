#pragma once

#include <cstdint>

namespace gustann {

#ifdef GUSTANN_LATENCY_PROBE

static constexpr int GUSTANN_PROBE_MAX_SAMPLES = 128;

__device__ __forceinline__ int64_t gustann_globaltimer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return static_cast<int64_t>(t);
}

struct GustannProbeSample {
  int32_t hop_index = 0;
  int32_t sublane = 0;
  int64_t t_kernel_start = 0;
  int64_t t_merge_start = 0;
  int64_t t_merge_done = 0;
  int64_t t_visit_start = 0;
  int64_t t_compute_done = 0;
};

struct GustannProbeQueryTrace {
  int32_t query_id = -1;
  int32_t num_samples = 0;
  GustannProbeSample samples[GUSTANN_PROBE_MAX_SAMPLES];
};

#endif  // GUSTANN_LATENCY_PROBE

}  // namespace gustann
