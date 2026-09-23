#include "index/hbc_progress.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace {

struct HbcProgressState {
  std::string label;
  size_t total = 0;
  size_t current = 0;
  size_t last_printed = std::numeric_limits<size_t>::max();
  bool active = false;
};

struct HbcProgressContext {
  std::vector<HbcProgressState> stack;
  bool line_dirty = false;
};

HbcProgressContext &MutableProgressContext() {
  static HbcProgressContext ctx;
  return ctx;
}

HbcProgressState *ActiveState(HbcProgressContext &ctx) {
  if (ctx.stack.empty()) {
    return nullptr;
  }
  return &ctx.stack.back();
}

void FlushLineLocked(HbcProgressContext &ctx) {
  if (ctx.line_dirty) {
    std::cout << std::endl;
    ctx.line_dirty = false;
  }
}

void HbcEmitProgressLocked(HbcProgressContext &ctx) {
  HbcProgressState *state = ActiveState(ctx);
  if (!state || !state->active) {
    return;
  }
  const std::string &label = state->label.empty() ? "bkt" : state->label;
  std::cout << "\r    [" << label << "] progress: " << state->current << "/"
            << state->total << std::flush;
  ctx.line_dirty = true;
  state->last_printed = state->current;
}

} // namespace

void HbcFlushProgressBeforeLog() {
  HbcProgressContext &ctx = MutableProgressContext();
  FlushLineLocked(ctx);
}

void HbcProgressStartLabeled(const std::string &label, size_t total) {
#ifdef _OPENMP
#pragma omp critical(hbc_log)
#endif
  {
    HbcProgressContext &ctx = MutableProgressContext();
    FlushLineLocked(ctx);

    HbcProgressState state;
    state.label = label;
    state.total = total;
    state.current = 0;
    state.active = total > 0;
    state.last_printed = std::numeric_limits<size_t>::max();
    ctx.stack.push_back(std::move(state));

    if (ctx.stack.back().active) {
      HbcEmitProgressLocked(ctx);
    }
  }
}

void HbcProgressStart(size_t total) {
  HbcProgressStartLabeled("bkt", total);
}

void HbcProgressAdvance(size_t delta) {
#ifdef _OPENMP
#pragma omp critical(hbc_log)
#endif
  {
    HbcProgressContext &ctx = MutableProgressContext();
    HbcProgressState *state = ActiveState(ctx);
    if (state && state->active && state->total != 0) {
      size_t next = state->current + delta;
      if (next > state->total) {
        next = state->total;
      }
      if (next != state->current) {
        state->current = next;
        HbcEmitProgressLocked(ctx);
      }
    }
  }
}

void HbcProgressFinish() {
#ifdef _OPENMP
#pragma omp critical(hbc_log)
#endif
  {
    HbcProgressContext &ctx = MutableProgressContext();
    if (!ctx.stack.empty()) {
      HbcProgressState &state = ctx.stack.back();
      if (state.active) {
        state.current = state.total;
        HbcEmitProgressLocked(ctx);
      }
      FlushLineLocked(ctx);
      ctx.stack.pop_back();
      if (!ctx.stack.empty()) {
        HbcEmitProgressLocked(ctx);
      }
    }
  }
}
