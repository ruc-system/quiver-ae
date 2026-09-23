#!/usr/bin/env bash
# Hello-world smoke test: one Quiver search on SIFT, one occupancy point.
#   ./ae/scripts/hello_world.sh
# Runs in ~1 minute and prints Recall@10 and QPS. It does not reproduce any
# paper figure; it only proves the machine, SPDK bindings and index are usable.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

AE_SCALE=1m
REPEAT=1
NUM_BLOCKS_LIST=108
# SPDK/EAL cold start can exceed the global 22-second first-point watchdog.
HANG_FLOOR_FIRST="${HANG_FLOOR_FIRST:-60}"

require_bin quiver_search
ae_lock >/dev/null
use_dataset sift1m
HELLO_WORLD_EF="${HELLO_WORLD_EF:-19}"
OUT="$(prepare_out hello_world)"
cd "${ROOT}"

LOG="${OUT}/quiver_${DATASET}_ef${HELLO_WORLD_EF}.log"
run_search "${LOG}" \
  "${BIN_DIR}/bin/quiver_search" \
  --index-dir "${INDEX_DIR}" \
  --query "${QUERY}" \
  --ground-truth "${GT}" \
  --data-type "${DATA_TYPE}" \
  --topk "${TOPK}" \
  --ef-search "${HELLO_WORLD_EF}" \
  --repeat "${REPEAT}" \
  --pipe-width "${PIPE_WIDTH_QUIVER}" \
  --queries-per-block "${QUERIES_PER_BLOCK}" \
  --poll-threads "${POLL_THREADS}" \
  --early-exit-policy none \
  --ssd-list-file "${SSD_LIST}" \
  --num-blocks-list "${NUM_BLOCKS_LIST}" \
  >/dev/null

echo "===== hello-world result ====="
python3 - "${LOG}" <<'PY'
import re
import sys

text = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[1]).read())

def value(pattern):
    match = re.search(pattern, text)
    if not match:
        sys.exit(f"missing result field in {sys.argv[1]}: {pattern}")
    return float(match.group(1))

recall = value(r"Recall @ 10:\s*([\d.]+)")
qps = value(r"QPS:\s*([\d.]+)")
avg = value(r"QueryLatency\(ms\):\s*avg=([\d.]+)")
p99 = value(r"QueryLatency\(ms\):.*?\bp99=([\d.]+)")

print("Dataset: SIFT-1M")
print("Sweep: num_blocks=108 queries_per_block=2")
print(f"  Recall @10 = {recall:.4f}")
print(f"  QPS = {qps:.1f}  avg = {avg:.2f} ms  P99 = {p99:.2f} ms")
PY
echo "==============================="
