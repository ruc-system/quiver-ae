#!/usr/bin/env bash
# fig-motivation/io-latency.pdf — 4 SSD, build-ae (LIGHT_BREAKDOWN).
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"
source "${DIR}/paper_knobs.sh"
apply_paper_knobs

BIN_DIR="${INSTRUMENTED_BIN_DIR}"
# This is the only figure that plots per-hop wait/compute intervals.
export AE_HOP_SAMPLES=1
# LIGHT_BREAKDOWN adds enough work at 1B scale that the production startup
# timeout can fire before the first report is emitted.
export HANG_FLOOR_FIRST="${HANG_FLOOR_FIRST:-300}"
export HANG_FLOOR="${HANG_FLOOR:-300}"

require_bin flashanns_search
require_bin gustann_search
require_bin quiver_search
ae_lock
use_dataset sift
OUT="$(prepare_out io_latency)"
cd "${ROOT}"

# Per-hop stall only compares across systems at matched throughput: these are
# the paper's iso-throughput points (POINT_* in common.sh), not one occupancy.
FLASH_BLOCKS="${FLASH_BLOCKS:-${POINT_FLASH_BLOCKS}}"
# Figure 3 uses GustANN's high-multiplexing operating point so panel (b)
# captures the driver/stream queuing tail. Figure 7(b) intentionally keeps
# POINT_GUST_MINI_BATCH=48 for its separate fixed-point breakdown.
GUST_MINI_BATCH="${GUST_MINI_BATCH:-1120}"
QUIVER_BLOCKS="${QUIVER_BLOCKS:-${POINT_QUIVER_BLOCKS}}"
QUIVER_Q="${QUIVER_Q:-${QUERIES_PER_BLOCK}}"
EF="${EF:-${EF_RECALL_90}}"

run_search "${OUT}/flashanns_${DATASET}_ef${EF}_nb${FLASH_BLOCKS}.log" \
  "${BIN_DIR}/bin/flashanns_search" \
  --index-dir "${INDEX_DIR}" \
  --query "${QUERY}" \
  --ground-truth "${GT}" \
  --data-type "${DATA_TYPE}" \
  --topk "${TOPK}" \
  --ef-search "${EF}" \
  --repeat "${REPEAT}" \
  --pipe-width "${PIPE_WIDTH_FLASH}" \
  --poll-threads "${POLL_THREADS}" \
  --ssd-list-file "${SSD_LIST}" \
  --num-blocks "${FLASH_BLOCKS}"

run_search "${OUT}/gustann_${DATASET}_ef${EF}_mb${GUST_MINI_BATCH}.log" \
  "${BIN_DIR}/bin/gustann_search" \
  --index-dir "${INDEX_DIR}" \
  --query "${QUERY}" \
  --ground-truth "${GT}" \
  --data-type "${DATA_TYPE}" \
  --topk "${TOPK}" \
  --ef-search "${EF}" \
  --repeat "${REPEAT}" \
  --pipe-width "${PIPE_WIDTH_GUST}" \
  --worker-threads "${WORKER_THREADS}" \
  --runner-contexts "${RUNNER_CONTEXTS}" \
  --ssd-list-file "${SSD_LIST}" \
  --mini-batch "${GUST_MINI_BATCH}"

run_search "${OUT}/quiver_${DATASET}_ef${EF}_q${QUIVER_Q}_nb${QUIVER_BLOCKS}.log" \
  "${BIN_DIR}/bin/quiver_search" \
  --index-dir "${INDEX_DIR}" \
  --query "${QUERY}" \
  --ground-truth "${GT}" \
  --data-type "${DATA_TYPE}" \
  --topk "${TOPK}" \
  --ef-search "${EF}" \
  --repeat "${REPEAT}" \
  --pipe-width "${PIPE_WIDTH_QUIVER}" \
  --queries-per-block "${QUIVER_Q}" \
  --poll-threads "${POLL_THREADS}" \
  --early-exit-policy none \
  --ssd-list-file "${SSD_LIST}" \
  --num-blocks "${QUIVER_BLOCKS}"

echo "DONE ${OUT}"
