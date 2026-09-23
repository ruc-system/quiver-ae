#!/usr/bin/env bash
# Paper Figure 5 (e2e). Official AE path is this reduced occupancy sweep
# (Q=1: 3 nbs, Q=2: 8 nbs, Flash: 10 nbs, Gust: 8 mbs; DEEP adds Q=3/4).
# The plotter takes the Pareto. This is not a hidden wider grid.
#
# SIFT and DEEP share:
#   Q=1  num_blocks = 108 x 1..3
#   Q=2  num_blocks = 108 x 2..9
#   FlashANNS num_blocks = 108 x 1..10, pipe_width=2
#   GustANN mini_batch = 16,24,32,48,64,96,128,192, pipe_width=1
# DEEP additionally measures Q=3 and Q=4 at num_blocks=864,972.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"
source "${DIR}/paper_knobs.sh"
apply_paper_knobs

require_bin flashanns_search
require_bin gustann_search
require_bin quiver_search
ae_lock
OUT="$(prepare_out e2e)"
cd "${ROOT}"

E2E_DATASETS="${E2E_DATASETS:-sift deep}"
E2E_RECALLS="${E2E_RECALLS:-90 96}"
E2E_SYSTEMS="${E2E_SYSTEMS:-flashanns gustann quiver}"

has_word() {
  local wanted="$1" word
  for word in $2; do
    [[ "${word}" == "${wanted}" ]] && return 0
  done
  return 1
}

run_flash() {
  local ef="$1"
  run_search "${OUT}/flashanns_${DATASET}_ef${ef}_pw2.log" \
    "${BIN_DIR}/bin/flashanns_search" \
    --index-dir "${INDEX_DIR}" --query "${QUERY}" --ground-truth "${GT}" \
    --data-type "${DATA_TYPE}" --topk "${TOPK}" --ef-search "${ef}" \
    --repeat "${REPEAT}" --pipe-width 2 --poll-threads "${POLL_THREADS}" \
    --ssd-list-file "${SSD_LIST}" --num-blocks-list "${NUM_BLOCKS_LIST}"
}

run_gust() {
  local ef="$1"
  run_search "${OUT}/gustann_${DATASET}_ef${ef}_pw1.log" \
    "${BIN_DIR}/bin/gustann_search" \
    --index-dir "${INDEX_DIR}" --query "${QUERY}" --ground-truth "${GT}" \
    --data-type "${DATA_TYPE}" --topk "${TOPK}" --ef-search "${ef}" \
    --repeat "${REPEAT}" --pipe-width 1 \
    --worker-threads "${WORKER_THREADS}" --runner-contexts "${RUNNER_CONTEXTS}" \
    --ssd-list-file "${SSD_LIST}" --mini-batch-list "${MINI_BATCH_LIST}"
}

run_quiver() {
  local ef="$1" q="$2" nbs="$3"
  run_search "${OUT}/quiver_${DATASET}_ef${ef}_q${q}_pw2.log" \
    "${BIN_DIR}/bin/quiver_search" \
    --index-dir "${INDEX_DIR}" --query "${QUERY}" --ground-truth "${GT}" \
    --data-type "${DATA_TYPE}" --topk "${TOPK}" --ef-search "${ef}" \
    --repeat "${REPEAT}" --pipe-width 2 --queries-per-block "${q}" \
    --poll-threads "${POLL_THREADS}" --early-exit-policy none \
    --ssd-list-file "${SSD_LIST}" --num-blocks-list "${nbs}"
}

run_one() {
  local family="$1"
  local ef="$2"
  local extra_q="${3:-}"
  use_dataset "${family}"

  has_word flashanns "${E2E_SYSTEMS}" && run_flash "${ef}"
  has_word gustann "${E2E_SYSTEMS}" && run_gust "${ef}"
  if has_word quiver "${E2E_SYSTEMS}"; then
    run_quiver "${ef}" "${QUIVER_LOWLAT_Q}" "${QUIVER_LOWLAT_NB}"
    run_quiver "${ef}" "${QUIVER_MID_Q}" "${QUIVER_MID_NB}"
    local q
    for q in ${extra_q}; do
      run_quiver "${ef}" "${q}" "${QUIVER_DEEP_EXTRA_NB}"
    done
  fi
}

# SIFT_DS / DEEP_DS default to sift/deep (follow AE_SCALE). Override with
# sift1b+deep1m to mix scales in one Figure 5 run.
SIFT_DS="${SIFT_DS:-sift}"
DEEP_DS="${DEEP_DS:-deep}"

if has_word sift "${E2E_DATASETS}"; then
  use_dataset "${SIFT_DS}"
  has_word 90 "${E2E_RECALLS}" && run_one "${SIFT_DS}" "${EF_RECALL_90}"
  has_word 96 "${E2E_RECALLS}" && run_one "${SIFT_DS}" "${EF_RECALL_96}"
fi
if [[ "${SKIP_DEEP:-0}" != 1 ]] && has_word deep "${E2E_DATASETS}"; then
  use_dataset "${DEEP_DS}"
  has_word 90 "${E2E_RECALLS}" &&
    run_one "${DEEP_DS}" "${EF_RECALL_90}" "${QUIVER_DEEP_EXTRA_Q}"
  has_word 96 "${E2E_RECALLS}" &&
    run_one "${DEEP_DS}" "${EF_RECALL_96}" "${QUIVER_DEEP_EXTRA_Q}"
fi

echo "DONE ${OUT}"
