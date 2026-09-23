#!/usr/bin/env bash
# fig-motivation/latency_qps.pdf — 4 SSD, sift, AE_SCALE=1b|1m.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"
source "${DIR}/paper_knobs.sh"
apply_paper_knobs

require_bin flashanns_search
require_bin gustann_search
require_bin quiver_search
ae_lock
use_dataset sift
OUT="$(prepare_out latency_qps)"
cd "${ROOT}"

EF="${EF:-${EF_RECALL_90}}"

for pw in ${FLASH_PW_LIST}; do
  run_search "${OUT}/flashanns_${DATASET}_ef${EF}_pw${pw}.log" \
    "${BIN_DIR}/bin/flashanns_search" \
    --index-dir "${INDEX_DIR}" \
    --query "${QUERY}" \
    --ground-truth "${GT}" \
    --data-type "${DATA_TYPE}" \
    --topk "${TOPK}" \
    --ef-search "${EF}" \
    --repeat "${REPEAT}" \
    --pipe-width "${pw}" \
    --poll-threads "${POLL_THREADS}" \
    --ssd-list-file "${SSD_LIST}" \
    --num-blocks-list "${NUM_BLOCKS_LIST}"
done

for pw in ${GUST_PW_LIST}; do
  run_search "${OUT}/gustann_${DATASET}_ef${EF}_pw${pw}.log" \
    "${BIN_DIR}/bin/gustann_search" \
    --index-dir "${INDEX_DIR}" \
    --query "${QUERY}" \
    --ground-truth "${GT}" \
    --data-type "${DATA_TYPE}" \
    --topk "${TOPK}" \
    --ef-search "${EF}" \
    --repeat "${REPEAT}" \
    --pipe-width "${pw}" \
    --worker-threads "${WORKER_THREADS}" \
    --runner-contexts "${RUNNER_CONTEXTS}" \
    --ssd-list-file "${SSD_LIST}" \
    --mini-batch-list "${MINI_BATCH_LIST}"
done

for pw in ${QUIVER_PW_LIST}; do
  for q in ${QUIVER_Q_LIST}; do
    case "${q}" in
      1) quiver_nbs="${QUIVER_LOWLAT_NB:-108,216,324}" ;;
      2) quiver_nbs="${QUIVER_MID_NB:-216,324,432,540,648,756,864,972}" ;;
      *) continue ;;
    esac
    run_search "${OUT}/quiver_${DATASET}_ef${EF}_q${q}_pw${pw}.log" \
      "${BIN_DIR}/bin/quiver_search" \
      --index-dir "${INDEX_DIR}" \
      --query "${QUERY}" \
      --ground-truth "${GT}" \
      --data-type "${DATA_TYPE}" \
      --topk "${TOPK}" \
      --ef-search "${EF}" \
      --repeat "${REPEAT}" \
      --pipe-width "${pw}" \
      --queries-per-block "${q}" \
      --poll-threads "${POLL_THREADS}" \
      --early-exit-policy none \
      --ssd-list-file "${SSD_LIST}" \
      --num-blocks-list "${quiver_nbs}"
  done
done

echo "DONE ${OUT}"
