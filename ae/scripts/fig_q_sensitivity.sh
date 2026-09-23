#!/usr/bin/env bash
# Paper Figure 8. The x-axis Q is queries_per_block * pipe_width:
#   Q=1  FlashANNS (single context)
#   Q=2,4,6,8  Quiver with queries_per_block=1,2,3,4 and pipe_width=2
# Panel (b) profiles Quiver Q=1,2,4,6,8 at pipe-width=1 / num-block=108.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"
source "${DIR}/paper_knobs.sh"
apply_paper_knobs

# Panel (b) uses the instrumented binary. Keep one conservative watchdog
# policy for both panels so switching builds cannot trigger a false timeout.
export HANG_FLOOR_FIRST="${HANG_FLOOR_FIRST:-300}"
export HANG_FLOOR="${HANG_FLOOR:-300}"

require_bin flashanns_search
require_bin quiver_search
require_bin flashanns_search "${INSTRUMENTED_BIN_DIR}"
require_bin quiver_search "${INSTRUMENTED_BIN_DIR}"
ae_lock
use_dataset sift
OUT="$(prepare_out q_sensitivity)"
cd "${ROOT}"

QPB_LIST="${QPB_LIST:-1 2 3 4}"
EF_LIST="${EF_LIST:-${EF_RECALL_90} ${EF_RECALL_92} ${EF_RECALL_94} ${EF_RECALL_96}}"
FIG8_NUM_BLOCKS_LIST="${FIG8_NUM_BLOCKS_LIST:-108,216,324,432,540,648,756,864,972}"
FIG8_RUN_A="${FIG8_RUN_A:-1}"
FIG8_RUN_B="${FIG8_RUN_B:-1}"
# Official AE path: the paper-selected occupancy for each (recall, Q) bar.
# Set FIG8_SELECTED_ONLY=0 only if you want the nine-point occupancy grid.
FIG8_SELECTED_ONLY="${FIG8_SELECTED_ONLY:-1}"

if [[ "${FIG8_RUN_A}" == 1 ]]; then
for ef in ${EF_LIST}; do
  flash_nbs="${FIG8_NUM_BLOCKS_LIST}"
  selected_qpbs="${QPB_LIST}"
  if [[ "${FIG8_SELECTED_ONLY}" == 1 ]]; then
    case "${ef}" in
      "${EF_RECALL_90}") flash_nbs=432; selected_qpbs="1 2 3 4" ;;
      "${EF_RECALL_92}") flash_nbs=432; selected_qpbs="1 2 3 4" ;;
      "${EF_RECALL_94}") flash_nbs=324; selected_qpbs="1 2 3 4" ;;
      "${EF_RECALL_96}") flash_nbs=216; selected_qpbs="1 2" ;;
      *) echo "no submitted Figure 8(a) point for ef=${ef}" >&2; exit 1 ;;
    esac
  fi
  run_search "${OUT}/flashanns_${DATASET}_ef${ef}_pw2.log" \
    "${BIN_DIR}/bin/flashanns_search" \
    --index-dir "${INDEX_DIR}" \
    --query "${QUERY}" \
    --ground-truth "${GT}" \
    --data-type "${DATA_TYPE}" \
    --topk "${TOPK}" \
    --ef-search "${ef}" \
    --repeat "${REPEAT}" \
    --pipe-width 2 \
    --poll-threads "${POLL_THREADS}" \
    --ssd-list-file "${SSD_LIST}" \
    --num-blocks-list "${flash_nbs}"

  for qpb in ${selected_qpbs}; do
    quiver_nbs="${FIG8_NUM_BLOCKS_LIST}"
    if [[ "${FIG8_SELECTED_ONLY}" == 1 ]]; then
      case "${ef}:${qpb}" in
        "${EF_RECALL_90}:1") quiver_nbs=756 ;;
        "${EF_RECALL_90}:2") quiver_nbs=432 ;;
        "${EF_RECALL_90}:3") quiver_nbs=216 ;;
        "${EF_RECALL_90}:4") quiver_nbs=108 ;;
        "${EF_RECALL_92}:1") quiver_nbs=648 ;;
        "${EF_RECALL_92}:2") quiver_nbs=324 ;;
        "${EF_RECALL_92}:3"|"${EF_RECALL_92}:4") quiver_nbs=108 ;;
        "${EF_RECALL_94}:1") quiver_nbs=432 ;;
        "${EF_RECALL_94}:2") quiver_nbs=216 ;;
        "${EF_RECALL_94}:3"|"${EF_RECALL_94}:4") quiver_nbs=108 ;;
        "${EF_RECALL_96}:1") quiver_nbs=324 ;;
        "${EF_RECALL_96}:2") quiver_nbs=108 ;;
        *) echo "no submitted Figure 8(a) point for ef=${ef}, qpb=${qpb}" >&2; exit 1 ;;
      esac
    fi
    run_search "${OUT}/quiver_${DATASET}_q${qpb}_ef${ef}_pw2.log" \
      "${BIN_DIR}/bin/quiver_search" \
      --index-dir "${INDEX_DIR}" \
      --query "${QUERY}" \
      --ground-truth "${GT}" \
      --data-type "${DATA_TYPE}" \
      --topk "${TOPK}" \
      --ef-search "${ef}" \
      --repeat "${REPEAT}" \
      --pipe-width 2 \
      --queries-per-block "${qpb}" \
      --poll-threads "${POLL_THREADS}" \
      --early-exit-policy none \
      --ssd-list-file "${SSD_LIST}" \
      --num-blocks-list "${quiver_nbs}"
  done
done
fi

if [[ "${FIG8_RUN_B}" == 1 ]]; then
QUIVER_AE_VARIANT=cta
# Panel (b) is a separate Quiver occupancy run: pipe-width=1, num-block=108,
# Q = queries_per_block. The paper measured Q=1,2,4,6 this way and filled
# Q=8; AE measures all five Q values with the same knobs.
for qpb in 1 2 4 6 8; do
  run_search "${OUT}/cta_quiver_${DATASET}_q${qpb}_ef${EF_RECALL_90}_pw1.log" \
    "${INSTRUMENTED_BIN_DIR}/bin/quiver_search" \
    --index-dir "${INDEX_DIR}" \
    --query "${QUERY}" \
    --ground-truth "${GT}" \
    --data-type "${DATA_TYPE}" \
    --topk "${TOPK}" \
    --ef-search "${EF_RECALL_90}" \
    --repeat "${REPEAT}" \
    --pipe-width 1 \
    --queries-per-block "${qpb}" \
    --poll-threads "${POLL_THREADS}" \
    --early-exit-policy none \
    --ssd-list-file "${SSD_LIST}" \
    --num-blocks-list 108
done
unset QUIVER_AE_VARIANT
fi

echo "DONE ${OUT}"
