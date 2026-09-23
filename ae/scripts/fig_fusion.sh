#!/usr/bin/env bash
# Paper Figure 6 — N=1 per family (PCI_SIFT_1SSD / PCI_DEEP_1SSD).
# Quiver 1B occupancy sweep vs FusionANNS 100M. The plotter takes the Pareto
# front of measured (P99, QPS) points and does not re-filter by Recall@10.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"
source "${DIR}/paper_knobs.sh"
apply_paper_knobs

case "${AE_SCALE}" in
  1b|1B|"") ;;
  *)
    echo "fig_fusion.sh is configured for mixed scale: Quiver 1B vs FusionANNS 100M" >&2
    echo "set AE_SCALE=1b, or update both sides together" >&2
    exit 1
    ;;
esac

FUSION_DATASETS="${FUSION_DATASETS:-sift deep}"
FUSION_SYSTEMS="${FUSION_SYSTEMS:-quiver fusionanns}"

has_word() {
  local wanted="$1" word
  for word in $2; do
    [[ "${word}" == "${wanted}" ]] && return 0
  done
  return 1
}

has_word quiver "${FUSION_SYSTEMS}" && require_bin quiver_search
ae_lock
OUT="$(prepare_out fusion)"
cd "${ROOT}"

fusion_query_format() {
  case "$1" in
    *.u8bin|*.bvecs) echo u8bin ;;
    *.fvecs) echo fvecs ;;
    *.fbin) echo fbin ;;
    *) echo fbin ;;
  esac
}

fusion_gt_format() {
  case "$1" in
    *.ivecs) echo ivecs ;;
    *) echo bin ;;
  esac
}

run_one() {
  local family="$1"
  local ef="$2"
  use_dataset "${family}"
  SPDK_BASE_LBA="${SPDK_BASE_LBA_1SSD}"
  use_1ssd "${OUT}/ssd_list_${DATASET}.txt"

  local pw q fusion_dataset fusion_index fusion_query fusion_gt
  case "${family}" in
    sift)
      fusion_dataset=sift100m
      fusion_index="${FUSIONANNS_SIFT100M_INDEX}"
      fusion_query="${FUSIONANNS_SIFT100M_QUERY}"
      fusion_gt="${FUSIONANNS_SIFT100M_GT}"
      ;;
    deep)
      fusion_dataset=deep100m
      fusion_index="${FUSIONANNS_DEEP100M_INDEX}"
      fusion_query="${FUSIONANNS_DEEP100M_QUERY}"
      fusion_gt="${FUSIONANNS_DEEP100M_GT}"
      ;;
    *)
      echo "unsupported FusionANNS family: ${family}" >&2
      return 1
      ;;
  esac
  local fusion_quiver_q_list quiver_nbs
  case "${family}" in
    sift) fusion_quiver_q_list="${FUSION_QUIVER_Q_LIST_SIFT:-1 2}" ;;
    deep) fusion_quiver_q_list="${FUSION_QUIVER_Q_LIST_DEEP:-1 2 3 4}" ;;
  esac
  if has_word quiver "${FUSION_SYSTEMS}"; then
    for pw in ${QUIVER_PW_LIST}; do
      for q in ${fusion_quiver_q_list}; do
        case "${q}" in
          1) quiver_nbs="${QUIVER_LOWLAT_NB:-108,216,324}" ;;
          2) quiver_nbs="${QUIVER_MID_NB:-216,324,432,540,648,756,864,972}" ;;
          3|4) quiver_nbs="${QUIVER_DEEP_EXTRA_NB:-864,972}" ;;
        esac
        run_search "${OUT}/quiver_${DATASET}_ef${ef}_q${q}_pw${pw}_1ssd.log" \
          "${BIN_DIR}/bin/quiver_search" \
          --index-dir "${INDEX_DIR}" \
          --query "${QUERY}" \
          --ground-truth "${GT}" \
          --data-type "${DATA_TYPE}" \
          --topk "${TOPK}" \
          --ef-search "${ef}" \
          --repeat "${REPEAT}" \
          --pipe-width "${pw}" \
          --queries-per-block "${q}" \
          --poll-threads "${POLL_THREADS}" \
          --early-exit-policy none \
          --ssd-list-file "${SSD_LIST}" \
          --num-blocks-list "${quiver_nbs}"
      done
    done
  fi

  if has_word fusionanns "${FUSION_SYSTEMS}"; then
    require_file "${fusion_query}"
    require_file "${fusion_gt}"
    local fusion_qfmt
    fusion_qfmt="$(fusion_query_format "${fusion_query}")"
    run_fusionanns_query "${OUT}/fusionanns_${fusion_dataset}_nprobe${FUSIONANNS_NPROBE}.log" \
      "${fusion_index}" \
      --query "${fusion_query}" \
      --query-format "${fusion_qfmt}" \
      --groundtruth "${fusion_gt}" \
      --groundtruth-format "$(fusion_gt_format "${fusion_gt}")" \
      --index-path "${fusion_index}" \
      --topk "${TOPK}" \
      --recall_k "${TOPK}" \
      --nprobe "${FUSIONANNS_NPROBE}" \
      --rerank_size "${FUSIONANNS_RERANK}" \
      --queries "${FUSIONANNS_QUERIES}" \
      --batch_size 32 \
      --threads "${FUSIONANNS_THREADS}" \
      --sweep_threads "${FUSIONANNS_THREAD_LIST}" \
      --sweep_nprobe "${FUSIONANNS_NPROBE_LIST}" \
      --sweep_rerank "${FUSIONANNS_RERANK_LIST}" \
      --warmup "${FUSIONANNS_WARMUP}" \
      --repeat 1 \
      --io_backend "${FUSIONANNS_IO_BACKEND}" \
      --cache_mb "${FUSIONANNS_CACHE_MB}" \
      --summary_csv "${OUT}/fusionanns_${fusion_dataset}_summary.csv" \
      --disable-query-stats \
      --no-use-cagra \
      --no-gpu-topk
  fi
}

for family in ${FUSION_DATASETS}; do
  use_dataset "${family}"
  run_one "${family}" "${EF_RECALL_90}"
done

echo "DONE ${OUT}"
echo "sift 1-SSD pci=${PCI_SIFT_1SSD}  deep 1-SSD pci=${PCI_DEEP_1SSD}"
echo "mixed scale: Quiver=${AE_SCALE}; FusionANNS=100m (explicitly labeled in plot)"
