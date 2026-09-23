#!/usr/bin/env bash
# AE_SCALE=1m ./ae/scripts/fusionanns_index.sh sift|deep
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FAMILY="${1:?usage: $0 sift|deep}"
use_dataset "${FAMILY}"

if [[ ! -x "${FUSIONANNS_DIR}/bin/build_index" ]]; then
  echo "missing ${FUSIONANNS_DIR}/bin/build_index (compile with ae/scripts/build.sh)" >&2
  exit 1
fi

base_fmt() {
  case "$1" in
    *.fvecs) echo fvecs ;;
    *.u8bin|*.bvecs) echo u8bin ;;
    *) echo fbin ;;
  esac
}

# Classic fvecs for 1M smoke; billion-scale usually fbin/u8bin next to Quiver data.
case "${DATASET}" in
  sift1m)
    # AE data is DiskANN fbin; fvecs is a fallback.
    if [[ -n "${FUSIONANNS_SIFT1M_BASE:-}" ]]; then
      BASE="${FUSIONANNS_SIFT1M_BASE}"
    elif [[ -f "${DATASET_ROOT}/sift1m/base.fbin" ]]; then
      BASE="${DATASET_ROOT}/sift1m/base.fbin"
    elif [[ -f "${DATASET_ROOT}/sift1m/sift_base.fvecs" ]]; then
      BASE="${DATASET_ROOT}/sift1m/sift_base.fvecs"
    else
      BASE="${DATASET_ROOT}/sift1m/base.fbin"
    fi
    if [[ -n "${FUSIONANNS_SIFT1M_LEARN:-}" ]]; then
      LEARN="${FUSIONANNS_SIFT1M_LEARN}"
    elif [[ -f "${DATASET_ROOT}/sift1m/sift_learn.fvecs" ]]; then
      LEARN="${DATASET_ROOT}/sift1m/sift_learn.fvecs"
    else
      LEARN=""
    fi
    NLIST="${NLIST:-1024}"
    ;;
  deep1m)
    if [[ -n "${FUSIONANNS_DEEP1M_BASE:-}" ]]; then
      BASE="${FUSIONANNS_DEEP1M_BASE}"
    elif [[ -f "${DATASET_ROOT}/deep1m/base.fbin" ]]; then
      BASE="${DATASET_ROOT}/deep1m/base.fbin"
    elif [[ -f "${DATASET_ROOT}/deep1m/base.fvecs" ]]; then
      BASE="${DATASET_ROOT}/deep1m/base.fvecs"
    else
      BASE="${DATASET_ROOT}/deep1m/base.fbin"
    fi
    LEARN="${FUSIONANNS_DEEP1M_LEARN:-}"
    NLIST="${NLIST:-1024}"
    ;;
  sift1b)
    BASE="${FUSIONANNS_SIFT1B_BASE:-${DATASET_ROOT}/sift1b/base.u8bin}"
    LEARN="${FUSIONANNS_SIFT1B_LEARN:-}"
    NLIST="${NLIST:-4096}"
    ;;
  deep1b)
    BASE="${FUSIONANNS_DEEP1B_BASE:-${DATASET_ROOT}/deep1b/base.fbin}"
    LEARN="${FUSIONANNS_DEEP1B_LEARN:-}"
    NLIST="${NLIST:-4096}"
    ;;
esac

require_file "${BASE}"
mkdir -p "${FUSIONANNS_INDEX_PATH}"
PQ_M="${PQ_M:-16}"
NBITS="${NBITS:-8}"

LEARN_ARGS=()
if [[ -n "${LEARN}" && -f "${LEARN}" ]]; then
  LEARN_ARGS=(--learn "${LEARN}" --learn-format "$(base_fmt "${LEARN}")")
fi

echo "build_index ${DATASET} -> ${FUSIONANNS_INDEX_PATH}"
"${FUSIONANNS_DIR}/bin/build_index" \
  --base "${BASE}" \
  --base-format "$(base_fmt "${BASE}")" \
  "${LEARN_ARGS[@]}" \
  --nlist "${NLIST}" \
  --m "${PQ_M}" \
  --nbits "${NBITS}" \
  --page 4096 \
  --preload \
  --output-dir "${FUSIONANNS_INDEX_PATH}"

echo "DONE ${FUSIONANNS_INDEX_PATH}"
