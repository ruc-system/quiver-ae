#!/usr/bin/env bash
# Compile AE binaries, then optional SSD write / FusionANNS indexes. No figures.
#   ./ae/scripts/prepare.sh
#   ./ae/scripts/prepare.sh --write
#   ./ae/scripts/prepare.sh --fusion-index
#   ./ae/scripts/prepare.sh --no-build   # check / write only
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"

DO_BUILD=1
DO_WRITE=0
DO_FUSION_INDEX=0
for a in "$@"; do
  case "$a" in
    --write) DO_WRITE=1 ;;
    --fusion-index|--fusion_index) DO_FUSION_INDEX=1 ;;
    --no-build) DO_BUILD=0 ;;
    *)
      echo "usage: $0 [--no-build] [--write] [--fusion-index]" >&2
      exit 1
      ;;
  esac
done

if [[ "${DO_BUILD}" -eq 1 ]]; then
  "${DIR}/build.sh"
fi

if [[ "${DO_WRITE}" -eq 1 ]]; then
  "${DIR}/write_all.sh"
fi

if [[ "${DO_FUSION_INDEX}" -eq 1 ]]; then
  echo "===== FusionANNS index AE_SCALE=${AE_SCALE} ====="
  "${DIR}/fusionanns_index.sh" sift
  "${DIR}/fusionanns_index.sh" deep
fi

echo "prepare done → ${DIR}/run_all.sh  then  ${DIR}/plot_all.py ${AE_DIR}/results"
