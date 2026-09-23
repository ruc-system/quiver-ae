#!/usr/bin/env bash
# 4-SSD then 1-SSD writes.  ./ae/scripts/write_all.sh [--4ssd|--oneshot]
# AE_SCALE=1m writes only *1m images (skips missing 1b).
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DIR}/common.sh"

mode="${1:-all}"
case "${AE_SCALE}" in
  1m|1M) names=(sift1m deep1m) ;;
  *) names=(deep1b sift1b sift1m deep1m) ;;
esac

write_one() {
  local name="$1"
  local extra="${2:-}"
  use_dataset "${name}"
  if [[ ! -e "${DISK_INDEX}" ]]; then
    echo "skip ${name}: missing ${DISK_INDEX}"
    return 0
  fi
  echo "===== write ${name} ${extra:-4ssd} ====="
  if [[ -n "${extra}" ]]; then
    "${DIR}/write_index.sh" "${name}" "${extra}"
  else
    "${DIR}/write_index.sh" "${name}"
  fi
}

if [[ "${mode}" == "all" || "${mode}" == "--4ssd" || "${mode}" == "4ssd" ]]; then
  for name in "${names[@]}"; do
    write_one "${name}"
  done
fi

if [[ "${mode}" == "all" || "${mode}" == "--oneshot" || "${mode}" == "oneshot" ]]; then
  for name in "${names[@]}"; do
    write_one "${name}" --oneshot
  done
fi
