#!/usr/bin/env bash
# SPDK write. --oneshot uses PCI_SIFT_1SSD or PCI_DEEP_1SSD from use_dataset.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 sift1b|deep1b|sift1m|deep1m [--oneshot]" >&2
  exit 1
fi

NAME="$1"
ONESHOT=0
if [[ "${2:-}" == "--oneshot" || "${2:-}" == "--1ssd" ]]; then
  ONESHOT=1
fi

use_dataset "${NAME}"
require_file "${DISK_INDEX}"
find_spdk_write

ae_lock
cd "${ROOT}"

OUT_LIST="${SSD_LIST}"
if [[ "${ONESHOT}" -eq 1 ]]; then
  SPDK_BASE_LBA="${SPDK_BASE_LBA_1SSD}"
  use_1ssd "${AE_DIR}/.cache/ssd_list_${DATASET}_1ssd.txt"
  OUT_LIST="${SSD_LIST}"
fi

PCI_ALLOWED="$(sed 's/#.*//' "${OUT_LIST}" | awk 'NF {print $1}' | xargs)"
SPDK_WRITE_READ_BUFFER_MB="${SPDK_WRITE_READ_BUFFER_MB:-64}"

echo "dataset=${DATASET}"
echo "disk_index=${DISK_INDEX}"
echo "ssd_list=${OUT_LIST}"
echo "SPDK_BASE_LBA=${SPDK_BASE_LBA}"
echo "PCI_ALLOWED=${PCI_ALLOWED}"

sudo -n env HUGEMEM=4096 PCI_ALLOWED="${PCI_ALLOWED}" DEV_TYPE=NVME \
  "${ROOT}/deps/spdk/scripts/setup.sh" config

sudo -n env \
  SPDK_BASE_LBA="${SPDK_BASE_LBA}" \
  SPDK_WRITE_READ_BUFFER_MB="${SPDK_WRITE_READ_BUFFER_MB}" \
  "${SPDK_WRITE}" "${DISK_INDEX}" "${OUT_LIST}"

echo "DONE write ${DATASET} BASE=${SPDK_BASE_LBA} list=${OUT_LIST}"
