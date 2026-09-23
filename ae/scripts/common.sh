#!/usr/bin/env bash
# Shared AE knobs. Machine-specific overrides live in ae/config.env.
#
# AE_SCALE=1b|1m (default 1b). use_dataset sift|deep follows AE_SCALE;
# sift1b|deep1b|sift1m|deep1m are explicit.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file; do not execute it" >&2
  exit 1
fi

AE_SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AE_DIR="$(cd -- "${AE_SCRIPTS_DIR}/.." && pwd)"
ROOT="$(cd -- "${AE_DIR}/.." && pwd)"

if [[ -f "${AE_DIR}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${AE_DIR}/config.env"
fi

# Production search: BIN_DIR=build/. Instrumented: INSTRUMENTED_BIN_DIR=build-ae/.
BIN_DIR="${BIN_DIR:-${ROOT}/build}"
INSTRUMENTED_BIN_DIR="${INSTRUMENTED_BIN_DIR:-${ROOT}/build-ae}"
FUSIONANNS_DIR="${FUSIONANNS_DIR:-${ROOT}/third-party/fusionanns}"
INDEX_ROOT="${INDEX_ROOT:-${ROOT}/index}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/dataset}"
AE_SCALE="${AE_SCALE:-1b}"
# Optional persistent debug history. When set, every attempted sweep group is
# kept under <root>/<figure>/runs/<timestamp>/groups/<stem>/ and complete
# groups can be reused by later invocations. The reviewer path leaves this
# unset and keeps the existing staging/publish behavior.
AE_DEBUG_ROOT="${AE_DEBUG_ROOT:-}"
AE_DEBUG_FORCE="${AE_DEBUG_FORCE:-0}"

GPU_ID="${GPU_ID:-0}"

# Keep the SPDK poll threads and their DMA buffers on the node the SSDs are
# attached to. Unset AE_NUMA_NODE to run unpinned.
AE_NUMA_PREFIX=""
if [[ -n "${AE_NUMA_NODE:-}" ]]; then
  if command -v numactl >/dev/null 2>&1; then
    AE_NUMA_PREFIX="numactl --cpunodebind=${AE_NUMA_NODE} --membind=${AE_NUMA_NODE}"
  else
    echo "AE_NUMA_NODE=${AE_NUMA_NODE} set but numactl is missing; running unpinned" >&2
  fi
fi

# PCIe from config.env. If PCI_4SSD unset, read SSD_LIST (one PCI per line).
SSD_LIST="${SSD_LIST:-${ROOT}/ssd_list.txt}"
_ae_pci_words() {
  echo "${1:-}" | tr ',;' ' '
}
if [[ -z "${PCI_4SSD:-}" && -f "${SSD_LIST}" ]]; then
  PCI_4SSD="$(awk 'NF && $1 !~ /^#/ {print $1}' "${SSD_LIST}" | paste -sd' ' -)"
fi
read -r -a PCI_4SSD_ARR <<< "$(_ae_pci_words "${PCI_4SSD:-}")"
PCI_SIFT_1SSD="${PCI_SIFT_1SSD:-${PCI_4SSD_ARR[0]:-}}"
PCI_DEEP_1SSD="${PCI_DEEP_1SSD:-${PCI_4SSD_ARR[1]:-}}"
SSD_LIST_4="${SSD_LIST_4:-${AE_DIR}/.cache/ssd_list_4.txt}"
SSD_LIST_SIFT_1SSD="${SSD_LIST_SIFT_1SSD:-${AE_DIR}/.cache/ssd_list_sift1.txt}"
SSD_LIST_DEEP_1SSD="${SSD_LIST_DEEP_1SSD:-${AE_DIR}/.cache/ssd_list_deep1.txt}"
mkdir -p "${AE_DIR}/.cache"
if [[ ${#PCI_4SSD_ARR[@]} -gt 0 ]]; then
  printf '%s\n' "${PCI_4SSD_ARR[@]}" > "${SSD_LIST_4}"
  SSD_LIST="${SSD_LIST_4}"
fi
[[ -n "${PCI_SIFT_1SSD}" ]] && printf '%s\n' "${PCI_SIFT_1SSD}" > "${SSD_LIST_SIFT_1SSD}"
[[ -n "${PCI_DEEP_1SSD}" ]] && printf '%s\n' "${PCI_DEEP_1SSD}" > "${SSD_LIST_DEEP_1SSD}"
REPEAT="${REPEAT:-20}"
POLL_THREADS="${POLL_THREADS:-6}"
TOPK="${TOPK:-10}"
WORKER_THREADS="${WORKER_THREADS:-2}"
RUNNER_CONTEXTS="${RUNNER_CONTEXTS:-20}"
NUM_BLOCKS_LIST="${NUM_BLOCKS_LIST:-108,216,324,432,540,648,756,864,972,1080}"
MINI_BATCH_LIST="${MINI_BATCH_LIST:-16,24,32,48,64,96,128,192,256,384,512,768,1024,1120}"

# Throughput-latency curves are the Pareto front over each system's tuning
# knobs, not a single occupancy sweep. Quiver trades Q against latency the way
# the baselines trade batch size, so a fixed Q shows only one slice of its
# curve: on SIFT ef=45 the paper's low-latency end comes from Q=1 and its
# peak-throughput end from Q=4 (results/sift1b-quiver/q{1..4}_ef45_*.log).
# Every system gets the same pipe-width budget so the fronts stay comparable.
QUIVER_Q_LIST="${QUIVER_Q_LIST:-1 2 3 4}"
QUIVER_PW_LIST="${QUIVER_PW_LIST:-1 2}"
FLASH_PW_LIST="${FLASH_PW_LIST:-1 2}"
GUST_PW_LIST="${GUST_PW_LIST:-1 2}"

# Single-operating-point figures (breakdown, GPU utilization) pick one point
# per system instead of a curve. Defaults are the paper's iso-throughput set
# from results/tech-breakdown-recall90-70k/ (Recall@10=0.90, 55-72k QPS):
# Quiver nb=216 Q=2, FlashANNS nb=756, GustANN mini_batch=48, +S nb=324 Q=2.
# They were tuned on 1B / 4 SSD; re-match them if AE_SCALE=1m shifts the knee.
PIPE_WIDTH_QUIVER="${PIPE_WIDTH_QUIVER:-2}"
PIPE_WIDTH_FLASH="${PIPE_WIDTH_FLASH:-2}"
PIPE_WIDTH_GUST="${PIPE_WIDTH_GUST:-1}"
QUERIES_PER_BLOCK="${QUERIES_PER_BLOCK:-2}"
POINT_QUIVER_BLOCKS="${POINT_QUIVER_BLOCKS:-216}"
POINT_FLASH_BLOCKS="${POINT_FLASH_BLOCKS:-756}"
POINT_PLUS_S_BLOCKS="${POINT_PLUS_S_BLOCKS:-324}"
POINT_GUST_MINI_BATCH="${POINT_GUST_MINI_BATCH:-48}"
LOCK_FILE="${LOCK_FILE:-/tmp/xxx.lock}"
QUIVER_SPDK_BLOCK_OFFSET="${QUIVER_SPDK_BLOCK_OFFSET:-0}"

# LBA defaults (4 KiB); override them in ae/config.env for another layout.
DEEP1B_SPDK_BASE_LBA="${DEEP1B_SPDK_BASE_LBA:-0}"
DEEP1M_SPDK_BASE_LBA="${DEEP1M_SPDK_BASE_LBA:-75497472}"          # 288 GiB
SIFT1B_SPDK_BASE_LBA="${SIFT1B_SPDK_BASE_LBA:-83886080}"          # 320 GiB
SIFT1M_SPDK_BASE_LBA="${SIFT1M_SPDK_BASE_LBA:-142606336}"         # 544 GiB
# 1-SSD defaults start at 576 GiB (packed after 4-SSD). Override to 2 TiB on
# exclusive-machine layouts via ae/config.env.
SIFT1M_1SSD_SPDK_BASE_LBA="${SIFT1M_1SSD_SPDK_BASE_LBA:-150994944}"  # 576 GiB
DEEP1M_1SSD_SPDK_BASE_LBA="${DEEP1M_1SSD_SPDK_BASE_LBA:-150994944}"
SIFT1B_1SSD_SPDK_BASE_LBA="${SIFT1B_1SSD_SPDK_BASE_LBA:-159383552}"  # 576 GiB + 32 GiB
DEEP1B_1SSD_SPDK_BASE_LBA="${DEEP1B_1SSD_SPDK_BASE_LBA:-159383552}"

# Index / query paths (override in config.env when layouts differ across machines).
SIFT1B_INDEX_DIR="${SIFT1B_INDEX_DIR:-${INDEX_ROOT}/sift1b-R128-QD32/}"
DEEP1B_INDEX_DIR="${DEEP1B_INDEX_DIR:-${INDEX_ROOT}/deep1b-R128-QD32/}"
SIFT1M_INDEX_DIR="${SIFT1M_INDEX_DIR:-${INDEX_ROOT}/sift1m-R128-QD32/}"
DEEP1M_INDEX_DIR="${DEEP1M_INDEX_DIR:-${INDEX_ROOT}/deep1m-R128-QD32/}"
SIFT1B_DISK_INDEX="${SIFT1B_DISK_INDEX:-${SIFT1B_INDEX_DIR}/1B_disk.index}"
DEEP1B_DISK_INDEX="${DEEP1B_DISK_INDEX:-${DEEP1B_INDEX_DIR}/1B_disk.index}"
SIFT1M_DISK_INDEX="${SIFT1M_DISK_INDEX:-${SIFT1M_INDEX_DIR}/ann_disk.index}"
DEEP1M_DISK_INDEX="${DEEP1M_DISK_INDEX:-${DEEP1M_INDEX_DIR}/ann_disk.index}"

SIFT1B_QUERY="${SIFT1B_QUERY:-${DATASET_ROOT}/sift1b/query.u8bin}"
SIFT1B_GT="${SIFT1B_GT:-${DATASET_ROOT}/sift1b/gt.ibin}"
DEEP1B_QUERY="${DEEP1B_QUERY:-${DATASET_ROOT}/deep1b/query.fbin}"
DEEP1B_GT="${DEEP1B_GT:-${DATASET_ROOT}/deep1b/gt.ibin}"
SIFT1M_QUERY="${SIFT1M_QUERY:-${DATASET_ROOT}/sift1m/query.fbin}"
SIFT1M_GT="${SIFT1M_GT:-${DATASET_ROOT}/sift1m/gt.ibin}"
DEEP1M_QUERY="${DEEP1M_QUERY:-${DATASET_ROOT}/deep1m/query.fbin}"
DEEP1M_GT="${DEEP1M_GT:-${DATASET_ROOT}/deep1m/gt.ibin}"
FUSIONANNS_SIFT100M_QUERY="${FUSIONANNS_SIFT100M_QUERY:-${DATASET_ROOT}/sift100m/query.public.10K.u8bin}"
FUSIONANNS_SIFT100M_GT="${FUSIONANNS_SIFT100M_GT:-${DATASET_ROOT}/sift100m/groundtruth.100m.ibin}"
FUSIONANNS_DEEP100M_QUERY="${FUSIONANNS_DEEP100M_QUERY:-${DATASET_ROOT}/deep100m/query.public.10K.fbin}"
FUSIONANNS_DEEP100M_GT="${FUSIONANNS_DEEP100M_GT:-${DATASET_ROOT}/deep100m/groundtruth.100m.bin}"
# deep1m-R128-QD32 ships query/gt next to the index when dataset/deep1m is absent.
if [[ ! -f "${DEEP1M_QUERY}" && -f "${DEEP1M_INDEX_DIR}/query.bin" ]]; then
  DEEP1M_QUERY="${DEEP1M_INDEX_DIR}/query.bin"
fi
if [[ ! -f "${DEEP1M_GT}" && -f "${DEEP1M_INDEX_DIR}/gt.bin" ]]; then
  DEEP1M_GT="${DEEP1M_INDEX_DIR}/gt.bin"
fi

# FusionANNS offline indexes (host FS, not SPDK). Prefer INDEX_ROOT layout.
FUSIONANNS_SIFT1B_INDEX="${FUSIONANNS_SIFT1B_INDEX:-${INDEX_ROOT}/fusionanns-sift1b}"
FUSIONANNS_DEEP1B_INDEX="${FUSIONANNS_DEEP1B_INDEX:-${INDEX_ROOT}/fusionanns-deep1b}"
FUSIONANNS_SIFT1M_INDEX="${FUSIONANNS_SIFT1M_INDEX:-${INDEX_ROOT}/fusionanns-sift1m}"
FUSIONANNS_DEEP1M_INDEX="${FUSIONANNS_DEEP1M_INDEX:-${INDEX_ROOT}/fusionanns-deep1m}"
FUSIONANNS_SIFT100M_INDEX="${FUSIONANNS_SIFT100M_INDEX:-${INDEX_ROOT}/fusionanns-sift100m}"
FUSIONANNS_DEEP100M_INDEX="${FUSIONANNS_DEEP100M_INDEX:-${INDEX_ROOT}/fusionanns-deep100m}"
FUSIONANNS_NPROBE="${FUSIONANNS_NPROBE:-1024}"
FUSIONANNS_RERANK="${FUSIONANNS_RERANK:-256}"
FUSIONANNS_THREADS="${FUSIONANNS_THREADS:-8}"
# Fusion figure sweeps search quality and client concurrency. The paper plot
# trusts the configured target bucket and does not re-filter measured recall.
FUSIONANNS_THREAD_LIST="${FUSIONANNS_THREAD_LIST:-1,2,4,8,16,32,64,128}"
FUSIONANNS_NPROBE_LIST="${FUSIONANNS_NPROBE_LIST:-1024,2048}"
FUSIONANNS_RERANK_LIST="${FUSIONANNS_RERANK_LIST:-256,512}"
FUSIONANNS_IO_BACKEND="${FUSIONANNS_IO_BACKEND:-io_uring}"
FUSIONANNS_QUERIES="${FUSIONANNS_QUERIES:-10000}"
FUSIONANNS_WARMUP="${FUSIONANNS_WARMUP:-10000}"
FUSIONANNS_CACHE_MB="${FUSIONANNS_CACHE_MB:-16384}"

use_dataset() {
  local name="$1"
  case "${name}" in
    sift|deep)
      case "${AE_SCALE}" in
        1m|1M) name="${name}1m" ;;
        1b|1B|"") name="${name}1b" ;;
        *)
          echo "unknown AE_SCALE=${AE_SCALE} (expected 1b or 1m)" >&2
          exit 1
          ;;
      esac
      ;;
  esac

  case "${name}" in
    sift1b)
      DATASET=sift1b
      INDEX_DIR="${SIFT1B_INDEX_DIR}"
      DISK_INDEX="${SIFT1B_DISK_INDEX}"
      QUERY="${SIFT1B_QUERY}"
      GT="${SIFT1B_GT}"
      DATA_TYPE=uint8
      SPDK_BASE_LBA="${SIFT1B_SPDK_BASE_LBA}"
      SPDK_BASE_LBA_1SSD="${SIFT1B_1SSD_SPDK_BASE_LBA}"
      PCI_1SSD="${PCI_SIFT_1SSD}"
      FUSIONANNS_INDEX_PATH="${FUSIONANNS_SIFT1B_INDEX}"
      # Paper-era ef↔threshold from results/* (SIFT measured ~0.89/0.91/0.93/0.95/0.96/0.98).
      # Threshold set: 0.88/0.90/0.92/0.94/0.96/0.98. Paper e2e "0.95" = the 0.96 bucket.
      EF_RECALL_88=40
      EF_RECALL_90=45
      EF_RECALL_92=55
      EF_RECALL_94=65
      EF_RECALL_96=80
      EF_RECALL_98=120
      ;;
    deep1b)
      DATASET=deep1b
      INDEX_DIR="${DEEP1B_INDEX_DIR}"
      DISK_INDEX="${DEEP1B_DISK_INDEX}"
      QUERY="${DEEP1B_QUERY}"
      GT="${DEEP1B_GT}"
      DATA_TYPE=float
      SPDK_BASE_LBA="${DEEP1B_SPDK_BASE_LBA}"
      SPDK_BASE_LBA_1SSD="${DEEP1B_1SSD_SPDK_BASE_LBA}"
      PCI_1SSD="${PCI_DEEP_1SSD}"
      FUSIONANNS_INDEX_PATH="${FUSIONANNS_DEEP1B_INDEX}"
      # From results/deep1b-gustann-thresholds/thresholds_*.csv
      EF_RECALL_88=60
      EF_RECALL_90=70
      EF_RECALL_92=85
      EF_RECALL_94=105
      EF_RECALL_96=145
      EF_RECALL_98=255
      ;;
    sift1m)
      DATASET=sift1m
      INDEX_DIR="${SIFT1M_INDEX_DIR}"
      DISK_INDEX="${SIFT1M_DISK_INDEX}"
      QUERY="${SIFT1M_QUERY}"
      GT="${SIFT1M_GT}"
      DATA_TYPE=float
      SPDK_BASE_LBA="${SIFT1M_SPDK_BASE_LBA}"
      SPDK_BASE_LBA_1SSD="${SIFT1M_1SSD_SPDK_BASE_LBA}"
      PCI_1SSD="${PCI_SIFT_1SSD}"
      FUSIONANNS_INDEX_PATH="${FUSIONANNS_SIFT1M_INDEX}"
      EF_RECALL_88=40
      EF_RECALL_90=45
      EF_RECALL_92=55
      EF_RECALL_94=65
      EF_RECALL_96=80
      EF_RECALL_98=120
      ;;
    deep1m)
      DATASET=deep1m
      INDEX_DIR="${DEEP1M_INDEX_DIR}"
      DISK_INDEX="${DEEP1M_DISK_INDEX}"
      QUERY="${DEEP1M_QUERY}"
      GT="${DEEP1M_GT}"
      DATA_TYPE=float
      SPDK_BASE_LBA="${DEEP1M_SPDK_BASE_LBA}"
      SPDK_BASE_LBA_1SSD="${DEEP1M_1SSD_SPDK_BASE_LBA}"
      PCI_1SSD="${PCI_DEEP_1SSD}"
      FUSIONANNS_INDEX_PATH="${FUSIONANNS_DEEP1M_INDEX}"
      EF_RECALL_88=60
      EF_RECALL_90=70
      EF_RECALL_92=85
      EF_RECALL_94=105
      EF_RECALL_96=145
      EF_RECALL_98=255
      ;;
    *)
      echo "unknown dataset: $1 (expected sift|deep|sift1b|deep1b|sift1m|deep1m)" >&2
      exit 1
      ;;
  esac
}

prepare_out() {
  local fig="$1"
  if [[ -z "${fig}" || "${fig}" == *"/"* || "${fig}" == "." || "${fig}" == ".." || "${fig}" == "Pre-executed-logs" ]]; then
    echo "refusing output directory: ${fig}" >&2
    exit 1
  fi
  # Debug runs are immutable attempts. Reviewer runs publish one directory per
  # figure through run_all.sh's same-filesystem staging tree.
  local output_root="${AE_OUTPUT_ROOT:-${AE_DIR}/results}"
  if [[ -n "${AE_DEBUG_ROOT}" ]]; then
    output_root="${AE_DEBUG_ROOT}"
    local run_id="${AE_DEBUG_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    OUT="${output_root}/${fig}/runs/${run_id}"
  else
    OUT="${output_root}/${fig}"
    rm -rf "${OUT}"
  fi
  mkdir -p "${OUT}"
  {
    echo "AE_SCALE=${AE_SCALE}"
    echo "AE_DEBUG_ROOT=${AE_DEBUG_ROOT}"
    echo "AE_DEBUG_RUN_ID=${AE_DEBUG_RUN_ID:-}"
    echo "BIN_DIR=${BIN_DIR}"
    echo "PCI_4SSD=${PCI_4SSD:-}"
    echo "PCI_SIFT_1SSD=${PCI_SIFT_1SSD:-}"
    echo "PCI_DEEP_1SSD=${PCI_DEEP_1SSD:-}"
    echo "SSD_LIST=${SSD_LIST}"
    echo "ROOT=${ROOT}"
  } > "${OUT}/env.txt"
  printf '%s\n' "${OUT}"
}

ae_expected_points() {
  local expected=1 arg value
  while [[ $# -gt 0 ]]; do
    arg="$1"
    case "${arg}" in
      --num-blocks-list|--mini-batch-list)
        value="${2:-}"
        shift 2
        expected="$(awk -F, '{print NF}' <<< "${value}")"
        ;;
      --num-blocks-list=*|--mini-batch-list=*)
        value="${arg#*=}"
        shift
        expected="$(awk -F, '{print NF}' <<< "${value}")"
        ;;
      *)
        shift
        ;;
    esac
  done
  printf '%s\n' "${expected}"
}

ae_validate_metrics() {
  local csv="$1" stem="$2" expected="$3"
  python3 - "${csv}" "${stem}" "${expected}" <<'PY'
import csv
import math
import sys
from pathlib import Path

path, stem, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if not path.is_file():
    raise SystemExit(1)
with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != expected or any(row.get("stem") != stem for row in rows):
    raise SystemExit(1)
keys = set()
for row in rows:
    try:
        qps = float(row["qps"])
        avg = float(row["avg_ms"])
        p99 = float(row["p99_ms"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(1)
    if not all(math.isfinite(value) and value > 0 for value in (qps, avg, p99)):
        raise SystemExit(1)
    keys.add((
        row.get("num_blocks", ""), row.get("mini_batch", ""),
        row.get("queries_per_block", ""), row.get("pipe_width", ""),
    ))
if len(keys) != expected:
    raise SystemExit(1)
PY
}

ae_debug_signature() {
  local variant="${QUIVER_AE_VARIANT:-}" joined binary_stamp=""
  if [[ -e "${1:-}" ]]; then
    binary_stamp="$(stat -Lc '%Y:%s' "$1")"
  fi
  printf -v joined '%q ' "$@"
  printf '%s' "binary=${binary_stamp} variant=${variant} dataset=${DATASET:-} lba=${SPDK_BASE_LBA:-} ${joined}" |
    sha256sum | awk '{print $1}'
}

ae_find_cached_group() {
  local figure_root="$1" stem="$2" signature="$3" expected="$4"
  local manifest group saved_status saved_signature saved_expected cached=""
  shopt -s nullglob
  for manifest in "${figure_root}"/runs/*/groups/"${stem}"/manifest.env; do
    group="$(dirname "${manifest}")"
    saved_status="$(awk -F= '$1=="status" {print $2}' "${manifest}")"
    saved_signature="$(awk -F= '$1=="signature" {print $2}' "${manifest}")"
    saved_expected="$(awk -F= '$1=="expected_points" {print $2}' "${manifest}")"
    if [[ "${saved_status}" == complete &&
          "${saved_signature}" == "${signature}" &&
          "${saved_expected}" == "${expected}" ]] &&
       ae_validate_metrics "${group}/metrics.csv" "${stem}" "${expected}"; then
      cached="${group}"
    fi
  done
  shopt -u nullglob
  printf '%s\n' "${cached}"
}

# Shared SPDK: wait if ${LOCK_FILE} exists, create when free, delete on exit.
ae_lock() {
  local wait_s="${LOCK_WAIT_SEC:-15}"
  echo "SPDK lock ${LOCK_FILE} (pid $$) ..."
  while true; do
    if (set -o noclobber; echo "$$" > "${LOCK_FILE}") 2>/dev/null; then
      trap ae_unlock EXIT
      trap 'ae_unlock; exit 130' INT
      trap 'ae_unlock; exit 143' TERM
      echo "acquired ${LOCK_FILE}"
      return 0
    fi
    local holder
    holder="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
    if [[ -n "${holder}" && "${holder}" =~ ^[0-9]+$ ]] && ! kill -0 "${holder}" 2>/dev/null; then
      echo "stale lock holder pid=${holder} is dead; removing ${LOCK_FILE}"
      rm -f "${LOCK_FILE}"
      continue
    fi
    echo "SPDK busy lock=${LOCK_FILE} holder=${holder:-unknown}; wait ${wait_s}s"
    sleep "${wait_s}"
  done
}

ae_unlock() {
  if [[ "$(cat "${LOCK_FILE}" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "${LOCK_FILE}"
  fi
}

require_bin() {
  local name="$1"
  local dir="${2:-${BIN_DIR}}"
  if [[ ! -x "${dir}/bin/${name}" ]]; then
    echo "missing executable: ${dir}/bin/${name}" >&2
    echo "hint: compile first with ae/scripts/build.sh (do not compile from run_all.sh)" >&2
    exit 1
  fi
}

find_spdk_write() {
  if [[ -x "${BIN_DIR}/bin/spdk_write" ]]; then
    SPDK_WRITE="${BIN_DIR}/bin/spdk_write"
  elif [[ -x "${ROOT}/build/bin/spdk_write" ]]; then
    SPDK_WRITE="${ROOT}/build/bin/spdk_write"
  else
    echo "missing spdk_write (build BIN_DIR=${BIN_DIR})" >&2
    exit 1
  fi
}

require_file() {
  local f="$1"
  if [[ ! -e "$f" ]]; then
    echo "missing file: $f" >&2
    exit 1
  fi
}

# Point SSD_LIST at the 1-SSD PCI for the current DATASET (sift* vs deep*).
# Does not change SPDK_BASE_LBA; caller sets it to *_1SSD_*.
use_1ssd() {
  local dest="${1:-}"
  if [[ -z "${PCI_1SSD:-}" ]]; then
    echo "PCI_1SSD empty for DATASET=${DATASET:-} (set PCI_SIFT_1SSD / PCI_DEEP_1SSD in ae/config.env)" >&2
    exit 1
  fi
  if [[ -z "${dest}" ]]; then
    case "${DATASET}" in
      sift*) dest="${SSD_LIST_SIFT_1SSD}" ;;
      deep*) dest="${SSD_LIST_DEEP_1SSD}" ;;
      *) dest="${AE_DIR}/.cache/ssd_list_1.txt" ;;
    esac
  fi
  mkdir -p "$(dirname "${dest}")"
  printf '%s\n' "${PCI_1SSD}" > "${dest}"
  SSD_LIST="${dest}"
}

# Map search ef → threshold label (not measured Recall@10).
# Canonical buckets: 0.88/0.90/0.92/0.94/0.96/0.98 from paper-era sweeps.
# Paper e2e text "0.95" is the 0.96 bucket (ef=80 SIFT / ef=145 DEEP); plot remaps.
ae_target_recall_from_ef() {
  local ef="$1"
  if [[ -n "${EF_RECALL_88:-}" && "${ef}" == "${EF_RECALL_88}" ]]; then
    echo "0.88"
  elif [[ -n "${EF_RECALL_90:-}" && "${ef}" == "${EF_RECALL_90}" ]]; then
    echo "0.90"
  elif [[ -n "${EF_RECALL_92:-}" && "${ef}" == "${EF_RECALL_92}" ]]; then
    echo "0.92"
  elif [[ -n "${EF_RECALL_94:-}" && "${ef}" == "${EF_RECALL_94}" ]]; then
    echo "0.94"
  elif [[ -n "${EF_RECALL_96:-}" && "${ef}" == "${EF_RECALL_96}" ]]; then
    echo "0.96"
  elif [[ -n "${EF_RECALL_98:-}" && "${ef}" == "${EF_RECALL_98}" ]]; then
    echo "0.98"
  else
    echo ""
  fi
}

# Point search binaries at this run's CSVs. Plotters read these, not logs.
ae_export_result_csv() {
  local log="$1"
  shift
  local out_dir stem ef=""
  out_dir="$(cd -- "$(dirname -- "${log}")" && pwd)"
  stem="$(basename "${log}" .log)"
  export QUIVER_AE_METRICS_CSV="${out_dir}/metrics.csv"
  export QUIVER_AE_BREAKDOWN_CSV="${out_dir}/breakdown.csv"
  # Per-query breakdown rows: no plot reads them, and they cost ~2MB per run.
  export QUIVER_AE_BREAKDOWN_SAMPLES_CSV=""
  export QUIVER_AE_CTA_SAMPLES_CSV="${out_dir}/cta_samples.csv"
  # Per-hop wait/compute intervals are only read by the io_latency plot, and
  # they run to hundreds of MB on a full occupancy sweep. Opt in.
  if [[ "${AE_HOP_SAMPLES:-0}" == "1" ]]; then
    export QUIVER_AE_HOP_SAMPLES_CSV="${out_dir}/hop_samples.csv"
  else
    export QUIVER_AE_HOP_SAMPLES_CSV=""
  fi
  export QUIVER_AE_STEM="${stem}"
  export QUIVER_AE_DATASET="${DATASET:-}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --ef-search)
        ef="${2:-}"
        shift 2
        ;;
      --ef-search=*)
        ef="${1#*=}"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
  export QUIVER_AE_EF="${ef}"
  export QUIVER_AE_TARGET_RECALL="$(ae_target_recall_from_ef "${ef}")"
}

ae_print_layout() {
  echo "---- PCIe ----"
  echo "  4-SSD     PCI_4SSD=${PCI_4SSD:-<unset>}"
  echo "  sift 1-SSD  PCI_SIFT_1SSD=${PCI_SIFT_1SSD:-<unset>}"
  echo "  deep 1-SSD  PCI_DEEP_1SSD=${PCI_DEEP_1SSD:-<unset>}"
  echo "---- 4-SSD LBA (N=4) ----"
  echo "  deep1b  DEEP1B_SPDK_BASE_LBA=${DEEP1B_SPDK_BASE_LBA}"
  echo "  deep1m  DEEP1M_SPDK_BASE_LBA=${DEEP1M_SPDK_BASE_LBA}"
  echo "  sift1b  SIFT1B_SPDK_BASE_LBA=${SIFT1B_SPDK_BASE_LBA}"
  echo "  sift1m  SIFT1M_SPDK_BASE_LBA=${SIFT1M_SPDK_BASE_LBA}"
  echo "---- 1-SSD LBA (N=1, two disks) ----"
  echo "  sift1m  SIFT1M_1SSD_SPDK_BASE_LBA=${SIFT1M_1SSD_SPDK_BASE_LBA}  pci=${PCI_SIFT_1SSD}"
  echo "  sift1b  SIFT1B_1SSD_SPDK_BASE_LBA=${SIFT1B_1SSD_SPDK_BASE_LBA}  pci=${PCI_SIFT_1SSD}"
  echo "  deep1m  DEEP1M_1SSD_SPDK_BASE_LBA=${DEEP1M_1SSD_SPDK_BASE_LBA}  pci=${PCI_DEEP_1SSD}"
  echo "  deep1b  DEEP1B_1SSD_SPDK_BASE_LBA=${DEEP1B_1SSD_SPDK_BASE_LBA}  pci=${PCI_DEEP_1SSD}"
}

# Kill sudo search and its children (do not use setsid: bash & already has a pgid).
ae_kill_search() {
  local pid="$1"
  local kids
  kids="$(pgrep -P "${pid}" 2>/dev/null || true)"
  sudo -n kill -TERM "${pid}" ${kids} 2>/dev/null || true
  sleep 2
  kids="$(pgrep -P "${pid}" 2>/dev/null || true)"
  sudo -n kill -KILL "${pid}" ${kids} 2>/dev/null || true
  kill -KILL "${pid}" 2>/dev/null || true
}

# Watch one process that sweeps occupancies in-process (do not split the list).
# Hang limit: first occupancy includes SPDK/EAL; later max(HANG_FLOOR,
# HANG_MULT * prev). A healthy 1B point with REPEAT=20 is usually well
# under two minutes including SPDK; 300s is a conservative floor so a
# slow but live occupancy is not killed. On timeout the watcher kills
# the process and run_search retries the same command.
ae_watch_search() {
  local log="$1"
  local pid="$2"
  local hang_mult="${HANG_MULT:-4}"
  local hang_floor="${HANG_FLOOR:-300}"
  local hang_floor_first="${HANG_FLOOR_FIRST:-300}"
  local poll_s="${HANG_POLL_SEC:-1}"
  local last_dur=0
  local point_t
  point_t="$(date +%s)"
  local in_point=1
  local label="startup/first-point"
  local sweep_n=0
  local recall_n=0
  local last_sweep_n=0
  local last_recall_n=0

  ae_point_limit() {
    local limit="${hang_floor_first}"
    if (( last_dur > 0 )); then
      limit=$((hang_mult * last_dur))
      if (( limit < hang_floor )); then
        limit="${hang_floor}"
      fi
    fi
    echo "${limit}"
  }

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${poll_s}"
    sweep_n="$(grep -cE 'Sweep:' "${log}" 2>/dev/null || true)"
    recall_n="$(grep -cE 'Recall @' "${log}" 2>/dev/null || true)"
    sweep_n="${sweep_n:-0}"
    recall_n="${recall_n:-0}"
    local now
    now="$(date +%s)"

    if (( sweep_n > last_sweep_n )); then
      if (( in_point == 1 && last_sweep_n > 0 )); then
        last_dur=$((now - point_t))
        if (( last_dur < 1 )); then
          last_dur=1
        fi
      fi
      local sweep_line
      sweep_line="$(grep -E 'Sweep:' "${log}" | tail -n 1 | tr -d '\033' | sed 's/\[[0-9;]*m//g')"
      if [[ "${sweep_line}" =~ num_blocks=([0-9]+) ]]; then
        label="num_blocks=${BASH_REMATCH[1]}"
      elif [[ "${sweep_line}" =~ mini.batch=([0-9]+) ]]; then
        label="mini_batch=${BASH_REMATCH[1]}"
      else
        label="sweep#${sweep_n}"
      fi
      in_point=1
      point_t="${now}"
      last_sweep_n="${sweep_n}"
      echo "watch: start ${label} (prev_dur=${last_dur}s limit=$(ae_point_limit)s)"
    fi

    if (( recall_n > last_recall_n )); then
      last_recall_n="${recall_n}"
      # A lagging Recall belongs to the previous occupancy. Only the current
      # point is done when Recall count has caught up to Sweep count.
      if (( sweep_n == 0 || recall_n >= sweep_n )); then
        last_dur=$((now - point_t))
        if (( last_dur < 1 )); then
          last_dur=1
        fi
        echo "watch: done ${label} in ${last_dur}s next_limit=$(ae_point_limit)s"
        in_point=0
      fi
    fi

    if (( in_point == 1 )); then
      local elapsed=$((now - point_t))
      local limit
      limit="$(ae_point_limit)"
      if (( elapsed > limit )); then
        {
          echo
          echo "HANG: ${label} elapsed=${elapsed}s limit=${limit}s prev_dur=${last_dur}s (HANG_MULT=${hang_mult})"
          echo "hint: occupancy hung in a consecutive sweep; do not treat isolated re-runs as proof"
        } | tee -a "${log}"
        ae_kill_search "${pid}"
        wait "${pid}" 2>/dev/null || true
        return 124
      fi
    fi
  done
  wait "${pid}"
  return $?
}

# Drop this stem's rows from shared CSVs so a timeout retry does not
# append a second partial sweep next to the hung one.
ae_drop_stem_rows() {
  local dir="$1"
  local stem="$2"
  local name
  for name in metrics.csv breakdown.csv cta_samples.csv hop_samples.csv; do
    [[ -f "${dir}/${name}" ]] || continue
    python3 - "${dir}/${name}" "${stem}" <<'PY'
import csv
import sys
from pathlib import Path

path, stem = Path(sys.argv[1]), sys.argv[2]
with path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    rows = list(reader)
if not fieldnames:
    raise SystemExit(0)
kept = [row for row in rows if row.get("stem") != stem]
with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept)
PY
  done
}

# Search binaries colorize stdout. Strip those codes so reviewer logs are
# plain text (plotters already ignore them).
ae_strip_ansi_log() {
  local log="$1"
  [[ -f "${log}" ]] || return 0
  python3 - "${log}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(errors="replace")
clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
if clean != text:
    path.write_text(clean)
PY
}

# $1 log path, remaining args are the program and its flags.
run_search() {
  local log="$1"
  shift
  local expected=1 signature="" group_dir="" figure_root="" cached=""
  local csv_log="${log}"
  if [[ -n "${AE_DEBUG_ROOT}" ]]; then
    local run_dir stem
    run_dir="$(cd -- "$(dirname -- "${log}")" && pwd)"
    stem="$(basename "${log}" .log)"
    figure_root="$(cd -- "${run_dir}/../.." && pwd)"
    expected="$(ae_expected_points "$@")"
    signature="$(ae_debug_signature "$@")"
    if [[ "${AE_DEBUG_FORCE}" != 1 ]]; then
      cached="$(ae_find_cached_group "${figure_root}" "${stem}" "${signature}" "${expected}")"
    fi
    if [[ -n "${cached}" ]]; then
      echo "reuse complete sweep group ${cached}"
      local cached_log
      cached_log="$(awk -F= '$1=="log_path" {sub(/^log_path=/, ""); print}' \
        "${cached}/manifest.env")"
      if [[ -z "${cached_log}" && -f "${cached}/${stem}.log" ]]; then
        cached_log="${cached}/${stem}.log"
      fi
      if [[ -n "${cached_log}" && -f "${cached_log}" ]]; then
        cp -- "${cached_log}" "${log}"
        ae_strip_ansi_log "${log}"
      fi
      return 0
    fi
    group_dir="${run_dir}/groups/${stem}"
    mkdir -p "${group_dir}"
    # Keep the caller-visible log where requested; only normalized CSVs and
    # the completion manifest are grouped for immutable history/reuse.
    csv_log="${group_dir}/${stem}.log"
  fi
  ae_export_result_csv "${csv_log}" "$@"
  local csv_dir
  csv_dir="$(cd -- "$(dirname -- "${csv_log}")" && pwd)"
  local stem
  stem="$(basename "${log}" .log)"
  local retries="${HANG_RETRIES:-2}"
  local attempt=0
  local rc=0
  while true; do
    {
      echo "dataset=${DATASET:-}"
      echo "AE_SCALE=${AE_SCALE}"
      echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
      echo "SPDK_BASE_LBA=${SPDK_BASE_LBA}"
      echo "QUIVER_SPDK_BLOCK_OFFSET=${QUIVER_SPDK_BLOCK_OFFSET}"
      echo "ssd_list=${SSD_LIST}"
      echo "PCI_1SSD=${PCI_1SSD:-}"
      echo "BIN_DIR=${BIN_DIR}"
      printf 'cmd:'
      printf ' %q' "$@"
      printf '\n'
    } | tee "${log}"
    sudo -n stdbuf -oL -eL env \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      SPDK_QPAIRS_PER_SSD=1 \
      SPDK_NUM_QP=1 \
      QUIVER_SPDK_DIAG=0 \
      SPDK_BASE_LBA="${SPDK_BASE_LBA}" \
      QUIVER_SPDK_BLOCK_OFFSET="${QUIVER_SPDK_BLOCK_OFFSET}" \
      QUIVER_AE_METRICS_CSV="${QUIVER_AE_METRICS_CSV:-}" \
      QUIVER_AE_BREAKDOWN_CSV="${QUIVER_AE_BREAKDOWN_CSV:-}" \
      QUIVER_AE_BREAKDOWN_SAMPLES_CSV="${QUIVER_AE_BREAKDOWN_SAMPLES_CSV:-}" \
      QUIVER_AE_CTA_SAMPLES_CSV="${QUIVER_AE_CTA_SAMPLES_CSV:-}" \
      QUIVER_AE_HOP_SAMPLES_CSV="${QUIVER_AE_HOP_SAMPLES_CSV:-}" \
      QUIVER_AE_STEM="${QUIVER_AE_STEM:-}" \
      QUIVER_AE_DATASET="${QUIVER_AE_DATASET:-}" \
      QUIVER_AE_EF="${QUIVER_AE_EF:-}" \
      QUIVER_AE_TARGET_RECALL="${QUIVER_AE_TARGET_RECALL:-}" \
      QUIVER_AE_VARIANT="${QUIVER_AE_VARIANT:-}" \
      ${AE_NUMA_PREFIX} \
      "$@" >> "${log}" 2>&1 &
    local pid=$!
    rc=0
    ae_watch_search "${log}" "${pid}" || rc=$?
    ae_strip_ansi_log "${log}"
    if [[ "${rc}" -eq 0 ]]; then
      break
    fi
    if [[ "${rc}" -eq 124 && "${attempt}" -lt "${retries}" ]]; then
      attempt=$((attempt + 1))
      echo "timeout: killed hung point, retry ${attempt}/${retries} after 5s  log=${log}" | tee -a "${log}"
      cp -- "${log}" "${log}.hang${attempt}" 2>/dev/null || true
      ae_drop_stem_rows "${csv_dir}" "${stem}"
      sleep 5
      continue
    fi
    if [[ -n "${group_dir}" ]]; then
      {
        echo "status=failed"
        echo "signature=${signature}"
        echo "expected_points=${expected}"
        echo "exit_code=${rc}"
      } > "${group_dir}/manifest.env"
    fi
    echo "search failed rc=${rc} log=${log}" >&2
    return "${rc}"
  done
  if [[ -n "${group_dir}" ]]; then
    if ! ae_validate_metrics "${group_dir}/metrics.csv" \
         "$(basename "${log}" .log)" "${expected}"; then
      echo "incomplete sweep group: ${group_dir} (expected ${expected} valid points)" >&2
      return 1
    fi
    {
      echo "status=complete"
      echo "signature=${signature}"
      echo "expected_points=${expected}"
      echo "log_path=${log}"
      echo "completed_at=$(date -Iseconds)"
    } > "${group_dir}/manifest.env"
  fi
}

# FusionANNS query_server (host FS index, not SPDK). Skips if index missing.
run_fusionanns_query() {
  local log="$1"
  local index_path="$2"
  shift 2
  local qs="${FUSIONANNS_DIR}/bin/query_server"
  if [[ ! -x "${qs}" ]]; then
    echo "skip FusionANNS: missing ${qs} (compile with ae/scripts/build.sh)" | tee "${log}"
    return 0
  fi
  if [[ ! -d "${index_path}" ]]; then
    echo "skip FusionANNS: missing index ${index_path}" | tee "${log}"
    return 0
  fi
  {
    echo "dataset=${DATASET:-}"
    echo "fusionanns_index=${index_path}"
    printf 'cmd:'
    printf ' %q' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${qs}" "$@"
    printf '\n'
  } | tee "${log}"
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${qs}" "$@" >> "${log}" 2>&1
  ae_strip_ansi_log "${log}"
}
