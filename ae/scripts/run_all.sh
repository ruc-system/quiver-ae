#!/usr/bin/env bash
# Test only (no compile). Compile first: ./ae/scripts/build.sh
# ./ae/scripts/run_all.sh [all|e2e|...]
# Production binaries use build/; instrumentation figures use build-ae/.
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${DIR}/../.." && pwd)"
source "${DIR}/paper_knobs.sh"
apply_paper_knobs
want="${1:-all}"

ALL_FIGS=(latency_qps io_latency e2e fusion ablation q_sensitivity)
declare -A PAPER_FIGURE=(
  [latency_qps]=1 [io_latency]=3 [e2e]=5
  [fusion]=6 [ablation]=7 [q_sensitivity]=8
)

selected=()
for name in "${ALL_FIGS[@]}"; do
  if [[ "${want}" == "all" || "${want}" == "${name}" || "${want}" == "fig_${name}.sh" ]]; then
    selected+=("${name}")
  fi
done
if [[ "${#selected[@]}" -eq 0 ]]; then
  echo "unknown experiment: ${want}" >&2
  echo "accepted: all ${ALL_FIGS[*]}" >&2
  exit 1
fi

total="${#selected[@]}"
started="$(date +%s)"
stage_root=""
if [[ -n "${AE_DEBUG_ROOT:-}" ]]; then
  AE_DEBUG_ROOT="$(realpath -m "${AE_DEBUG_ROOT}")"
  AE_DEBUG_RUN_ID="${AE_DEBUG_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
  export AE_DEBUG_ROOT AE_DEBUG_RUN_ID
  mkdir -p "${AE_DEBUG_ROOT}"
  echo "debug history: ${AE_DEBUG_ROOT} (run ${AE_DEBUG_RUN_ID})"
else
  stage_root="${ROOT}/.cache/ae-staging/run.$$"
  mkdir -p "${stage_root}"
  trap 'rm -rf "${stage_root}"' EXIT
fi

publish_result() {
  local name="$1"
  local incoming="${stage_root}/${name}"
  local result_root="${ROOT}/ae/results"
  local final="${result_root}/${name}"
  local backup="${result_root}/.${name}.previous.$$"

  [[ -f "${incoming}/env.txt" ]] || {
    echo "incomplete staged result: ${incoming}" >&2
    return 1
  }
  mkdir -p "${result_root}"
  rm -rf "${backup}"
  if [[ -e "${final}" ]]; then
    mv "${final}" "${backup}"
  fi
  if mv "${incoming}" "${final}"; then
    rm -rf "${backup}"
    echo "published ${final}"
    return 0
  fi
  [[ ! -e "${final}" && -e "${backup}" ]] && mv "${backup}" "${final}"
  return 1
}

i=0
for name in "${selected[@]}"; do
  i=$((i + 1))
  echo
  echo "===== [${i}/${total}] ${name} -> paper Figure ${PAPER_FIGURE[${name}]} (AE_SCALE=${AE_SCALE:-1b}) ====="
  step_started="$(date +%s)"
  if [[ -n "${AE_DEBUG_ROOT:-}" ]]; then
    "${DIR}/fig_${name}.sh"
    echo "kept debug attempt under ${AE_DEBUG_ROOT}/${name}/runs/${AE_DEBUG_RUN_ID}"
  else
    AE_OUTPUT_ROOT="${stage_root}" "${DIR}/fig_${name}.sh"
    publish_result "${name}"
  fi
  echo "----- ${name} done in $(( ($(date +%s) - step_started) / 60 )) min -----"
done

echo
echo "All ${total} experiment(s) finished in $(( ($(date +%s) - started) / 60 )) min."
echo "Next: ${DIR}/plot_all.py"
