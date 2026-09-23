#!/usr/bin/env bash
# Compile every binary AE tests need. Does not write SSD images or run figures.
#   ./ae/scripts/build.sh
# SKIP_FUSIONANNS=1 skips FusionANNS xmake.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/common.sh"

if ! command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
  if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
  fi
fi

nproc_j="$(nproc)"

ensure_cmake_tree() {
  local build_dir="$1"
  shift
  mkdir -p "${build_dir}"
  cmake -S "${ROOT}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-80}" \
    -DQUIVER_ENABLE_SPDK=ON \
    "$@"
}

echo "===== production ${BIN_DIR} ====="
ensure_cmake_tree "${BIN_DIR}" \
  -DQUIVER_ENABLE_LIGHT_BREAKDOWN=OFF \
  -DQUIVER_BUILD_STRAWMEN=OFF
cmake --build "${BIN_DIR}" -j"${nproc_j}" --target \
  quiver_search flashanns_search gustann_search spdk_write

echo "===== instrumented ${INSTRUMENTED_BIN_DIR} ====="
ensure_cmake_tree "${INSTRUMENTED_BIN_DIR}" \
  -DQUIVER_ENABLE_LIGHT_BREAKDOWN=ON \
  -DQUIVER_BUILD_STRAWMEN=ON
cmake --build "${INSTRUMENTED_BIN_DIR}" -j"${nproc_j}" --target \
  quiver_search flashanns_search gustann_search strawmen_pq_in_smem

if [[ "${SKIP_FUSIONANNS:-0}" == 1 ]]; then
  echo "skip FusionANNS (SKIP_FUSIONANNS=1)"
else
  echo "===== FusionANNS ${FUSIONANNS_DIR} ====="
  "${DIR}/fusionanns_build.sh"
fi

missing=0
for b in quiver_search flashanns_search gustann_search spdk_write; do
  if [[ ! -x "${BIN_DIR}/bin/${b}" ]]; then
    echo "missing ${BIN_DIR}/bin/${b}" >&2
    missing=1
  fi
done
for b in quiver_search flashanns_search gustann_search strawmen_pq_in_smem; do
  if [[ ! -x "${INSTRUMENTED_BIN_DIR}/bin/${b}" ]]; then
    echo "missing ${INSTRUMENTED_BIN_DIR}/bin/${b}" >&2
    missing=1
  fi
done
if [[ "${SKIP_FUSIONANNS:-0}" != 1 ]]; then
  for b in query_server build_index; do
    if [[ ! -x "${FUSIONANNS_DIR}/bin/${b}" ]]; then
      echo "missing ${FUSIONANNS_DIR}/bin/${b}" >&2
      missing=1
    fi
  done
fi
[[ "${missing}" -eq 0 ]]
echo "build done (compile only; run tests with ${DIR}/run_all.sh)"
