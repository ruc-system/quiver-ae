#!/usr/bin/env bash
# xmake FusionANNS (sm_80). FUSIONANNS_CUGENCODES overrides arch.
# FAISS is vendored under third-party/fusionanns/extern/faiss (xmake package).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FUSIONANNS_CUGENCODES="${FUSIONANNS_CUGENCODES:-}"
XMAKE_ROOT="${XMAKE_ROOT:-y}"
export XMAKE_ROOT

if ! command -v xmake >/dev/null 2>&1; then
  echo "xmake not in PATH" >&2
  exit 1
fi
if [[ ! -f "${FUSIONANNS_DIR}/xmake.lua" ]]; then
  echo "missing ${FUSIONANNS_DIR}/xmake.lua" >&2
  exit 1
fi
required_cmake_files=(
  "extern/faiss/CMakeLists.txt"
  "extern/faiss/faiss/CMakeLists.txt"
  "extern/SPTAG/CMakeLists.txt"
  "extern/SPTAG/AnnService/CMakeLists.txt"
  "extern/SPTAG/ThirdParty/zstd/build/cmake/CMakeLists.txt"
)
for cmake_file in "${required_cmake_files[@]}"; do
  if [[ ! -f "${FUSIONANNS_DIR}/${cmake_file}" ]]; then
    echo "missing vendored FusionANNS dependency file: ${cmake_file}" >&2
    exit 1
  fi
done

if [[ -n "${FUSIONANNS_CUGENCODES}" ]]; then
  python3 - "${FUSIONANNS_DIR}/xmake.lua" "${FUSIONANNS_CUGENCODES}" <<'PY'
import re, sys
path, codes = sys.argv[1], sys.argv[2]
quoted = ", ".join(f'"{c.strip()}"' for c in codes.split(",") if c.strip())
text = open(path).read()
repl, n = re.subn(
    r'local cuda_gencodes = \{[^}]*\}',
    f'local cuda_gencodes = {{ {quoted} }}',
    text, count=1)
if n != 1:
    sys.exit("could not patch cuda_gencodes")
open(path, "w").write(repl)
PY
fi

cd "${FUSIONANNS_DIR}"
xmake f -m release -y
xmake build -y build_index query_server
test -x "${FUSIONANNS_DIR}/bin/build_index"
test -x "${FUSIONANNS_DIR}/bin/query_server"
echo "FusionANNS binaries in ${FUSIONANNS_DIR}/bin"
