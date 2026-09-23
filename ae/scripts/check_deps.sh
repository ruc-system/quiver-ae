#!/usr/bin/env bash
# Report which build/plot dependencies are missing and how to install them.
# Checks only; it never installs anything, because the AE server is already
# configured and package installs need root.
#   ./ae/scripts/check_deps.sh
set -euo pipefail

ok=0
fail=0
declare -a HINTS=()

pass() { printf '  \033[32mOK\033[0m       %s\n' "$1"; ok=$((ok + 1)); }
miss() {  # $1=what  $2=install hint
  printf '  \033[31mMISSING\033[0m  %s\n' "$1"
  fail=$((fail + 1))
  HINTS+=("$2")
}

need_cmd() {  # $1=command  $2=label  $3=install hint
  if command -v "$1" >/dev/null 2>&1; then
    pass "$2 ($(command -v "$1"))"
  else
    miss "$2" "$3"
  fi
}

echo "---- build toolchain ----"
need_cmd nvcc "CUDA toolkit (nvcc)" \
  "install CUDA 12.x and add it to PATH, e.g. export PATH=/usr/local/cuda-12.8/bin:\$PATH"
if command -v nvcc >/dev/null 2>&1; then
  ver="$(nvcc --release 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  case "${ver}" in
    12.*) pass "CUDA version ${ver}" ;;
    "")   miss "CUDA version could not be parsed" "check that nvcc runs" ;;
    *)    miss "CUDA version ${ver} (need 12.x)" "install CUDA 12.x; the paper builds use 12.8" ;;
  esac
fi
need_cmd cmake "CMake" "apt-get install -y cmake   # >= 3.20"
need_cmd g++ "C++ compiler (g++)" "apt-get install -y build-essential"
need_cmd make "make" "apt-get install -y build-essential"
if command -v g++ >/dev/null 2>&1 && echo 'int main(){}' | g++ -fopenmp -x c++ - -o /dev/null 2>/dev/null; then
  pass "OpenMP support in g++"
else
  miss "OpenMP support in g++" "apt-get install -y libomp-dev"
fi
need_cmd nvidia-smi "NVIDIA driver (nvidia-smi)" "install the NVIDIA driver matching CUDA 12.x"

echo "---- SPDK ----"
need_cmd meson "meson (SPDK build)" "pip3 install --user meson ninja"
need_cmd ninja "ninja (SPDK build)" "pip3 install --user ninja"
if [[ "$(id -u)" -eq 0 ]] || sudo -n true 2>/dev/null; then
  pass "passwordless sudo (SPDK binds NVMe devices)"
else
  miss "passwordless sudo" "SPDK needs root to bind NVMe devices; grant NOPASSWD sudo"
fi

echo "---- plotting ----"
need_cmd python3 "Python 3" "apt-get install -y python3"
if python3 -c 'import matplotlib' 2>/dev/null; then
  pass "matplotlib ($(python3 -c 'import matplotlib; print(matplotlib.__version__)'))"
else
  miss "matplotlib" "pip3 install matplotlib"
fi

echo "---- FusionANNS baseline (Exp #4 only) ----"
need_cmd xmake "xmake" "curl -fsSL https://xmake.io/shget.text | bash"
if ls /usr/include/boost/version.hpp /usr/local/include/boost/version.hpp >/dev/null 2>&1; then
  pass "Boost headers"
else
  miss "Boost headers" "apt-get install -y libboost-all-dev"
fi
if ldconfig -p 2>/dev/null | grep -q libopenblas; then
  pass "OpenBLAS"
else
  miss "OpenBLAS" "apt-get install -y libopenblas-dev"
fi

echo "---- ${ok} ok  ${fail} missing ----"
if [[ "${fail}" -gt 0 ]]; then
  echo
  echo "To install the missing pieces:"
  printf '  %s\n' "${HINTS[@]}"
  echo
  echo "FusionANNS items only matter for Exp #4 (paper Figure 6)."
  exit 1
fi
echo "All dependencies present. Next: ./ae/scripts/build.sh"
