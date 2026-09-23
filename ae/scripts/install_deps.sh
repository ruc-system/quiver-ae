#!/usr/bin/env bash
# Install host packages used to build Quiver, SPDK, the plotter, and FusionANNS.
# CUDA 12.x is not installed here; put its bin directory on PATH first.
#   ./ae/scripts/install_deps.sh
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "install_deps.sh supports apt-based systems; use ae/scripts/check_deps.sh elsewhere" >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  python3 python3-matplotlib \
  meson ninja-build \
  libaio-dev libnuma-dev nasm autoconf automake libtool \
  libboost-all-dev libopenblas-dev liburing-dev curl

if ! command -v xmake >/dev/null 2>&1; then
  curl -fsSL https://xmake.io/shget.text | bash
  echo "xmake installed; open a new shell so ~/.local/bin is on PATH"
fi

echo "host packages installed. Next: ./ae/scripts/check_deps.sh"
