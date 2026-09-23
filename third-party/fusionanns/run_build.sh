#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH=/data/root/usr/local/local/lib64:/data/miniconda/lib
mkdir -p "${SCRIPT_DIR}/indices_random110m"
exec "${SCRIPT_DIR}/bin/build_index" \
    --base "${SCRIPT_DIR}/data/random110m/base.fvecs" \
    --base-format fvecs \
    --nlist 1700000 \
    --m 64 \
    --nbits 8 \
    --output-dir "${SCRIPT_DIR}/indices_random110m" \
    --preload \
    --chunk 1000000 \
    --page 4096
