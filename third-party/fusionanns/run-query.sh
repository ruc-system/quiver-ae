#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH=/data/root/usr/local/local/lib64:/data/miniconda/lib
exec "${SCRIPT_DIR}/bin/query_server" \
    --query "${SCRIPT_DIR}/data/sift/sift_query.fvecs" \
    --query-format fvecs \
    --groundtruth "${SCRIPT_DIR}/data/sift/sift_groundtruth.ivecs" \
    --groundtruth-format ivecs \
    --index-path "${SCRIPT_DIR}/indices_sift1m/" \
    --warmup 10000 \
    --threads 48 \
    --nprobe 48 \
    --rerank_size 512 \
    --batch_size 32 \
    --disable-query-stats \
    --rerank_stable_iters 2 \
    --rerank_delta 0 \
    --no-gpu-topk \
    --no-use-cagra \
    --io_backend pread 2>&1
