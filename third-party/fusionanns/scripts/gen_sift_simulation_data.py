#!/usr/bin/env python3
"""
Generate simulation data from SIFT1B subset for FusionANNS centroid recall testing.

Source: /data/workspace/dataset/sift1b_0.001/
  - base.bin:  1M x 128 uint8 (8-byte header: nrows, dim)
  - query.bin: 10K x 128 uint8
  - gt.bin:    10K x 100 int32

Outputs (data/sift_simulation/):
  centroids.fvecs             — 1.7M x 256 float32
  queries.fvecs               — 10K x 256 float32
  groundtruth_centroids.ivecs — 10K x top-10000 brute-force GT
"""

import argparse
import os
import struct
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers (reused from gen_simulation_data.py)
# ---------------------------------------------------------------------------

def write_fvecs_chunked(path, data, chunk_size=100000):
    """Write float32 matrix (n, d) in fvecs format, in chunks to limit memory."""
    n, d = data.shape
    assert data.dtype == np.float32
    dim_bytes = struct.pack('<i', d)
    row_bytes = 4 + d * 4
    with open(path, 'wb') as f:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = data[start:end]
            cn = chunk.shape[0]
            buf = bytearray(cn * row_bytes)
            offset = 0
            for i in range(cn):
                buf[offset:offset + 4] = dim_bytes
                offset += 4
                buf[offset:offset + d * 4] = chunk[i].tobytes()
                offset += d * 4
            f.write(buf)
            if end % 500000 == 0 or end == n:
                print(f"    ... written {end}/{n} vectors")
    print(f"  Written {path} ({n} vectors, dim={d}, {os.path.getsize(path) / 1e9:.2f} GB)")


def write_ivecs(path, data):
    """Write int32 matrix (n, d) in ivecs format."""
    n, d = data.shape
    assert data.dtype == np.int32
    row_bytes = 4 + d * 4
    buf = bytearray(n * row_bytes)
    dim_bytes = struct.pack('<i', d)
    offset = 0
    for i in range(n):
        buf[offset:offset + 4] = dim_bytes
        offset += 4
        buf[offset:offset + d * 4] = data[i].tobytes()
        offset += d * 4
    with open(path, 'wb') as f:
        f.write(buf)
    print(f"  Written {path} ({n} vectors, dim={d}, {os.path.getsize(path) / 1e9:.2f} GB)")


def compute_groundtruth(queries, centroids, k, chunk_size=20000):
    """
    Brute-force top-k nearest centroids for each query.
    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b trick.
    Processes centroids in chunks to limit memory.
    """
    nq, d = queries.shape
    nc = centroids.shape[0]
    print(f"  Computing GT: {nq} queries x {nc} centroids, top-{k}")

    q_norms = np.sum(queries ** 2, axis=1)  # (nq,)

    gt_ids = np.full((nq, k), -1, dtype=np.int32)
    gt_dists = np.full((nq, k), np.inf, dtype=np.float32)

    t0 = time.time()
    for chunk_start in range(0, nc, chunk_size):
        chunk_end = min(chunk_start + chunk_size, nc)
        c_chunk = centroids[chunk_start:chunk_end]  # (cs, d)

        c_norms = np.sum(c_chunk ** 2, axis=1)  # (cs,)

        # ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q*c
        dists = q_norms[:, None] + c_norms[None, :] - 2.0 * (queries @ c_chunk.T)
        dists = dists.astype(np.float32)

        chunk_ids = np.arange(chunk_start, chunk_end, dtype=np.int32)

        combined_dists = np.concatenate([gt_dists, dists], axis=1)
        combined_ids = np.concatenate(
            [gt_ids, np.tile(chunk_ids, (nq, 1))], axis=1
        )

        part_idx = np.argpartition(combined_dists, k, axis=1)[:, :k]

        row_idx = np.arange(nq)[:, None]
        gt_dists = combined_dists[row_idx, part_idx]
        gt_ids = combined_ids[row_idx, part_idx]

        elapsed = time.time() - t0
        progress = chunk_end / nc * 100
        print(f"    chunk [{chunk_start}:{chunk_end}] "
              f"({progress:.1f}%) elapsed={elapsed:.1f}s")

    # Final sort within each query's top-k by distance
    for i in range(nq):
        order = np.argsort(gt_dists[i])
        gt_ids[i] = gt_ids[i][order]
        gt_dists[i] = gt_dists[i][order]

    total_time = time.time() - t0
    print(f"  GT computation done in {total_time:.1f}s")
    return gt_ids


# ---------------------------------------------------------------------------
# SIFT1B binary reader
# ---------------------------------------------------------------------------

def read_sift1b_bin(path):
    """
    Read SIFT1B-format binary: 8-byte header (nrows: int32, dim: int32)
    followed by nrows*dim uint8 values. Returns float32 array.
    """
    with open(path, 'rb') as f:
        nrows, dim = struct.unpack('<ii', f.read(8))
        print(f"  Reading {path}: {nrows} x {dim} uint8")
        raw = np.frombuffer(f.read(nrows * dim), dtype=np.uint8)
    data = raw.reshape(nrows, dim).astype(np.float32)
    return data


def read_sift1b_gt(path):
    """
    Read SIFT1B ground-truth binary: 8-byte header (nrows: int32, k: int32)
    followed by nrows*k int32 values.
    """
    with open(path, 'rb') as f:
        nrows, k = struct.unpack('<ii', f.read(8))
        print(f"  Reading {path}: {nrows} x {k} int32")
        raw = np.frombuffer(f.read(nrows * k * 4), dtype=np.int32)
    return raw.reshape(nrows, k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SIFT1B-based simulation data for FusionANNS")
    parser.add_argument("--src-dir", type=str,
                        default="/data/workspace/dataset/sift1b_0.001",
                        help="Source SIFT1B subset directory")
    parser.add_argument("--output-dir", type=str,
                        default="data/sift_simulation",
                        help="Output directory")
    parser.add_argument("--topk", type=int, default=10000,
                        help="GT top-k")
    parser.add_argument("--gt-chunk", type=int, default=20000,
                        help="Centroid chunk size for GT computation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Read source data
    # ------------------------------------------------------------------
    print("[1/5] Reading source data...")
    base = read_sift1b_bin(os.path.join(args.src_dir, "base.bin"))
    queries_raw = read_sift1b_bin(os.path.join(args.src_dir, "query.bin"))
    n_base, dim_src = base.shape
    n_query = queries_raw.shape[0]
    print(f"  base: {n_base} x {dim_src}, queries: {n_query} x {dim_src}")

    # ------------------------------------------------------------------
    # Step 2: Build centroids (1.7M x 256)
    #   dim-duplicate each vector: [v, v] → 256-d
    #   concat(all 1M, first 700K) = 1.7M
    # ------------------------------------------------------------------
    print("[2/5] Building centroids: 1.7M x 256...")
    base_256 = np.concatenate([base, base], axis=1)  # 1M x 256
    extra = base_256[:700000]  # 700K x 256
    centroids = np.concatenate([base_256, extra], axis=0)  # 1.7M x 256
    print(f"  centroids shape: {centroids.shape}")

    # Free intermediate arrays, keep centroids
    del base, base_256, extra

    centroids_path = os.path.join(args.output_dir, "centroids.fvecs")
    print("[3/5] Writing centroids.fvecs...")
    write_fvecs_chunked(centroids_path, centroids)

    # ------------------------------------------------------------------
    # Step 3: Build queries (10K x 256)
    # ------------------------------------------------------------------
    print("[4/5] Building and writing queries: 10K x 256...")
    queries = np.concatenate([queries_raw, queries_raw], axis=1)  # 10K x 256
    del queries_raw
    queries_path = os.path.join(args.output_dir, "queries.fvecs")
    write_fvecs_chunked(queries_path, queries)

    # ------------------------------------------------------------------
    # Step 4: Compute ground truth
    # ------------------------------------------------------------------
    print(f"[5/5] Computing brute-force top-{args.topk} ground truth...")
    gt_ids = compute_groundtruth(queries, centroids, args.topk,
                                 chunk_size=args.gt_chunk)
    gt_path = os.path.join(args.output_dir, "groundtruth_centroids.ivecs")
    write_ivecs(gt_path, gt_ids)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nDone! Generated files:")
    for f in ["centroids.fvecs", "queries.fvecs", "groundtruth_centroids.ivecs"]:
        fpath = os.path.join(args.output_dir, f)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  {fpath} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
