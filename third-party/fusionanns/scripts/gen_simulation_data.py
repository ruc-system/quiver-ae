#!/usr/bin/env python3
"""
Generate simulation data for FusionANNS centroid recall testing.

Outputs:
  data/simulation/centroids.fvecs  — 1.7M random centroids (dim=256)
  data/simulation/queries.fvecs    — 10K random queries   (dim=256)
  data/simulation/groundtruth_centroids.ivecs — brute-force top-10000 GT
"""

import argparse
import os
import struct
import sys
import time

import numpy as np


def write_fvecs(path, data):
    """Write float32 matrix (n, d) in fvecs format."""
    n, d = data.shape
    assert data.dtype == np.float32
    # Each row: [dim (int32)] [d floats]
    row_bytes = 4 + d * 4  # dim header + float data
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

    # Pre-compute query norms: (nq,)
    q_norms = np.sum(queries ** 2, axis=1)  # (nq,)

    # We maintain top-k per query using a running sort approach
    # Start with infinite distances
    gt_ids = np.full((nq, k), -1, dtype=np.int32)
    gt_dists = np.full((nq, k), np.inf, dtype=np.float32)

    t0 = time.time()
    for chunk_start in range(0, nc, chunk_size):
        chunk_end = min(chunk_start + chunk_size, nc)
        c_chunk = centroids[chunk_start:chunk_end]  # (cs, d)
        cs = c_chunk.shape[0]

        # Centroid norms: (cs,)
        c_norms = np.sum(c_chunk ** 2, axis=1)  # (cs,)

        # Distance matrix: (nq, cs)
        # ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q*c
        dists = q_norms[:, None] + c_norms[None, :] - 2.0 * (queries @ c_chunk.T)
        dists = dists.astype(np.float32)

        # Global IDs for this chunk
        chunk_ids = np.arange(chunk_start, chunk_end, dtype=np.int32)

        # For each query, merge current chunk results with running top-k
        # Concatenate running top-k with chunk results, then take top-k
        combined_dists = np.concatenate([gt_dists, dists], axis=1)  # (nq, k + cs)
        combined_ids = np.concatenate(
            [gt_ids, np.tile(chunk_ids, (nq, 1))], axis=1
        )  # (nq, k + cs)

        # Partial sort: find k smallest per row
        part_idx = np.argpartition(combined_dists, k, axis=1)[:, :k]  # (nq, k)

        # Gather (compatible with numpy < 1.15 which lacks take_along_axis)
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


def main():
    parser = argparse.ArgumentParser(description="Generate simulation data for FusionANNS")
    parser.add_argument("--nlist", type=int, default=1700000, help="Number of centroids")
    parser.add_argument("--dim", type=int, default=256, help="Vector dimension")
    parser.add_argument("--nq", type=int, default=10000, help="Number of queries")
    parser.add_argument("--topk", type=int, default=10000, help="GT top-k")
    parser.add_argument("--output-dir", type=str, default="data/simulation",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gt-chunk", type=int, default=20000,
                        help="Centroid chunk size for GT computation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Step 1: Generate centroids
    print(f"[1/3] Generating {args.nlist} centroids (dim={args.dim})...")
    centroids = rng.randn(args.nlist, args.dim).astype(np.float32)
    centroids_path = os.path.join(args.output_dir, "centroids.fvecs")
    write_fvecs_chunked(centroids_path, centroids)

    # Step 2: Generate queries
    print(f"[2/3] Generating {args.nq} queries (dim={args.dim})...")
    queries = rng.randn(args.nq, args.dim).astype(np.float32)
    queries_path = os.path.join(args.output_dir, "queries.fvecs")
    write_fvecs(queries_path, queries)

    # Step 3: Compute ground truth
    print(f"[3/3] Computing brute-force top-{args.topk} ground truth...")
    gt_ids = compute_groundtruth(queries, centroids, args.topk,
                                 chunk_size=args.gt_chunk)
    gt_path = os.path.join(args.output_dir, "groundtruth_centroids.ivecs")
    write_ivecs(gt_path, gt_ids)

    print("\nDone! Generated files:")
    for f in ["centroids.fvecs", "queries.fvecs", "groundtruth_centroids.ivecs"]:
        fpath = os.path.join(args.output_dir, f)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  {fpath} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
