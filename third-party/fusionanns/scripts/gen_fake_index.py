#!/usr/bin/env python3
"""
Generate fake index files for FusionANNS end-to-end latency testing.

This creates minimal index artifacts so query_server can run the
graph_search + GPU PQ scoring stages without needing real 110M vectors.

Outputs (under --output-dir):
  posting_lists_metadata  — 1.7M posting lists, ~65 vectors each
  pq_codes               — 110M PQ codes (M=64, uint8)
  vector_location.map    — trivial sequential mapping
  packed_raw_vectors.bin  — sparse file (placeholder, not read with rerank=0)

The pq_codec must be generated separately via C++ (faiss::write_ProductQuantizer).
"""

import argparse
import os
import struct
import sys
import time

import numpy as np


def gen_posting_lists_metadata(path, nlist, total_vectors):
    """
    Generate posting list metadata in the mmapped binary format expected by
    PostingListAccessor.

    Binary format:
      [nlist (uint32)]
      [table: nlist entries of (offset: uint64, count: uint32, reserved: uint32)]
      [posting data: int32[] — vector IDs for all lists concatenated]
    """
    print(f"  Generating posting list metadata: nlist={nlist}, total_vectors={total_vectors}")

    # Distribute vectors roughly evenly across lists
    base_count = total_vectors // nlist
    remainder = total_vectors % nlist

    # Build the table and posting data
    offsets = []
    counts = []
    current_offset = 0  # offset in int32 elements

    for i in range(nlist):
        count = base_count + (1 if i < remainder else 0)
        offsets.append(current_offset)
        counts.append(count)
        current_offset += count

    assert current_offset == total_vectors, f"{current_offset} != {total_vectors}"

    # Write binary file
    with open(path, 'wb') as f:
        # Header: nlist
        f.write(struct.pack('<I', nlist))

        # Table: nlist entries of (offset_u64, count_u32, reserved_u32)
        for i in range(nlist):
            f.write(struct.pack('<QII', offsets[i], counts[i], 0))

        # Posting data: sequential vector IDs
        chunk_size = 10_000_000  # write 10M IDs at a time
        vid = 0
        while vid < total_vectors:
            end = min(vid + chunk_size, total_vectors)
            ids = np.arange(vid, end, dtype=np.int32)
            f.write(ids.tobytes())
            vid = end
            if vid % 50_000_000 == 0 or vid == total_vectors:
                print(f"    ... written {vid}/{total_vectors} posting IDs")

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Written {path} ({size_mb:.1f} MB)")


def gen_pq_codes(path, total_vectors, m_sub=64):
    """
    Generate random PQ codes: total_vectors x m_sub uint8 values.
    Each vector is encoded as m_sub bytes (one per sub-quantizer).
    Written as a flat binary file.
    """
    print(f"  Generating PQ codes: {total_vectors} x {m_sub} = "
          f"{total_vectors * m_sub / 1e9:.2f} GB")

    chunk_size = 5_000_000  # 5M vectors per chunk
    rng = np.random.RandomState(123)

    with open(path, 'wb') as f:
        written = 0
        while written < total_vectors:
            n = min(chunk_size, total_vectors - written)
            codes = rng.randint(0, 256, size=(n, m_sub), dtype=np.uint8)
            f.write(codes.tobytes())
            written += n
            if written % 20_000_000 == 0 or written == total_vectors:
                print(f"    ... written {written}/{total_vectors} PQ codes")

    size_gb = os.path.getsize(path) / 1e9
    print(f"  Written {path} ({size_gb:.2f} GB)")


def gen_vector_location_map(path, total_vectors, dim, page_size=4096):
    """
    Generate vector_location.map: maps vector ID -> (page_id, offset_in_page).

    VectorLocation struct: { uint32_t page_id; uint16_t offset_in_page; }
    Total 6 bytes per entry.

    Vectors are packed sequentially into pages.
    """
    print(f"  Generating vector location map: {total_vectors} vectors")

    vec_bytes = dim * 4  # float32
    vecs_per_page = page_size // vec_bytes
    if vecs_per_page == 0:
        # Vector larger than one page — each vector starts at page boundary
        vecs_per_page = 1

    # Write in chunks
    chunk_size = 10_000_000
    with open(path, 'wb') as f:
        for start in range(0, total_vectors, chunk_size):
            end = min(start + chunk_size, total_vectors)
            n = end - start
            buf = bytearray(n * 6)  # 6 bytes per VectorLocation
            offset = 0
            for vid in range(start, end):
                page_id = vid // vecs_per_page
                offset_in_page = (vid % vecs_per_page) * vec_bytes
                struct.pack_into('<IH', buf, offset, page_id, offset_in_page)
                offset += 6
            f.write(buf)
            if end % 50_000_000 == 0 or end == total_vectors:
                print(f"    ... written {end}/{total_vectors} entries")

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Written {path} ({size_mb:.1f} MB)")


def gen_sparse_packed_vectors(path, total_vectors, dim):
    """
    Create a sparse file for packed_raw_vectors.bin.
    Total size = total_vectors * dim * 4 bytes.
    Using ftruncate to create a sparse file (no actual disk usage).
    """
    total_bytes = total_vectors * dim * 4
    print(f"  Creating sparse packed vectors file: {total_bytes / 1e9:.1f} GB (sparse)")
    with open(path, 'wb') as f:
        f.seek(total_bytes - 1)
        f.write(b'\x00')
    print(f"  Written {path} (sparse, logical size={total_bytes / 1e9:.1f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fake FusionANNS index for latency testing")
    parser.add_argument("--nlist", type=int, default=1700000,
                        help="Number of posting lists (centroids)")
    parser.add_argument("--total-vectors", type=int, default=110000000,
                        help="Total number of vectors")
    parser.add_argument("--dim", type=int, default=256,
                        help="Vector dimension")
    parser.add_argument("--m-sub", type=int, default=64,
                        help="PQ sub-quantizer count")
    parser.add_argument("--output-dir", type=str, default="indices_simulation",
                        help="Output index directory")
    parser.add_argument("--skip-pq-codes", action="store_true",
                        help="Skip generating large PQ codes file")
    parser.add_argument("--skip-packed-vectors", action="store_true",
                        help="Skip generating packed vectors file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # 1. Posting list metadata
    print("[1/4] Generating posting list metadata...")
    gen_posting_lists_metadata(
        os.path.join(args.output_dir, "posting_lists_metadata"),
        args.nlist, args.total_vectors)

    # 2. PQ codes
    if not args.skip_pq_codes:
        print("\n[2/4] Generating PQ codes...")
        gen_pq_codes(
            os.path.join(args.output_dir, "pq_codes"),
            args.total_vectors, args.m_sub)
    else:
        print("\n[2/4] Skipped PQ codes generation")

    # 3. Vector location map
    print("\n[3/4] Generating vector location map...")
    gen_vector_location_map(
        os.path.join(args.output_dir, "vector_location.map"),
        args.total_vectors, args.dim)

    # 4. Sparse packed vectors
    if not args.skip_packed_vectors:
        print("\n[4/4] Creating sparse packed vectors file...")
        gen_sparse_packed_vectors(
            os.path.join(args.output_dir, "packed_raw_vectors.bin"),
            args.total_vectors, args.dim)
    else:
        print("\n[4/4] Skipped packed vectors file")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s! Index artifacts in {args.output_dir}/")
    print("\nNote: You still need to generate pq_codec using C++:")
    print("  ./bin/build_centroid_graph_standalone can be extended, or use")
    print("  a separate tool to train faiss::ProductQuantizer on centroids")
    print("  and call faiss::write_ProductQuantizer().")


if __name__ == "__main__":
    main()
