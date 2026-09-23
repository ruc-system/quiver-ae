#!/usr/bin/env python3
"""生成随机 FP32 fvecs 数据集
用法: python gen_random_fvecs.py --output <path> --num <N> --dim <D> [--chunk <C>]
"""

import argparse
import os
import struct
import time

import numpy as np


def write_fvecs_chunk(fp, data: np.ndarray):
    """将一批向量以 fvecs 格式写入文件。
    fvecs 格式: 每行 = [int32 dim] + [float32 × dim]
    """
    n, d = data.shape
    # 构造每行的头部 (dim as int32)
    dim_bytes = struct.pack('<i', d)
    dim_col = np.frombuffer(dim_bytes * n, dtype=np.int32).reshape(n, 1)
    # 将 dim 列和数据列拼接 (在字节层面)
    row_bytes = np.empty((n, 1 + d), dtype=np.float32)
    row_bytes[:, 0] = dim_col[:, 0].view(np.float32)
    row_bytes[:, 1:] = data
    fp.write(row_bytes.tobytes())


def main():
    parser = argparse.ArgumentParser(description="生成随机 fvecs 数据集")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--num", type=int, default=110_000_000, help="向量数量 (默认 110M)")
    parser.add_argument("--dim", type=int, default=256, help="向量维度 (默认 256)")
    parser.add_argument("--chunk", type=int, default=1_000_000, help="每次写入的向量数 (默认 1M)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    total = args.num
    dim = args.dim
    chunk = args.chunk
    
    # 预计文件大小
    file_size_gb = total * (4 + dim * 4) / (1024**3)
    print(f"准备生成 {total:,} 个 {dim} 维 FP32 向量")
    print(f"预计文件大小: {file_size_gb:.1f} GB")
    print(f"每批写入: {chunk:,} 个向量")
    print(f"输出路径: {args.output}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    np.random.seed(args.seed)
    
    t0 = time.time()
    written = 0
    with open(args.output, "wb") as fp:
        while written < total:
            batch = min(chunk, total - written)
            # 生成标准正态分布随机向量
            data = np.random.randn(batch, dim).astype(np.float32)
            write_fvecs_chunk(fp, data)
            written += batch
            elapsed = time.time() - t0
            speed = written / elapsed
            eta = (total - written) / speed if speed > 0 else 0
            print(f"\r  已写入: {written:>12,} / {total:,}  "
                  f"({100*written/total:5.1f}%)  "
                  f"速度: {speed/1e6:.2f}M vec/s  "
                  f"ETA: {eta/60:.1f} min", end="", flush=True)
    
    elapsed = time.time() - t0
    actual_size = os.path.getsize(args.output)
    print(f"\n完成! 用时: {elapsed:.1f}s, 文件大小: {actual_size/(1024**3):.2f} GB")


if __name__ == "__main__":
    main()
