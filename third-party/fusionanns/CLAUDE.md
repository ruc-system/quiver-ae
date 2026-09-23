# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Primary build system**: xmake

- `xmake`: Configure toolchains, resolve dependencies (faiss/openblas), and build all targets in debug mode
- `xmake f -m release`: Reconfigure for release build (adjust CUDA gencodes in xmake.lua for your GPU)
- `xmake build build_index`: Compile the offline index builder, outputs to bin/
- `xmake build query_server`: Compile the query server, outputs to bin/
- `xmake run build_index`: Execute the index builder from repo root, artifacts go to indices/
- `xmake run query_server [args]`: Run the query server with optional arguments

**Dependencies**: faiss, openblas (system packages), CUDA toolkit

**Optional cuVS (GPU TopK + CAGRA)**: For GPU-accelerated TopK and CAGRA centroid graph:
```bash
conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.9
conda activate base
xmake f -c --libcuvs_root=$CONDA_PREFIX --cuda=$CONDA_PREFIX
xmake build query_server
xmake run query_server   # 应看到 [cuVS] GPU TopK enabled (select_k)
```
**注意**：
- 启用 cuVS 时必须使用 `--cuda=$CONDA_PREFIX`，确保 CUDA 工具链与 conda 的 libcuvs 版本一致。
- `-c` 会清空配置缓存，若之前配置时 CONDA_PREFIX 未设置，必须用 `-c` 重新配置，否则会构建出未链接 libcuvs 的二进制。

## Code Architecture

**FusionANNS** is a hybrid CPU-GPU approximate nearest neighbor search system with SSD-optimized storage.

### Module Structure

- **`app/`**: CLI entry points
  - `build_index.cpp`: Offline index builder using FusionAnnsBuilder
  - `query_server.cpp`: Online query server with GPU acceleration and heuristic reranking

- **`src/common/`**: Shared utilities
  - `types.h`: Core data structures (VectorLocation, SSD_PAGE_SIZE)
  - `io.h`/`io.cpp`: File I/O utilities for binary formats (fvecs, ivecs, metadata)
  - `common.h`: General utilities

- **`src/index/`**: Offline index building
  - `builder.h`/`builder.cpp`: FusionAnnsBuilder class implements the offline pipeline
  - Creates optimized layouts, centroids, PQ codes, and metadata

- **`src/online/`**: Online query processing
  - `gpu_kernels.cu`/`gpu_kernels.cuh`: CUDA kernels for PQ distance computation
  - `gpu_pool.h`/`gpu_pool.cpp`: GPU memory pool management (GpuBlockPool, GpuBlock)
  - `io_manager.h`/`io_manager.cpp`: SSD-aware I/O with multi-shard LRU cache

### Key Design Patterns

**Two-phase architecture**: Offline index building (CPU) + online querying (GPU-accelerated)

**GPU resource management**: GpuBlockPool allocates fixed-size blocks with CUDA streams for concurrent processing. Each block contains distance tables, candidate IDs, and scratch space.

**Storage optimization**: Vectors are packed into SSD_PAGE_SIZE (4KB) pages. VectorLocation maps vector IDs to page_id + offset_in_page.

**Caching strategy**: IOManager uses 64-shard LRU cache with direct I/O for SSD reads, minimizing page cache interference.

**Search pipeline**:
1. Graph search on centroids (HNSW)
2. GPU-based PQ distance computation for candidates
3. Heuristic reranking with exact distances and early termination

### Critical Implementation Details

**CUDA architecture**: Configure cuda_gencodes in xmake.lua for target GPU (default: sm_89, compute_89)

**Memory layout**: Vectors stored contiguously in packed format, accessed via location mapping for cache efficiency

**Concurrency**: Query server supports multi-threaded execution with per-thread GPU blocks and shared index structures

**Error handling**: CUDA_CHECK macro for GPU operations, exception-based error propagation

## Python Environment

**Virtual environment**: `.venv/` (Python 3.8)

- Activate: `source .venv/bin/activate`
- Python binary: `.venv/bin/python3`
- Run scripts: `.venv/bin/python3 <script.py>`
- Installed packages: matplotlib, seaborn, pandas, numpy

## Coding Conventions

- C++17 with CUDA extensions
- Two-space indentation, same-line braces
- PascalCase for classes (FusionAnnsBuilder), snake_case for functions (create_optimized_layout)
- Trailing underscores for member variables (base_file_, dim_)
- Lowercase filenames with underscores
- Project-relative includes
- Run `clang-format -style=LLVM` before commits

## Data Flow

**Build phase**: SIFT vectors → FusionAnnsBuilder → indices/ (centroids, PQ codes, metadata, packed vectors, location map)

**Query phase**: Query vectors → centroid graph → candidate IDs → GPU PQ distance → approximate results → heuristic rerank → final top-k

**Artifacts location**: All generated indices stored in `indices/`, sample data in `data/sift/`