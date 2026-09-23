# FusionANNS 系统架构文档

## 概述

FusionANNS 是一个高性能的近似最近邻搜索系统，专为十亿级向量数据集设计。系统采用 **SPANN 风格**的架构设计，结合了 GPU 加速的 PQ 评分和基于 SSD 的精确重排序。

### SPANN 风格的关键特征

| 特征 | 传统 IVF | SPANN / FusionANNS |
|------|----------|--------------------|
| 聚类数量 | ~4,096 | **N × 10%** (如 100M→10M) |
| 聚类大小 | ~25,000 向量 | **~10 向量** (复制前) |
| 复制因子 | 1x (无复制) | **7-8x** |
| 每个聚类实际大小 | ~25,000 | **~70-80 向量** |

**设计理念**：使用非常多的小聚类 + 高复制因子，使得边界向量被多个聚类覆盖，从而提高召回率。

## 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                        FusionANNS                                │
├─────────────────────────────────────────────────────────────────┤
│  构建阶段 (Offline)                                              │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌─────────────────┐  │
│  │   HBC   │→ │ Centroid │→ │ Boundary  │→ │ PQ 编码 +       │  │
│  │ 聚类    │  │ Navigator│  │ Replicator│  │ Packed Layout   │  │
│  └─────────┘  └──────────┘  └───────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  查询阶段 (Online)                                               │
│  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌─────────────────┐  │
│  │  Graph   │→ │ Gather  │→ │  GPU PQ   │→ │   Heuristic     │  │
│  │  Search  │  │ +Dedup  │  │  Scoring  │  │   Re-ranking    │  │
│  └──────────┘  └─────────┘  └───────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

# 第一部分：构建流程 (Offline Indexing)

## 1. 分层平衡聚类 (HBC - Hierarchical Balanced Clustering)

### 目的
将 N 个向量分配到 ~nlist 个聚类中，保证聚类大小相对平衡。

### 算法流程

```
输入: N 个 d 维向量, 目标聚类数 nlist, 分支因子 k
输出: nlist 个聚类，每个向量的聚类分配

1. 初始化
   - root ← 包含所有向量 ID 的根节点
   - queue ← [root]

2. 分裂迭代
   while len(leaves) < nlist:
       node ← queue.pop()  // 取最大的叶子节点
       if node.size < threshold:
           continue
       
       // 对 node 中的向量做 k-means
       centroids ← kmeans(node.vectors, k)
       children ← 按分配结果分裂为 k 个子节点
       
       // 平衡调整 (Lambda 因子)
       balance_children(children, lambda_factor)
       
       queue.extend(children)

3. 收集叶子节点
   leaves ← 所有没有子节点的节点
   for each leaf in leaves:
       leaf.centroid ← mean(leaf.vectors)
```

### 参数
- `nlist`: 目标聚类数，**推荐 N × 10%** (如 100M 向量 → ~10M 聚类)
- `branch`: K-means 分支因子，默认 32
- `lambda_factor`: 平衡因子，默认 0.05

### SPANN 风格的聚类特点
- **极多的小聚类**：每个聚类平均只有 ~10 个向量（复制前）
- **更精确的近邻近似**：小聚类意味着质心更接近真实向量
- **依赖边界复制**：通过高复制因子弥补小聚类的覆盖不足

### 输出文件
- `vector_to_leaf`: 每个向量的聚类分配 (N × int32)
- 叶子节点 centroids: nlist × d 的质心矩阵

---

## 2. 质心导航器 (Centroid Navigator - SPTAG BKT)

### 目的
构建一个高效的数据结构，支持快速找到与查询最近的 nprobe 个质心。

### 算法：SPTAG BKT (Balanced K-means Tree)

```
输入: nlist 个 d 维质心向量
输出: BKT 索引

1. 构建 BK-Tree
   - 使用 K-means 递归分裂质心集合
   - 每个内部节点有 BKTKmeansK 个子节点
   - 叶子节点包含 BKTLeafSize 个质心

2. 构建邻域图 (Neighborhood Graph)
   for each centroid c:
       c.neighbors ← 最近的 NeighborhoodSize 个质心
   
   // RNG/KNN 图精化
   refine_graph(iterations=RefineIterations)

3. 搜索时
   - 从 BK-Tree 找初始候选
   - 在邻域图上扩展搜索
   - 使用优先队列维护最近的 nprobe 个质心
```

### 参数
- `MaxCheck`: 最多检查的节点数，默认 2048
- `NeighborhoodSize`: 每个质心的邻居数，默认 64
- `CEF`: 候选边扩展因子，默认 256

### 输出文件
- `proximity_graph_index/`: SPTAG 索引目录

---

## 3. 边界复制 (Boundary Replication)

### 目的
处理位于聚类边界的向量，将其复制到多个相邻聚类的 posting list 中，提高召回率。

### 算法流程

```
输入: N 个向量, 主聚类分配, 质心导航器, epsilon, max_replicas
输出: 每个聚类的 posting list (包含复制的向量)

for each vector v:
    primary ← v 的主聚类
    candidates ← centroid_nav.search(v, beam=64)
    
    // 距离阈值过滤
    threshold ← dist(v, candidates[0]) × (1 + epsilon)²
    filtered ← [c for c in candidates if dist(v, c) ≤ threshold]
    
    // 基于 RNG 的选择 (避免冗余复制)
    selected ← [filtered[0]]  // 最近的质心
    for c in filtered[1:]:
        dominated ← any(dist(c, s) < dist(v, c) for s in selected)
        if not dominated:
            selected.append(c)
    
    // 分配到 posting lists
    for cluster_id in selected[:max_replicas]:
        posting_lists[cluster_id].append(v.id)
```

### 参数
- `epsilon`: 距离阈值扩展系数，默认 10.0 (较大值允许更多复制)
- `max_replicas`: 最大复制数，默认 8
- `beam`: 候选搜索数量，默认 256

### 复制效果
- **平均复制因子 ≈ 7-8x**
- 每个向量被复制到 7-8 个相邻聚类的 posting list 中
- 每个聚类的实际大小 = 原始大小 × 复制因子 ≈ 10 × 7 = **~70 向量**
- 极大提高边界区域的召回率，代价是增加存储和候选数量

---

## 4. 向量编码 (PQ - Product Quantization)

### 算法流程

```
输入: N 个 d 维向量
输出: N × m 的 PQ codes (每个 uint8)

1. 训练 PQ 码本
   train_data ← sample(base_vectors, 1M)
   pq = ProductQuantizer(d, m, nbits=8)
   pq.train(train_data)
   
   // 每个子空间: dsub = d/m 维, 256 个聚类中心

2. 编码所有向量
   for each vector v:
       codes[v.id] = pq.encode(v)  // m 个 uint8
```

### 参数
- `m`: 子空间数量，默认 32
- `nbits`: 每个子空间的比特数，默认 8 (256 个聚类)
- 每个向量的压缩大小: m × 1 byte = 32 bytes

### 输出文件
- `pq_codec`: FAISS ProductQuantizer 对象
- `pq_codes`: N × m 的 PQ 码 (按原始向量 ID 顺序存储)

---

## 5. Packed Layout

### 目的
将原始向量按 4KB 页对齐存储，优化 SSD Direct I/O 读取。

### 存储布局

```
┌────────────────────────────────────────┐
│           packed_raw_vectors.bin        │
├────────────────────────────────────────┤
│ Page 0: [vec_a, vec_b, vec_c, padding] │  4KB
│ Page 1: [vec_d, vec_e, vec_f, padding] │  4KB
│ Page 2: [vec_g, vec_h, ...]            │  4KB
│ ...                                    │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│           vector_location.map          │
├────────────────────────────────────────┤
│ vec_id → (page_id, offset_in_page)     │
└────────────────────────────────────────┘
```

### 分配策略
1. 同一聚类的向量尽量放在连续页
2. 尾部向量（不足一页）跨聚类合并
3. 使用贪心算法最小化碎片

---

## 6. Posting List 元数据

### 存储格式

```
posting_lists_metadata:
┌─────────────────────────────────────┐
│ Header:                              │
│   nlist: uint64                      │
│   page_bytes: uint32                 │
│   total_postings: uint64             │
├─────────────────────────────────────┤
│ Entry[0]: {offset: uint64, count: u32}│
│ Entry[1]: {offset: uint64, count: u32}│
│ ...                                  │
├─────────────────────────────────────┤
│ 连续存储的所有 posting list IDs      │
│ [list_0_ids...][list_1_ids...]...   │
└─────────────────────────────────────┘
```

---

# 第二部分：查询流程 (Online Search)

## 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         Query Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query → [Graph Search] → [Gather+Dedup] → [GPU PQ] → [Rerank]  │
│          ~0.3ms          ~0.1ms          ~0.3ms      ~0.1ms     │
│                                                                  │
│  总延迟 ≈ 0.8~1.0 ms @ Recall ~0.90                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 图搜索 (Graph Search)

### 输入/输出
- 输入: 查询向量 q (d 维), nprobe
- 输出: 最近的 nprobe 个质心 ID 及其距离

### SPTAG 搜索算法

```python
def search(query, nprobe, max_check):
    # 1. BK-Tree 初始化
    candidates = bkt.search_tree(query, initial_pivots=50)
    priority_queue = MinHeap(candidates)
    visited = HashSet()
    results = MaxHeap(size=nprobe)  # 维护最近的 nprobe 个
    
    checked_count = 0
    while not priority_queue.empty() and checked_count < max_check:
        node = priority_queue.pop()
        
        if node.id in visited:
            continue
        visited.add(node.id)
        
        # 更新结果
        results.push(node)
        checked_count += 1
        
        # 扩展邻居
        for neighbor_id in graph[node.id]:
            if neighbor_id in visited:
                continue
            dist = compute_L2(query, centroids[neighbor_id])
            if dist < results.worst() or results.size < nprobe:
                priority_queue.push((dist, neighbor_id))
        
        # 动态添加更多 BK-Tree 候选
        if priority_queue.top().dist > bkt_queue.top().dist:
            candidates = bkt.search_tree(query, additional_pivots)
            priority_queue.extend(candidates)
    
    return results.get_top_k(nprobe)
```

### 参数影响
- `nprobe`: 返回的质心数量，影响召回率和候选数量
- `MaxCheck`: 最多检查的节点数，影响搜索时间

---

## 2. 候选收集与去重 (Gather + Dedup)

### 流程

```python
def gather_and_dedup(probe_result, posting_list_accessor):
    candidate_ids = []
    
    # 1. 收集所有 posting list 的向量 ID
    for centroid_id in probe_result.ids:
        view = posting_list_accessor.view(centroid_id)
        for id in view:
            candidate_ids.append(id)
    
    # 2. 去重 (使用 boost::unordered_flat_set)
    seen = FlatHashSet()
    unique_ids = []
    for id in candidate_ids:
        if id not in seen:
            seen.add(id)
            unique_ids.append(id)
    
    return unique_ids

# 典型数据 (SPANN 风格):
# nprobe = 160, 平均每个聚类 ~70 个向量 (包含复制)
# 收集: 160 × 70 ≈ ~11,000 个 ID (含重复)
# 去重后: ~10,000 个唯一 ID (因为复制导致的重叠)
# 
# 注意: 由于 7-8x 复制因子，同一个向量可能出现在多个 posting list 中
# 去重对于 SPANN 风格系统非常重要
```

---

## 3. GPU PQ 评分 (GPU PQ Scoring)

### 流程

```
┌────────────────────────────────────────────────────────────────┐
│                      GPU PQ Scoring Pipeline                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Host                           Device (GPU)                    │
│  ┌─────────────┐               ┌─────────────────────────────┐ │
│  │ Query (d)   │ ──H2D──────→ │ Build Distance Table        │ │
│  └─────────────┘               │ LUT[m][256] for each query  │ │
│                                └─────────────────────────────┘ │
│  ┌─────────────┐               ┌─────────────────────────────┐ │
│  │Candidate IDs│ ──H2D──────→ │ PQ Code Lookup              │ │
│  │   (N_cand)  │               │ codes[id] → m × uint8       │ │
│  └─────────────┘               └─────────────────────────────┘ │
│                                              ↓                  │
│                                ┌─────────────────────────────┐ │
│                                │ Accumulate Distances        │ │
│                                │ dist = Σ LUT[j][codes[id][j]]│ │
│  ┌─────────────┐               └─────────────────────────────┘ │
│  │ All dists   │ ←──D2H──── 返回所有候选的距离 ────────────────│
│  │ (N_cand)    │                                               │
│  └─────────────┘                                               │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                      CPU TopK Selection                         │
├────────────────────────────────────────────────────────────────┤
│  std::nth_element(results, rerank_size)  // O(N) 选择          │
│  std::sort(results[:rerank_size])        // 排序 top-k         │
└────────────────────────────────────────────────────────────────┘
```

**注意**: TopK 选择在 **CPU** 上完成，GPU 只负责计算 PQ 距离。

### CUDA Kernel 伪代码

```cuda
// 1. 构建 LUT (每个 query 一次)
__global__ void build_distance_table(
    float* query,           // [d]
    float* codebook,        // [m×256×dsub]
    float* lut              // [m×256]
) {
    int subspace = blockIdx.x;
    int code = threadIdx.x;
    
    // 计算 query 子向量到该聚类中心的距离
    float dist = 0;
    for (int i = 0; i < dsub; i++) {
        float diff = query[subspace*dsub + i] - 
                     codebook[subspace*256*dsub + code*dsub + i];
        dist += diff * diff;
    }
    lut[subspace*256 + code] = dist;
}

// 2. 累加 PQ 距离
__global__ void accumulate_pq_distance(
    uint8_t* pq_codes,      // [N × m]
    float* lut,             // [m × 256]
    int* candidate_ids,     // [N_cand]
    float* distances        // [N_cand]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_cand) return;
    
    int id = candidate_ids[tid];
    float dist = 0;
    for (int j = 0; j < m; j++) {
        uint8_t code = pq_codes[id * m + j];
        dist += lut[j * 256 + code];
    }
    distances[tid] = dist;
}
```

### CPU TopK 选择 (CPUSelectTopR)

```cpp
// 在 CPU 上选择 top rerank_size 个候选
if (rerank_size < approximate_results.size()) {
    // O(N) 的 nth_element 找出第 rerank_size 小的元素
    std::nth_element(approximate_results.begin(),
                     approximate_results.begin() + rerank_size,
                     approximate_results.end(),
                     [](const auto& a, const auto& b) {
                         return a.dist < b.dist;
                     });
    approximate_results.resize(rerank_size);
}
// 对 top-k 排序
std::sort(approximate_results.begin(), approximate_results.end());
```

### 输出
- top `rerank_size` 个候选的 (id, approx_distance)，按距离排序

---

## 4. 启发式重排序 (Heuristic Re-ranking)

### 目的
用原始向量计算精确 L2 距离，从 rerank_size 个候选中选出 top-k。

### 算法 (论文 Algorithm 1)

```python
def heuristic_rerank(query, approx_results, io_manager, top_k, 
                     batch_size, epsilon, beta):
    """
    Args:
        query: 查询向量 (d 维)
        approx_results: PQ 评分后的 (id, approx_dist) 列表，按距离排序
        io_manager: 异步 I/O 管理器
        top_k: 返回的最终结果数
        batch_size: 每次 I/O 读取的向量数
        epsilon: 稳定性阈值
        beta: 连续稳定次数阈值
    
    Returns:
        top_k 个最近邻的 ID
    """
    
    max_heap = MaxHeap(capacity=top_k)  # 堆顶是当前 top-k 中最远的
    last_top_k_ids = set()
    stability_counter = 0
    
    for batch_start in range(0, len(approx_results), batch_size):
        batch = approx_results[batch_start : batch_start + batch_size]
        
        # I/O: 读取原始向量 (io_uring Direct I/O)
        vectors = io_manager.fetch_batch([r.id for r in batch])
        
        # 计算精确 L2 距离
        for id, vec in zip([r.id for r in batch], vectors):
            dist = L2_distance(query, vec)
            
            if max_heap.size < top_k:
                max_heap.push((dist, id))
            elif dist < max_heap.top().dist:
                max_heap.pop()
                max_heap.push((dist, id))
        
        # 稳定性检测
        if max_heap.size >= top_k:
            current_top_k_ids = set(h.id for h in max_heap)
            
            # 计算变化率 Δ
            if last_top_k_ids:
                changed = len(current_top_k_ids - last_top_k_ids)
                delta = changed / top_k
            else:
                delta = 1.0
            
            # 更新稳定计数器
            if delta < epsilon:
                stability_counter += 1
            else:
                stability_counter = 0
            
            last_top_k_ids = current_top_k_ids
            
            # 早停判断
            if stability_counter >= beta:
                break
    
    return [h.id for h in max_heap.sorted()]
```

### 参数
- `rerank_size`: 进入重排序的候选数量，默认 128
- `batch_size`: 每批 I/O 读取的向量数，默认 32
- `epsilon (delta)`: 稳定性阈值，默认 0.05
- `beta (stable_iters)`: 连续稳定轮数阈值，默认 5

### I/O 优化
- 使用 io_uring with SQPOLL
- Direct I/O (绕过 page cache)
- 批量请求合并
- 内存缓存层 (PageCacheBackend)

---

# 第三部分：数据流图

## 构建阶段数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        建索引数据流                              │
└─────────────────────────────────────────────────────────────────┘

base.fvecs (N × d floats)
        │
        ▼
┌─────────────────┐
│ HBC Clustering  │ ───── vector_to_leaf (N × int32)
│ (K-means tree)  │       leaf_centroids (nlist × d)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ SPTAG BKT Build │ ───── proximity_graph_index/
│ (centroid graph)│        ├── tree.bin
└─────────────────┘        ├── graph.bin
        │                  └── vectors.bin
        ▼
┌─────────────────┐
│ Boundary        │ ───── posting_lists (nlist 个 list)
│ Replication     │       avg ~7-8× replication
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ PQ Encoding     │ ───── pq_codec (FAISS PQ)
│ (m=32, 256 sub) │       pq_codes (N × m bytes)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Packed Layout   │ ───── packed_raw_vectors.bin
│ (4KB aligned)   │       vector_location.map
└─────────────────┘

posting_lists_metadata ─────────────────────────────────┐
                                                        ▼
                                                 Online Query
```

---

## 查询阶段数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        查询数据流                                │
└─────────────────────────────────────────────────────────────────┘

Query Vector (d floats)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Graph Search (SPTAG)                                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  BKT traversal + Neighborhood graph expansion            │   │
│  │  Priority queue: (distance, centroid_id)                 │   │
│  │  Output: nprobe nearest centroids                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
        │
        │ ProbeResult: [(centroid_id, distance)] × nprobe
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Gather Candidates                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  for each centroid in probe_result:                      │   │
│  │      candidates += posting_list[centroid]                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
        │
        │ ~11,000 vector IDs (with duplicates from replication)
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Deduplicate (boost::unordered_flat_set)                         │
└─────────────────────────────────────────────────────────────────┘
        │
        │ ~10,000 unique vector IDs
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ GPU PQ Scoring                                                  │
│  ┌────────────────────┐  ┌────────────────────────────────┐    │
│  │ H2D: query, IDs    │→ │ Build LUT: [m × 256]           │    │
│  └────────────────────┘  │ Lookup codes: [N × m]          │    │
│                          │ Accumulate: dist[i] = Σ LUT    │    │
│                          │ CUB TopK: top rerank_size      │    │
│  ┌────────────────────┐  └────────────────────────────────┘    │
│  │ D2H: (id, dist)    │← ──────────────────────────────────    │
│  └────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
        │
        │ rerank_size × (id, approx_dist), sorted
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Heuristic Re-ranking                                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  for batch in candidates (batch_size=32):                │   │
│  │      vectors = io_uring_fetch(batch.ids)  // SSD read   │   │
│  │      exact_dists = L2(query, vectors)     // AVX512     │   │
│  │      heap.update(batch.ids, exact_dists)                 │   │
│  │                                                          │   │
│  │      if stability_detected():  // Δ < ε for β rounds    │   │
│  │          break  // Early termination                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
        │
        │ top_k × (id, exact_dist)
        ▼
    Final Results
```

---

# 第四部分：性能特征

## 典型性能 (SIFT100M, 10M 质心, 100M 向量)

| 指标 | 数值 |
|------|------|
| QPS (effective) | ~1000-1100 |
| 平均延迟 | ~0.9-1.0 ms |
| P99 延迟 | ~4-5 ms |
| Recall@10 | ~0.90 |

## 时间分解

| 阶段 | 时间占比 | 主要瓶颈 |
|------|----------|----------|
| Graph Search | ~34% | CPU 距离计算 + 优先队列 |
| Gather+Dedup | ~15% | 哈希表去重 |
| GPU PQ | ~30% | H2D/D2H 传输 + kernel |
| Re-rank | ~10% | SSD I/O |
| 其他 | ~11% | 调度、同步 |

---

# 第五部分：关键文件索引

```
fusionanns/
├── src/
│   ├── index/
│   │   ├── builder.cpp          # 构建主流程
│   │   ├── hbc_builder.cpp      # HBC 聚类
│   │   ├── replicator.cpp       # 边界复制
│   │   ├── centroid_navigator_sptag.cpp  # SPTAG 封装
│   ├── online/
│   │   ├── benchmark_runner.cpp # 查询主流程
│   │   ├── pq_lut_scorer.h      # GPU PQ 评分
│   │   ├── gpu_kernels.cu       # CUDA kernels
│   │   ├── io_manager.cpp       # io_uring 管理
│   │   └── posting_list_accessor.cpp  # Posting list 访问
│   └── common/
│       └── types.h              # 通用类型定义
├── extern/
│   └── SPTAG/                   # SPTAG 库
└── app/
    └── query_server.cpp         # 查询服务入口
```

---

# 第六部分：参数调优指南

## 召回率 vs QPS 权衡

| 参数 | 增加 → | 召回率 | QPS |
|------|--------|--------|-----|
| `nprobe` | ↑ | ↑ | ↓ |
| `rerank_size` | ↑ | ↑ (上限) | ↓ |
| `MaxCheck` | ↑ | ↑ (边际) | ↓ |
| `epsilon` | ↑ | ↔ | ↑ (更快早停) |
| `beta` | ↓ | ↓ (可能) | ↑ |

## 推荐配置

| 场景 | nprobe | rerank_size | 预期 Recall | 预期 QPS |
|------|--------|-------------|-------------|----------|
| 高召回 | 200 | 256 | ~0.92 | ~800 |
| 均衡 | 160 | 128 | ~0.90 | ~1000 |
| 高吞吐 | 128 | 64 | ~0.87 | ~1300 |
