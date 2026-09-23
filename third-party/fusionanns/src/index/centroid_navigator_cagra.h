#pragma once

#ifdef FUSIONANNS_USE_CUVS_CAGRA

#include "index/centroid_navigator.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// 前向声明，避免在头文件中引入重量级 CUDA/RAFT 头文件
struct CUstream_st;
typedef CUstream_st *cudaStream_t;

namespace raft {
class device_resources;
}
namespace cuvs::neighbors::cagra {
template <typename T, typename IdxT> struct index;
}

/// CAGRA search 延迟 breakdown 数据（单次 batch 调用）
struct CagraSearchProfile {
  double acquire_slot_us = 0.0;    // 获取 resource slot 等待时间
  double alloc_query_us = 0.0;     // 分配 device query buffer
  double h2d_query_us = 0.0;       // H2D query memcpy
  double alloc_output_us = 0.0;    // 分配 device output buffers
  double kernel_launch_us = 0.0;   // CAGRA search kernel 执行（含 launch overhead）
  double stream_sync_us = 0.0;     // stream sync 等待
  double d2h_result_us = 0.0;      // D2H result memcpy
  double assemble_us = 0.0;        // 结果组装
  double total_us = 0.0;           // 总时间
  double gpu_kernel_us = 0.0;      // GPU 端 kernel 实际执行时间（globaltimer）
  double collect_wait_us = 0.0;    // batch collector 中等待凑满 batch 的时间
  int num_queries = 0;
  int batch_size = 0;              // 实际 batch 大小
};

/// CAGRA search 延迟 breakdown 累计统计
struct CagraProfileStats {
  std::atomic<int> count{0};
  std::atomic<double> sum_acquire_slot_us{0.0};
  std::atomic<double> sum_alloc_query_us{0.0};
  std::atomic<double> sum_h2d_query_us{0.0};
  std::atomic<double> sum_alloc_output_us{0.0};
  std::atomic<double> sum_kernel_launch_us{0.0};
  std::atomic<double> sum_stream_sync_us{0.0};
  std::atomic<double> sum_d2h_result_us{0.0};
  std::atomic<double> sum_assemble_us{0.0};
  std::atomic<double> sum_total_us{0.0};
  std::atomic<double> sum_gpu_kernel_us{0.0};
  std::atomic<double> sum_collect_wait_us{0.0};
  std::atomic<double> max_acquire_slot_us{0.0};
  std::atomic<double> max_total_us{0.0};
  std::atomic<int> sum_batch_size{0};
  std::atomic<int> batch_count{0};

  void accumulate(const CagraSearchProfile &p);
  std::string to_string() const;
  void reset();
};

class CAGRACentroidNavigator;

/// CAGRA Batch Collector：将多线程的逐条查询自动合并为 GPU batch 搜索
///
/// 工作流程：
/// 1. Worker 线程调用 submit(query, nprobe) → 返回 std::future<ProbeResult>
/// 2. 当 pending 查询数达到 batch_size 或等待超时(flush_timeout)时，
///    自动触发 flush：一次 CAGRA batch kernel 处理所有 pending 查询
/// 3. 结果通过 promise 分发给各个 worker 线程
///
/// 这样 CAGRA kernel 可以同时处理 N 个查询，GPU 利用率远高于逐条提交。
class CagraBatchCollector {
public:
  /// @param navigator  CAGRA 导航器（必须已 init_resource_pool）
  /// @param batch_size 目标 batch 大小（达到即立刻 flush）
  /// @param nprobe     每次搜索的 probe 数
  /// @param flush_timeout_us  等不到满 batch 时的超时（微秒），0 = 不超时
  CagraBatchCollector(CAGRACentroidNavigator &navigator, int batch_size,
                      int nprobe, int flush_timeout_us = 200);
  ~CagraBatchCollector();

  // 不可拷贝 / 不可移动
  CagraBatchCollector(const CagraBatchCollector &) = delete;
  CagraBatchCollector &operator=(const CagraBatchCollector &) = delete;

  /// 提交一条查询，返回 future<ProbeResult>。线程安全。
  std::future<ProbeResult> submit(const float *query);

  /// 手动触发 flush（清空所有 pending 查询）。通常在 warmup/shutdown 时使用。
  void flush();

  /// 关停 collector（flush 剩余 + 停止 flush 线程）
  void shutdown();

  /// 获取 batch_size
  int batch_size() const { return batch_size_; }

private:
  struct PendingQuery {
    std::vector<float> query_data;  // 查询向量副本（dim_ 维）
    std::promise<ProbeResult> promise;
  };

  void flush_loop();                    // 后台 flush 线程
  void flush_pending(std::unique_lock<std::mutex> &lock);  // 执行一次 flush

  CAGRACentroidNavigator &navigator_;
  int batch_size_;
  int nprobe_;
  int flush_timeout_us_;

  std::mutex mutex_;
  std::condition_variable cv_;          // notify: 有新查询 / shutdown
  std::condition_variable flush_cv_;    // notify: flush 完成（用于 manual flush 等待）
  std::vector<PendingQuery> pending_;
  bool shutdown_ = false;
  std::thread flush_thread_;
};

/// 基于 cuVS CAGRA 的 GPU 质心图导航器
///
/// 使用 per-slot resource pool（每个 slot 拥有独立的 stream + device_resources），
/// 多线程并发搜索无需互斥锁，完全消除串行化瓶颈。
/// CAGRA index 是只读的，可安全被多个 slot 并发访问。
class CAGRACentroidNavigator final : public ICentroidNavigator {
public:
  CAGRACentroidNavigator();
  ~CAGRACentroidNavigator() override;

  // 不可拷贝 / 不可移动（持有 GPU 资源）
  CAGRACentroidNavigator(const CAGRACentroidNavigator &) = delete;
  CAGRACentroidNavigator &operator=(const CAGRACentroidNavigator &) = delete;

  void build(const float *centroids, int nlist, int dim,
             int num_threads = 0) override;
  void save(const std::string &path) const override;
  void load(const std::string &path) override;

  /// 单查询搜索（无锁：从 resource pool 获取 slot）
  ProbeResult search_one(const float *query, int nprobe) const override;

  /// 批量搜索（无锁：从 resource pool 获取 slot）
  std::vector<ProbeResult>
  search_batch(const float *queries, int num_queries,
               int nprobe) const override;

  /// 批量搜索（flat 输出，无锁）
  void search_batch_flat(const float *queries, int num_queries, int nprobe,
                         std::vector<int32_t> &out_ids,
                         std::vector<float> &out_dists) const override;

  int get_dim() const override { return dim_; }

  /// 预分配 resource pool。num_slots 个独立的 {stream, device_resources}。
  /// max_nprobe: 预分配 buffer 的最大 nprobe 值。
  /// max_batch_size: 每个 slot 预分配的最大 batch 大小。
  /// 应在 benchmark 开始前、warmup 之前调用。
  void init_resource_pool(int num_slots, int max_nprobe = 256,
                          int max_batch_size = 1);

  /// 创建 batch collector（需要先 init_resource_pool）
  std::unique_ptr<CagraBatchCollector>
  create_batch_collector(int batch_size, int nprobe,
                         int flush_timeout_us = 200);

  // ========== Profiling API ==========
  void enable_profiling(bool enable) { profiling_enabled_ = enable; }
  bool profiling_enabled() const { return profiling_enabled_; }
  CagraProfileStats &get_profile_stats() { return profile_stats_; }
  const CagraProfileStats &get_profile_stats() const { return profile_stats_; }
  void reset_profile_stats() { profile_stats_.reset(); }

private:
  void ensure_ready() const;
  void init_resources();  // 初始化构建/加载用的主 GPU 资源

  std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> index_;
  int dim_ = 0;

  // 构建/加载时使用的主资源（单线程操作）
  std::unique_ptr<raft::device_resources> res_;
  cudaStream_t stream_ = nullptr;

  // ========== Per-slot resource pool ==========
  struct ResourceSlot {
    std::unique_ptr<raft::device_resources> res;
    cudaStream_t stream = nullptr;
    unsigned long long *d_timer_buf = nullptr;  // [2]: start, end (for profiling)
    std::atomic<bool> in_use{false};            // lock-free acquire/release
    // 预分配 device buffer（消除 per-search cudaMalloc 开销）
    float *d_query = nullptr;         // [max_batch_size * dim]
    uint32_t *d_neighbors = nullptr;  // [max_batch_size * max_nprobe]
    float *d_distances = nullptr;     // [max_batch_size * max_nprobe]
    int max_nprobe = 0;               // 预分配的最大 nprobe
    int max_batch_size = 0;           // 预分配的最大 batch 大小
  };
  mutable std::vector<std::unique_ptr<ResourceSlot>> resource_pool_;
  int pool_size_ = 0;

  // 获取/释放 resource slot（lock-free spin）
  ResourceSlot *acquire_slot() const;
  void release_slot(ResourceSlot *slot) const;

  // 使用指定 slot 执行搜索的内部实现
  std::vector<ProbeResult>
  search_batch_with_slot(ResourceSlot *slot, const float *queries,
                         int num_queries, int nprobe,
                         CagraSearchProfile *prof_out) const;

  // ========== Profiling state ==========
  std::atomic<bool> profiling_enabled_{false};
  mutable CagraProfileStats profile_stats_;

  // CagraBatchCollector 是友元，需要访问 search_batch_with_slot
  friend class CagraBatchCollector;
};

#endif // FUSIONANNS_USE_CUVS_CAGRA
