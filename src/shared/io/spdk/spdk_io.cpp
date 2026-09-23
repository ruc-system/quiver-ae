#include "../io_stats.hpp"
#include "../loader.hpp"
#include "../spdk_lba_offset.hpp"
#include "submit_queue.hpp"

#include "spdk/env.h"
#include "spdk_wrapper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>

#include "../../common/runtime.hpp"

using shared::spdk_submit_queue::MPSCQueue;
using shared::spdk_submit_queue::Queue;

namespace shared {
namespace {

std::mutex g_spdk_wrapper_mu;
std::shared_ptr<spdk_wrapper::SpdkWrapper> g_shared_spdk_wrapper;
std::vector<std::string> g_shared_spdk_ssds;

std::shared_ptr<spdk_wrapper::SpdkWrapper> get_or_create_shared_spdk_wrapper(
    const std::vector<std::string>& ssds) {
  std::lock_guard<std::mutex> lk(g_spdk_wrapper_mu);
  if (g_shared_spdk_wrapper) {
    if (g_shared_spdk_ssds != ssds) {
      throw std::runtime_error(
          "SPDK SSD list mismatch across runs in same process");
    }
    return g_shared_spdk_wrapper;
  }
  auto wrapper = spdk_wrapper::SpdkWrapper::create(32);
  wrapper->Init(ssds);
  g_shared_spdk_ssds = ssds;
  g_shared_spdk_wrapper = std::shared_ptr<spdk_wrapper::SpdkWrapper>(
      wrapper.release());
  return g_shared_spdk_wrapper;
}

bool spdk_diag_enabled_from_env() {
  const char* env = std::getenv("QUIVER_SPDK_DIAG");
  return env != nullptr && std::atoi(env) != 0;
}

int hot_sample_interval_ms_from_env() {
  const char* env = std::getenv("QUIVER_SPDK_HOT_SAMPLE_MS");
  if (env == nullptr) {
    return 100;
  }
  return std::max(1, std::atoi(env));
}

int spdk_max_retry_from_env() {
  const char* env = std::getenv("QUIVER_SPDK_MAX_RETRY");
  if (env == nullptr) {
    return 256;
  }
  return std::max(1, std::atoi(env));
}

uint64_t steady_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

class SpdkIOImpl : public IndexLoader {
 public:
  void submit_task(const std::vector<IoRequest> &req, int tid,
                   int cid) override {
    io_stats_record_enqueued(stats_.get(), req.size());
    auto now_ns = std::chrono::duration<uint64_t, std::nano>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    for (auto [blk, dst] : req) {
      const long logical_block = static_cast<long>(blk) + block_offset_;
      int ssd = logical_block % num_ssd;
      long off = base_lba_ + logical_block / num_ssd;
      int rid = route_runner(off);
      Req rq(cid, off, dst);
      rq.enqueue_ns = now_ns;
      submit_queue[ssd][rid]->push(rq, tid);
    }
    ready[cid].fetch_add(req.size());
  }

  bool poll_task(int cid) override { return ready[cid].load() == 0; }

  void submit_direct(const std::vector<DirectIoRequest>& reqs, int tid) override {
    io_stats_record_enqueued(stats_.get(), reqs.size());
    uint64_t now_ns = 0;
    if (spdk_diag_enabled_) {
      now_ns = std::chrono::duration<uint64_t, std::nano>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    for (auto& r : reqs) {
      const long logical_block =
          static_cast<long>(r.block_id) + block_offset_;
      int ssd = logical_block % num_ssd;
      long off = base_lba_ + logical_block / num_ssd;
      int rid = route_runner(off);
      Req rq(off, r.dest, r.status_ptr, r.done_value, r.complete_ns);
      rq.enqueue_ns = now_ns;
      submit_queue[ssd][rid]->push(rq, tid);
    }
  }

  LoaderStatsSnapshot snapshot_stats() const override {
    if (stats_ == nullptr) {
      return {};
    }
    flush_hot_sample(true);
    return stats_->snapshot();
  }

  void init(const std::vector<std::string> &ssds, int submit_queue_capacity,
            int thread_cnt, int task_context_count,
            int runner_io_context_count, int runners_per_ssd) {
    spdk = get_or_create_shared_spdk_wrapper(ssds);
    ready = new std::atomic<int>[task_context_count];
    for (int i = 0; i < task_context_count; i++) {
      ready[i].store(0, std::memory_order_relaxed);
    }
    num_ssd = ssds.size();
    base_lba_ = spdk_base_lba_from_env();
    block_offset_ = spdk_block_offset_from_env();
    runner_io_context_count_ = runner_io_context_count;
    if (runner_io_context_count_ < 1) {
      throw std::runtime_error(
          "runner_io_context_count must be positive, got " +
          std::to_string(runner_io_context_count_));
    }
    runners_per_ssd_ = runners_per_ssd;
    if (runners_per_ssd_ < 1 || runners_per_ssd_ > 32) {
      throw std::runtime_error(
          "runners_per_ssd must be in [1,32], got " +
          std::to_string(runners_per_ssd_));
    }

    const int total_runners = num_ssd * runners_per_ssd_;
    const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    const int max_core =
        runners_per_ssd_ == 1 ? (num_ssd - 1) * 2 + 3
                              : 4 + (total_runners - 1) * 2;
    if (online_cpus > 0 && max_core >= online_cpus) {
      throw std::runtime_error(
          "runners_per_ssd=" + std::to_string(runners_per_ssd_) +
          " with num_ssd=" + std::to_string(num_ssd) +
          " needs max runner core " + std::to_string(max_core) +
          ", but only " + std::to_string(online_cpus) +
          " CPUs are online");
    }
    spdk_diag_enabled_ = spdk_diag_enabled_from_env();
    hot_sample_interval_ms_ = hot_sample_interval_ms_from_env();
    max_retry_on_error_ = spdk_max_retry_from_env();

    stats_ = std::make_unique<LoaderStatsAccumulator>(num_ssd);
    reset_hot_sample_baseline();
    submit_queue.assign(num_ssd, {});
    for (int i = 0; i < num_ssd; i++) {
      submit_queue[i].reserve(runners_per_ssd_);
      for (int r = 0; r < runners_per_ssd_; r++) {
        submit_queue[i].push_back(
            new MPSCQueue<Req>(submit_queue_capacity, thread_cnt));
      }
    }
    if (spdk_diag_enabled_) {
      printf("[SPDK] runners_per_ssd=%d -> total runners=%d max_retry=%d\n",
             runners_per_ssd_, num_ssd * runners_per_ssd_, max_retry_on_error_);
    }
    worker_thread();
  }

  void clear_stats() override {
    if (stats_) {
      stats_->clear();
      reset_hot_sample_baseline();
    }
  }

  ~SpdkIOImpl() override {
    finished.store(true);
    for (auto &th : worker) {
      th.join();
    }
  }

 private:
  struct Ctx {
    int ns_id;
    int qp_id;
    int runner_in_ssd;
    int cid;
    long r;
    void *buffer;
    std::atomic<int> *read_cnt;
    Queue<Ctx *> *wait_queue;
    Queue<Ctx *> *idle_queue;
    SpdkIOImpl *ctx;
    volatile int32_t* status_ptr;
    int32_t status_value;
    volatile int64_t* complete_ns;
    int retry_count;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
  };

  struct Req {
    int cid;
    long r;
    void *dst;
    volatile int32_t* status_ptr;
    int32_t status_value;
    volatile int64_t* complete_ns;
    uint64_t enqueue_ns;  // CLOCK_MONOTONIC nanoseconds when pushed to MPSC
    Req() : cid(0), r(0), dst(nullptr), status_ptr(nullptr), status_value(0), complete_ns(nullptr), enqueue_ns(0) {}
    Req(int cid, long r, void *dst)
        : cid(cid), r(r), dst(dst), status_ptr(nullptr), status_value(0), complete_ns(nullptr), enqueue_ns(0) {}
    Req(long r, void *dst, volatile int32_t* sp, int32_t sv,
        volatile int64_t* cn)
        : cid(-1), r(r), dst(dst), status_ptr(sp), status_value(sv), complete_ns(cn), enqueue_ns(0) {}
  };

  int num_ssd;
  long base_lba_ = 0;
  long block_offset_ = 0;
  int runners_per_ssd_ = 1;
  int runner_io_context_count_ = 1024;
  bool spdk_diag_enabled_ = false;
  int max_retry_on_error_ = 256;
  std::shared_ptr<spdk_wrapper::SpdkWrapper> spdk;
  // submit_queue[ssd_id][runner_in_ssd] -> per-runner MPSC inbox
  std::vector<std::vector<MPSCQueue<Req> *>> submit_queue;
  std::atomic<int> *ready;
  std::vector<std::thread> worker;

  std::unique_ptr<LoaderStatsAccumulator> stats_;
  int hot_sample_interval_ms_ = 100;
  mutable std::mutex hot_sample_mu_;
  mutable uint64_t hot_last_sample_ns_ = 0;
  mutable uint64_t hot_last_submitted_ = 0;

  // Per-IO latency histogram (unconditional, always-on diagnostics).
  // 16 log-scale buckets: [0,1us), [1,2us), [2,4us), ... [16ms,+)
  // Layout: io_lat_hist[ssd][bucket], io_lat_sum_ns[ssd], io_lat_cnt[ssd]
  static constexpr int IO_LAT_BUCKETS = 16;
  std::vector<std::atomic<uint64_t>*> io_lat_hist;   // [ssd] -> array[BUCKETS]
  std::vector<std::atomic<uint64_t>*> io_lat_sum_ns;  // [ssd] -> single atomic
  std::vector<std::atomic<uint64_t>*> io_lat_cnt;     // [ssd] -> single atomic

  static int io_lat_bucket(uint64_t ns) {
    uint64_t us = ns / 1000;
    if (us < 1) return 0;
    // floor(log2(us)) + 1, clamped to BUCKETS-1
    int b = 64 - __builtin_clzll(us);
    return b < IO_LAT_BUCKETS ? b : IO_LAT_BUCKETS - 1;
  }

  // Queue latency histogram: time from MPSC push to runner pop.
  // Same bucket structure as IO latency.
  std::vector<std::atomic<uint64_t>*> q_lat_hist;    // [ssd] -> array[BUCKETS]
  std::vector<std::atomic<uint64_t>*> q_lat_sum_ns;   // [ssd]
  std::vector<std::atomic<uint64_t>*> q_lat_cnt;      // [ssd]

  // Runner batch size stats: how many IOs submitted per find_ready round.
  std::vector<std::atomic<uint64_t>*> runner_batch_cnt;    // [ssd] -> count of rounds
  std::vector<std::atomic<uint64_t>*> runner_batch_total;  // [ssd] -> sum of batch sizes

  std::atomic<int> idle_cnt;
  static const int PG_SIZE = 4096;

  void reset_hot_sample_baseline() const {
    std::lock_guard<std::mutex> lk(hot_sample_mu_);
    hot_last_sample_ns_ = steady_now_ns();
    hot_last_submitted_ = 0;
  }

  void flush_hot_sample(bool force) const {
    if (stats_ == nullptr) {
      return;
    }
    const uint64_t now_ns = steady_now_ns();
    const uint64_t submitted = stats_->snapshot().pages_submitted;
    std::lock_guard<std::mutex> lk(hot_sample_mu_);
    if (hot_last_sample_ns_ == 0) {
      hot_last_sample_ns_ = now_ns;
      hot_last_submitted_ = submitted;
      return;
    }
    const uint64_t elapsed_ns = now_ns - hot_last_sample_ns_;
    if (!force &&
        elapsed_ns < static_cast<uint64_t>(hot_sample_interval_ms_) * 1000000ULL) {
      return;
    }
    const uint64_t delta = submitted - hot_last_submitted_;
    hot_last_sample_ns_ = now_ns;
    hot_last_submitted_ = submitted;
    if (delta == 0 || elapsed_ns == 0) {
      return;
    }
    io_stats_record_hot_sample(stats_.get(), delta,
                               static_cast<double>(elapsed_ns) / 1e9);
  }

  static void callback(void *ctx, const struct spdk_nvme_cpl *cpl) {
    Ctx *cur = (Ctx *)ctx;
    io_stats_record_completed(cur->ctx->stats_.get(), cur->ns_id);
    if (cur->ctx->spdk_diag_enabled_) {
      cur->ctx->runner_completed[cur->ns_id][cur->runner_in_ssd]
          .fetch_add(1, std::memory_order_relaxed);
    }
    if (spdk_nvme_cpl_is_error(cpl)) {
      io_stats_record_nvme_error(cur->ctx->stats_.get());
      cur->retry_count++;
      std::fprintf(stderr,
                   "[SPDK] completion error: ns=%d runner=%d qp=%d lba=%ld retry=%d/%d\n",
                   cur->ns_id, cur->runner_in_ssd, cur->qp_id, cur->r,
                   cur->retry_count, cur->ctx->max_retry_on_error_);
      if (cur->retry_count > cur->ctx->max_retry_on_error_) {
        std::fprintf(stderr,
                     "[SPDK] completion retry exhausted, aborting to avoid silent hang.\n");
        std::abort();
      }
      submit(cur);
      return;
    }

    cur->retry_count = 0;
    if (cur->ctx->spdk_diag_enabled_) {
      auto end = std::chrono::high_resolution_clock::now();
      auto ns = std::chrono::duration<uint64_t, std::nano>(end - cur->start).count();
      int b = io_lat_bucket(ns);
      cur->ctx->io_lat_hist[cur->ns_id][b].fetch_add(1, std::memory_order_relaxed);
      cur->ctx->io_lat_sum_ns[cur->ns_id]->fetch_add(ns, std::memory_order_relaxed);
      cur->ctx->io_lat_cnt[cur->ns_id]->fetch_add(1, std::memory_order_relaxed);
    }

    if (cur->complete_ns != nullptr) {
      const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
      __atomic_store_n(cur->complete_ns, now_ns, __ATOMIC_RELEASE);
    }
    if (cur->status_ptr) {
      __atomic_store_n(cur->status_ptr, cur->status_value, __ATOMIC_RELEASE);
    } else {
      cur->ctx->ready[cur->cid].fetch_sub(1);
    }
    if (cur->ctx->find_ready(cur->ns_id, cur->runner_in_ssd, cur->cid, cur->r,
                             cur->buffer, cur->status_ptr, cur->status_value,
                             cur->complete_ns)) {
      submit(cur);
    } else {
      cur->ctx->idle_cnt.fetch_add(1);
      cur->idle_queue->push(cur);
    }
  }

  static bool submit(Ctx *ctx) {
    int ret = ctx->ctx->spdk->TrySubmitReadCommand(
        ctx->buffer, PG_SIZE, ctx->r, callback, ctx, ctx->ns_id, ctx->qp_id);
    if (ret != 0) {
      if (ret == -ENOMEM) {
        io_stats_record_submit_backpressure(ctx->ctx->stats_.get());
        ctx->wait_queue->push(ctx);
        return false;
      }
      io_stats_record_nvme_error(ctx->ctx->stats_.get());
      ctx->retry_count++;
      std::fprintf(stderr,
                   "[SPDK] submit error: ns=%d runner=%d qp=%d lba=%ld ret=%d retry=%d/%d\n",
                   ctx->ns_id, ctx->runner_in_ssd, ctx->qp_id, ctx->r, ret,
                   ctx->retry_count, ctx->ctx->max_retry_on_error_);
      if (ctx->retry_count > ctx->ctx->max_retry_on_error_) {
        std::fprintf(stderr,
                     "[SPDK] submit retry exhausted, aborting to avoid silent hang.\n");
        std::abort();
      }
      ctx->wait_queue->push(ctx);
      return false;
    }
    if (ctx->ctx->spdk_diag_enabled_) {
      ctx->start = std::chrono::high_resolution_clock::now();
    }
    ctx->read_cnt->fetch_add(1);
    if (ctx->ctx->spdk_diag_enabled_) {
      ctx->ctx->runner_submitted[ctx->ns_id][ctx->runner_in_ssd]
          .fetch_add(1, std::memory_order_relaxed);
    }
    io_stats_record_submitted(ctx->ctx->stats_.get(), ctx->ns_id);
    return true;
  }

  // Route a per-SSD page offset to one of the runners on that SSD.
  // Mid-bit hashing keeps small consecutive bursts on the same runner
  // (preserving short-range locality friendly for SSD prefetch) while still
  // distributing larger batches across all runners.
  inline int route_runner(long off) const {
    if (runners_per_ssd_ <= 1) return 0;
    return static_cast<int>((off >> 3) % runners_per_ssd_);
  }

  bool find_ready(int ns_id, int runner_in_ssd, int &cid, long &blk,
                  void *&dest, volatile int32_t *&sp, int32_t &sv,
                  volatile int64_t*& complete_ns) {
    Req res;

    if (submit_queue[ns_id][runner_in_ssd]->pop(res)) {
      dest = res.dst;
      blk = res.r;
      cid = res.cid;
      sp = res.status_ptr;
      sv = res.status_value;
      complete_ns = res.complete_ns;
      if (spdk_diag_enabled_ && res.enqueue_ns > 0) {
        auto pop_ns = std::chrono::duration<uint64_t, std::nano>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t qlat_ns = pop_ns - res.enqueue_ns;
        int b = io_lat_bucket(qlat_ns);
        q_lat_hist[ns_id][b].fetch_add(1, std::memory_order_relaxed);
        q_lat_sum_ns[ns_id]->fetch_add(qlat_ns, std::memory_order_relaxed);
        q_lat_cnt[ns_id]->fetch_add(1, std::memory_order_relaxed);
      }
      return true;
    }
    return false;
  }

  std::atomic<int> *read_cnt;
  // Per-runner submitted (cumulative) and in-flight (submitted - completed)
  // counters for live diagnostics. Layout: [ssd][runner].
  std::vector<std::vector<std::atomic<long>>> runner_submitted;
  std::vector<std::vector<std::atomic<long>>> runner_completed;
  std::atomic<int> finished;

  void worker_thread() {
    read_cnt = new std::atomic<int>[num_ssd];
    for (int i = 0; i < num_ssd; i++) {
      read_cnt[i].store(0, std::memory_order_relaxed);
    }
    idle_cnt = 0;

    finished = false;
    runner_submitted = std::vector<std::vector<std::atomic<long>>>(num_ssd);
    runner_completed = std::vector<std::vector<std::atomic<long>>>(num_ssd);
    for (int s = 0; s < num_ssd; s++) {
      runner_submitted[s] = std::vector<std::atomic<long>>(runners_per_ssd_);
      runner_completed[s] = std::vector<std::atomic<long>>(runners_per_ssd_);
      for (int r = 0; r < runners_per_ssd_; r++) {
        runner_submitted[s][r].store(0);
        runner_completed[s][r].store(0);
      }
    }

    // Per-IO latency histogram init
    io_lat_hist.resize(num_ssd);
    io_lat_sum_ns.resize(num_ssd);
    io_lat_cnt.resize(num_ssd);
    for (int s = 0; s < num_ssd; s++) {
      io_lat_hist[s] = new std::atomic<uint64_t>[IO_LAT_BUCKETS]();
      io_lat_sum_ns[s] = new std::atomic<uint64_t>(0);
      io_lat_cnt[s] = new std::atomic<uint64_t>(0);
    }

    // Queue latency histogram + runner batch size init
    q_lat_hist.resize(num_ssd);
    q_lat_sum_ns.resize(num_ssd);
    q_lat_cnt.resize(num_ssd);
    runner_batch_cnt.resize(num_ssd);
    runner_batch_total.resize(num_ssd);
    for (int s = 0; s < num_ssd; s++) {
      q_lat_hist[s] = new std::atomic<uint64_t>[IO_LAT_BUCKETS]();
      q_lat_sum_ns[s] = new std::atomic<uint64_t>(0);
      q_lat_cnt[s] = new std::atomic<uint64_t>(0);
      runner_batch_cnt[s] = new std::atomic<uint64_t>(0);
      runner_batch_total[s] = new std::atomic<uint64_t>(0);
    }

    auto monitor = [&]() {
      uint64_t diag_last_ns = steady_now_ns();
      int last_idle = 0;
      std::vector<long> last(num_ssd, 0);
      std::vector<std::vector<long>> last_runner_cmp(
          num_ssd, std::vector<long>(runners_per_ssd_, 0));
      std::vector<uint64_t> last_cnt(num_ssd, 0);
      std::vector<uint64_t> last_sum(num_ssd, 0);
      std::vector<std::vector<uint64_t>> last_hist(
          num_ssd, std::vector<uint64_t>(IO_LAT_BUCKETS, 0));
      std::vector<uint64_t> last_q_cnt(num_ssd, 0);
      std::vector<uint64_t> last_q_sum(num_ssd, 0);
      std::vector<std::vector<uint64_t>> last_q_hist(
          num_ssd, std::vector<uint64_t>(IO_LAT_BUCKETS, 0));
      std::vector<uint64_t> last_batch_cnt(num_ssd, 0);
      std::vector<uint64_t> last_batch_total(num_ssd, 0);
      while (!finished.load()) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(hot_sample_interval_ms_));

        flush_hot_sample(true);
        const uint64_t diag_now_ns = steady_now_ns();
        const double elapsed =
            static_cast<double>(diag_now_ns - diag_last_ns) / 1e9;
        diag_last_ns = diag_now_ns;
        std::vector<long> rds;
        for (int i = 0; i < num_ssd; i++) {
          long cur = read_cnt[i];
          long r = cur - last[i];
          last[i] = cur;
          rds.push_back(r);
        }

        long r = std::accumulate(rds.begin(), rds.end(), 0L);

        if (spdk_diag_enabled_) {
          printf("Bandwidth: %lf GB/s, %ld\n",
                 r * PG_SIZE / elapsed / 1024 / 1024 / 1024, r);
          printf("IOPS: ");
          for (int i = 0; i < num_ssd; i++) {
            printf("%ld, ", rds[i]);
          }
          printf("\n");

          printf("PerRunner:");
          for (int s = 0; s < num_ssd; s++) {
            for (int rr = 0; rr < runners_per_ssd_; rr++) {
              long sub_cur = runner_submitted[s][rr].load(std::memory_order_relaxed);
              long cmp_cur = runner_completed[s][rr].load(std::memory_order_relaxed);
              long iops = cmp_cur - last_runner_cmp[s][rr];
              long depth = sub_cur - cmp_cur;
              last_runner_cmp[s][rr] = cmp_cur;
              printf(" [s%d r%d iops=%ld depth=%ld]", s, rr, iops, depth);
            }
          }
          printf("\n");
        }

        int idle_cur = idle_cnt.load();
        int val = idle_cur - last_idle;
        last_idle = idle_cur;
        if (spdk_diag_enabled_) {
          printf("Idle cnt: %d\n", val);

          printf("Queue Occupacy: ");
          for (auto &per_ssd : submit_queue) {
            for (auto &q : per_ssd) {
              for (auto &sq : q->q) {
                auto head = sq->head.load();
                auto tail = sq->tail.load();
                int size;
                if (head <= tail) {
                  size = tail - head;
                } else {
                  size = tail - head + sq->cap;
                }
                printf("%d, ", size);
              }
            }
          }
          printf("\n");
        }

        if (spdk_diag_enabled_) {
          static const char* bucket_labels[IO_LAT_BUCKETS] = {
            "0-1us", "1-2us", "2-4us", "4-8us", "8-16us", "16-32us",
            "32-64us", "64-128us", "128-256us", "256-512us", "512-1ms",
            "1-2ms", "2-4ms", "4-8ms", "8-16ms", "16ms+"
          };
          static const double bucket_mid_us[IO_LAT_BUCKETS] = {
            0.5, 1.5, 3, 6, 12, 24, 48, 96, 192, 384, 768,
            1536, 3072, 6144, 12288, 24576
          };
          for (int s = 0; s < num_ssd; s++) {
            uint64_t cnt_cur = io_lat_cnt[s]->load(std::memory_order_relaxed);
            uint64_t sum_cur = io_lat_sum_ns[s]->load(std::memory_order_relaxed);
            uint64_t delta_cnt = cnt_cur - last_cnt[s];
            uint64_t delta_sum = sum_cur - last_sum[s];
            last_cnt[s] = cnt_cur;
            last_sum[s] = sum_cur;

            std::vector<uint64_t> delta_hist(IO_LAT_BUCKETS);
            uint64_t delta_total = 0;
            printf("IO_Lat[s%d]:", s);
            for (int b = 0; b < IO_LAT_BUCKETS; b++) {
              uint64_t v = io_lat_hist[s][b].load(std::memory_order_relaxed);
              delta_hist[b] = v - last_hist[s][b];
              last_hist[s][b] = v;
              delta_total += delta_hist[b];
              if (delta_hist[b] > 0)
                printf(" %s=%lu", bucket_labels[b], delta_hist[b]);
            }

            double avg_us = delta_cnt > 0 ? (delta_sum / (double)delta_cnt / 1000.0) : 0;
            double p50 = 0, p90 = 0, p99 = 0;
            uint64_t cum = 0;
            for (int b = 0; b < IO_LAT_BUCKETS; b++) {
              cum += delta_hist[b];
              if (p50 == 0 && cum >= delta_total * 0.50) p50 = bucket_mid_us[b];
              if (p90 == 0 && cum >= delta_total * 0.90) p90 = bucket_mid_us[b];
              if (p99 == 0 && cum >= delta_total * 0.99) p99 = bucket_mid_us[b];
            }
            printf(" | avg=%.1fus p50=%.0fus p90=%.0fus p99=%.0fus n=%lu\n",
                   avg_us, p50, p90, p99, delta_cnt);
          }

          for (int s = 0; s < num_ssd; s++) {
            uint64_t cnt_cur = q_lat_cnt[s]->load(std::memory_order_relaxed);
            uint64_t sum_cur = q_lat_sum_ns[s]->load(std::memory_order_relaxed);
            uint64_t delta_cnt = cnt_cur - last_q_cnt[s];
            uint64_t delta_sum = sum_cur - last_q_sum[s];
            last_q_cnt[s] = cnt_cur;
            last_q_sum[s] = sum_cur;

            std::vector<uint64_t> delta_hist(IO_LAT_BUCKETS);
            uint64_t delta_total = 0;
            printf("Q_Lat[s%d]:", s);
            for (int b = 0; b < IO_LAT_BUCKETS; b++) {
              uint64_t v = q_lat_hist[s][b].load(std::memory_order_relaxed);
              delta_hist[b] = v - last_q_hist[s][b];
              last_q_hist[s][b] = v;
              delta_total += delta_hist[b];
              if (delta_hist[b] > 0)
                printf(" %s=%lu", bucket_labels[b], delta_hist[b]);
            }

            double avg_us = delta_cnt > 0 ? (delta_sum / (double)delta_cnt / 1000.0) : 0;
            double p50 = 0, p90 = 0, p99 = 0;
            uint64_t cum = 0;
            for (int b = 0; b < IO_LAT_BUCKETS; b++) {
              cum += delta_hist[b];
              if (p50 == 0 && cum >= delta_total * 0.50) p50 = bucket_mid_us[b];
              if (p90 == 0 && cum >= delta_total * 0.90) p90 = bucket_mid_us[b];
              if (p99 == 0 && cum >= delta_total * 0.99) p99 = bucket_mid_us[b];
            }
            printf(" | avg=%.1fus p50=%.0fus p90=%.0fus p99=%.0fus n=%lu\n",
                   avg_us, p50, p90, p99, delta_cnt);
          }

          for (int s = 0; s < num_ssd; s++) {
            uint64_t bc = runner_batch_cnt[s]->load(std::memory_order_relaxed);
            uint64_t bt = runner_batch_total[s]->load(std::memory_order_relaxed);
            uint64_t dc = bc - last_batch_cnt[s];
            uint64_t dt = bt - last_batch_total[s];
            last_batch_cnt[s] = bc;
            last_batch_total[s] = bt;
            double avg_batch = dc > 0 ? (double)dt / dc : 0;
            printf("Batch[s%d]: avg=%.1f rounds=%lu ios=%lu\n", s, avg_batch, dc, dt);
          }
        }
      }
    };

    auto runner = [&](int nid, int r_in_ssd) {
      // qp策略：runners_per_ssd>1 时每个 runner 独占 1 qp（global qp_id =
      // r_in_ssd）；runners_per_ssd==1 时退回 SPDK_QPAIRS_PER_SSD 兼容路径。
      int num_qp;
      int qp_base;
      if (runners_per_ssd_ > 1) {
        num_qp = 1;
        qp_base = r_in_ssd;
      } else {
        const char* env = getenv("SPDK_QPAIRS_PER_SSD");
        num_qp = env ? std::max(1, std::min(atoi(env), 32)) : 1;
        qp_base = 0;
      }
      // runners_per_ssd==1 preserves the legacy binding (core=ssd*2+3)
      // for performance regression compatibility. Multi-runner mode fans out
      // from core 4 with global_runner_id = nid * R + r_in_ssd.
      int global_r = nid * runners_per_ssd_ + r_in_ssd;
      int core = runners_per_ssd_ == 1 ? nid * 2 + 3 : 4 + global_r * 2;
      shared::bind_core(core);
      if (spdk_diag_enabled_) {
        printf("[SPDK runner ssd=%d r=%d] TID: %d core: %d qp_base=%d num_qp=%d\n",
               nid, r_in_ssd, gettid(), sched_getcpu(), qp_base, num_qp);
      }

      const int runner_io_contexts = runner_io_context_count_;
      std::vector<Ctx> ctxs(runner_io_contexts);
      Queue<Ctx *> idle_queue(runner_io_contexts), wait_queue(runner_io_contexts);

      char *buff = (char *)spdk_dma_zmalloc(
          runner_io_contexts * PG_SIZE, PG_SIZE, nullptr);

      for (int i = 0; i < runner_io_contexts; i++) {
        ctxs[i].ns_id = nid;
        ctxs[i].qp_id = qp_base + (i % num_qp);
        ctxs[i].runner_in_ssd = r_in_ssd;
        ctxs[i].wait_queue = &wait_queue;
        ctxs[i].idle_queue = &idle_queue;
        ctxs[i].read_cnt = &read_cnt[nid];
        ctxs[i].ctx = this;
        ctxs[i].buffer = buff + i * PG_SIZE;
        ctxs[i].cid = 0;
        ctxs[i].r = i;
        ctxs[i].status_ptr = nullptr;
        ctxs[i].status_value = 0;
        ctxs[i].complete_ns = nullptr;
        ctxs[i].retry_count = 0;
        ctxs[i].idle_queue->push(&ctxs[i]);
      }

      while (!finished.load()) {
        while (!wait_queue.empty()) {
          auto c = wait_queue.pop();
          if (!submit(c)) {
            break;
          }
        }

        for (int qp = 0; qp < num_qp; qp++) {
          spdk->PollCompleteQueue(nid, qp_base + qp);
        }

        long r;
        void *dst;
        int tid;
        volatile int32_t* sp = nullptr;
        int32_t sv = 0;
        volatile int64_t* complete_ns = nullptr;

        int batch_this_round = 0;
        while (!idle_queue.empty() &&
               find_ready(nid, r_in_ssd, tid, r, dst, sp, sv, complete_ns)) {
          auto c = idle_queue.pop();
          c->cid = tid;
          c->r = r;
          c->buffer = dst;
          c->status_ptr = sp;
          c->status_value = sv;
          c->complete_ns = complete_ns;
          c->retry_count = 0;
          submit(c);
          if (spdk_diag_enabled_) {
            batch_this_round++;
          }
        }
        if (spdk_diag_enabled_ && batch_this_round > 0) {
          runner_batch_total[nid]->fetch_add(batch_this_round, std::memory_order_relaxed);
          runner_batch_cnt[nid]->fetch_add(1, std::memory_order_relaxed);
        }
      }
    };

    worker.emplace_back(monitor);
    for (int nid = 0; nid < num_ssd; nid++) {
      for (int r = 0; r < runners_per_ssd_; r++) {
        worker.emplace_back(runner, nid, r);
      }
    }
  }

  uint8_t *create_buffer(int64_t size) override {
    return (uint8_t *)spdk_dma_zmalloc_socket(sizeof(uint8_t) * size,
                                              PAGE_SIZE, NULL, 1);
  }

  void destroy_buffer(uint8_t *buf) override { spdk_free(buf); }

  spdk_wrapper::SpdkWrapper *get_spdk() { return spdk.get(); }
  int get_num_ssd() const { return num_ssd; }
};

std::shared_ptr<IndexLoader> create_spdk_loader(const std::vector<std::string> &ssds,
                                                int submit_queue_capacity,
                                                int thread_cnt,
                                                int task_context_count,
                                                int runner_io_context_count,
                                                int runners_per_ssd) {
  std::shared_ptr<SpdkIOImpl> spdk_io = std::make_shared<SpdkIOImpl>();
  spdk_io->init(ssds, submit_queue_capacity, thread_cnt, task_context_count,
                runner_io_context_count, runners_per_ssd);
  return spdk_io;
}

}  // namespace shared
