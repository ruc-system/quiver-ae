#include "../io_stats.hpp"
#include "../loader.hpp"
#include "../../../../shared/io/spdk_lba_offset.hpp"
#include "../../../../shared/io/spdk/submit_queue.hpp"

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

int hot_sample_interval_ms_from_env() {
  const char* env = std::getenv("QUIVER_SPDK_HOT_SAMPLE_MS");
  if (env == nullptr) {
    return 100;
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
    for (auto [blk, dst] : req) {
      const long logical_block = static_cast<long>(blk) + block_offset_;
      int ssd = logical_block % num_ssd;
      long off = base_lba_ + logical_block / num_ssd;
      submit_queue[ssd]->push(Req(cid, off, dst), tid);
    }
    ready[cid].fetch_add(req.size());
  }

  void submit_traced_task(const std::vector<TracedIoRequest> &req, int tid,
                          int cid) override {
    io_stats_record_enqueued(stats_.get(), req.size());
    const int64_t now_ns = static_cast<int64_t>(steady_now_ns());
    for (const auto &request : req) {
      const long logical_block =
          static_cast<long>(request.block_id) + block_offset_;
      int ssd = logical_block % num_ssd;
      long off = base_lba_ + logical_block / num_ssd;
      if (request.enqueue_ns != nullptr && request.trace_index >= 0) {
        request.enqueue_ns[request.trace_index] = now_ns;
      }
      submit_queue[ssd]->push(Req(cid, off, request.dest, request.trace_index,
                                  request.enqueue_ns, request.submit_ns,
                                  request.complete_ns),
                              tid);
    }
    ready[cid].fetch_add(req.size());
  }

  bool poll_task(int cid) override { return ready[cid].load() == 0; }

  void submit_direct(const std::vector<DirectIoRequest>& reqs, int tid) override {
    io_stats_record_enqueued(stats_.get(), reqs.size());
    for (auto& r : reqs) {
      const long logical_block =
          static_cast<long>(r.block_id) + block_offset_;
      int ssd = logical_block % num_ssd;
      long off = base_lba_ + logical_block / num_ssd;
      submit_queue[ssd]->push(Req(off, r.dest, r.status_ptr, r.done_value), tid);
    }
  }

  LoaderStatsSnapshot snapshot_stats() const override {
    if (stats_ == nullptr) {
      return {};
    }
    flush_hot_sample(true);
    return stats_->snapshot();
  }

  void init(const std::vector<std::string> &ssds, int queue_cap,
            int thread_cnt, int ctx_cnt) {
    spdk = get_or_create_shared_spdk_wrapper(ssds);
    ready = new std::atomic<int>[ctx_cnt];
    for (int i = 0; i < ctx_cnt; i++) {
      ready[i].store(0, std::memory_order_relaxed);
    }
    num_ssd = ssds.size();
    base_lba_ = spdk_base_lba_from_env();
    block_offset_ = spdk_block_offset_from_env();

    hot_sample_interval_ms_ = hot_sample_interval_ms_from_env();
    stats_ = std::make_unique<LoaderStatsAccumulator>(num_ssd);
    reset_hot_sample_baseline();



    for (int i = 0; i < num_ssd; i++) {
      submit_queue.push_back(new MPSCQueue<Req>(queue_cap, thread_cnt));
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
    int cid;
    long r;
    void *buffer;
    std::atomic<int> *read_cnt;
    Queue<Ctx *> *wait_queue;
    Queue<Ctx *> *idle_queue;
    SpdkIOImpl *ctx;
    volatile int32_t* status_ptr;
    int32_t status_value;
    int trace_index;
    int64_t *enqueue_ns;
    int64_t *submit_ns;
    int64_t *complete_ns;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
  };

  struct Req {
    int cid;
    long r;
    void *dst;
    volatile int32_t* status_ptr;
    int32_t status_value;
    int trace_index;
    int64_t *enqueue_ns;
    int64_t *submit_ns;
    int64_t *complete_ns;
    Req()
        : cid(0), r(0), dst(nullptr), status_ptr(nullptr), status_value(0),
          trace_index(-1), enqueue_ns(nullptr), submit_ns(nullptr),
          complete_ns(nullptr) {}
    Req(int cid, long r, void *dst)
        : cid(cid), r(r), dst(dst), status_ptr(nullptr), status_value(0),
          trace_index(-1), enqueue_ns(nullptr), submit_ns(nullptr),
          complete_ns(nullptr) {}
    Req(int cid, long r, void *dst, int trace_index, int64_t *enqueue_ns,
        int64_t *submit_ns, int64_t *complete_ns)
        : cid(cid), r(r), dst(dst), status_ptr(nullptr), status_value(0),
          trace_index(trace_index), enqueue_ns(enqueue_ns),
          submit_ns(submit_ns), complete_ns(complete_ns) {}
    Req(long r, void *dst, volatile int32_t* sp, int32_t sv)
        : cid(-1), r(r), dst(dst), status_ptr(sp), status_value(sv),
          trace_index(-1), enqueue_ns(nullptr), submit_ns(nullptr),
          complete_ns(nullptr) {}
  };

  int num_ssd;
  long base_lba_ = 0;
  long block_offset_ = 0;
  std::shared_ptr<spdk_wrapper::SpdkWrapper> spdk;
  std::vector<MPSCQueue<Req> *> submit_queue;
  std::atomic<int> *ready;
  std::vector<std::thread> worker;

  std::vector<int> fail;
  std::unique_ptr<LoaderStatsAccumulator> stats_;
  int hot_sample_interval_ms_ = 100;
  mutable std::mutex hot_sample_mu_;
  mutable uint64_t hot_last_sample_ns_ = 0;
  mutable uint64_t hot_last_submitted_ = 0;

  int b;
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
    if (spdk_nvme_cpl_is_error(cpl)) {
      io_stats_record_nvme_error(cur->ctx->stats_.get());
      printf("Error!!!!");
      return;
    }

    if (cur->status_ptr) {
      __atomic_store_n(cur->status_ptr, cur->status_value, __ATOMIC_RELEASE);
    } else {
      if (cur->complete_ns != nullptr && cur->trace_index >= 0) {
        cur->complete_ns[cur->trace_index] =
            static_cast<int64_t>(steady_now_ns());
      }
      if (cur->ctx->ready[cur->cid].fetch_sub(1) == 1) {
        cur->ctx->fail[cur->ns_id]++;
      }
    }
    if (cur->ctx->find_ready(cur->ns_id, cur->cid, cur->r, cur->buffer,
                             cur->status_ptr, cur->status_value,
                             cur->trace_index, cur->enqueue_ns,
                             cur->submit_ns, cur->complete_ns)) {
      submit(cur);
    } else {
      cur->ctx->idle_cnt.fetch_add(1);
      cur->idle_queue->push(cur);
    }
  }

  static bool submit(Ctx *ctx) {
    if (ctx->submit_ns != nullptr && ctx->trace_index >= 0) {
      ctx->submit_ns[ctx->trace_index] = static_cast<int64_t>(steady_now_ns());
    }
    int ret = ctx->ctx->spdk->TrySubmitReadCommand(
        ctx->buffer, PG_SIZE, ctx->r, callback, ctx, ctx->ns_id, ctx->qp_id);
    if (ret != 0) {
      if (ret == -ENOMEM) {
        io_stats_record_submit_backpressure(ctx->ctx->stats_.get());
        ctx->wait_queue->push(ctx);
        return false;
      } else {
        io_stats_record_nvme_error(ctx->ctx->stats_.get());
        printf("Error!!!\n");
      }
    }
    ctx->read_cnt->fetch_add(1);
    io_stats_record_submitted(ctx->ctx->stats_.get(), ctx->ns_id);
    return true;
  }

  bool find_ready(int ns_id, int &cid, long &blk, void *&dest,
                  volatile int32_t*& sp, int32_t& sv, int &trace_index,
                  int64_t *&enqueue_ns, int64_t *&submit_ns,
                  int64_t *&complete_ns) {
    Req res;

    if (submit_queue[ns_id]->pop(res)) {
      dest = res.dst;
      blk = res.r;
      cid = res.cid;
      sp = res.status_ptr;
      sv = res.status_value;
      trace_index = res.trace_index;
      enqueue_ns = res.enqueue_ns;
      submit_ns = res.submit_ns;
      complete_ns = res.complete_ns;
      return true;
    }
    return false;
  }

  std::atomic<int> *read_cnt;
  std::atomic<int> finished;

  void worker_thread() {
    read_cnt = new std::atomic<int>[num_ssd];
    for (int i = 0; i < num_ssd; i++) {
      read_cnt[i].store(0, std::memory_order_relaxed);
    }
    idle_cnt = 0;

    finished = false;
    fail.resize(num_ssd);
    const bool monitor_enabled = [] {
      const char *env = std::getenv("QUIVER_SPDK_DIAG");
      return env != nullptr && std::atoi(env) != 0;
    }();

    auto monitor = [this, monitor_enabled]() {
      uint64_t diag_last_ns = steady_now_ns();
      int last_idle = 0;
      std::vector<long> last(num_ssd);
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

        if (monitor_enabled) {
          printf("Bandwidth: %lf GB/s, %ld\n",
                 r * PG_SIZE / elapsed / 1024 / 1024 / 1024, r);
          printf("IOPS: ");
          for (int i = 0; i < num_ssd; i++) {
            printf("%ld, ", rds[i]);
          }
          printf("\n");
        }

        int idle_cur = idle_cnt.load();
        int val = idle_cur - last_idle;
        last_idle = idle_cur;
        if (monitor_enabled) {
          printf("Idle cnt: %d\n", val);

          printf("Queue Occupacy: ");
          for (auto &q : submit_queue) {
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
          printf("\n");
        }

      }
    };

    auto runner = [&](int nid) {
      const int num_qp = []() {
        const char* env = getenv("SPDK_NUM_QP");
        return env ? std::max(1, std::min(atoi(env), 32)) : 1;
      }();
      const int BATCH_SIZE = 1024;
      shared::bind_core(nid * 2 + 3);
      printf("TID: %d %d\n", gettid(), sched_getcpu());

      std::vector<Ctx> ctxs(BATCH_SIZE);
      Queue<Ctx *> idle_queue(BATCH_SIZE), wait_queue(BATCH_SIZE);

      char *buff = (char *)spdk_dma_zmalloc(BATCH_SIZE * PG_SIZE, PG_SIZE, nullptr);

      for (int i = 0; i < BATCH_SIZE; i++) {
        ctxs[i].ns_id = nid;
        ctxs[i].qp_id = i % num_qp;
        ctxs[i].wait_queue = &wait_queue;
        ctxs[i].idle_queue = &idle_queue;
        ctxs[i].read_cnt = &read_cnt[nid];
        ctxs[i].ctx = this;
        ctxs[i].buffer = buff + i * PG_SIZE;
        ctxs[i].cid = 0;
        ctxs[i].r = i;
        ctxs[i].status_ptr = nullptr;
        ctxs[i].status_value = 0;
        ctxs[i].trace_index = -1;
        ctxs[i].enqueue_ns = nullptr;
        ctxs[i].submit_ns = nullptr;
        ctxs[i].complete_ns = nullptr;
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
          spdk->PollCompleteQueue(nid, qp);
        }

        long r;
        void *dst;
        int tid;
        volatile int32_t* sp = nullptr;
        int32_t sv = 0;
        int trace_index = -1;
        int64_t *enqueue_ns = nullptr;
        int64_t *submit_ns = nullptr;
        int64_t *complete_ns = nullptr;

        while (!idle_queue.empty() &&
               find_ready(nid, tid, r, dst, sp, sv, trace_index, enqueue_ns,
                          submit_ns, complete_ns)) {
          auto c = idle_queue.pop();
          c->cid = tid;
          c->r = r;
          c->buffer = dst;
          c->status_ptr = sp;
          c->status_value = sv;
          c->trace_index = trace_index;
          c->enqueue_ns = enqueue_ns;
          c->submit_ns = submit_ns;
          c->complete_ns = complete_ns;
          submit(c);
        }
      }
    };

    worker.emplace_back(monitor);
    for (int i = 0; i < num_ssd; i++) {
      worker.emplace_back(runner, i);
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
                                                int queue_cap, int thread_cnt,
                                                int ctx_cnt,
                                                int runners_per_ssd) {
  // Baseline ignores runners_per_ssd; accepted for source-compat with quiver.
  (void)runners_per_ssd;
  std::shared_ptr<SpdkIOImpl> spdk_io = std::make_shared<SpdkIOImpl>();
  spdk_io->init(ssds, queue_cap, thread_cnt, ctx_cnt);
  return spdk_io;
}

}  // namespace shared
