#include "io_manager.h"
#include "../common/io.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <array>
#include <deque>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include <boost/lockfree/queue.hpp>

#include <fcntl.h>
#include <liburing.h>

// 兼容旧版内核头文件（Linux < 6.0）
#ifndef IORING_SETUP_SINGLE_ISSUER
#define IORING_SETUP_SINGLE_ISSUER (1U << 4)
#endif
#ifndef IORING_SETUP_COOP_TASKRUN
#define IORING_SETUP_COOP_TASKRUN (1U << 8)
#endif

#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

size_t next_power_of_two(size_t n) {
  if (n <= 1) {
    return 1;
  }
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  if constexpr (sizeof(size_t) == 8) {
    n |= n >> 32;
  }
  return n + 1;
}

static constexpr uint32_t kEmptyPage = 0xFFFFFFFFu;
static constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull;

enum class FrameState : uint32_t {
  Empty = 0,
  Loading = 1,
  Ready = 2,
  Evicting = 3
};

struct alignas(64) Frame {
  std::atomic<uint32_t> page_id{kEmptyPage};
  std::atomic<uint32_t> state{static_cast<uint32_t>(FrameState::Empty)};
  std::atomic<uint8_t> refbit{0};
  std::atomic<uint16_t> pin{0};
  char *data = nullptr;
};

struct WaitSlot {
  std::mutex mutex;
  std::condition_variable cv;
  uint64_t version = 0;
};

class WaitTable {
public:
  WaitTable() : slots_(0) {}
  explicit WaitTable(size_t slots) { reset(slots); }

  void reset(size_t slots) {
    slots_ = next_power_of_two(std::max<size_t>(1, slots));
    table_.reset();
    table_ = std::make_unique<WaitSlot[]>(slots_);
  }

  WaitSlot &slot(uint32_t page_id) { return table_[page_id & (slots_ - 1)]; }

  void notify(uint32_t page_id) {
    WaitSlot &s = slot(page_id);
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      ++s.version;
    }
    s.cv.notify_all();
  }

private:
  size_t slots_;
  std::unique_ptr<WaitSlot[]> table_;
};

class ConcurrentPageMap {
public:
  ConcurrentPageMap() { reset(8); }
  explicit ConcurrentPageMap(size_t capacity) { reset(capacity); }

  void reset(size_t capacity) {
    size_t desired = next_power_of_two(std::max<size_t>(capacity * 2, 8));
    mask_ = desired - 1;
    capacity_ = desired;
    entries_.reset(new std::atomic<uint64_t>[capacity_]);
    for (size_t i = 0; i < capacity_; ++i) {
      entries_[i].store(kEmptyEntry, std::memory_order_relaxed);
    }
  }

  bool find(uint32_t page_id, uint32_t &idx) const {
    size_t start = bucket(page_id);
    for (size_t probe = 0; probe <= mask_; ++probe) {
      size_t pos = (start + probe) & mask_;
      uint64_t entry = entries_[pos].load(std::memory_order_acquire);
      if (entry == kEmptyEntry) {
        return false;
      }
      if (entry == kTombstone) {
        continue;
      }
      if (page_of(entry) == page_id) {
        idx = index_of(entry);
        return true;
      }
    }
    return false;
  }

  bool insert(uint32_t page_id, uint32_t idx, uint32_t &existing) {
    size_t start = bucket(page_id);
    uint64_t encoded = encode(page_id, idx);
    for (size_t probe = 0; probe <= mask_; ++probe) {
      size_t pos = (start + probe) & mask_;
      uint64_t cur = entries_[pos].load(std::memory_order_acquire);
      if (cur == kEmptyEntry || cur == kTombstone) {
        if (entries_[pos].compare_exchange_strong(cur, encoded,
                                                  std::memory_order_acq_rel)) {
          return true;
        }
        continue;
      }
      if (page_of(cur) == page_id) {
        existing = index_of(cur);
        return false;
      }
    }
    return false;
  }

  bool erase(uint32_t page_id, uint32_t idx) {
    size_t start = bucket(page_id);
    for (size_t probe = 0; probe <= mask_; ++probe) {
      size_t pos = (start + probe) & mask_;
      uint64_t cur = entries_[pos].load(std::memory_order_acquire);
      if (cur == kEmptyEntry) {
        return false;
      }
      if (cur == kTombstone) {
        continue;
      }
      if (page_of(cur) == page_id && index_of(cur) == idx) {
        uint64_t expected = cur;
        return entries_[pos].compare_exchange_strong(expected, kTombstone,
                                                     std::memory_order_acq_rel);
      }
    }
    return false;
  }

private:
  size_t bucket(uint32_t page_id) const {
    uint64_t z = static_cast<uint64_t>(page_id) * kMul;
    return (z >> 32) & mask_;
  }

  static constexpr uint64_t kEmptyEntry = std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t kTombstone =
      std::numeric_limits<uint64_t>::max() - 1;

  static uint64_t encode(uint32_t page_id, uint32_t idx) {
    return (static_cast<uint64_t>(page_id) << 32) | static_cast<uint64_t>(idx);
  }

  static uint32_t page_of(uint64_t entry) {
    return static_cast<uint32_t>(entry >> 32);
  }
  static uint32_t index_of(uint64_t entry) {
    return static_cast<uint32_t>(entry & 0xFFFFFFFFu);
  }

  size_t mask_ = 0;
  size_t capacity_ = 0;
  std::unique_ptr<std::atomic<uint64_t>[]> entries_;
};

class IPageReader {
public:
  virtual ~IPageReader() = default;

  virtual void read_page_sync(uint32_t page_id, void *dst) = 0;

  virtual void read_pages_async(const std::vector<uint32_t> &page_ids,
                                const std::vector<void *> &dsts,
                                std::promise<int> &done) {
    try {
      for (size_t i = 0; i < page_ids.size(); ++i) {
        read_page_sync(page_ids[i], dsts[i]);
      }
      done.set_value(0);
    } catch (const std::system_error &se) {
      done.set_value(se.code().value());
    } catch (...) {
      done.set_value(EIO);
    }
  }

  virtual bool supports_async() const { return false; }
};

} // namespace

class IOManager::Backend {
public:
  Backend(const std::vector<VectorLocation> &map, int dim,
          IOManager::VectorStorage storage)
      : location_map_(map), dim_(dim), storage_(storage) {}
  virtual ~Backend() = default;

  virtual void get_vectors(const std::vector<long> &ids,
                           std::vector<float> &out_data,
                           uint32_t *out_unique_pages) = 0;
  virtual IOManager::IOGlobal snapshot() const = 0;

protected:
  size_t encoded_vector_bytes() const {
    return static_cast<size_t>(dim_) *
           (storage_ == IOManager::VectorStorage::Uint8 ? sizeof(uint8_t)
                                                        : sizeof(float));
  }

  void decode_vector(const char *src, float *dst) const {
    if (storage_ == IOManager::VectorStorage::Uint8) {
      const auto *values = reinterpret_cast<const uint8_t *>(src);
      for (int d = 0; d < dim_; ++d) {
        dst[d] = static_cast<float>(values[d]);
      }
      return;
    }
    std::memcpy(dst, src, static_cast<size_t>(dim_) * sizeof(float));
  }

  const std::vector<VectorLocation> &location_map_;
  int dim_ = 128;
  IOManager::VectorStorage storage_ = IOManager::VectorStorage::Float32;
};

namespace {

class PreadPageReader : public IPageReader {
public:
  explicit PreadPageReader(const std::string &packed_vectors_file) {
    initialize(packed_vectors_file);
  }
  ~PreadPageReader() override { finalize(); }

  void read_page_sync(uint32_t page_id, void *dst) override {
    off_t off = static_cast<off_t>(page_id) * static_cast<off_t>(SSD_PAGE_SIZE);
    ssize_t n = ::pread(fd_, dst, SSD_PAGE_SIZE, off);
    if (n != static_cast<ssize_t>(SSD_PAGE_SIZE)) {
      int err = n < 0 ? errno : EIO;
      throw std::system_error(err, std::generic_category(),
                              "pread page failed");
    }
  }

  bool supports_async() const override { return false; }

private:
  void initialize(const std::string &packed_vectors_file) {
    int flags = O_RDONLY;
#ifdef O_DIRECT
    flags |= O_DIRECT;
#else
    dio_ = false;
#endif

    fd_ = ::open(packed_vectors_file.c_str(), flags);
#ifdef O_DIRECT
    if (fd_ < 0 && (flags & O_DIRECT)) {
      dio_ = false;
      fd_ = ::open(packed_vectors_file.c_str(), O_RDONLY);
    }
#endif

    if (fd_ < 0) {
      throw std::runtime_error("Cannot open file: " + packed_vectors_file +
                               ", err=" + std::string(::strerror(errno)));
    }

    std::cout << "  > IOManager backend: pread (Direct I/O: "
              << (dio_ ? "ON" : "OFF (fallback)") << ")" << std::endl;
  }

  void finalize() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd_{-1};
  bool dio_{true};
};

class IoUringPageReader : public IPageReader {
public:
  explicit IoUringPageReader(const std::string &packed_vectors_file) {
    initialize(packed_vectors_file);
  }

  ~IoUringPageReader() override { finalize(); }

  void read_page_sync(uint32_t page_id, void *dst) override {
    off_t off = static_cast<off_t>(page_id) * static_cast<off_t>(SSD_PAGE_SIZE);
    ssize_t n = ::pread(fd_, dst, SSD_PAGE_SIZE, off);
    if (n != static_cast<ssize_t>(SSD_PAGE_SIZE)) {
      int err = n < 0 ? errno : EIO;
      throw std::system_error(err, std::generic_category(),
                              "pread page failed");
    }
  }

  bool supports_async() const override { return true; }

  void read_pages_async(const std::vector<uint32_t> &page_ids,
                        const std::vector<void *> &dsts,
                        std::promise<int> &done) override {
    if (page_ids.size() != dsts.size()) {
      done.set_value(EINVAL);
      return;
    }
    if (page_ids.empty()) {
      done.set_value(0);
      return;
    }
    if (!ready_) {
      done.set_value(EIO);
      return;
    }
    auto *batch = new ReadBatch();
    batch->items.resize(page_ids.size());
    batch->pending.store(static_cast<uint32_t>(page_ids.size()),
                         std::memory_order_relaxed);
    batch->error.store(0, std::memory_order_relaxed);
    batch->user_promise = &done;

    for (size_t i = 0; i < page_ids.size(); ++i) {
      batch->items[i].page_id = page_ids[i];
      batch->items[i].dst = dsts[i];
      batch->items[i].owner = batch;
    }

    if (!enqueue_batch(batch)) {
      batch->user_promise->set_value(ECANCELED);
      delete batch;
      return;
    }
  }

private:
  struct ReadBatch;

  struct ReadItem {
    uint32_t page_id = 0;
    void *dst = nullptr;
    ReadBatch *owner = nullptr;
  };

  struct ReadBatch {
    std::vector<ReadItem> items;
    std::atomic<uint32_t> pending{0};
    std::atomic<int> error{0};
    std::promise<int> *user_promise = nullptr;
  };

  static int translate_error(int res) {
    if (res >= 0) {
      return EIO;
    }
    return -res;
  }

  void initialize(const std::string &packed_vectors_file) {
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));

    auto try_init = [&](unsigned flags, const char *label) -> bool {
      memset(&params, 0, sizeof(params));
      params.flags = flags;
      params.sq_thread_idle = 2000;
      int ret = io_uring_queue_init_params(queue_depth_, &ring_, &params);
      if (ret == 0) {
        ready_ = true;
        used_flags_ = flags;
        used_label_ = label;
        return true;
      }
      return false;
    };

    // Try progressively simpler flag sets.
    const unsigned flag_sets[] = {
        IORING_SETUP_SQPOLL | IORING_SETUP_SINGLE_ISSUER |
            IORING_SETUP_COOP_TASKRUN,
        IORING_SETUP_SQPOLL | IORING_SETUP_SINGLE_ISSUER,
        IORING_SETUP_SQPOLL,
        0u,
    };
    const char *labels[] = {"SQPOLL|SINGLE_ISSUER|COOP_TASKRUN",
                            "SQPOLL|SINGLE_ISSUER", "SQPOLL", "none"};

    bool ok = false;
    for (size_t i = 0; i < sizeof(flag_sets) / sizeof(flag_sets[0]); ++i) {
      if (try_init(flag_sets[i], labels[i])) {
        ok = true;
        break;
      }
    }

    if (!ok) {
      throw std::runtime_error("Failed to initialize io_uring queue: " +
                               std::string(::strerror(errno)));
    }

    int flags = O_RDONLY;
#ifdef O_DIRECT
    flags |= O_DIRECT;
#else
    dio_ = false;
#endif

    fd_ = ::open(packed_vectors_file.c_str(), flags);
#ifdef O_DIRECT
    if (fd_ < 0 && (flags & O_DIRECT)) {
      dio_ = false;
      fd_ = ::open(packed_vectors_file.c_str(), O_RDONLY);
    }
#endif

    if (fd_ < 0) {
      throw std::runtime_error("Cannot open file: " + packed_vectors_file +
                               ", err=" + std::string(::strerror(errno)));
    }

    std::cout << "  > IOManager backend: io_uring (depth " << queue_depth_
              << ", flags=" << used_label_
              << ", Direct I/O: " << (dio_ ? "ON" : "OFF (fallback)") << ")"
              << std::endl;

    wake_fd_ = ::eventfd(0, 0);
    if (wake_fd_ < 0) {
      throw std::runtime_error("Failed to create eventfd for io_uring thread");
    }

    stop_.store(false, std::memory_order_release);
    io_thread_ = std::thread(&IoUringPageReader::io_loop, this);
  }

  void finalize() {
    stop_.store(true, std::memory_order_release);
    if (wake_fd_ >= 0) {
      uint64_t one = 1;
      ::eventfd_write(wake_fd_, one);
    }
    if (io_thread_.joinable()) {
      io_thread_.join();
    }
    if (ready_) {
      io_uring_queue_exit(&ring_);
      ready_ = false;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (wake_fd_ >= 0) {
      ::close(wake_fd_);
      wake_fd_ = -1;
    }
  }

  bool enqueue_batch(ReadBatch *batch) {
    if (stop_.load(std::memory_order_acquire)) {
      return false;
    }
    size_t retries = 0;
    while (!queue_.push(batch)) {
      if (stop_.load(std::memory_order_acquire)) {
        return false;
      }
      if (++retries > 1024) {
        return false;
      }
      std::this_thread::yield();
    }
    if (wake_fd_ >= 0) {
      uint64_t one = 1;
      ::eventfd_write(wake_fd_, one);
    }
    return true;
  }

  void io_loop() {
    std::vector<ReadBatch *> pending_batches;
    pending_batches.reserve(64);
    while (true) {
      pending_batches.clear();
      ReadBatch *b = nullptr;
      while (queue_.pop(b)) {
        if (b) {
          pending_batches.push_back(b);
        }
      }

      if (!pending_batches.empty()) {
        submit_batches(pending_batches);
      }

      if (outstanding_.load(std::memory_order_acquire) > 0) {
        reap_completions(true);
      }

      if (stop_.load(std::memory_order_acquire)) {
        drain_queue_on_stop();
        if (outstanding_.load(std::memory_order_acquire) == 0) {
          break;
        }
      }

      if (pending_batches.empty() &&
          outstanding_.load(std::memory_order_acquire) == 0 &&
          !stop_.load(std::memory_order_acquire)) {
        wait_for_wake();
      }
    }
    reap_completions(true);
    drain_queue_on_stop();
  }

  void submit_batches(const std::vector<ReadBatch *> &batches) {
    auto fail_all = [&](int err) {
      for (ReadBatch *b : batches) {
        fail_batch(b, err);
      }
    };

    uint32_t prepared = 0;

    auto flush_prepared = [&]() -> bool {
      while (prepared > 0) {
        int rc = io_uring_submit(&ring_);
        if (rc < 0) {
          fail_all(translate_error(rc));
          return false;
        }
        if (rc == 0) {
          reap_completions(true);
          continue;
        }
        outstanding_.fetch_add(static_cast<uint64_t>(rc),
                               std::memory_order_acq_rel);
        prepared -= static_cast<uint32_t>(rc);
      }
      return true;
    };

    for (ReadBatch *batch : batches) {
      for (ReadItem &itm_ref : batch->items) {
        ReadItem *item = &itm_ref;
        while (true) {
          io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
          if (!sqe) {
            if (prepared > 0) {
              if (!flush_prepared()) {
                return;
              }
            } else {
              reap_completions(true);
            }
            continue;
          }
          off_t off = static_cast<off_t>(item->page_id) *
                      static_cast<off_t>(SSD_PAGE_SIZE);
          io_uring_prep_read(sqe, fd_, item->dst, SSD_PAGE_SIZE, off);
          io_uring_sqe_set_data(sqe, item);
          ++prepared;
          break;
        }
      }
    }

    if (prepared > 0) {
      flush_prepared();
    }
  }

  void reap_completions(bool allow_block) {
    std::array<io_uring_cqe *, 64> cqes{};
    while (true) {
      unsigned ready = io_uring_peek_batch_cqe(&ring_, cqes.data(), cqes.size());
      if (ready == 0) {
        if (allow_block &&
            outstanding_.load(std::memory_order_acquire) > 0) {
          io_uring_cqe *one = nullptr;
          int rc = io_uring_wait_cqe(&ring_, &one);
          if (rc < 0) {
            int err = translate_error(rc);
            handle_wait_error(err);
            return;
          }
          if (one) {
            cqes[0] = one;
            ready = 1;
          }
        } else {
          return;
        }
      }

      if (ready == 0) {
        return;
      }

      for (unsigned i = 0; i < ready; ++i) {
        process_cqe(cqes[i]);
      }
    }
  }

  void wait_for_wake() {
    if (wake_fd_ < 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      return;
    }
    uint64_t val = 0;
    while (true) {
      int rc = ::eventfd_read(wake_fd_, &val);
      if (rc == 0) {
        return;
      }
      if (rc < 0 && errno == EINTR) {
        continue;
      }
      if (rc < 0 && errno == EAGAIN) {
        return;
      }
      return;
    }
  }

  void handle_wait_error(int err) {
    std::deque<ReadBatch *> pending;
    ReadBatch *b = nullptr;
    while (queue_.pop(b)) {
      if (b) {
        pending.push_back(b);
      }
    }
    while (!pending.empty()) {
      ReadBatch *batch = pending.front();
      pending.pop_front();
      fail_batch(batch, err);
    }
  }

  void process_cqe(io_uring_cqe *cqe) {
    if (!cqe) {
      return;
    }
    ReadItem *item = static_cast<ReadItem *>(io_uring_cqe_get_data(cqe));
    if (!item) {
      io_uring_cqe_seen(&ring_, cqe);
      return;
    }

    int res = cqe->res;
    int err = 0;
    if (res != static_cast<int>(SSD_PAGE_SIZE)) {
      err = translate_error(res);
    }

    ReadBatch *owner = item->owner;
    if (err != 0) {
      int expected = 0;
      owner->error.compare_exchange_strong(expected, err,
                                           std::memory_order_acq_rel);
    }

    uint32_t remaining =
        owner->pending.fetch_sub(1, std::memory_order_acq_rel);
    if (remaining == 1) {
      int final_err = owner->error.load(std::memory_order_acquire);
      owner->user_promise->set_value(final_err);
      delete owner;
    }

    outstanding_.fetch_sub(1, std::memory_order_acq_rel);
    io_uring_cqe_seen(&ring_, cqe);
  }

  void fail_batch(ReadBatch *batch, int err) {
    if (!batch) {
      return;
    }
    batch->error.store(err, std::memory_order_release);
    batch->pending.store(0, std::memory_order_release);
    if (batch->user_promise) {
      batch->user_promise->set_value(err);
    }
    delete batch;
  }

  void drain_queue_on_stop() {
    std::deque<ReadBatch *> leftover;
    ReadBatch *b = nullptr;
    while (queue_.pop(b)) {
      if (b) {
        leftover.push_back(b);
      }
    }
    while (!leftover.empty()) {
      ReadBatch *batch = leftover.front();
      leftover.pop_front();
      fail_batch(batch, ECANCELED);
    }
  }

  io_uring ring_{};
  unsigned queue_depth_ = 512;
  int fd_{-1};
  bool dio_{true};
  bool ready_{false};
  std::thread io_thread_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> outstanding_{0};
  unsigned used_flags_ = 0;
  const char *used_label_ = "unknown";
  int wake_fd_{-1};

  static constexpr size_t kQueueCapacity = 8192;
  boost::lockfree::queue<ReadBatch *, boost::lockfree::capacity<kQueueCapacity>>
      queue_;
};

using aligned_buf_ptr = std::unique_ptr<char, void (*)(void *)>;

aligned_buf_ptr make_aligned_page() {
  void *p = nullptr;
  if (posix_memalign(&p, SSD_PAGE_SIZE, SSD_PAGE_SIZE) != 0 || p == nullptr) {
    throw std::bad_alloc();
  }
  return aligned_buf_ptr(reinterpret_cast<char *>(p), std::free);
}

class PageCacheBackend : public IOManager::Backend {
public:
  PageCacheBackend(const std::vector<VectorLocation> &map, int dim,
                   std::unique_ptr<IPageReader> reader, size_t max_pages,
                   IOManager::VectorStorage storage)
      : IOManager::Backend(map, dim, storage), reader_(std::move(reader)),
        max_pages_(max_pages), cache_enabled_(max_pages > 0),
        wait_table_(4096) {
    if (!reader_) {
      throw std::invalid_argument("PageCacheBackend requires a reader");
    }
    if (!cache_enabled_) {
      return;
    }
    page_table_ = std::make_unique<ConcurrentPageMap>(max_pages_);
    // Use a bounded number of wait slots to avoid huge mutex/cv arrays that
    // would trash caches and NUMA locality; large enough to spread contention.
    constexpr size_t kWaitSlots = 65536;
    size_t wait_slots = kWaitSlots;
    wait_table_.reset(wait_slots);
    frame_count_ = max_pages_;
    frames_ = std::make_unique<Frame[]>(frame_count_);
    for (size_t i = 0; i < frame_count_; ++i) {
      aligned_buf_ptr buf = make_aligned_page();
      Frame &fr = frames_[i];
      fr.data = buf.release();
      fr.page_id.store(kEmptyPage, std::memory_order_relaxed);
      fr.state.store(static_cast<uint32_t>(FrameState::Empty),
                     std::memory_order_relaxed);
      fr.refbit.store(0, std::memory_order_relaxed);
      fr.pin.store(0, std::memory_order_relaxed);
    }
    std::cout << "  > PageCacheBackend cache_enabled=" << cache_enabled_
              << " max_pages=" << max_pages_ << " frame_count=" << frame_count_
              << " bytes=" << (frame_count_ * SSD_PAGE_SIZE) / (1024.0 * 1024.0)
              << " MiB" << std::endl;
  }

  ~PageCacheBackend() override {
    for (size_t i = 0; i < frame_count_; ++i) {
      std::free(frames_[i].data);
      frames_[i].data = nullptr;
    }
  }

  void get_vectors(const std::vector<long> &ids, std::vector<float> &out_data,
                   uint32_t *out_unique_pages) override {
    if (out_unique_pages) {
      *out_unique_pages = 0;
    }
    if (ids.empty()) {
      return;
    }

    out_data.resize(ids.size() * dim_);

    std::vector<uint32_t> pages;
    pages.reserve(ids.size());
    for (long id : ids) {
      pages.push_back(location_map_[id].page_id);
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

    if (!cache_enabled_) {
      read_without_cache(pages, ids, out_data);
      if (out_unique_pages) {
        *out_unique_pages = static_cast<uint32_t>(pages.size());
      }
      return;
    }

    std::unordered_map<uint32_t, Frame *> page_views;
    page_views.reserve(pages.size());
    acquire_pages(pages, page_views);

    for (size_t i = 0; i < ids.size(); ++i) {
      long vec_id = ids[i];
      const auto &loc = location_map_[vec_id];
      auto it = page_views.find(loc.page_id);
      if (it == page_views.end()) {
        throw std::runtime_error("Page missing after load: " +
                                 std::to_string(loc.page_id));
      }
      const char *src = it->second->data + loc.offset_in_page;
      decode_vector(src, out_data.data() + i * dim_);
    }

    for (auto &kv : page_views) {
      unpin(kv.second);
    }

    if (out_unique_pages) {
      *out_unique_pages = static_cast<uint32_t>(pages.size());
    }
  }

  IOManager::IOGlobal snapshot() const override {
    IOManager::IOGlobal g;
    g.hits = hits_.load(std::memory_order_relaxed);
    g.misses = misses_.load(std::memory_order_relaxed);
    g.io_count = io_count_.load(std::memory_order_relaxed);
    g.evictions = evictions_.load(std::memory_order_relaxed);
    return g;
  }

private:
  struct LoaderItem {
    uint32_t page_id = 0;
    uint32_t frame_idx = 0;
    Frame *frame = nullptr;
  };

  void read_without_cache(const std::vector<uint32_t> &pages,
                          const std::vector<long> &ids,
                          std::vector<float> &out_data) {
    std::unordered_map<uint32_t, aligned_buf_ptr> temp;
    temp.reserve(pages.size());
    for (uint32_t page_id : pages) {
      aligned_buf_ptr buf = make_aligned_page();
      reader_->read_page_sync(page_id, buf.get());
      temp.emplace(page_id, std::move(buf));
    }

    misses_.fetch_add(pages.size(), std::memory_order_relaxed);
    io_count_.fetch_add(pages.size(), std::memory_order_relaxed);

    for (size_t i = 0; i < ids.size(); ++i) {
      long vec_id = ids[i];
      const auto &loc = location_map_[vec_id];
      auto it = temp.find(loc.page_id);
      if (it == temp.end()) {
        throw std::runtime_error("Temporary page missing: " +
                                 std::to_string(loc.page_id));
      }
      const char *src = it->second.get() + loc.offset_in_page;
      decode_vector(src, out_data.data() + i * dim_);
    }
  }

  Frame *try_hit(uint32_t page_id) {
    uint32_t idx = 0;
    if (!page_table_->find(page_id, idx)) {
      return nullptr;
    }
    Frame &fr = frames_[idx];
    fr.pin.fetch_add(1, std::memory_order_acq_rel);

    if (fr.state.load(std::memory_order_acquire) !=
            static_cast<uint32_t>(FrameState::Ready) ||
        fr.page_id.load(std::memory_order_acquire) != page_id) {
      fr.pin.fetch_sub(1, std::memory_order_acq_rel);
      return nullptr;
    }

    fr.refbit.store(1, std::memory_order_relaxed);
    hits_.fetch_add(1, std::memory_order_relaxed);
    return &fr;
  }

  void unpin(Frame *frame) {
    if (!frame) {
      return;
    }
    frame->pin.fetch_sub(1, std::memory_order_acq_rel);
  }

  void acquire_pages(const std::vector<uint32_t> &pages,
                     std::unordered_map<uint32_t, Frame *> &out) {
    std::vector<LoaderItem> loaders;
    loaders.reserve(pages.size());

    for (uint32_t page_id : pages) {
      Frame *hit = try_hit(page_id);
      if (hit) {
        out.emplace(page_id, hit);
        continue;
      }
      schedule_load(page_id, loaders, out);
    }

    if (!loaders.empty()) {
      perform_loads(loaders, out);
    }
  }

  void schedule_load(uint32_t page_id, std::vector<LoaderItem> &loaders,
                     std::unordered_map<uint32_t, Frame *> &out) {
    while (true) {
      uint32_t idx = claim_frame(page_id);
      uint32_t existing = 0;
      if (page_table_->insert(page_id, idx, existing)) {
        Frame &fr = frames_[idx];
        fr.refbit.store(0, std::memory_order_relaxed);
        fr.pin.store(1, std::memory_order_release);
        loaders.push_back({page_id, idx, &fr});
        return;
      }

      release_frame(idx);
      Frame *ready = wait_for_page(page_id);
      if (ready) {
        out.emplace(page_id, ready);
        return;
      }
    }
  }

  void perform_loads(const std::vector<LoaderItem> &loaders,
                     std::unordered_map<uint32_t, Frame *> &out) {
    std::vector<uint32_t> ids;
    std::vector<void *> dests;
    ids.reserve(loaders.size());
    dests.reserve(loaders.size());

    for (const auto &item : loaders) {
      ids.push_back(item.page_id);
      dests.push_back(item.frame->data);
    }

    misses_.fetch_add(loaders.size(), std::memory_order_relaxed);
    io_count_.fetch_add(loaders.size(), std::memory_order_relaxed);

    int err = 0;
    if (reader_->supports_async() && loaders.size() > 1) {
      std::promise<int> promise;
      auto future = promise.get_future();
      reader_->read_pages_async(ids, dests, promise);
      err = future.get();
    } else {
      try {
        for (size_t i = 0; i < ids.size(); ++i) {
          reader_->read_page_sync(ids[i], dests[i]);
        }
      } catch (const std::system_error &se) {
        err = se.code().value();
      } catch (...) {
        err = EIO;
      }
    }

    if (err != 0) {
      for (const auto &item : loaders) {
        page_table_->erase(item.page_id, item.frame_idx);
        Frame &fr = frames_[item.frame_idx];
        fr.page_id.store(kEmptyPage, std::memory_order_release);
        fr.state.store(static_cast<uint32_t>(FrameState::Empty),
                       std::memory_order_release);
        fr.refbit.store(0, std::memory_order_relaxed);
        fr.pin.store(0, std::memory_order_relaxed);
        notify_waiters(item.page_id);
      }
      throw std::runtime_error("Failed to load pages: " +
                               std::string(::strerror(err)));
    }

    for (const auto &item : loaders) {
      Frame &fr = frames_[item.frame_idx];
      fr.refbit.store(1, std::memory_order_relaxed);
      fr.state.store(static_cast<uint32_t>(FrameState::Ready),
                     std::memory_order_release);
      out.emplace(item.page_id, &fr);
      notify_waiters(item.page_id);
    }
  }

  uint32_t claim_frame(uint32_t page_id) {
    while (true) {
      uint32_t idx = static_cast<uint32_t>(
          clock_hand_.fetch_add(1, std::memory_order_acq_rel) % max_pages_);
      Frame &fr = frames_[idx];
      FrameState state =
          static_cast<FrameState>(fr.state.load(std::memory_order_acquire));

      if (state == FrameState::Empty) {
        uint32_t expected = static_cast<uint32_t>(FrameState::Empty);
        if (fr.state.compare_exchange_strong(
                expected, static_cast<uint32_t>(FrameState::Loading),
                std::memory_order_acq_rel)) {
          fr.page_id.store(page_id, std::memory_order_release);
          fr.refbit.store(0, std::memory_order_relaxed);
          fr.pin.store(0, std::memory_order_relaxed);
          return idx;
        }
        continue;
      }

      if (state != FrameState::Ready) {
        continue;
      }

      if (fr.refbit.exchange(0, std::memory_order_acq_rel) == 1) {
        continue;
      }

      uint32_t expected = static_cast<uint32_t>(FrameState::Ready);
      if (!fr.state.compare_exchange_strong(
              expected, static_cast<uint32_t>(FrameState::Evicting),
              std::memory_order_acq_rel)) {
        continue;
      }

      if (fr.pin.load(std::memory_order_acquire) != 0) {
        fr.state.store(static_cast<uint32_t>(FrameState::Ready),
                       std::memory_order_release);
        continue;
      }

      uint32_t evict_page = fr.page_id.load(std::memory_order_acquire);
      if (page_table_->erase(evict_page, idx)) {
        evictions_.fetch_add(1, std::memory_order_relaxed);
      } else {
        fr.state.store(static_cast<uint32_t>(FrameState::Ready),
                       std::memory_order_release);
        fr.refbit.store(1, std::memory_order_relaxed);
        fr.pin.store(0, std::memory_order_relaxed);
        continue;
      }

      fr.page_id.store(page_id, std::memory_order_release);
      fr.refbit.store(0, std::memory_order_relaxed);
      fr.pin.store(0, std::memory_order_relaxed);
      fr.state.store(static_cast<uint32_t>(FrameState::Loading),
                     std::memory_order_release);
      return idx;
    }
  }

  void release_frame(uint32_t idx) {
    Frame &fr = frames_[idx];
    fr.page_id.store(kEmptyPage, std::memory_order_release);
    fr.state.store(static_cast<uint32_t>(FrameState::Empty),
                   std::memory_order_release);
    fr.refbit.store(0, std::memory_order_relaxed);
    fr.pin.store(0, std::memory_order_relaxed);
  }

  Frame *wait_for_page(uint32_t page_id) {
    WaitSlot &slot = wait_table_.slot(page_id);
    std::unique_lock<std::mutex> lock(slot.mutex);
    uint64_t version = slot.version;
    while (true) {
      lock.unlock();
      Frame *hit = try_hit(page_id);
      if (hit) {
        waits_.fetch_add(1, std::memory_order_relaxed);
        return hit;
      }
      uint32_t idx = 0;
      if (!page_table_->find(page_id, idx)) {
        return nullptr;
      }
      lock.lock();
      slot.cv.wait(lock, [&] { return slot.version != version; });
      version = slot.version;
    }
  }

  void notify_waiters(uint32_t page_id) { wait_table_.notify(page_id); }

  std::unique_ptr<IPageReader> reader_;
  size_t max_pages_ = 0;
  bool cache_enabled_ = true;
  std::unique_ptr<ConcurrentPageMap> page_table_;
  std::unique_ptr<Frame[]> frames_;
  size_t frame_count_ = 0;
  WaitTable wait_table_;
  std::atomic<uint64_t> clock_hand_{0};
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> io_count_{0};
  std::atomic<uint64_t> evictions_{0};
  std::atomic<uint64_t> waits_{0};
};

class MMapBackend : public IOManager::Backend {
public:
  MMapBackend(const std::vector<VectorLocation> &map, int dim,
              const std::string &packed_vectors_file,
              IOManager::VectorStorage storage)
      : IOManager::Backend(map, dim, storage) {
    initialize(packed_vectors_file);
  }

  ~MMapBackend() override { finalize(); }

  void get_vectors(const std::vector<long> &ids, std::vector<float> &out_data,
                   uint32_t *out_unique_pages) override {
    if (out_unique_pages) {
      *out_unique_pages = 0;
    }
    if (ids.empty()) {
      return;
    }

    out_data.resize(ids.size() * dim_);
    std::vector<uint32_t> unique_pages;
    unique_pages.reserve(ids.size());
    for (long id : ids) {
      unique_pages.push_back(location_map_[id].page_id);
    }
    std::sort(unique_pages.begin(), unique_pages.end());
    unique_pages.erase(std::unique(unique_pages.begin(), unique_pages.end()),
                       unique_pages.end());

    if (out_unique_pages) {
      *out_unique_pages = static_cast<uint32_t>(unique_pages.size());
    }

    hits_.fetch_add(unique_pages.size(), std::memory_order_relaxed);

    const size_t bytes = encoded_vector_bytes();
    for (size_t i = 0; i < ids.size(); ++i) {
      long vec_id = ids[i];
      const auto &loc = location_map_[vec_id];
      size_t offset = static_cast<size_t>(loc.page_id) * SSD_PAGE_SIZE +
                      static_cast<size_t>(loc.offset_in_page);
      if (offset + bytes > mapped_size_) {
        throw std::out_of_range("Vector offset exceeds mapped range");
      }
      const char *src = mapped_base_ + offset;
      decode_vector(src, out_data.data() + i * dim_);
    }
  }

  IOManager::IOGlobal snapshot() const override {
    IOManager::IOGlobal g;
    g.hits = hits_.load(std::memory_order_relaxed);
    return g;
  }

private:
  void initialize(const std::string &packed_vectors_file) {
    fd_ = ::open(packed_vectors_file.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("Cannot open file: " + packed_vectors_file +
                               ", err=" + std::string(::strerror(errno)));
    }

    struct stat st {};
    if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
      throw std::runtime_error("Failed to stat file: " + packed_vectors_file);
    }
    mapped_size_ = static_cast<size_t>(st.st_size);

    void *addr = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr == MAP_FAILED) {
      throw std::runtime_error("Failed to mmap file: " + packed_vectors_file +
                               ", err=" + std::string(::strerror(errno)));
    }
    mapped_base_ = static_cast<const char *>(addr);
#ifdef POSIX_MADV_RANDOM
    ::posix_madvise(const_cast<char *>(mapped_base_), mapped_size_,
                    POSIX_MADV_RANDOM);
#elif defined(MADV_RANDOM)
    ::madvise(const_cast<char *>(mapped_base_), mapped_size_, MADV_RANDOM);
#endif

    std::cout << "  > IOManager backend: Memory-mapped" << std::endl;
  }

  void finalize() {
    if (mapped_base_) {
      ::munmap(const_cast<char *>(mapped_base_), mapped_size_);
      mapped_base_ = nullptr;
      mapped_size_ = 0;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd_{-1};
  const char *mapped_base_ = nullptr;
  size_t mapped_size_ = 0;

  std::atomic<uint64_t> hits_{0};
};

} // namespace

IOManager::IOManager(const std::string &map_file,
                     const std::string &packed_vectors_file, int dim,
                     BackendKind kind, size_t cache_pages,
                     VectorStorage storage)
    : dim_(dim) {
  load_binary<VectorLocation>(map_file, location_map_);
  if (location_map_.empty()) {
    throw std::runtime_error("Vector location map is empty");
  }

  switch (kind) {
  case BackendKind::Pread:
    backend_ = std::make_unique<PageCacheBackend>(
        location_map_, dim_,
        std::make_unique<PreadPageReader>(packed_vectors_file), cache_pages,
        storage);
    break;
  case BackendKind::MMap:
    backend_ =
        std::make_unique<MMapBackend>(location_map_, dim_, packed_vectors_file,
                                     storage);
    break;
  case BackendKind::IoUring:
    backend_ = std::make_unique<PageCacheBackend>(
        location_map_, dim_,
        std::make_unique<IoUringPageReader>(packed_vectors_file), cache_pages,
        storage);
    break;
  default:
    throw std::invalid_argument("Unsupported IO backend kind");
  }
}

IOManager::~IOManager() = default;

void IOManager::get_vectors(const std::vector<long> &ids,
                            std::vector<float> &out_data,
                            uint32_t *out_unique_pages) {
  backend_->get_vectors(ids, out_data, out_unique_pages);
}

IOManager::IOGlobal IOManager::snapshot() const { return backend_->snapshot(); }
