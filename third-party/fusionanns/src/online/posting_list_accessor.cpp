#include "posting_list_accessor.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fusionann::online {

PostingListAccessor::PostingListAccessor(const std::string &path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error("Cannot open metadata file: " + path);
  }
  struct stat st {};
  if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
    ::close(fd_);
    throw std::runtime_error("Failed to stat metadata file: " + path);
  }
  mapped_size_ = static_cast<size_t>(st.st_size);
  base_ = static_cast<const char *>(
      ::mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED, fd_, 0));
  if (base_ == MAP_FAILED) {
    ::close(fd_);
    base_ = nullptr;
    throw std::runtime_error("Failed to mmap metadata file: " + path);
  }
  if (mlock(base_, mapped_size_) != 0) {
    // mlock 失败不致命——仅在容器/cgroup 环境中 RLIMIT_MEMLOCK 过低时发生
    // 元数据仍可通过 mmap 正常访问，只是可能被换出影响性能
    fprintf(stderr,
            "[WARN] mlock metadata file failed (%s): %s (size=%zu). "
            "Continuing without memory locking.\n",
            path.c_str(), strerror(errno), mapped_size_);
  }
  ::close(fd_);
  fd_ = -1;

  const char *ptr = base_;
  const char *end = base_ + mapped_size_;
  if (ptr + sizeof(uint32_t) > end) {
    throw std::runtime_error("Metadata header truncated");
  }
  uint32_t nlist = 0;
  std::memcpy(&nlist, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);
  entries_.resize(nlist);

  struct ListEntry {
    uint64_t offset;
    uint32_t count;
    uint32_t reserved;
  };

  size_t table_bytes = static_cast<size_t>(nlist) * sizeof(ListEntry);
  if (ptr + static_cast<ptrdiff_t>(table_bytes) > end) {
    throw std::runtime_error("Metadata table truncated");
  }

  const auto *table = reinterpret_cast<const ListEntry *>(ptr);
  ptr += table_bytes;

  size_t total_ids = 0;
  for (uint32_t cid = 0; cid < nlist; ++cid) {
    const ListEntry &entry = table[cid];
    if (entry.count == 0) {
      entries_[cid] = Entry{nullptr, 0};
      continue;
    }
    size_t element_offset = static_cast<size_t>(entry.offset);
    size_t element_count = static_cast<size_t>(entry.count);
    size_t byte_offset = element_offset * sizeof(int32_t);
    size_t byte_count = element_count * sizeof(int32_t);
    if (byte_offset + byte_count > static_cast<size_t>(end - ptr)) {
      throw std::runtime_error("Metadata entry out of range");
    }
    const int32_t *ids = reinterpret_cast<const int32_t *>(ptr + byte_offset);
    entries_[cid] = Entry{ids, element_count};
    total_ids += element_count;
  }
  total_postings_ = total_ids;

  // Compute max vector ID across all posting lists.
  int32_t max_id = 0;
  for (uint32_t cid = 0; cid < nlist; ++cid) {
    const Entry &e = entries_[cid];
    for (size_t j = 0; j < e.count; ++j) {
      if (e.ids[j] > max_id)
        max_id = e.ids[j];
    }
  }
  max_vector_id_ = static_cast<size_t>(max_id);
}

PostingListAccessor::~PostingListAccessor() {
  if (base_) {
    ::munmap(const_cast<char *>(base_), mapped_size_);
    base_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

struct PostingListAccessor::HotspotTracker {
  explicit HotspotTracker(size_t nlist)
      : list_count(nlist),
        hits(nlist ? std::make_unique<std::atomic<uint64_t>[]>(nlist)
                   : nullptr) {
    reset();
  }

  void record(int list_id) {
    if (!enabled.load(std::memory_order_relaxed)) {
      return;
    }
    size_t idx = static_cast<size_t>(list_id);
    if (idx >= list_count || !hits) {
      return;
    }
    hits[idx].fetch_add(1, std::memory_order_relaxed);
    total_hits.fetch_add(1, std::memory_order_relaxed);
  }

  void set_enabled(bool value) {
    enabled.store(value, std::memory_order_release);
  }

  bool is_enabled() const { return enabled.load(std::memory_order_acquire); }

  void reset() {
    total_hits.store(0, std::memory_order_relaxed);
    if (!hits) {
      return;
    }
    for (size_t i = 0; i < list_count; ++i) {
      hits[i].store(0, std::memory_order_relaxed);
    }
  }

  uint64_t total() const { return total_hits.load(std::memory_order_relaxed); }

  std::vector<PostingListAccessor::HotspotEntry> top(size_t limit) const {
    std::vector<PostingListAccessor::HotspotEntry> result;
    if (!hits) {
      return result;
    }
    result.reserve(list_count);
    for (size_t i = 0; i < list_count; ++i) {
      uint64_t count = hits[i].load(std::memory_order_relaxed);
      if (count == 0) {
        continue;
      }
      result.push_back(
          PostingListAccessor::HotspotEntry{static_cast<int>(i), count});
    }
    if (result.empty()) {
      return result;
    }
    std::sort(result.begin(), result.end(),
              [](const PostingListAccessor::HotspotEntry &a,
                 const PostingListAccessor::HotspotEntry &b) {
                return a.hits > b.hits;
              });
    if (limit > 0 && result.size() > limit) {
      result.resize(limit);
    }
    return result;
  }

  std::atomic<bool> enabled{false};
  std::atomic<uint64_t> total_hits{0};
  size_t list_count = 0;
  std::unique_ptr<std::atomic<uint64_t>[]> hits;
};

PostingListAccessor::View PostingListAccessor::view(int list_id) const {
  if (list_id < 0 || static_cast<size_t>(list_id) >= entries_.size()) {
    return {};
  }
  const Entry &entry = entries_[static_cast<size_t>(list_id)];
  if (hotspot_tracker_) {
    hotspot_tracker_->record(list_id);
  }
  return {entry.ids, entry.count};
}

size_t PostingListAccessor::nlist() const { return entries_.size(); }

size_t PostingListAccessor::total_postings() const { return total_postings_; }

size_t PostingListAccessor::max_vector_id() const { return max_vector_id_; }

bool PostingListAccessor::enable_hotspot_tracking() {
  if (!hotspot_tracker_) {
    hotspot_tracker_ = std::make_unique<HotspotTracker>(entries_.size());
  } else {
    hotspot_tracker_->reset();
  }
  hotspot_tracker_->set_enabled(true);
  return true;
}

void PostingListAccessor::disable_hotspot_tracking() {
  if (hotspot_tracker_) {
    hotspot_tracker_->set_enabled(false);
  }
}

void PostingListAccessor::reset_hotspot_tracking() {
  if (hotspot_tracker_) {
    hotspot_tracker_->reset();
  }
}

bool PostingListAccessor::hotspot_tracking_enabled() const {
  return hotspot_tracker_ && hotspot_tracker_->is_enabled();
}

std::vector<PostingListAccessor::HotspotEntry>
PostingListAccessor::top_hotspot_lists(size_t limit) const {
  if (!hotspot_tracker_) {
    return {};
  }
  return hotspot_tracker_->top(limit);
}

uint64_t PostingListAccessor::hotspot_total_hits() const {
  if (!hotspot_tracker_) {
    return 0;
  }
  return hotspot_tracker_->total();
}

} // namespace fusionann::online
