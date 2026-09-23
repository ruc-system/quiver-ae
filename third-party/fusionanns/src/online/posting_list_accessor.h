#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fusionann::online {

class PostingListAccessor {
public:
  struct View {
    const int32_t *data = nullptr;
    size_t size = 0;
  };

  struct HotspotEntry {
    int list_id = -1;
    uint64_t hits = 0;
  };

  explicit PostingListAccessor(const std::string &path);
  ~PostingListAccessor();

  PostingListAccessor(const PostingListAccessor &) = delete;
  PostingListAccessor &operator=(const PostingListAccessor &) = delete;

  View view(int list_id) const;
  size_t nlist() const;
  size_t total_postings() const;
  size_t max_vector_id() const;
  bool enable_hotspot_tracking();
  void disable_hotspot_tracking();
  void reset_hotspot_tracking();
  bool hotspot_tracking_enabled() const;
  std::vector<HotspotEntry> top_hotspot_lists(size_t limit) const;
  uint64_t hotspot_total_hits() const;

private:
  struct Entry {
    const int32_t *ids = nullptr;
    size_t count = 0;
  };

  struct HotspotTracker;

  int fd_{-1};
  const char *base_ = nullptr;
  size_t mapped_size_ = 0;
  std::vector<Entry> entries_;
  size_t total_postings_ = 0;
  size_t max_vector_id_ = 0;
  mutable std::unique_ptr<HotspotTracker> hotspot_tracker_;
};

} // namespace fusionann::online
