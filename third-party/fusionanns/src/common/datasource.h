#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct IDataSource {
  virtual ~IDataSource() = default;
  virtual size_t size() const = 0;
  virtual int dim() const = 0;
  virtual void read_block(size_t start, size_t count, float *out) = 0;
  virtual void read_gather(const int64_t *ids, size_t count, float *out) = 0;
};

struct FvecsSource final : IDataSource {
  explicit FvecsSource(const std::string &path, size_t n_hint = 0,
                       bool use_mmap = false);
  ~FvecsSource() override;

  size_t size() const override { return n_; }
  int dim() const override { return d_; }
  void read_block(size_t start, size_t count, float *out) override;
  void read_gather(const int64_t *ids, size_t count, float *out) override;

private:
  std::string path_;
  size_t n_;
  int d_;
  size_t record_bytes_;
  int fd_ = -1;
  void *mmap_ptr_ = nullptr;
  size_t mmap_size_ = 0;
};

struct U8binSource final : IDataSource {
  explicit U8binSource(const std::string &path, bool use_mmap = false);
  ~U8binSource() override;

  size_t size() const override { return static_cast<size_t>(n_); }
  int dim() const override { return static_cast<int>(d_); }
  void read_block(size_t start, size_t count, float *out) override;
  void read_gather(const int64_t *ids, size_t count, float *out) override;

private:
  std::string path_;
  uint32_t n_;
  uint32_t d_;
  static constexpr size_t kHeaderBytes = sizeof(uint32_t) * 2;
  int fd_ = -1;
  void *mmap_ptr_ = nullptr;
  size_t mmap_size_ = 0;
};

// DiskANN / Quiver fbin: uint32 npts, uint32 dim, then npts*dim float32.
struct FbinSource final : IDataSource {
  explicit FbinSource(const std::string &path, bool use_mmap = false);
  ~FbinSource() override;

  size_t size() const override { return static_cast<size_t>(n_); }
  int dim() const override { return static_cast<int>(d_); }
  void read_block(size_t start, size_t count, float *out) override;
  void read_gather(const int64_t *ids, size_t count, float *out) override;

private:
  std::string path_;
  uint32_t n_;
  uint32_t d_;
  static constexpr size_t kHeaderBytes = sizeof(uint32_t) * 2;
  int fd_ = -1;
  void *mmap_ptr_ = nullptr;
  size_t mmap_size_ = 0;
};

struct BvecsSource final : IDataSource {
  BvecsSource(const std::string &path, size_t n_hint, int d_expect = 128,
              bool use_mmap = false);
  ~BvecsSource() override;

  size_t size() const override { return n_; }
  int dim() const override { return d_; }
  void read_block(size_t start, size_t count, float *out) override;
  void read_gather(const int64_t *ids, size_t count, float *out) override;

private:
  std::string path_;
  size_t n_;
  int d_;
  size_t record_bytes_;
  int fd_ = -1;
  void *mmap_ptr_ = nullptr;
  size_t mmap_size_ = 0;
};

enum class BaseFormat { FVECs, U8BIN, BVECS, FBIN };

std::unique_ptr<IDataSource> make_source(BaseFormat fmt,
                                         const std::string &path,
                                         size_t n_hint = 0, int d_expect = 128,
                                         bool use_mmap = false);

class LRUDataSource final : public IDataSource {
public:
  LRUDataSource(std::unique_ptr<IDataSource> source, size_t cache_size_mb);

  size_t size() const override;
  int dim() const override;
  void read_block(size_t start, size_t count, float *out) override;
  void read_gather(const int64_t *ids, size_t count, float *out) override;

private:
  struct Page {
    std::vector<float> data;
    size_t page_id;
  };

  const float *get_page(size_t page_id);

  std::unique_ptr<IDataSource> source_;
  size_t cache_size_bytes_;
  size_t page_size_vectors_ = 1024; // 1024 vectors per page
  size_t page_size_bytes_;
  size_t max_pages_;

  // LRU implementation
  struct Node {
    size_t page_id;
    std::unique_ptr<Page> page;
    Node *prev = nullptr;
    Node *next = nullptr;
  };

  std::map<size_t, Node *> page_map_;
  Node *head_ = nullptr;
  Node *tail_ = nullptr;
  size_t current_pages_ = 0;

  void move_to_head(Node *node);
  void remove_tail();
};
