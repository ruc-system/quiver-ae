#include "common/datasource.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <map>
FvecsSource::FvecsSource(const std::string &path, size_t n_hint, bool use_mmap)
    : path_(path) {
  if (use_mmap) {
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
      throw std::runtime_error("failed to open fvecs file for mmap: " + path_);
    }
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
      close(fd_);
      throw std::runtime_error("failed to stat fvecs file: " + path_);
    }
    mmap_size_ = sb.st_size;
    if (mmap_size_ < sizeof(int)) {
      close(fd_);
      throw std::runtime_error("fvecs file too small: " + path_);
    }
    mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("failed to mmap fvecs file: " + path_);
    }
    madvise(mmap_ptr_, mmap_size_, MADV_RANDOM);

    int d = 0;
    std::memcpy(&d, mmap_ptr_, sizeof(int));
    if (d <= 0 || d > 4096) {
      munmap(mmap_ptr_, mmap_size_);
      close(fd_);
      throw std::runtime_error("invalid fvecs dimension");
    }
    d_ = d;
    record_bytes_ = sizeof(int) + static_cast<size_t>(d_) * sizeof(float);
    size_t guess = mmap_size_ / record_bytes_;
    n_ = n_hint > 0 ? std::min(n_hint, guess) : guess;
  } else {
    std::ifstream f(path_, std::ios::binary | std::ios::ate);
    if (!f) {
      throw std::runtime_error("failed to open fvecs file: " + path_);
    }
    if (f.tellg() < static_cast<std::streampos>(sizeof(int))) {
      throw std::runtime_error("fvecs file too small: " + path_);
    }
    f.seekg(0);
    int d = 0;
    f.read(reinterpret_cast<char *>(&d), sizeof(int));
    if (d <= 0 || d > 4096) {
      throw std::runtime_error("invalid fvecs dimension");
    }
    d_ = d;
    record_bytes_ = sizeof(int) + static_cast<size_t>(d_) * sizeof(float);

    f.seekg(0, std::ios::end);
    size_t bytes = static_cast<size_t>(f.tellg());
    size_t guess = bytes / record_bytes_;
    n_ = n_hint > 0 ? std::min(n_hint, guess) : guess;
  }
}

FvecsSource::~FvecsSource() {
  if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
    munmap(mmap_ptr_, mmap_size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

void FvecsSource::read_block(size_t start, size_t count, float *out) {
  if (mmap_ptr_) {
    const char *base = static_cast<const char *>(mmap_ptr_);
    const size_t rec_bytes = record_bytes_;
    const int dim = d_;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      size_t offset = (start + i) * rec_bytes;
      // int d = 0;
      // std::memcpy(&d, base + offset, sizeof(int));
      // if (d != dim) {
      //   throw std::runtime_error("fvecs dimension mismatch");
      // }
      std::memcpy(out + i * dim, base + offset + sizeof(int),
                  sizeof(float) * dim);
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open fvecs file: " + path_);
    }
    f.seekg(static_cast<std::streamoff>(start * record_bytes_), std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
      int d = 0;
      f.read(reinterpret_cast<char *>(&d), sizeof(int));
      if (d != d_) {
        throw std::runtime_error("fvecs dimension mismatch");
      }
      f.read(reinterpret_cast<char *>(out + i * d_), sizeof(float) * d_);
    }
  }
}

void FvecsSource::read_gather(const int64_t *ids, size_t count, float *out) {
  if (mmap_ptr_) {
    const char *base = static_cast<const char *>(mmap_ptr_);
    const size_t rec_bytes = record_bytes_;
    const int dim = d_;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      size_t offset = static_cast<size_t>(ids[i]) * rec_bytes;
      // int d = 0;
      // std::memcpy(&d, base + offset, sizeof(int));
      // if (d != dim) {
      //   throw std::runtime_error("fvecs dimension mismatch");
      // }
      std::memcpy(out + i * dim, base + offset + sizeof(int),
                  sizeof(float) * dim);
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open fvecs file: " + path_);
    }
    for (size_t i = 0; i < count; ++i) {
      size_t offset = static_cast<size_t>(ids[i]) * record_bytes_;
      f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
      int d = 0;
      f.read(reinterpret_cast<char *>(&d), sizeof(int));
      if (d != d_) {
        throw std::runtime_error("fvecs dimension mismatch");
      }
      f.read(reinterpret_cast<char *>(out + i * d_), sizeof(float) * d_);
    }
  }
}

U8binSource::U8binSource(const std::string &path, bool use_mmap)
    : path_(path), n_(0), d_(0) {
  if (use_mmap) {
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
      throw std::runtime_error("failed to open u8bin file for mmap: " + path_);
    }
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
      close(fd_);
      throw std::runtime_error("failed to stat u8bin file: " + path_);
    }
    mmap_size_ = sb.st_size;
    if (mmap_size_ < kHeaderBytes) {
      close(fd_);
      throw std::runtime_error("u8bin file too small: " + path_);
    }
    mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("failed to mmap u8bin file: " + path_);
    }
    madvise(mmap_ptr_, mmap_size_, MADV_RANDOM);

    const char *base = static_cast<const char *>(mmap_ptr_);
    std::memcpy(&n_, base, sizeof(uint32_t));
    std::memcpy(&d_, base + sizeof(uint32_t), sizeof(uint32_t));

    if (d_ == 0 || d_ > 4096) {
      munmap(mmap_ptr_, mmap_size_);
      close(fd_);
      throw std::runtime_error("invalid u8bin dimension");
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open u8bin file: " + path_);
    }
    f.read(reinterpret_cast<char *>(&n_), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(&d_), sizeof(uint32_t));
    if (d_ == 0 || d_ > 4096) {
      throw std::runtime_error("invalid u8bin dimension");
    }
  }
}

U8binSource::~U8binSource() {
  if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
    munmap(mmap_ptr_, mmap_size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

void U8binSource::read_block(size_t start, size_t count, float *out) {
  size_t stride = static_cast<size_t>(d_);
  if (mmap_ptr_) {
    const uint8_t *base = static_cast<const uint8_t *>(mmap_ptr_);
    const size_t header = kHeaderBytes;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      float *dst = out + i * stride;
      const uint8_t *src = base + header + (start + i) * stride;
      for (size_t j = 0; j < stride; ++j) {
        dst[j] = static_cast<float>(src[j]);
      }
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open u8bin file: " + path_);
    }
    size_t offset = kHeaderBytes + start * stride;
    f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::vector<uint8_t> buffer(stride);
    for (size_t i = 0; i < count; ++i) {
      f.read(reinterpret_cast<char *>(buffer.data()), stride);
      float *dst = out + i * stride;
      for (size_t j = 0; j < stride; ++j) {
        dst[j] = static_cast<float>(buffer[j]);
      }
    }
  }
}

void U8binSource::read_gather(const int64_t *ids, size_t count, float *out) {
  size_t stride = static_cast<size_t>(d_);
  if (mmap_ptr_) {
    const uint8_t *base = static_cast<const uint8_t *>(mmap_ptr_);
    const size_t header = kHeaderBytes;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      size_t offset = header + static_cast<size_t>(ids[i]) * stride;
      const uint8_t *src = base + offset;
      float *dst = out + i * stride;
      for (size_t j = 0; j < stride; ++j) {
        dst[j] = static_cast<float>(src[j]);
      }
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open u8bin file: " + path_);
    }
    std::vector<uint8_t> buffer(stride);
    for (size_t i = 0; i < count; ++i) {
      size_t offset = kHeaderBytes + static_cast<size_t>(ids[i]) * stride;
      f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
      f.read(reinterpret_cast<char *>(buffer.data()), stride);
      float *dst = out + i * stride;
      for (size_t j = 0; j < stride; ++j) {
        dst[j] = static_cast<float>(buffer[j]);
      }
    }
  }
}

FbinSource::FbinSource(const std::string &path, bool use_mmap)
    : path_(path), n_(0), d_(0) {
  if (use_mmap) {
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
      throw std::runtime_error("failed to open fbin file for mmap: " + path_);
    }
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
      close(fd_);
      throw std::runtime_error("failed to stat fbin file: " + path_);
    }
    mmap_size_ = sb.st_size;
    if (mmap_size_ < kHeaderBytes) {
      close(fd_);
      throw std::runtime_error("fbin file too small: " + path_);
    }
    mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("failed to mmap fbin file: " + path_);
    }
    madvise(mmap_ptr_, mmap_size_, MADV_RANDOM);

    const char *base = static_cast<const char *>(mmap_ptr_);
    std::memcpy(&n_, base, sizeof(uint32_t));
    std::memcpy(&d_, base + sizeof(uint32_t), sizeof(uint32_t));

    if (d_ == 0 || d_ > 4096) {
      munmap(mmap_ptr_, mmap_size_);
      close(fd_);
      throw std::runtime_error("invalid fbin dimension");
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open fbin file: " + path_);
    }
    f.read(reinterpret_cast<char *>(&n_), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(&d_), sizeof(uint32_t));
    if (d_ == 0 || d_ > 4096) {
      throw std::runtime_error("invalid fbin dimension");
    }
  }
}

FbinSource::~FbinSource() {
  if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
    munmap(mmap_ptr_, mmap_size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

void FbinSource::read_block(size_t start, size_t count, float *out) {
  const size_t stride = static_cast<size_t>(d_);
  const size_t vec_bytes = stride * sizeof(float);
  if (mmap_ptr_) {
    const char *base = static_cast<const char *>(mmap_ptr_);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      const char *src =
          base + kHeaderBytes + (start + i) * vec_bytes;
      std::memcpy(out + i * stride, src, vec_bytes);
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open fbin file: " + path_);
    }
    f.seekg(static_cast<std::streamoff>(kHeaderBytes + start * vec_bytes),
            std::ios::beg);
    f.read(reinterpret_cast<char *>(out),
           static_cast<std::streamsize>(count * vec_bytes));
  }
}

void FbinSource::read_gather(const int64_t *ids, size_t count, float *out) {
  const size_t stride = static_cast<size_t>(d_);
  const size_t vec_bytes = stride * sizeof(float);
  if (mmap_ptr_) {
    const char *base = static_cast<const char *>(mmap_ptr_);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      const char *src =
          base + kHeaderBytes + static_cast<size_t>(ids[i]) * vec_bytes;
      std::memcpy(out + i * stride, src, vec_bytes);
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open fbin file: " + path_);
    }
    for (size_t i = 0; i < count; ++i) {
      const size_t offset =
          kHeaderBytes + static_cast<size_t>(ids[i]) * vec_bytes;
      f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
      f.read(reinterpret_cast<char *>(out + i * stride),
             static_cast<std::streamsize>(vec_bytes));
    }
  }
}

BvecsSource::BvecsSource(const std::string &path, size_t n_hint, int d_expect,
                         bool use_mmap)
    : path_(path), n_(n_hint), d_(d_expect) {
  record_bytes_ = sizeof(int) + static_cast<size_t>(d_);
  if (d_ <= 0 || d_ > 4096) {
    throw std::runtime_error("invalid bvecs dimension expectation");
  }

  if (use_mmap) {
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
      throw std::runtime_error("failed to open bvecs file for mmap: " + path_);
    }
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
      close(fd_);
      throw std::runtime_error("failed to stat bvecs file: " + path_);
    }
    mmap_size_ = sb.st_size;
    if (mmap_size_ == 0) {
      close(fd_);
      throw std::runtime_error("empty bvecs file: " + path_);
    }
    mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("failed to mmap bvecs file: " + path_);
    }
    madvise(mmap_ptr_, mmap_size_, MADV_RANDOM);
    if (n_ == 0) {
      n_ = mmap_size_ / record_bytes_;
    }
  } else {
    if (n_ == 0) {
      std::ifstream f(path_, std::ios::binary | std::ios::ate);
      if (!f) {
        throw std::runtime_error("failed to open bvecs file: " + path_);
      }
      f.seekg(0, std::ios::end);
      size_t bytes = static_cast<size_t>(f.tellg());
      if (bytes == 0) {
        throw std::runtime_error("empty bvecs file: " + path_);
      }
      n_ = bytes / record_bytes_;
    }
  }
}

BvecsSource::~BvecsSource() {
  if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
    munmap(mmap_ptr_, mmap_size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

void BvecsSource::read_block(size_t start, size_t count, float *out) {
  if (mmap_ptr_) {
    const uint8_t *base = static_cast<const uint8_t *>(mmap_ptr_);
    const size_t rec_bytes = record_bytes_;
    const int dim = d_;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      size_t offset = (start + i) * rec_bytes;
      // int d_on_disk = 0;
      // std::memcpy(&d_on_disk, base + offset, sizeof(int));
      // if (d_on_disk != dim) {
      //   throw std::runtime_error("bvecs dimension mismatch");
      // }
      const uint8_t *src = base + offset + sizeof(int);
      float *dst = out + i * dim;
      for (int j = 0; j < dim; ++j) {
        dst[j] = static_cast<float>(src[j]);
      }
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open bvecs file: " + path_);
    }
    f.seekg(static_cast<std::streamoff>(start * record_bytes_), std::ios::beg);
    std::vector<uint8_t> buffer(d_);
    for (size_t i = 0; i < count; ++i) {
      int d_on_disk = 0;
      f.read(reinterpret_cast<char *>(&d_on_disk), sizeof(int));
      if (d_on_disk != d_) {
        throw std::runtime_error("bvecs dimension mismatch");
      }
      f.read(reinterpret_cast<char *>(buffer.data()), d_);
      float *dst = out + i * d_;
      for (int j = 0; j < d_; ++j) {
        dst[j] = static_cast<float>(buffer[j]);
      }
    }
  }
}

void BvecsSource::read_gather(const int64_t *ids, size_t count, float *out) {
  if (mmap_ptr_) {
    const uint8_t *base = static_cast<const uint8_t *>(mmap_ptr_);
    const size_t rec_bytes = record_bytes_;
    const int dim = d_;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      size_t offset = static_cast<size_t>(ids[i]) * rec_bytes;
      // int d_on_disk = 0;
      // std::memcpy(&d_on_disk, base + offset, sizeof(int));
      // if (d_on_disk != dim) {
      //   throw std::runtime_error("bvecs dimension mismatch");
      // }
      const uint8_t *src = base + offset + sizeof(int);
      float *dst = out + i * dim;
      for (int j = 0; j < dim; ++j) {
        dst[j] = static_cast<float>(src[j]);
      }
    }
  } else {
    std::ifstream f(path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("failed to open bvecs file: " + path_);
    }
    std::vector<uint8_t> buffer(d_);
    for (size_t i = 0; i < count; ++i) {
      size_t offset = static_cast<size_t>(ids[i]) * record_bytes_;
      f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
      int d_on_disk = 0;
      f.read(reinterpret_cast<char *>(&d_on_disk), sizeof(int));
      if (d_on_disk != d_) {
        throw std::runtime_error("bvecs dimension mismatch");
      }
      f.read(reinterpret_cast<char *>(buffer.data()), d_);
      float *dst = out + i * d_;
      for (int j = 0; j < d_; ++j) {
        dst[j] = static_cast<float>(buffer[j]);
      }
    }
  }
}

std::unique_ptr<IDataSource> make_source(BaseFormat fmt,
                                         const std::string &path, size_t n_hint,
                                         int d_expect, bool use_mmap) {
  switch (fmt) {
  case BaseFormat::FVECs:
    return std::make_unique<FvecsSource>(path, n_hint, use_mmap);
  case BaseFormat::U8BIN:
    return std::make_unique<U8binSource>(path, use_mmap);
  case BaseFormat::BVECS:
    return std::make_unique<BvecsSource>(path, n_hint, d_expect, use_mmap);
  case BaseFormat::FBIN:
    return std::make_unique<FbinSource>(path, use_mmap);
  }
  throw std::runtime_error("unknown base format");
}

LRUDataSource::LRUDataSource(std::unique_ptr<IDataSource> source,
                             size_t cache_size_mb)
    : source_(std::move(source)) {
  if (!source_) {
    throw std::invalid_argument("source cannot be null");
  }
  cache_size_bytes_ = cache_size_mb * 1024 * 1024;
  size_t dim_bytes = static_cast<size_t>(source_->dim()) * sizeof(float);
  page_size_bytes_ = page_size_vectors_ * dim_bytes;
  max_pages_ = std::max<size_t>(1, cache_size_bytes_ / page_size_bytes_);
}

size_t LRUDataSource::size() const { return source_->size(); }

int LRUDataSource::dim() const { return source_->dim(); }

const float *LRUDataSource::get_page(size_t page_id) {
  auto it = page_map_.find(page_id);
  if (it != page_map_.end()) {
    move_to_head(it->second);
    return it->second->page->data.data();
  }

  if (current_pages_ >= max_pages_) {
    remove_tail();
  }

  auto page = std::make_unique<Page>();
  page->page_id = page_id;
  page->data.resize(page_size_vectors_ * static_cast<size_t>(source_->dim()));

  size_t start = page_id * page_size_vectors_;
  size_t count = std::min(page_size_vectors_, source_->size() - start);
  source_->read_block(start, count, page->data.data());

  auto node = new Node();
  node->page_id = page_id;
  node->page = std::move(page);
  node->next = head_;
  if (head_) {
    head_->prev = node;
  }
  head_ = node;
  if (!tail_) {
    tail_ = node;
  }
  page_map_[page_id] = node;
  current_pages_++;

  return node->page->data.data();
}

void LRUDataSource::move_to_head(Node *node) {
  if (node == head_) {
    return;
  }
  if (node == tail_) {
    tail_ = node->prev;
    tail_->next = nullptr;
  } else {
    node->prev->next = node->next;
    node->next->prev = node->prev;
  }
  node->next = head_;
  node->prev = nullptr;
  head_->prev = node;
  head_ = node;
}

void LRUDataSource::remove_tail() {
  if (!tail_) {
    return;
  }
  Node *node = tail_;
  page_map_.erase(node->page_id);
  if (tail_ == head_) {
    head_ = nullptr;
    tail_ = nullptr;
  } else {
    tail_ = tail_->prev;
    tail_->next = nullptr;
  }
  delete node;
  current_pages_--;
}

void LRUDataSource::read_block(size_t start, size_t count, float *out) {
  size_t dim_stride = static_cast<size_t>(source_->dim());
  size_t processed = 0;
  while (processed < count) {
    size_t current_idx = start + processed;
    size_t page_id = current_idx / page_size_vectors_;
    size_t offset_in_page = current_idx % page_size_vectors_;
    size_t vectors_in_page =
        std::min(count - processed, page_size_vectors_ - offset_in_page);

    const float *page_data = get_page(page_id);
    std::memcpy(out + processed * dim_stride,
                page_data + offset_in_page * dim_stride,
                vectors_in_page * dim_stride * sizeof(float));
    processed += vectors_in_page;
  }
}

void LRUDataSource::read_gather(const int64_t *ids, size_t count, float *out) {
  size_t dim_stride = static_cast<size_t>(source_->dim());
  for (size_t i = 0; i < count; ++i) {
    size_t idx = static_cast<size_t>(ids[i]);
    size_t page_id = idx / page_size_vectors_;
    size_t offset_in_page = idx % page_size_vectors_;

    const float *page_data = get_page(page_id);
    std::memcpy(out + i * dim_stride, page_data + offset_in_page * dim_stride,
                dim_stride * sizeof(float));
  }
}
