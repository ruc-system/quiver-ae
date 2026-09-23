#include "../src/shared/io/spdk/spdk_wrapper.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int64_t kBlockSize = 4096;

struct ReadCtx {
  bool done = false;
  bool error = false;
};

uint64_t fnv1a(const unsigned char* data, size_t size) {
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

bool is_zero_block(const unsigned char* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (data[i] != 0) {
      return false;
    }
  }
  return true;
}

void print_sample(const unsigned char* data, size_t size) {
  size_t first_nonzero = size;
  for (size_t i = 0; i < size; ++i) {
    if (data[i] != 0) {
      first_nonzero = i;
      break;
    }
  }

  if (first_nonzero == size) {
    std::printf(" first_nonzero=none sample=all_zero");
    return;
  }

  std::printf(" first_nonzero=%zu sample=", first_nonzero);
  size_t end = std::min(first_nonzero + 16, size);
  for (size_t i = first_nonzero; i < end; ++i) {
    std::printf("%02x", data[i]);
  }
}

void print_signatures(const unsigned char* data, size_t size) {
  bool printed = false;
  auto print_sig = [&](const char* sig) {
    if (!printed) {
      std::printf(" signatures=");
      printed = true;
    } else {
      std::printf(",");
    }
    std::printf("%s", sig);
  };

  if (size >= 512 && data[510] == 0x55 && data[511] == 0xaa) {
    print_sig("MBR");
  }
  if (size >= 520 && std::equal(data + 512, data + 520,
                                reinterpret_cast<const unsigned char*>("EFI PART"))) {
    print_sig("GPT");
  }
  if (size >= 1082 && data[1080] == 0x53 && data[1081] == 0xef) {
    print_sig("ext4");
  }
  if (!printed) {
    std::printf(" signatures=none");
  }
}

std::vector<std::string> load_ssds(const char* path) {
  std::ifstream file_stream;
  std::istream* stream = &std::cin;
  if (std::string(path) != "-") {
    file_stream.open(path);
    if (!file_stream) {
      std::fprintf(stderr, "SSD list file not found: %s\n", path);
      std::exit(1);
    }
    stream = &file_stream;
  }

  std::vector<std::string> ssds;
  std::string s;
  while (*stream >> s) {
    ssds.push_back(s);
  }
  if (ssds.empty()) {
    std::fprintf(stderr, "SSD list file is empty: %s\n", path);
    std::exit(1);
  }
  return ssds;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::printf("%s <ssd_list> [blocks_to_probe] [start_lba]\n", argv[0]);
    return 1;
  }

  int blocks_to_probe = 64;
  if (argc >= 3) {
    blocks_to_probe = std::atoi(argv[2]);
    if (blocks_to_probe <= 0) {
      std::fprintf(stderr, "blocks_to_probe must be positive\n");
      return 1;
    }
  }
  int64_t start_lba = 0;
  if (argc == 4) {
    start_lba = std::atoll(argv[3]);
    if (start_lba < 0) {
      std::fprintf(stderr, "start_lba must be non-negative\n");
      return 1;
    }
  }

  auto ssds = load_ssds(argv[1]);
  std::unique_ptr<spdk_wrapper::SpdkWrapper> spdk =
      spdk_wrapper::SpdkWrapper::create(1);
  spdk->Init(ssds);

  unsigned char* buf = static_cast<unsigned char*>(
      spdk_dma_zmalloc(kBlockSize, kBlockSize, nullptr));
  if (buf == nullptr) {
    std::fprintf(stderr, "spdk_dma_zmalloc failed\n");
    return 1;
  }

  for (size_t ns_id = 0; ns_id < ssds.size(); ++ns_id) {
    int nonzero_blocks = 0;
    std::printf("SSD %zu %s\n", ns_id, ssds[ns_id].c_str());
    for (int64_t offset = 0; offset < blocks_to_probe; ++offset) {
      int64_t lba = start_lba + offset;
      std::fill(buf, buf + kBlockSize, 0);
      ReadCtx ctx;
      spdk->SubmitReadCommand(
          buf, kBlockSize, lba,
          [](void* raw_ctx, const struct spdk_nvme_cpl* cpl) {
            auto* ctx = static_cast<ReadCtx*>(raw_ctx);
            ctx->error = spdk_nvme_cpl_is_error(cpl);
            ctx->done = true;
          },
          &ctx, static_cast<int>(ns_id), 0);

      while (!ctx.done) {
        spdk->PollCompleteQueue(static_cast<int>(ns_id), 0);
      }

      bool zero = is_zero_block(buf, kBlockSize);
      if (!zero) {
        ++nonzero_blocks;
      }

      std::printf("  lba=%ld zero=%s hash=%016lx", static_cast<long>(lba),
                  zero ? "yes" : "no", fnv1a(buf, kBlockSize));
      print_sample(buf, kBlockSize);
      print_signatures(buf, kBlockSize);
      if (ctx.error) {
        std::printf(" read_error=yes");
      }
      std::printf("\n");
    }
    std::printf("SUMMARY %s nonzero_blocks=%d/%d\n", ssds[ns_id].c_str(),
                nonzero_blocks, blocks_to_probe);
  }

  spdk_dma_free(buf);
  return 0;
}
