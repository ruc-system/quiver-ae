#include "../src/shared/io/spdk/spdk_wrapper.h"
#include "../src/shared/io/spdk_lba_offset.hpp"
#include "../src/shared/io/spdk_write_buffer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

constexpr long PG_SIZE = 4096;

namespace {

int get_env_int(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || value <= 0 || value > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "Invalid %s=%s, using %d\n", name, raw,
                 default_value);
    return default_value;
  }
  return static_cast<int>(value);
}

struct WriteContext {
  long* total_in_flight = nullptr;
  std::vector<long>* per_disk_in_flight = nullptr;
  std::vector<long>* per_disk_completed = nullptr;
  std::vector<long>* per_disk_errors = nullptr;
  int ns_id = 0;
  long block_id = 0;
  long lba = 0;
};

void print_in_flight_report(long cnt, long total_in_flight,
                            const std::vector<long>& per_disk_in_flight,
                            const std::vector<long>& per_disk_completed,
                            const std::vector<long>& per_disk_errors) {
  std::fprintf(stderr, "[spdk_write] waiting at block=%ld total_in_flight=%ld\n",
               cnt, total_in_flight);
  for (size_t i = 0; i < per_disk_in_flight.size(); ++i) {
    std::fprintf(stderr,
                 "  disk=%zu in_flight=%ld completed=%ld errors=%ld\n", i,
                 per_disk_in_flight[i], per_disk_completed[i],
                 per_disk_errors[i]);
  }
  std::fflush(stderr);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("%s: <input> <ssd_list>\n", argv[0]);
    return -1;
  }

  std::unique_ptr<spdk_wrapper::SpdkWrapper> spdk;
  spdk = spdk_wrapper::SpdkWrapper::create(32);

  std::vector<std::string> ssds;
  std::fstream stream(argv[2]);
  printf("%s\n", argv[2]);

  if (stream) {
    std::string s;
    while (stream >> s) {
      printf("USE SSD: \"%s\"\n", s.c_str());
      ssds.push_back(s);
    }
  } else {
    printf("SSD File not found!");
    exit(-1);
  }

  spdk->Init(ssds);

  FILE* f0 = fopen(argv[1], "rb");
  if (f0 == nullptr) {
    std::perror(argv[1]);
    return -1;
  }
  fseek(f0, PG_SIZE, SEEK_SET);

  const int batch_size = get_env_int("SPDK_WRITE_BATCH_SIZE", 128);
  const int timeout_ms = get_env_int("SPDK_WRITE_WAIT_TIMEOUT_MS", 10000);
  const std::size_t read_buffer_bytes =
      shared::spdk_write_read_buffer_bytes_from_env(
          "SPDK_WRITE_READ_BUFFER_MB", 64, PG_SIZE);
  char* buff = (char*)spdk_dma_zmalloc(batch_size * PG_SIZE, PG_SIZE, NULL);
  if (buff == nullptr) {
    std::fprintf(stderr, "spdk_dma_zmalloc failed\n");
    return -1;
  }
  std::vector<char> read_buffer(read_buffer_bytes);

  int shards = ssds.size();
  if (shards <= 0) {
    std::fprintf(stderr, "SSD list is empty\n");
    return -1;
  }
  long in_flight = 0;
  std::vector<long> per_disk_in_flight(shards, 0);
  std::vector<long> per_disk_completed(shards, 0);
  std::vector<long> per_disk_errors(shards, 0);
  std::vector<WriteContext> ctxs(batch_size);
  long cnt = 0;
  const long base_lba = shared::spdk_base_lba_from_env();
  printf("SPDK_BASE_LBA: %ld\n", base_lba);
  printf("SPDK_WRITE_BATCH_SIZE: %d\n", batch_size);
  printf("SPDK_WRITE_WAIT_TIMEOUT_MS: %d\n", timeout_ms);
  printf("SPDK_WRITE_READ_BUFFER_BYTES: %zu\n", read_buffer_bytes);
  fflush(stdout);

  while (true) {
    const std::size_t bytes_read =
        fread(read_buffer.data(), 1, read_buffer.size(), f0);
    if (bytes_read < static_cast<std::size_t>(PG_SIZE)) break;

    const std::size_t full_pages = bytes_read / PG_SIZE;
    for (std::size_t page = 0; page < full_pages; ++page) {
      int block = cnt % batch_size;
      std::memcpy(buff + block * PG_SIZE, read_buffer.data() + page * PG_SIZE,
                  PG_SIZE);
      int ns_id = cnt % shards;
      long lba = base_lba + cnt / shards;
      auto& ctx = ctxs[block];
      ctx.total_in_flight = &in_flight;
      ctx.per_disk_in_flight = &per_disk_in_flight;
      ctx.per_disk_completed = &per_disk_completed;
      ctx.per_disk_errors = &per_disk_errors;
      ctx.ns_id = ns_id;
      ctx.block_id = cnt;
      ctx.lba = lba;
      spdk->SubmitWriteCommand(
          buff + block * PG_SIZE, PG_SIZE, lba,
          [](void* ctx, const struct spdk_nvme_cpl* cpl) {
            auto* write_ctx = static_cast<WriteContext*>(ctx);
            if (spdk_nvme_cpl_is_error(cpl)) {
              std::fprintf(stderr,
                           "[spdk_write] completion error disk=%d block=%ld "
                           "lba=%ld sct=%u sc=%u\n",
                           write_ctx->ns_id, write_ctx->block_id,
                           write_ctx->lba, cpl->status.sct, cpl->status.sc);
              (*write_ctx->per_disk_errors)[write_ctx->ns_id]++;
            }
            (*write_ctx->total_in_flight)--;
            (*write_ctx->per_disk_in_flight)[write_ctx->ns_id]--;
            (*write_ctx->per_disk_completed)[write_ctx->ns_id]++;
          },
          &ctx, ns_id, 0);
      in_flight++;
      per_disk_in_flight[ns_id]++;
      if (++cnt % batch_size == 0) {
        auto wait_start = std::chrono::steady_clock::now();
        auto last_report = wait_start;
        while (in_flight > 0) {
          for (int i = 0; i < shards; i++) {
            spdk->PollCompleteQueue(i, 0);
          }
          auto now = std::chrono::steady_clock::now();
          auto elapsed_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  now - wait_start)
                  .count();
          auto since_report_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  now - last_report)
                  .count();
          if (since_report_ms >= timeout_ms) {
            print_in_flight_report(cnt, in_flight, per_disk_in_flight,
                                   per_disk_completed, per_disk_errors);
            last_report = now;
          }
          if (elapsed_ms >= timeout_ms * 6L) {
            std::fprintf(stderr,
                         "[spdk_write] timeout waiting for completions; "
                         "aborting to avoid an infinite poll loop\n");
            print_in_flight_report(cnt, in_flight, per_disk_in_flight,
                                   per_disk_completed, per_disk_errors);
            return -2;
          }
        }
      }
      if (cnt % 1000000 == 0) {
        printf("%ld Blocks\n", cnt);
        fflush(stdout);
      }
    }
    if (bytes_read < read_buffer.size()) break;
  }
  auto final_wait_start = std::chrono::steady_clock::now();
  auto final_last_report = final_wait_start;
  while (in_flight > 0) {
    for (int i = 0; i < shards; i++) {
      spdk->PollCompleteQueue(i, 0);
    }
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - final_wait_start)
            .count();
    auto since_report_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - final_last_report)
            .count();
    if (since_report_ms >= timeout_ms) {
      print_in_flight_report(cnt, in_flight, per_disk_in_flight,
                             per_disk_completed, per_disk_errors);
      final_last_report = now;
    }
    if (elapsed_ms >= timeout_ms * 6L) {
      std::fprintf(stderr,
                   "[spdk_write] timeout waiting for final completions; "
                   "aborting\n");
      print_in_flight_report(cnt, in_flight, per_disk_in_flight,
                             per_disk_completed, per_disk_errors);
      return -2;
    }
  }
  printf("%ld Block in Total!\n", cnt);
}
