#pragma once

#include "spdk/env.h"
#include "spdk/log.h"
#include "spdk/nvme.h"
#include "spdk/nvme_zns.h"
#include "spdk/stdinc.h"
#include "spdk/string.h"
#include "spdk/vmd.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace spdk_wrapper {

class SpdkWrapper {
 public:
  static std::unique_ptr<SpdkWrapper> create(int queue_cnt);
  virtual void Init(const std::vector<std::string> &) = 0;

  virtual int TrySubmitReadCommand(void *pinned_dst, const int64_t bytes,
                                   const int64_t lba, spdk_nvme_cmd_cb func,
                                   void *ctx, int ns_id, int qp_id) = 0;
  virtual void SubmitReadCommand(void *pinned_dst, const int64_t bytes,
                                 const int64_t lba, spdk_nvme_cmd_cb func,
                                 void *ctx, int ns_id, int qp_id) = 0;

  virtual void SubmitWriteCommand(const void *pinned_src, const int64_t bytes,
                                  const int64_t lba, spdk_nvme_cmd_cb func,
                                  void *ctx, int ns_id, int qp_id) = 0;

  virtual void PollCompleteQueue(int ns_id, int qp_id) = 0;
  virtual int GetLBASize() const = 0;
  virtual uint64_t GetLBANumber() const = 0;
  virtual ~SpdkWrapper() {}
};

}  // namespace spdk_wrapper
