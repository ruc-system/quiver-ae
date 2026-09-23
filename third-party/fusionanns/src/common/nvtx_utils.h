#pragma once

#include <cstdint>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define FUSIONANNS_NVTX_AVAILABLE 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define FUSIONANNS_NVTX_AVAILABLE 1
#endif
#endif

#ifndef FUSIONANNS_NVTX_AVAILABLE
#define FUSIONANNS_NVTX_AVAILABLE 0
#endif

#if FUSIONANNS_NVTX_AVAILABLE
#include <functional>
#include <thread>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

namespace fusionann::common {

#if FUSIONANNS_NVTX_AVAILABLE
inline std::uint32_t current_os_thread_id() noexcept {
#if defined(__linux__)
  return static_cast<std::uint32_t>(::syscall(SYS_gettid));
#else
  auto tid = std::this_thread::get_id();
  return static_cast<std::uint32_t>(std::hash<std::thread::id>{}(tid));
#endif
}

inline void nvtx_name_current_thread(const char *name) noexcept {
  if (!name)
    return;
  nvtxNameOsThreadA(current_os_thread_id(), name);
}
#else
inline void nvtx_name_current_thread(const char * /*name*/) noexcept {}
#endif

class NvtxScopedRange {
public:
  explicit NvtxScopedRange(const char *name,
                           std::uint32_t argb_color = 0xFF3AA0FF) noexcept {
#if FUSIONANNS_NVTX_AVAILABLE
    if (!name)
      name = "unnamed";
    nvtxEventAttributes_t attrs{};
    attrs.version = NVTX_VERSION;
    attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attrs.message.ascii = name;
    attrs.colorType = NVTX_COLOR_ARGB;
    attrs.color = argb_color;
    nvtxRangePushEx(&attrs);
    active_ = true;
#else
    (void)name;
    (void)argb_color;
#endif
  }

  NvtxScopedRange(const NvtxScopedRange &) = delete;
  NvtxScopedRange &operator=(const NvtxScopedRange &) = delete;

  NvtxScopedRange(NvtxScopedRange &&other) noexcept {
#if FUSIONANNS_NVTX_AVAILABLE
    active_ = other.active_;
    other.active_ = false;
#else
    (void)other;
#endif
  }

  NvtxScopedRange &operator=(NvtxScopedRange &&other) noexcept {
    if (this == &other)
      return *this;
#if FUSIONANNS_NVTX_AVAILABLE
    if (active_) {
      nvtxRangePop();
    }
    active_ = other.active_;
    other.active_ = false;
#else
    (void)other;
#endif
    return *this;
  }

  ~NvtxScopedRange() {
#if FUSIONANNS_NVTX_AVAILABLE
    if (active_) {
      nvtxRangePop();
    }
#endif
  }

private:
#if FUSIONANNS_NVTX_AVAILABLE
  bool active_ = false;
#endif
};

} // namespace fusionann::common

#if FUSIONANNS_NVTX_AVAILABLE
#define FUSIONANNS_NVTX_MARK(label) nvtxMarkA(label)
#else
#define FUSIONANNS_NVTX_MARK(label) ((void)0)
#endif

#define FUSIONANNS_NVTX_TOKEN_CAT(x, y) x##y
#define FUSIONANNS_NVTX_TOKEN(x, y) FUSIONANNS_NVTX_TOKEN_CAT(x, y)

#define FUSIONANNS_NVTX_RANGE(name)                                           \
  fusionann::common::NvtxScopedRange                                          \
  FUSIONANNS_NVTX_TOKEN(_fusionann_nvtx_scope_, __COUNTER__)(name)

#define FUSIONANNS_NVTX_RANGE_COLOR(name, argb)                               \
  fusionann::common::NvtxScopedRange                                          \
  FUSIONANNS_NVTX_TOKEN(_fusionann_nvtx_scope_, __COUNTER__)(name, argb)
