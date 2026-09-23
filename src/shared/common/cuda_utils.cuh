#pragma once

#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif

#include <cstddef>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

namespace shared {

inline constexpr int WARP_SIZE = 32;

inline void checkCuda(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream err;
    err << "Cuda Error: " << cudaGetErrorString(code) << " (" << file << ":"
        << line << ")";
    throw std::runtime_error(err.str());
  }
}

template <class T>
inline void copy_to_dev(const T *host_data, T *&dev_data, size_t len) {
  checkCuda(cudaMalloc(&dev_data, sizeof(T) * len), __FILE__, __LINE__);
  checkCuda(cudaMemcpy(dev_data, host_data, sizeof(T) * len,
                       cudaMemcpyHostToDevice),
            __FILE__, __LINE__);
}

}  // namespace shared

#define CHECK_CUDA(code)                                                   \
  { ::shared::checkCuda((code), __FILE__, __LINE__); }
