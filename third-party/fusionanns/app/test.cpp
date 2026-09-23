#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Error checking macros
#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " \
                << __LINE__ << std::endl;                                      \
      return -1;                                                               \
    }                                                                          \
  }

#define CHECK_CUBLAS(func)                                                     \
  {                                                                            \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl;           \
      return -1;                                                               \
    }                                                                          \
  }

int main() {
  int query_batch_size = 1000;
  int vector_dim = 128;
  int database_size = 1000000; // 1000万

  // 2. 映射到 GEMM 参数 (C = A * B)
  // 我们希望 C 的形状是 [128, 1000万]，即 [Batch, DB_Size]
  int m = query_batch_size;
  int k = vector_dim; // K 必须是维度，因为要在维度上做点积累加
  int n = database_size; // N 是列数，代表数据库向量的个数

  std::cout << "Benchmarking int8 matrix multiplication on GPU" << std::endl;
  // A [M, K] = [128, 128]
  // B [K, N] = [128, 10000000] (注意这里意味着数据库B需要在内存里转置存放)
  // C [M, N] = [128, 10000000]
  std::cout << "Matrix A (Query): " << m << "x" << k << std::endl;
  std::cout << "Matrix B (Data):  " << k << "x" << n << std::endl;
  std::cout << "Matrix C (Score): " << m << "x" << n << std::endl;

  // Host memory
  std::vector<int8_t> h_A(m * k);
  std::vector<int8_t> h_B(k * n);
  std::vector<int32_t> h_C(m * n);

  // Initialize with some data
  for (int i = 0; i < m * k; ++i)
    h_A[i] = (i % 3) - 1;
  for (int i = 0; i < k * n; ++i)
    h_B[i] = (i % 3) - 1;

  // Device memory
  int8_t *d_A, *d_B;
  int32_t *d_C;

  CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(int32_t)));

  CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(int8_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(int8_t),
                        cudaMemcpyHostToDevice));

  // cuBLAS handle
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // Scaling factors
  const int32_t alpha = 1;
  const int32_t beta = 0;

  // Warmup
  for (int i = 0; i < 10; ++i) {
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              d_B, CUDA_R_8I, n, d_A, CUDA_R_8I, k, &beta, d_C,
                              CUDA_R_32I, n, CUDA_R_32I,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark
  int num_iterations = 10000;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; ++i) {
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              d_B, CUDA_R_8I, n, d_A, CUDA_R_8I, k, &beta, d_C,
                              CUDA_R_32I, n, CUDA_R_32I,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  double avg_time_ms = (diff.count() * 1000.0) / num_iterations;

  // Calculate TOPS (Tera Operations Per Second)
  // 2 * M * N * K operations per GEMM
  double ops = 2.0 * m * n * k;
  double tops = (ops * num_iterations) / diff.count() / 1e12;

  std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
  std::cout << "Performance: " << tops << " TOPS" << std::endl;

  // Cleanup
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));

  return 0;
}