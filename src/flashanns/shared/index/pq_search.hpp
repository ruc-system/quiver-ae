#pragma once

#include <cstdint>
#include <string>

namespace shared {

struct PQData {
  int idx;
  float distance;
};

struct PQSearchData {
  static const int num_pivots = 256;
  int dim = 0;
  int num_chunks = 0;
  int num_pts = 0;
  float *centroid = nullptr;        // 1 * dim
  float *pivots = nullptr;          // num_pivots * dim
  float *pivots_t = nullptr;        // dim * num_pivots
  int *chunk_id = nullptr;          // dim
  uint8_t *compressed_data = nullptr;  // num_pts * num_chunks
  float *pq_dists = nullptr;        // num_blocks * num_pivots * num_chunks
  PQData *pq_retset = nullptr;

  __inline__ __device__ void init_query(float *q, int offset = 0);
  __inline__ __device__ float compute_dist(int idx, int offset = 0);
  __inline__ __device__ void compute_dist(int *idx, float *result, int cnt);
};

class PQSearch {
 public:
  PQSearch() = default;
  ~PQSearch();
  PQSearch(const PQSearch &) = delete;
  PQSearch &operator=(const PQSearch &) = delete;
  PQSearch(PQSearch &&) = delete;
  PQSearch &operator=(PQSearch &&) = delete;

  void read_data(std::string table_file, std::string vec_file);
  void init_device(int dim, int num_pts, int num_thread_blocks, int ef_search);
  inline PQSearchData *get_device_ptr() { return device_ptr; }
  const PQSearchData &get_data() const { return host_data; }
  PQSearchData device_data{};

 private:
  void cleanup_host() noexcept;
  void cleanup_device() noexcept;

  PQSearchData host_data{};
  PQSearchData *device_ptr = nullptr;
};

}  // namespace shared
