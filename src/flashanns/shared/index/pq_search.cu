#include "pq_search.hpp"

#include <cstring>
#include <fstream>

#include "../common/cuda_utils.cuh"
#include "../common/logging.hpp"

namespace shared {

template <class T>
static void read_bin(std::string file, int &npts, int &ndim, size_t offset,
                     T *&data);

namespace {

template <class T>
void delete_array_noexcept(T *&ptr) noexcept {
  delete[] ptr;
  ptr = nullptr;
}

template <class T>
void cuda_free_noexcept(T *&ptr) noexcept {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

}  // namespace

PQSearch::~PQSearch() {
  cleanup_device();
  cleanup_host();
}

void PQSearch::cleanup_host() noexcept {
  delete_array_noexcept(host_data.centroid);
  delete_array_noexcept(host_data.pivots);
  delete_array_noexcept(host_data.pivots_t);
  delete_array_noexcept(host_data.chunk_id);
  delete_array_noexcept(host_data.compressed_data);
  delete_array_noexcept(host_data.pq_retset);
  host_data.dim = 0;
  host_data.num_chunks = 0;
  host_data.num_pts = 0;
}

void PQSearch::cleanup_device() noexcept {
  cuda_free_noexcept(device_data.centroid);
  cuda_free_noexcept(device_data.pivots);
  cuda_free_noexcept(device_data.pivots_t);
  cuda_free_noexcept(device_data.chunk_id);
  cuda_free_noexcept(device_data.compressed_data);
  cuda_free_noexcept(device_data.pq_dists);
  cuda_free_noexcept(device_data.pq_retset);
  cuda_free_noexcept(device_ptr);
  device_data = {};
}

void PQSearch::read_data(std::string table_file, std::string vec_file) {
  cleanup_device();
  cleanup_host();

  DEBUG("Reading Compressed data");
  read_bin(vec_file, host_data.num_pts, host_data.num_chunks, 0,
           host_data.compressed_data);

  size_t *basic_offsets;
  int nr, nc;
  DEBUG("Reading metadata");
  read_bin(table_file, nr, nc, 0, basic_offsets);
  ASSERT((nr == 4 || nr == 5) && nc == 1);
  if (nr == 4) {
    DEBUG("Metadata: {} {} {} {}", basic_offsets[0], basic_offsets[1],
          basic_offsets[2], basic_offsets[3]);
  } else {
    DEBUG("Metadata: {} {} {} {} {}", basic_offsets[0], basic_offsets[1],
          basic_offsets[2], basic_offsets[3], basic_offsets[4]);
  }
  const int chunk_offsets_index = (nr == 5) ? 3 : 2;

  DEBUG("Reading pivots");
  read_bin(table_file, nr, host_data.dim, basic_offsets[0], host_data.pivots);
  ASSERT(nr == host_data.num_pivots);
  host_data.pivots_t = new float[(size_t)host_data.num_pivots * host_data.dim];
  for (int i = 0; i < host_data.num_pivots; i++) {
    for (int j = 0; j < host_data.dim; j++) {
      host_data.pivots_t[j * host_data.num_pivots + i] =
          host_data.pivots[i * host_data.dim + j];
    }
  }

  DEBUG("Reading centroid");
  read_bin(table_file, nr, nc, basic_offsets[1], host_data.centroid);
  ASSERT(nr == host_data.dim && nc == 1);

  DEBUG("Reading chunk offsets");
  int *chunk_offsets;
  read_bin(table_file, nr, nc, basic_offsets[chunk_offsets_index], chunk_offsets);
  ASSERT(nr == host_data.num_chunks + 1 && nc == 1);
  host_data.chunk_id = new int[host_data.dim];
  memset(host_data.chunk_id, -1, sizeof(int) * host_data.dim);
  for (int i = 0; i < host_data.num_chunks; i++) {
    for (int j = chunk_offsets[i]; j < chunk_offsets[i + 1]; j++) {
      host_data.chunk_id[j] = i;
    }
  }
  delete[] basic_offsets;
  delete[] chunk_offsets;
  DEBUG("PQ data load ok: dim: {}, num_pts: {}, num_chunks: {}", host_data.dim,
        host_data.num_pts, host_data.num_chunks);
}

void PQSearch::init_device(int dim, int num_pts, int num_thread_blocks,
                           int ef_search) {
  ASSERT(dim == host_data.dim);
  ASSERT(num_pts == host_data.num_pts);
  cleanup_device();

#ifdef MEM_PROFILE
  size_t free_mem, tot_mem;
  CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
  printf("Now %lf/%lf B free mem\n", 1.0 * free_mem, 1.0 * tot_mem);
#endif

  PQSearchData new_device{};
  new_device.dim = host_data.dim;
  new_device.num_chunks = host_data.num_chunks;
  new_device.num_pts = host_data.num_pts;

  try {
    copy_to_dev(host_data.centroid, new_device.centroid, host_data.dim);
    copy_to_dev(host_data.pivots, new_device.pivots,
                (size_t)host_data.num_pivots * host_data.dim);
    copy_to_dev(host_data.pivots_t, new_device.pivots_t,
                (size_t)host_data.num_pivots * host_data.dim);
    copy_to_dev(host_data.chunk_id, new_device.chunk_id, host_data.dim);

#ifdef MEM_PROFILE
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
    printf("Now %lf/%lf B free mem\n", 1.0 * free_mem, 1.0 * tot_mem);
#endif

    copy_to_dev(host_data.compressed_data, new_device.compressed_data,
                (size_t)host_data.num_pts * host_data.num_chunks);

#ifdef MEM_PROFILE
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
    printf("Now %lf/%lf B free mem\n", 1.0 * free_mem, 1.0 * tot_mem);
#endif

    CHECK_CUDA(cudaMalloc(&new_device.pq_dists,
                          sizeof(float) * num_thread_blocks *
                              host_data.num_pivots * host_data.num_chunks));
  } catch (...) {
    cuda_free_noexcept(new_device.centroid);
    cuda_free_noexcept(new_device.pivots);
    cuda_free_noexcept(new_device.pivots_t);
    cuda_free_noexcept(new_device.chunk_id);
    cuda_free_noexcept(new_device.compressed_data);
    cuda_free_noexcept(new_device.pq_dists);
    throw;
  }

  device_data = new_device;

  DEBUG("F");
  copy_to_dev(&device_data, device_ptr, 1);
  DEBUG("PQ data moved to device");

#ifdef MEM_PROFILE
  CHECK_CUDA(cudaMemGetInfo(&free_mem, &tot_mem));
  printf("Now %lf/%lf B free mem\n", 1.0 * free_mem, 1.0 * tot_mem);
#endif
}

template <class T>
static void read_bin(std::string filename, int &npts, int &ndim, size_t offset,
                     T *&data) {
  std::ifstream ifile(filename);
  if (!ifile.is_open()) {
    ERROR("Cannot open file: {}", filename);
    exit(-1);
  }
  ifile.seekg(offset, std::ios::beg);
  ifile.read((char *)&npts, sizeof(int));
  ifile.read((char *)&ndim, sizeof(int));
  size_t len = (size_t)npts * ndim;
  data = new T[len];
  ifile.read((char *)data, sizeof(T) * len);
  DEBUG("Read {} x {} data from file: {}", npts, ndim, filename);
}

}  // namespace shared
