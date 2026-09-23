#include "io.h"
#include <fstream>
#include <iostream>

// --- Helper Functions for I/O ---

// Loads data from a .fvecs file
// Format: [dim_1 (int32)] [vec_1 (float * dim_1)] [dim_2 (int32)] [vec_2 (float
// * dim_2)] ...
void load_fvecs(const std::string &filename, std::vector<float> &data,
                long &num, int &dim) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    exit(-1);
  }
  ifs.read(reinterpret_cast<char *>(&dim), sizeof(int));
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  num = file_size / ((dim * sizeof(float)) + sizeof(int));
  data.resize(num * dim);

  ifs.seekg(0, std::ios::beg);
  std::vector<float> temp_vec(dim);
  int temp_dim;
  for (long i = 0; i < num; ++i) {
    ifs.read(reinterpret_cast<char *>(&temp_dim), sizeof(int));
    if (temp_dim != dim) {
      std::cerr << "Inconsistent dimension found in " << filename << std::endl;
      exit(-1);
    }
    ifs.read(reinterpret_cast<char *>(temp_vec.data()), dim * sizeof(float));
    std::copy(temp_vec.begin(), temp_vec.end(), data.begin() + i * dim);
  }
}

// Saves the metadata (inverted lists) to a binary file
void save_metadata(const std::string &filename,
                   const std::map<int, std::vector<long>> &metadata) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << "Cannot create file: " << filename << std::endl;
    exit(-1);
  }
  int num_lists = metadata.size();
  ofs.write(reinterpret_cast<const char *>(&num_lists), sizeof(int));

  for (const auto &pair : metadata) {
    int list_id = pair.first;
    const auto &vec_ids = pair.second;
    long list_size = vec_ids.size();

    ofs.write(reinterpret_cast<const char *>(&list_id), sizeof(int));
    ofs.write(reinterpret_cast<const char *>(&list_size), sizeof(long));
    ofs.write(reinterpret_cast<const char *>(vec_ids.data()),
              list_size * sizeof(long));
  }
}

// --- Helper function to load any binary file into a vector ---
void load_metadata(const std::string &filename,
                   std::map<int, std::vector<long>> &metadata) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  int num_lists;
  ifs.read(reinterpret_cast<char *>(&num_lists), sizeof(int));

  for (int i = 0; i < num_lists; ++i) {
    int list_id;
    long list_size;
    ifs.read(reinterpret_cast<char *>(&list_id), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&list_size), sizeof(long));
    std::vector<long> vec_ids(list_size);
    ifs.read(reinterpret_cast<char *>(vec_ids.data()),
             list_size * sizeof(long));
    metadata[list_id] = std::move(vec_ids);
  }
}