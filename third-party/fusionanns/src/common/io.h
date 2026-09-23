#pragma once
#include <fstream>
#include <map>
#include <string>
#include <vector>

void load_fvecs(const std::string &filename, std::vector<float> &data,
                long &num, int &dim);

template <typename T>
void save_binary(const std::string &filename, const std::vector<T> &data) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot create file: " + filename);
  }
  ofs.write(reinterpret_cast<const char *>(data.data()),
            data.size() * sizeof(T));
}

void save_metadata(const std::string &filename,
                   const std::map<int, std::vector<long>> &metadata);

void load_metadata(const std::string &filename,
                   std::map<int, std::vector<long>> &metadata);

template <typename T>
void load_binary(const std::string &filename, std::vector<T> &data) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  data.resize(file_size / sizeof(T));
  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(data.data()), file_size);
}