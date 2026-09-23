#include "common/datasource.h"
#include "common/io.h"
#include "common/types.h"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>
#include <fstream>

#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

enum class GroundTruthFormat { Ivecs, Bin };

BaseFormat parse_base_format(const std::string &value) {
  std::string lower = value;
  std::transform(
      lower.begin(), lower.end(), lower.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "fvecs") {
    return BaseFormat::FVECs;
  }
  if (lower == "u8bin" || lower == "bin") {
    return BaseFormat::U8BIN;
  }
  if (lower == "bvecs") {
    return BaseFormat::BVECS;
  }
  if (lower == "fbin") {
    return BaseFormat::FBIN;
  }
  throw std::runtime_error("invalid dataset format: " + value);
}

GroundTruthFormat parse_groundtruth_format(const std::string &value) {
  std::string lower = value;
  std::transform(
      lower.begin(), lower.end(), lower.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "ivecs") {
    return GroundTruthFormat::Ivecs;
  }
  if (lower == "bin") {
    return GroundTruthFormat::Bin;
  }
  throw std::runtime_error("invalid ground truth format: " + value);
}

void load_groundtruth_ivecs(const std::string &path, std::vector<int> &data,
                            long &num, int &dim) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open ground truth file: " + path);
  }
  if (!ifs.read(reinterpret_cast<char *>(&dim), sizeof(int))) {
    throw std::runtime_error("Failed to read ivecs header: " + path);
  }
  if (dim <= 0) {
    throw std::runtime_error("Invalid ivecs dimension: " + path);
  }
  ifs.seekg(0, std::ios::end);
  std::streamoff bytes = ifs.tellg();
  if (bytes <= 0) {
    throw std::runtime_error("Empty ivecs file: " + path);
  }
  std::streamoff record_bytes =
      static_cast<std::streamoff>(sizeof(int) + dim * sizeof(int));
  num = static_cast<long>(bytes / record_bytes);
  data.resize(static_cast<size_t>(num) * static_cast<size_t>(dim));
  ifs.seekg(0, std::ios::beg);
  std::vector<int> buffer(static_cast<size_t>(dim));
  for (long i = 0; i < num; ++i) {
    int d_on_disk = 0;
    ifs.read(reinterpret_cast<char *>(&d_on_disk), sizeof(int));
    if (d_on_disk != dim) {
      throw std::runtime_error("Inconsistent ivecs dimension in " + path);
    }
    ifs.read(reinterpret_cast<char *>(buffer.data()),
             static_cast<std::streamsize>(dim) * sizeof(int));
    if (!ifs) {
      throw std::runtime_error("Truncated ivecs payload in " + path);
    }
    std::copy(buffer.begin(), buffer.end(),
              data.begin() + static_cast<size_t>(i) * static_cast<size_t>(dim));
  }
}

void load_groundtruth_bin(const std::string &path, std::vector<int> &data,
                          long &num, int &dim) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open ground truth file: " + path);
  }
  uint32_t n = 0;
  uint32_t k = 0;
  ifs.read(reinterpret_cast<char *>(&n), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));
  if (!ifs) {
    throw std::runtime_error("Failed to read bin header: " + path);
  }
  if (n == 0 || k == 0) {
    throw std::runtime_error("Invalid bin header (n or k == 0): " + path);
  }
  std::vector<uint32_t> raw(static_cast<size_t>(n) * static_cast<size_t>(k));
  ifs.read(reinterpret_cast<char *>(raw.data()),
           static_cast<std::streamsize>(raw.size() * sizeof(uint32_t)));
  if (ifs.gcount() !=
      static_cast<std::streamsize>(raw.size() * sizeof(uint32_t))) {
    throw std::runtime_error("Truncated bin payload in " + path);
  }
  data.resize(raw.size());
  std::transform(raw.begin(), raw.end(), data.begin(),
                 [](uint32_t v) { return static_cast<int>(v); });
  num = static_cast<long>(n);
  dim = static_cast<int>(k);
}

float l2_distance_sq(const float *lhs, const float *rhs, int dim) {
  float dist = 0.0f;
  for (int i = 0; i < dim; ++i) {
    float diff = lhs[i] - rhs[i];
    dist += diff * diff;
  }
  return dist;
}

struct PackedVectorReader {
  PackedVectorReader(const std::string &path, uint32_t page_bytes, int dim)
      : page_bytes_(page_bytes), dim_(dim),
        vec_bytes_(static_cast<size_t>(dim) * sizeof(float)),
        ifs_(path, std::ios::binary) {
    if (!ifs_) {
      throw std::runtime_error("Failed to open packed vectors: " + path);
    }
    if (page_bytes_ == 0) {
      throw std::runtime_error("page bytes must be > 0");
    }
    if (vec_bytes_ > page_bytes_) {
      throw std::runtime_error(
          "vector bytes exceed page bytes; incorrect --page?");
    }
    page_buffer_.resize(page_bytes_);
  }

  void read_batch(const std::vector<VectorLocation> &map, size_t start,
                  size_t count, float *out) {
    for (size_t i = 0; i < count; ++i) {
      const VectorLocation &loc = map[start + i];
      ensure_page_loaded(loc.page_id);
      size_t offset = static_cast<size_t>(loc.offset_in_page);
      if (offset + vec_bytes_ > page_bytes_) {
        std::ostringstream oss;
        oss << "Vector " << (start + i)
            << " spans beyond page boundary (page=" << loc.page_id
            << ", offset=" << loc.offset_in_page << ")";
        throw std::runtime_error(oss.str());
      }
      const char *src = page_buffer_.data() + offset;
      std::memcpy(out + i * static_cast<size_t>(dim_), src, vec_bytes_);
    }
  }

private:
  void ensure_page_loaded(uint32_t page_id) {
    if (current_page_ == page_id) {
      return;
    }
    std::streamoff offset = static_cast<std::streamoff>(page_id) *
                            static_cast<std::streamoff>(page_bytes_);
    ifs_.seekg(offset, std::ios::beg);
    if (!ifs_) {
      std::ostringstream oss;
      oss << "Failed to seek to page " << page_id << " (offset=" << offset
          << ")";
      throw std::runtime_error(oss.str());
    }
    ifs_.read(page_buffer_.data(), static_cast<std::streamsize>(page_bytes_));
    if (!ifs_ || ifs_.gcount() != static_cast<std::streamsize>(page_bytes_)) {
      std::ostringstream oss;
      oss << "Failed to read full page (page_id=" << page_id << ")";
      throw std::runtime_error(oss.str());
    }
    current_page_ = page_id;
  }

  uint32_t page_bytes_;
  int dim_;
  size_t vec_bytes_;
  std::ifstream ifs_;
  std::vector<char> page_buffer_;
  uint32_t current_page_ = std::numeric_limits<uint32_t>::max();
};

std::vector<int>
brute_force_topk(const float *query, PackedVectorReader &reader,
                 const std::vector<VectorLocation> &map, size_t base_size,
                 int dim, int top_k, size_t chunk_size,
                 std::vector<float> &chunk_buffer, size_t query_index,
                 size_t total_queries, bool show_progress) {
  using DistPair = std::pair<float, int>;
  std::priority_queue<DistPair> heap;
  chunk_size = std::max<size_t>(1, chunk_size);
  const size_t vec_stride = static_cast<size_t>(dim);
  size_t processed = 0;
  size_t report_step =
      std::max<size_t>(chunk_size, std::max<size_t>(1, base_size / 20));
  size_t next_report = report_step;
  size_t last_width = 0;
  const auto start_time = std::chrono::steady_clock::now();

  for (size_t start = 0; start < base_size; start += chunk_size) {
    size_t count = std::min(chunk_size, base_size - start);
    reader.read_batch(map, start, count, chunk_buffer.data());
    for (size_t i = 0; i < count; ++i) {
      const float *vec = chunk_buffer.data() + i * vec_stride;
      float dist = l2_distance_sq(query, vec, dim);
      if (static_cast<int>(heap.size()) < top_k) {
        heap.emplace(dist, static_cast<int>(start + i));
      } else if (dist < heap.top().first) {
        heap.pop();
        heap.emplace(dist, static_cast<int>(start + i));
      }
    }
    processed += count;
    if (show_progress && (processed >= next_report || processed == base_size)) {
      double percent = base_size == 0
                           ? 100.0
                           : (static_cast<double>(processed) * 100.0) /
                                 static_cast<double>(base_size);
      double elapsed_s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start_time)
                             .count();
      double vec_per_sec = elapsed_s > 0.0 ? processed / elapsed_s : 0.0;
      double mv_per_sec = vec_per_sec / 1e6;
      std::ostringstream oss;
      oss.setf(std::ios::fixed);
      oss << "  > Query " << (query_index + 1) << "/" << total_queries
          << " 暴力搜索进度: " << processed << "/" << base_size << " ("
          << std::setprecision(1) << percent << "%, " << mv_per_sec << " MV/s)";
      std::string line = oss.str();
      if (line.size() < last_width) {
        line.append(last_width - line.size(), ' ');
      } else {
        last_width = line.size();
      }
      std::cout << '\r' << line;
      if (processed == base_size) {
        std::cout << std::endl;
        last_width = 0;
      }
      next_report += report_step;
    }
  }
  std::vector<int> result(heap.size());
  for (int idx = static_cast<int>(heap.size()) - 1; idx >= 0; --idx) {
    result[static_cast<size_t>(idx)] = heap.top().second;
    heap.pop();
  }
  return result;
}

struct Options {
  std::string base_path = "data/sift/sift_base.fvecs";
  std::string base_format = "fvecs";
  std::string query_path = "data/sift/sift_query.fvecs";
  std::string query_format = "fvecs";
  std::string groundtruth_path = "data/sift/sift_groundtruth.ivecs";
  std::string groundtruth_format = "ivecs";
  std::string index_dir = "indices";
  uint32_t page_bytes = static_cast<uint32_t>(SSD_PAGE_SIZE);
  size_t check_count = 100000; // 0 表示检查全部向量
  int top_k = 100;
  size_t queries = 10; // 0 表示全部查询
  size_t chunk = 1024;
  bool verbose = false;
  bool progress = true;
  bool pq_correlation = false;
  std::string pq_query_path = "data/sift/query.public.10K.u8bin";
  std::string pq_query_format = "u8bin";
  size_t pq_query_index = 0;
  std::string pq_codec_path = "indices/pq_codec";
  std::string pq_codes_path = "indices/pq_codes";
  size_t pq_sample = 1'000'000;
  size_t pq_top_overlap = 1000;
  uint64_t pq_seed = 42;
  bool pq_progress = true;
};

} // namespace

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;

  Options opt;

  CLI::App app{"FusionANNS Index Checker / FusionANNS 索引验证小工具"};
  app.add_option(
      "--base", opt.base_path,
      "Base dataset path / 底库数据路径 (默认 data/sift/sift_base.fvecs)");
  app.add_option("--base-format", opt.base_format,
                 "Base format: fvecs|u8bin|bvecs|fbin / 底库格式");
  app.add_option("--query", opt.query_path,
                 "Query dataset path / 查询数据路径");
  app.add_option("--query-format", opt.query_format,
                 "Query format: fvecs|u8bin|bvecs|fbin / 查询格式");
  app.add_option("--groundtruth", opt.groundtruth_path,
                 "Ground truth path / 精确真值路径");
  app.add_option("--groundtruth-format", opt.groundtruth_format,
                 "Ground truth format: ivecs|bin / 真值格式");
  app.add_option("--index-dir", opt.index_dir,
                 "Directory containing built index / 存放索引输出的目录");
  app.add_option("--page", opt.page_bytes,
                 "Packed page bytes (must match builder --page) / 索引页大小");
  app.add_option(
      "--check-count", opt.check_count,
      "Number of base vectors to verify (0 = all) / 校验底库向量数量");
  app.add_option("--topk", opt.top_k,
                 "Recall@K for brute-force check / 暴力搜索 Top-K");
  app.add_option("--queries", opt.queries,
                 "Queries to evaluate (0 = all) / 参与验证的查询数量");
  app.add_option("--chunk", opt.chunk,
                 "Streaming chunk size during brute force / 暴力扫描批大小");
  app.add_flag("--verbose", opt.verbose, "Print extra details / 输出更多细节");
  app.add_flag("--progress,!--no-progress", opt.progress,
               "Show brute-force progress (默认开启) / 显示暴搜进度");
  app.add_flag("--pq-correlation", opt.pq_correlation,
               "Run PQ correlation checker / 执行 PQ 相关性检测");
  app.add_option("--pq-query", opt.pq_query_path,
                 "Correlation query dataset path / PQ校验用查询数据路径");
  app.add_option(
      "--pq-query-format", opt.pq_query_format,
      "Correlation query format: fvecs|u8bin|bvecs|fbin / PQ校验查询格式");
  app.add_option("--pq-query-idx", opt.pq_query_index,
                 "Correlation query index (default 0) / PQ校验查询序号");
  app.add_option("--pq-codec", opt.pq_codec_path,
                 "PQ codec path / PQ码本路径 (默认 indices/pq_codec)");
  app.add_option("--pq-codes", opt.pq_codes_path,
                 "PQ codes path / PQ编码路径 (默认 indices/pq_codes)");
  app.add_option(
      "--pq-sample", opt.pq_sample,
      "Sample size for correlation (default 1,000,000) / PQ校验抽样数量");
  app.add_option("--pq-top", opt.pq_top_overlap,
                 "Top-K overlap cutoff (default 1000) / PQ校验Top-K交集阈值");
  app.add_option("--pq-seed", opt.pq_seed,
                 "Random seed for sampling / PQ抽样随机种子");
  app.add_flag("--pq-progress,!--pq-no-progress", opt.pq_progress,
               "Show PQ sampling progress (默认开启) / 显示PQ抽样进度");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  if (opt.page_bytes == 0) {
    std::cerr << "错误: --page 必须大于 0" << std::endl;
    return 1;
  }
  if (opt.chunk == 0) {
    opt.chunk = 1024;
  }
  if (opt.top_k <= 0) {
    std::cerr << "错误: --topk 必须大于 0" << std::endl;
    return 1;
  }

  try {
    BaseFormat base_fmt = parse_base_format(opt.base_format);
    BaseFormat query_fmt = parse_base_format(opt.query_format);
    GroundTruthFormat gt_fmt = parse_groundtruth_format(opt.groundtruth_format);

    auto base_source = make_source(base_fmt, opt.base_path);
    size_t base_size = base_source->size();
    int dim = base_source->dim();

    auto query_source = make_source(query_fmt, opt.query_path, 0, dim);
    if (query_source->dim() != dim) {
      throw std::runtime_error("Query dimension mismatch vs base dataset");
    }
    size_t available_queries = query_source->size();

    std::string map_path = opt.index_dir + "/vector_location.map";
    std::string packed_path = opt.index_dir + "/packed_raw_vectors.bin";

    std::vector<VectorLocation> location_map;
    load_binary(map_path, location_map);
    if (location_map.size() != base_size) {
      std::ostringstream oss;
      oss << "Vector location map size (" << location_map.size()
          << ") mismatch with base dataset size (" << base_size << ")";
      throw std::runtime_error(oss.str());
    }

    PackedVectorReader reader(packed_path, opt.page_bytes, dim);

    std::cout << "--- 索引基础信息 ---" << std::endl;
    std::cout << "Base vectors 底库向量数: " << base_size << std::endl;
    std::cout << "Dimension 向量维度: " << dim << std::endl;
    std::cout << "Packed file 索引原始文件: " << packed_path << std::endl;
    std::cout << "Location map 映射表: " << map_path << std::endl;

    size_t verify_total = (opt.check_count == 0)
                              ? base_size
                              : std::min(opt.check_count, base_size);
    const size_t verify_chunk =
        verify_total > 0
            ? std::max<size_t>(1, std::min(opt.chunk, verify_total))
            : 0;
    std::vector<float> base_buffer(verify_chunk * static_cast<size_t>(dim));
    std::vector<float> index_buffer(verify_chunk * static_cast<size_t>(dim));

    double max_abs_diff = 0.0;
    double mean_abs_diff = 0.0;
    size_t compared = 0;

    std::vector<int> gt_data;
    long gt_num = 0;
    int gt_dim = 0;
    if (gt_fmt == GroundTruthFormat::Ivecs) {
      load_groundtruth_ivecs(opt.groundtruth_path, gt_data, gt_num, gt_dim);
    } else {
      load_groundtruth_bin(opt.groundtruth_path, gt_data, gt_num, gt_dim);
    }

    if (gt_num == 0 || gt_dim == 0) {
      throw std::runtime_error("Ground truth file is empty or malformed");
    }

    if (static_cast<size_t>(gt_num) < available_queries) {
      std::cout << "  > Ground truth queries < query dataset, 自动截断。"
                << std::endl;
      available_queries = static_cast<size_t>(gt_num);
    }

    size_t queries_to_run = opt.queries == 0
                                ? available_queries
                                : std::min(opt.queries, available_queries);

    int total_steps = 0;
    if (verify_total > 0) {
      total_steps++;
    }
    if (queries_to_run > 0) {
      total_steps++;
    }
    if (opt.pq_correlation) {
      total_steps++;
    }
    if (total_steps == 0) {
      total_steps = 1;
    }
    int current_step = 1;

    if (verify_total > 0) {
      std::cout << "\n[" << current_step << "/" << total_steps
                << "] 校验 packed_raw_vectors 是否完整还原底库..." << std::endl;
      for (size_t offset = 0; offset < verify_total; offset += verify_chunk) {
        size_t count = std::min(verify_chunk, verify_total - offset);
        base_source->read_block(offset, count, base_buffer.data());
        reader.read_batch(location_map, offset, count, index_buffer.data());
        for (size_t i = 0; i < count * static_cast<size_t>(dim); ++i) {
          double diff = std::abs(static_cast<double>(base_buffer[i]) -
                                 static_cast<double>(index_buffer[i]));
          max_abs_diff = std::max(max_abs_diff, diff);
          mean_abs_diff += diff;
        }
        compared += count * static_cast<size_t>(dim);
        if (opt.verbose) {
          std::cout << "  > 已比对 " << (offset + count) << " / "
                    << verify_total << " 向量" << std::endl;
        }
      }
      mean_abs_diff =
          compared > 0 ? mean_abs_diff / static_cast<double>(compared) : 0.0;
      std::cout << "  > 最大绝对误差 Max |Δ|: " << std::fixed
                << std::setprecision(6) << max_abs_diff << std::endl;
      std::cout << "  > 平均绝对误差 Mean |Δ|: " << std::fixed
                << std::setprecision(6) << mean_abs_diff << std::endl;
      if (max_abs_diff > 1e-5) {
        std::cout << "  > 警告: packed_raw_vectors 与原始数据存在明显差异!"
                  << std::endl;
      } else {
        std::cout << "  > OK: 索引中的原始向量与底库一致。" << std::endl;
      }
      current_step++;
    } else if (opt.verbose) {
      std::cout << "\n[信息] check-count 为 0，跳过 packed_raw_vectors 校验。"
                << std::endl;
    }

    if (opt.pq_correlation) {
      std::cout << "\n[" << current_step << "/" << total_steps
                << "] 检查 PQ 距离与真实 L2 距离的相关性..." << std::endl;

      BaseFormat pq_query_fmt = parse_base_format(opt.pq_query_format);
      auto pq_query_source =
          make_source(pq_query_fmt, opt.pq_query_path, 0, dim);
      if (pq_query_source->dim() != dim) {
        throw std::runtime_error("PQ correlation query dimension mismatch");
      }
      size_t pq_query_available = pq_query_source->size();
      if (pq_query_available == 0) {
        throw std::runtime_error("PQ correlation query dataset is empty");
      }
      if (opt.pq_query_index >= pq_query_available) {
        std::ostringstream oss;
        oss << "PQ correlation query index " << opt.pq_query_index
            << " 超出范围 (available=" << pq_query_available << ")";
        throw std::runtime_error(oss.str());
      }

      std::vector<float> corr_query(static_cast<size_t>(dim));
      pq_query_source->read_block(opt.pq_query_index, 1, corr_query.data());

      std::unique_ptr<faiss::ProductQuantizer> pq_holder(
          faiss::read_ProductQuantizer(opt.pq_codec_path.c_str()));
      if (!pq_holder) {
        throw std::runtime_error("Failed to load PQ codec from " +
                                 opt.pq_codec_path);
      }
      const faiss::ProductQuantizer &pq = *pq_holder;
      if (static_cast<int>(pq.dsub * pq.M) != dim) {
        std::ostringstream oss;
        oss << "PQ codec dimension mismatch: M=" << pq.M << " dsub=" << pq.dsub
            << " (M*dsub=" << pq.M * pq.dsub << ", expected dim=" << dim << ")";
        throw std::runtime_error(oss.str());
      }
      const size_t code_size = pq.code_size;
      if (code_size == 0) {
        throw std::runtime_error("PQ codec reports zero sub-quantizers");
      }

      std::ifstream pq_codes(opt.pq_codes_path, std::ios::binary);
      if (!pq_codes.is_open()) {
        throw std::runtime_error("Cannot open PQ codes file: " +
                                 opt.pq_codes_path);
      }
      pq_codes.seekg(0, std::ios::end);
      std::streamoff codes_bytes = pq_codes.tellg();
      if (codes_bytes < 0) {
        throw std::runtime_error("Failed to stat PQ codes file: " +
                                 opt.pq_codes_path);
      }
      if (codes_bytes % static_cast<std::streamoff>(code_size) != 0) {
        throw std::runtime_error("PQ codes file size is not aligned to code "
                                 "length: " +
                                 opt.pq_codes_path);
      }
      size_t codes_count =
          static_cast<size_t>(codes_bytes) / static_cast<size_t>(code_size);
      if (codes_count < base_size) {
        std::ostringstream oss;
        oss << "PQ codes count (" << codes_count << ") 小于底库向量数 ("
            << base_size << ")";
        throw std::runtime_error(oss.str());
      }
      pq_codes.seekg(0, std::ios::beg);

      size_t sample_goal =
          opt.pq_sample == 0 ? base_size : std::min(opt.pq_sample, base_size);
      if (sample_goal == 0) {
        std::cout << "  > 底库为空，跳过 PQ 相关性检测。" << std::endl;
        if (current_step < total_steps) {
          current_step++;
        }
      } else {
        std::vector<size_t> sample_ids;
        sample_ids.reserve(sample_goal);
        if (sample_goal == base_size) {
          sample_ids.resize(base_size);
          std::iota(sample_ids.begin(), sample_ids.end(), 0);
        } else {
          std::mt19937_64 rng(opt.pq_seed);
          std::uniform_int_distribution<size_t> dist(static_cast<size_t>(0),
                                                     base_size - 1);
          std::unordered_set<size_t> unique;
          unique.reserve(std::min(base_size, sample_goal * 2));
          while (sample_ids.size() < sample_goal) {
            size_t candidate = dist(rng);
            if (unique.insert(candidate).second) {
              sample_ids.push_back(candidate);
            }
          }
        }

        struct SampleSlot {
          size_t id;
          size_t slot;
        };
        std::vector<SampleSlot> sorted(sample_ids.size());
        for (size_t i = 0; i < sample_ids.size(); ++i) {
          sorted[i] = SampleSlot{sample_ids[i], i};
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const SampleSlot &lhs, const SampleSlot &rhs) {
                    return lhs.id < rhs.id;
                  });

        std::vector<float> distance_table(static_cast<size_t>(pq.M) *
                                          static_cast<size_t>(pq.ksub));
        pq.compute_distance_table(corr_query.data(), distance_table.data());

        struct CorrEntry {
          size_t vector_id;
          float dist_l2;
          float dist_pq;
        };
        std::vector<CorrEntry> entries(sample_ids.size());

        std::vector<float> gather_buffer;
        std::vector<uint8_t> codes_buffer;
        size_t processed = 0;
        size_t report_step =
            std::max<size_t>(1, std::max<size_t>(sample_ids.size() / 20, 1));
        size_t next_report = report_step;
        size_t last_width = 0;
        const auto start_time = std::chrono::steady_clock::now();

        size_t pos = 0;
        while (pos < sorted.size()) {
          size_t run_start = pos;
          size_t run_base_id = sorted[run_start].id;
          size_t run_end = run_start + 1;
          while (run_end < sorted.size() &&
                 sorted[run_end].id == run_base_id + (run_end - run_start)) {
            ++run_end;
          }
          size_t run_length = run_end - run_start;
          gather_buffer.resize(run_length * static_cast<size_t>(dim));
          reader.read_batch(location_map, run_base_id, run_length,
                            gather_buffer.data());

          pq_codes.clear();
          std::streamoff seek_offset =
              static_cast<std::streamoff>(run_base_id * code_size);
          pq_codes.seekg(seek_offset, std::ios::beg);
          codes_buffer.resize(run_length * code_size);
          pq_codes.read(reinterpret_cast<char *>(codes_buffer.data()),
                        static_cast<std::streamsize>(run_length * code_size));
          if (pq_codes.gcount() !=
              static_cast<std::streamsize>(run_length * code_size)) {
            throw std::runtime_error("PQ codes truncated near vector " +
                                     std::to_string(run_base_id));
          }

          auto compute_pq_distance = [&](const uint8_t *code_ptr) -> float {
            double acc = 0.0;
            if (pq.nbits == 8) {
              for (size_t m = 0; m < pq.M; ++m) {
                uint8_t cid = code_ptr[m];
                if (cid >= pq.ksub) {
                  throw std::runtime_error("PQ code byte exceeds ksub");
                }
                acc += static_cast<double>(distance_table[m * pq.ksub + cid]);
              }
            } else {
              faiss::PQDecoderGeneric decoder(code_ptr,
                                              static_cast<int>(pq.nbits));
              for (size_t m = 0; m < pq.M; ++m) {
                uint64_t cid = decoder.decode();
                if (cid >= pq.ksub) {
                  throw std::runtime_error("PQ code symbol exceeds ksub");
                }
                acc += static_cast<double>(distance_table[m * pq.ksub + cid]);
              }
            }
            return static_cast<float>(acc);
          };

          for (size_t i = 0; i < run_length; ++i) {
            size_t slot = sorted[run_start + i].slot;
            const float *vec =
                gather_buffer.data() + i * static_cast<size_t>(dim);
            float l2 = l2_distance_sq(corr_query.data(), vec, dim);
            const uint8_t *code =
                codes_buffer.data() + i * static_cast<size_t>(code_size);
            float pq_dist = compute_pq_distance(code);
            entries[slot] = CorrEntry{sorted[run_start + i].id, l2, pq_dist};
          }

          pos = run_end;
          processed += run_length;

          if (opt.pq_progress &&
              (processed >= next_report || processed == sample_ids.size())) {
            double percent = sample_ids.empty()
                                 ? 100.0
                                 : (static_cast<double>(processed) * 100.0) /
                                       static_cast<double>(sample_ids.size());
            double elapsed_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              start_time)
                    .count();
            double pairs_per_sec =
                elapsed_s > 0.0 ? processed / elapsed_s : 0.0;
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << "  > PQ抽样进度: " << processed << "/" << sample_ids.size()
                << " (" << std::setprecision(1) << percent << "%, "
                << std::setprecision(2) << (pairs_per_sec / 1e6) << " M对/s)";
            std::string line = oss.str();
            if (line.size() < last_width) {
              line.append(last_width - line.size(), ' ');
            } else {
              last_width = line.size();
            }
            std::cout << '\r' << line;
            if (processed == sample_ids.size()) {
              std::cout << std::endl;
              last_width = 0;
            }
            next_report += report_step;
          }
        }

        double sum_l2 = 0.0;
        double sum_pq = 0.0;
        double sum_l2_sq = 0.0;
        double sum_pq_sq = 0.0;
        double sum_cross = 0.0;
        for (const CorrEntry &entry : entries) {
          double l2 = static_cast<double>(entry.dist_l2);
          double pq_dist = static_cast<double>(entry.dist_pq);
          sum_l2 += l2;
          sum_pq += pq_dist;
          sum_l2_sq += l2 * l2;
          sum_pq_sq += pq_dist * pq_dist;
          sum_cross += l2 * pq_dist;
        }
        double denom = static_cast<double>(entries.size());
        double mean_l2 = sum_l2 / denom;
        double mean_pq = sum_pq / denom;
        double var_l2 = sum_l2_sq / denom - mean_l2 * mean_l2;
        double var_pq = sum_pq_sq / denom - mean_pq * mean_pq;
        double cov = sum_cross / denom - mean_l2 * mean_pq;
        double corr = 0.0;
        if (var_l2 > 0.0 && var_pq > 0.0) {
          corr = cov / std::sqrt(var_l2 * var_pq);
        }

        size_t top_k = opt.pq_top_overlap == 0
                           ? std::min<size_t>(1000, entries.size())
                           : std::min(opt.pq_top_overlap, entries.size());
        size_t overlap = 0;
        if (top_k > 0) {
          std::vector<size_t> order_l2(entries.size());
          std::iota(order_l2.begin(), order_l2.end(), 0);
          std::partial_sort(order_l2.begin(), order_l2.begin() + top_k,
                            order_l2.end(), [&](size_t a, size_t b) {
                              return entries[a].dist_l2 < entries[b].dist_l2;
                            });

          std::vector<size_t> order_pq(entries.size());
          std::iota(order_pq.begin(), order_pq.end(), 0);
          std::partial_sort(order_pq.begin(), order_pq.begin() + top_k,
                            order_pq.end(), [&](size_t a, size_t b) {
                              return entries[a].dist_pq < entries[b].dist_pq;
                            });

          std::unordered_set<size_t> l2_top_ids;
          l2_top_ids.reserve(top_k * 2);
          for (size_t i = 0; i < top_k; ++i) {
            l2_top_ids.insert(entries[order_l2[i]].vector_id);
          }
          for (size_t i = 0; i < top_k; ++i) {
            overlap += l2_top_ids.count(entries[order_pq[i]].vector_id) ? 1 : 0;
          }
        }
        double overlap_ratio = (top_k > 0) ? static_cast<double>(overlap) /
                                                 static_cast<double>(top_k)
                                           : 0.0;

        std::cout << "  > 抽样对数: " << entries.size() << std::endl;
        std::cout << "  > Pearson 相关系数: " << std::fixed
                  << std::setprecision(4) << corr << std::endl;
        if (top_k > 0) {
          std::cout << "  > Top-" << top_k << " 交集: " << overlap << " ("
                    << std::setprecision(1) << overlap_ratio * 100.0 << "%)"
                    << std::endl;
        } else {
          std::cout << "  > Top-K 交集计算被禁用 (pq-top=0)." << std::endl;
        }
        std::cout << "  > L2 均值/方差: " << std::setprecision(6) << mean_l2
                  << " / " << var_l2 << std::endl;
        std::cout << "  > PQ 均值/方差: " << std::setprecision(6) << mean_pq
                  << " / " << var_pq << std::endl;

        if (top_k > 0 && overlap < top_k / 10) {
          std::cout << "\n结论: PQ 距离与真实 L2 距离几乎无关，召回风险极高。"
                    << std::endl;
        } else if (corr < 0.2) {
          std::cout << "\n结论: PQ 相关性偏弱，请检查 PQ 训练或编码流程。"
                    << std::endl;
        } else {
          std::cout << "\n结论: PQ 距离与真实 L2 距离相关性良好。" << std::endl;
        }
      }

      if (current_step < total_steps) {
        current_step++;
      }
    }

    double recall_sum = 0.0;
    double recall_min = 1.0;
    size_t perfect = 0;

    if (queries_to_run == 0) {
      std::cout << "\n[" << current_step << "/" << total_steps
                << "] 无可用查询，跳过 Recall 验证。" << std::endl;
      if (current_step < total_steps) {
        current_step++;
      }
    } else {
      int eval_top_k =
          std::min(opt.top_k, std::min(static_cast<int>(gt_dim),
                                       static_cast<int>(base_size)));
      if (eval_top_k <= 0) {
        throw std::runtime_error("Invalid top_k setting after clamping");
      }

      std::cout << "\n[" << current_step << "/" << total_steps
                << "] 进行暴力搜索 Recall@" << eval_top_k << " 验证..."
                << std::endl;

      std::vector<float> query_buffer(queries_to_run *
                                      static_cast<size_t>(dim));
      query_source->read_block(0, queries_to_run, query_buffer.data());

      const size_t brute_chunk_size =
          std::max<size_t>(1, std::min(opt.chunk, base_size));
      std::vector<float> brute_chunk(brute_chunk_size *
                                     static_cast<size_t>(dim));

      for (size_t qi = 0; qi < queries_to_run; ++qi) {
        const float *query =
            query_buffer.data() + qi * static_cast<size_t>(dim);
        std::vector<int> predicted = brute_force_topk(
            query, reader, location_map, base_size, dim, eval_top_k,
            brute_chunk_size, brute_chunk, qi, queries_to_run, opt.progress);

        const int *gt_begin = gt_data.data() + qi * static_cast<size_t>(gt_dim);
        std::unordered_set<int> gt_set(gt_begin, gt_begin + eval_top_k);

        int hits = 0;
        for (int id : predicted) {
          hits += gt_set.count(id) ? 1 : 0;
        }
        double recall =
            static_cast<double>(hits) / static_cast<double>(eval_top_k);
        recall_sum += recall;
        recall_min = std::min(recall_min, recall);
        if (hits == eval_top_k) {
          perfect++;
        }

        if (opt.verbose && qi < 5) {
          std::cout << "  > Query " << qi << ": hits=" << hits << "/"
                    << eval_top_k << " recall=" << std::fixed
                    << std::setprecision(4) << recall << std::endl;
          if (hits != eval_top_k) {
            std::cout << "    预测Top: ";
            for (size_t i = 0; i < predicted.size(); ++i) {
              std::cout << predicted[i] << " ";
            }
            std::cout << "\n    真值Top: ";
            for (int i = 0; i < eval_top_k; ++i) {
              std::cout << gt_begin[i] << " ";
            }
            std::cout << std::endl;
          }
        }
      }

      double recall_avg = recall_sum / static_cast<double>(queries_to_run);
      std::cout << "\n--- 验证结果 ---" << std::endl;
      std::cout << "Queries 检测查询数: " << queries_to_run << std::endl;
      std::cout << "Perfect 完全匹配数: " << perfect << std::endl;
      std::cout << "Recall@" << eval_top_k << " 平均值: " << std::fixed
                << std::setprecision(4) << recall_avg << std::endl;
      std::cout << "Recall 最低值: " << std::fixed << std::setprecision(4)
                << recall_min << std::endl;

      if (recall_min < 0.999) {
        std::cout << "\n结论: Recall 存在异常，请检查索引输出或原始数据。"
                  << std::endl;
      } else {
        std::cout << "\n结论: 索引中的原始向量与真值完全一致，暴力召回通过。"
                  << std::endl;
      }

      current_step++;
    }
  } catch (const std::exception &e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
