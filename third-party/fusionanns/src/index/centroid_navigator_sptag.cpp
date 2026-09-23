#include "index/centroid_navigator_sptag.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <inc/Core/Common.h>
#include <inc/Core/SearchResult.h>
#include <inc/Core/VectorIndex.h>
#include <inc/Helper/StringConvert.h>

namespace {

void ensure_success(SPTAG::ErrorCode code, const char *context) {
  if (code != SPTAG::ErrorCode::Success) {
    throw std::runtime_error(std::string(context) + " failed: " +
                             SPTAG::Helper::Convert::ConvertToString(code));
  }
}

std::once_flag g_logger_once;

} // namespace

SPTAGCentroidNavigator::SPTAGCentroidNavigator() {
  std::call_once(g_logger_once, []() {
    SPTAG::SetLogger(std::make_shared<SPTAG::Helper::SimpleLogger>(
        SPTAG::Helper::LogLevel::LL_Warning));
  });
}

SPTAGCentroidNavigator::~SPTAGCentroidNavigator() = default;

void SPTAGCentroidNavigator::build(const float *centroids, int nlist, int dim,
                                   int num_threads) {
  if (centroids == nullptr || nlist <= 0 || dim <= 0) {
    throw std::invalid_argument(
        "Invalid input to SPTAGCentroidNavigator::build");
  }

  index_ = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT,
                                              SPTAG::VectorValueType::Float);
  if (!index_) {
    throw std::runtime_error("SPTAG VectorIndex creation failed");
  }

  dim_ = dim;

  configure_build_defaults(nlist, num_threads);

  ensure_success(index_->BuildIndex(centroids,
                                    static_cast<SPTAG::SizeType>(nlist),
                                    static_cast<SPTAG::DimensionType>(dim_)),
                 "SPTAG BuildIndex");
}

void SPTAGCentroidNavigator::save(const std::string &path) const {
  ensure_ready();

  std::filesystem::path folder(path);
  if (!std::filesystem::exists(folder)) {
    std::filesystem::create_directories(folder);
  }

  std::string folder_str = folder.string();
  ensure_success(index_->SaveIndex(folder_str), "SPTAG SaveIndex");
}

void SPTAGCentroidNavigator::load(const std::string &path) {
  std::filesystem::path folder(path);
  if (!std::filesystem::exists(folder)) {
    throw std::runtime_error("SPTAG index path does not exist: " + path);
  }

  std::shared_ptr<SPTAG::VectorIndex> loaded;
  ensure_success(SPTAG::VectorIndex::LoadIndex(folder.string(), loaded),
                 "SPTAG LoadIndex");
  if (!loaded) {
    throw std::runtime_error("SPTAG LoadIndex returned null instance");
  }

  index_ = std::move(loaded);
  dim_ = index_->GetFeatureDim();
}

ProbeResult SPTAGCentroidNavigator::search_one(const float *query,
                                               int nprobe) const {
  ensure_ready();
  if (query == nullptr || nprobe <= 0) {
    throw std::invalid_argument(
        "Invalid input to SPTAGCentroidNavigator::search_one");
  }

  const int neighbor_count = nprobe;
  std::vector<SPTAG::BasicResult> results(
      static_cast<size_t>(std::max(neighbor_count, 1)));
  SPTAG::QueryResult qres((char *)query, neighbor_count, /*withMeta=*/false,
                          results.data());
  ensure_success(index_->SearchIndex(qres), "SPTAG SearchIndex");

  ProbeResult out;
  out.ids.reserve(results.size());
  out.dists.reserve(results.size());

  for (const auto &res : results) {
    if (res.VID < 0) {
      continue;
    }
    out.ids.push_back(static_cast<long>(res.VID));
    out.dists.push_back(res.Dist);
  }

  return out;
}

std::vector<ProbeResult>
SPTAGCentroidNavigator::search_batch(const float *queries, int num_queries,
                                     int nprobe) const {
  ensure_ready();
  if (queries == nullptr || num_queries <= 0 || nprobe <= 0) {
    throw std::invalid_argument(
        "Invalid input to SPTAGCentroidNavigator::search_batch");
  }

  // 使用 SPTAG 批量搜索 API
  // 结果按 row-major 排列: [num_queries * nprobe]
  std::vector<SPTAG::BasicResult> results(static_cast<size_t>(num_queries) *
                                          static_cast<size_t>(nprobe));

  ensure_success(index_->SearchIndex(queries, num_queries, nprobe,
                                     /*p_withMeta=*/false, results.data()),
                 "SPTAG SearchIndex batch");

  // 转换结果格式
  std::vector<ProbeResult> out(num_queries);
  for (int q = 0; q < num_queries; ++q) {
    auto &probe_result = out[q];
    probe_result.ids.reserve(nprobe);
    probe_result.dists.reserve(nprobe);

    const size_t base = static_cast<size_t>(q) * static_cast<size_t>(nprobe);
    for (int k = 0; k < nprobe; ++k) {
      const auto &res = results[base + k];
      if (res.VID < 0) {
        continue;
      }
      probe_result.ids.push_back(static_cast<long>(res.VID));
      probe_result.dists.push_back(res.Dist);
    }
  }

  return out;
}

void SPTAGCentroidNavigator::search_batch_flat(
    const float *queries, int num_queries, int nprobe,
    std::vector<int32_t> &out_ids, std::vector<float> &out_dists) const {
  ensure_ready();
  if (queries == nullptr || num_queries <= 0 || nprobe <= 0) {
    throw std::invalid_argument(
        "Invalid input to SPTAGCentroidNavigator::search_batch_flat");
  }

  static thread_local std::vector<SPTAG::BasicResult> results;
  const size_t total =
      static_cast<size_t>(num_queries) * static_cast<size_t>(nprobe);
  results.resize(total);
  for (auto &res : results) {
    res.VID = -1;
    res.Dist = std::numeric_limits<float>::infinity();
  }

  ensure_success(index_->SearchIndex(queries, num_queries, nprobe,
                                     /*p_withMeta=*/false, results.data()),
                 "SPTAG SearchIndex batch");

  out_ids.resize(total);
  out_dists.resize(total);
  for (size_t i = 0; i < total; ++i) {
    const auto &res = results[i];
    if (res.VID < 0) {
      out_ids[i] = -1;
      out_dists[i] = std::numeric_limits<float>::infinity();
    } else {
      out_ids[i] = static_cast<int32_t>(res.VID);
      out_dists[i] = res.Dist;
    }
  }
}

void SPTAGCentroidNavigator::tune_for_nprobe(int nprobe) {
  if (nprobe <= 0) {
    return;
  }
  configure_search_defaults(nprobe);
}

void SPTAGCentroidNavigator::apply_thread_local_settings() const {
  if (!index_) {
    return;
  }
  ensure_success(index_->UpdateIndex(), "SPTAG UpdateIndex");
}

void SPTAGCentroidNavigator::ensure_ready() const {
  if (!index_ || !index_->IsReady()) {
    throw std::runtime_error("SPTAG index is not ready");
  }
}

void SPTAGCentroidNavigator::configure_build_defaults(int nlist, int nthreads) {
  if (!index_) {
    return;
  }
  ensure_success(index_->SetParameter("DistCalcMethod", "L2"),
                 "SetParameter DistCalcMethod");
  ensure_success(index_->SetParameter("BKTKmeansK", "16"),
                 "SetParameter BKTKmeansK");
  ensure_success(index_->SetParameter("BKTLeafSize", "8"),
                 "SetParameter BKTLeafSize");
  ensure_success(index_->SetParameter("NeighborhoodSize", "64"),
                 "SetParameter NeighborhoodSize");
  ensure_success(index_->SetParameter("RefineIterations", "2"),
                 "SetParameter RefineIterations");
  if (nthreads > 0) {
    ensure_success(
        index_->SetParameter("NumberOfThreads", std::to_string(nthreads)),
        "SetParameter NumberOfThreads");
  }
  ensure_success(index_->SetParameter("HashTableExponent", "2"),
                 "SetParameter HashTableExponent");
  ensure_success(index_->SetParameter("MaxCheck", "2048"),
                 "SetParameter MaxCheck");
  ensure_success(index_->SetParameter("CEF", "256"), "SetParameter CEF");
  ensure_success(index_->SetParameter("TPTNumber", "16"),
                 "SetParameter TPTNumber");
  ensure_success(index_->SetParameter("TPTLeafSize", "2000"),
                 "SetParameter TPTLeafSize");
  const int sample_cap = std::max(32, std::min(nlist, 2048));
  ensure_success(index_->SetParameter("Samples", std::to_string(sample_cap)),
                 "SetParameter Samples");
}

void SPTAGCentroidNavigator::configure_search_defaults(int nprobe) const {
  if (!index_) {
    return;
  }

  ensure_success(index_->SetParameter("NumberOfThreads", "1"),
                 "SetParameter NumberOfThreads");

  ensure_success(index_->UpdateIndex(), "SPTAG UpdateIndex");

  // MaxCheck: 搜索时最多访问的节点数
  // 理论最小值 ≈ nprobe * 4 (假设图深度 ≈ 4 层)
  // 保守系数 8~14，取决于召回率要求
  // nprobe >= 128 时可以用较小系数，因为图搜索覆盖更广
  const int maxcheck_factor = (nprobe >= 128) ? 10 : (nprobe >= 64 ? 12 : 14);
  const int min_maxcheck = std::max(512, nprobe * 6); // 下限改为 nprobe * 6
  const int target_checks = nprobe * maxcheck_factor;
  const int tuned_maxcheck = std::clamp(target_checks, min_maxcheck, 4096);

  // CEF: 搜索时的候选边扩展因子
  const int cef_factor = (nprobe >= 128) ? 3 : 4;
  const int min_cef = std::max(128, nprobe); // 下限改为 nprobe
  const int target_cef = nprobe * cef_factor;
  const int tuned_cef = std::clamp(target_cef, min_cef, 512);

  ensure_success(
      index_->SetParameter("MaxCheck", std::to_string(tuned_maxcheck)),
      "SetParameter MaxCheck");
  ensure_success(index_->SetParameter("CEF", std::to_string(tuned_cef)),
                 "SetParameter CEF");
}
