#pragma once

#ifdef FLASHANNS_LATENCY_PROBE

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <vector>
#include "io_control.cuh"

namespace flashanns {

inline double ns_to_us(int64_t ns) {
    return static_cast<double>(ns) / 1000.0;
}

struct FlashQueryStageBreak {
    double init_query;
    double boot_io;
    double boot_pq;
    double boot_merge;
    double boot_sort;
    double boot_fill;
    double pipe_io;
    double pipe_pq;
    double pipe_merge;
    double pipe_sort;
    double pipe_issue;
    double conv_spin;
    double epilogue;
    double total;
    double other;
};

struct FlashBlockTimelineBreak {
    double total;
    double no_compute_idle;
    double active_compute;
    double idle_ratio_pct;
};

static constexpr int FLASH_NUM_STAGES = 15;

inline FlashQueryStageBreak compute_stage_break(const FlashProbeQueryTrace& tr)
{
    FlashQueryStageBreak b{};
    b.init_query  = ns_to_us(tr.t_init_done       - tr.t_query_start);
    b.boot_io     = ns_to_us(tr.t_boot_io_done    - tr.t_init_done);
    b.boot_pq     = ns_to_us(tr.t_boot_pq_done    - tr.t_boot_io_done);
    b.boot_merge  = ns_to_us(tr.t_boot_merge_done - tr.t_boot_pq_done);
    b.boot_sort   = ns_to_us(tr.t_boot_sort_done  - tr.t_boot_merge_done);
    b.boot_fill   = ns_to_us(tr.t_boot_fill_done  - tr.t_boot_sort_done);
    b.pipe_io     = ns_to_us(tr.pipe_io_ns);
    b.pipe_pq     = ns_to_us(tr.pipe_pq_ns);
    b.pipe_merge  = ns_to_us(tr.pipe_merge_ns);
    b.pipe_sort   = ns_to_us(tr.pipe_sort_ns);
    b.pipe_issue  = ns_to_us(tr.pipe_issue_ns);
    b.conv_spin   = ns_to_us(tr.t_pipe_done        - tr.t_boot_fill_done
                              - tr.pipe_io_ns - tr.pipe_pq_ns
                              - tr.pipe_merge_ns - tr.pipe_sort_ns
                              - tr.pipe_issue_ns);
    b.epilogue    = ns_to_us(tr.t_query_done       - tr.t_pipe_done);
    b.total       = ns_to_us(tr.t_query_done       - tr.t_query_start);

    double known = b.init_query + b.boot_io + b.boot_pq + b.boot_merge
                 + b.boot_sort + b.boot_fill
                 + b.pipe_io + b.pipe_pq + b.pipe_merge + b.pipe_sort
                 + b.pipe_issue + b.conv_spin + b.epilogue;
    b.other = b.total - known;
    return b;
}

inline FlashBlockTimelineBreak compute_block_timeline_break(
    const FlashProbeQueryTrace& tr)
{
    FlashBlockTimelineBreak b{};
    b.total = ns_to_us(tr.t_query_done - tr.t_query_start);
    b.no_compute_idle = ns_to_us((tr.t_boot_io_done - tr.t_init_done)
                                 + tr.pipe_io_ns);
    b.active_compute = b.total - b.no_compute_idle;
    if (b.active_compute < 0) b.active_compute = 0;
    b.idle_ratio_pct = (b.total > 0)
                           ? 100.0 * b.no_compute_idle / b.total
                           : 0.0;
    return b;
}

inline double percentile(std::vector<double>& v, double pct) {
    if (v.empty()) return 0;
    size_t idx = (size_t)(v.size() * pct / 100.0);
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

inline void print_hop_latency_breakdown(
    const FlashProbeQueryTrace* traces, int num_traces)
{
    if (num_traces <= 0) return;

    int valid = 0;
    int max_hop = -1;
    for (int i = 0; i < num_traces; i++) {
        if (traces[i].t_query_done == 0) continue;
        valid++;
        max_hop = std::max(max_hop, traces[i].num_hops - 1);
    }
    if (valid == 0 || max_hop < 0) {
        printf("\n=== FlashANNS Per-Hop Latency Breakdown: no valid samples ===\n");
        return;
    }

    std::vector<std::vector<double>> wait_by_hop(max_hop + 1);
    std::vector<std::vector<double>> compute_by_hop(max_hop + 1);
    for (int i = 0; i < num_traces; i++) {
        const auto& tr = traces[i];
        if (tr.t_query_done == 0) continue;
        int hops = std::min(tr.num_hops, FLASH_PROBE_MAX_HOPS);
        for (int h = 0; h < hops; h++) {
            const auto& hop = tr.hops[h];
            wait_by_hop[h].push_back(ns_to_us(hop.t_ready_found - hop.t_wait_start));
            compute_by_hop[h].push_back(ns_to_us(hop.t_issue_done - hop.t_ready_found));
        }
    }

    printf("\n=== FlashANNS Per-Hop Latency Breakdown (%d traces) ===\n", valid);
    printf("  %-6s %8s %12s %12s %12s %12s %12s %12s\n",
           "Hop", "count", "wait_avg", "wait_p90", "wait_p99",
           "compute_avg", "compute_p90", "compute_p99");
    printf("  %-6s %8s %12s %12s %12s %12s %12s %12s\n",
           "------", "--------", "------------", "------------",
           "------------", "------------", "------------", "------------");

    for (int h = 0; h <= max_hop; h++) {
        auto& waits = wait_by_hop[h];
        auto& computes = compute_by_hop[h];
        if (waits.empty() || computes.empty()) continue;
        std::sort(waits.begin(), waits.end());
        std::sort(computes.begin(), computes.end());
        double wait_avg = std::accumulate(waits.begin(), waits.end(), 0.0) / waits.size();
        double compute_avg = std::accumulate(computes.begin(), computes.end(), 0.0) / computes.size();
        printf("  %-6d %8zu %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f\n",
               h, waits.size(), wait_avg, percentile(waits, 90),
               percentile(waits, 99), compute_avg,
               percentile(computes, 90), percentile(computes, 99));
    }
    printf("=== End FlashANNS Per-Hop Latency Breakdown ===\n\n");
}

inline void print_metric_summary(const char* name, std::vector<double> values)
{
    if (values.empty()) {
        printf("  %-18s %10d %12.1f %12.1f %12.1f %12.1f %12.1f\n",
               name, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    std::sort(values.begin(), values.end());
    double avg = std::accumulate(values.begin(), values.end(), 0.0) /
                 values.size();
    printf("  %-18s %10zu %12.1f %12.1f %12.1f %12.1f %12.1f\n",
           name, values.size(), avg, percentile(values, 50),
           percentile(values, 90), percentile(values, 99), values.back());
}

inline void print_end_to_end_io_breakdown(
    const std::vector<double>& end_to_end_io_us)
{
    if (end_to_end_io_us.empty()) {
        printf("\n=== FlashANNS IO Latency: no samples ===\n");
        return;
    }

    printf("\n=== FlashANNS IO Latency (%zu IO samples) ===\n",
           end_to_end_io_us.size());
    printf("  EndToEndIO: CPU submit_task -> poll_task ready\n");
    printf("  %-18s %10s %12s %12s %12s %12s %12s\n", "Metric", "count",
           "avg_us", "p50_us", "p90_us", "p99_us", "max_us");
    printf("  %-18s %10s %12s %12s %12s %12s %12s\n",
           "------------------", "----------", "------------",
           "------------", "------------", "------------", "------------");
    print_metric_summary("EndToEndIO", end_to_end_io_us);
    printf("=== End FlashANNS IO Latency ===\n\n");
}

inline void print_block_timeline_breakdown(
    const FlashProbeQueryTrace* traces, int num_traces)
{
    if (num_traces <= 0) return;

    std::vector<double> total;
    std::vector<double> no_compute_idle;
    std::vector<double> active_compute;
    std::vector<double> idle_ratio_pct;
    total.reserve(num_traces);
    no_compute_idle.reserve(num_traces);
    active_compute.reserve(num_traces);
    idle_ratio_pct.reserve(num_traces);

    for (int i = 0; i < num_traces; i++) {
        if (traces[i].t_query_done == 0) continue;
        FlashBlockTimelineBreak b = compute_block_timeline_break(traces[i]);
        total.push_back(b.total);
        no_compute_idle.push_back(b.no_compute_idle);
        active_compute.push_back(b.active_compute);
        idle_ratio_pct.push_back(b.idle_ratio_pct);
    }

    if (total.empty()) {
        printf("\n=== FlashANNS Block Timeline Breakdown: no valid traces ===\n");
        return;
    }

    printf("\n=== FlashANNS Block Timeline Breakdown (%zu traces) ===\n",
           total.size());
    printf("  Definition: block idle = no ready lane, block spins waiting for IO\n");
    printf("  %-22s %10s %12s %12s %12s %12s %12s\n", "Metric", "count",
           "avg", "p50", "p90", "p99", "max");
    printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
           "----------------------", "----------", "------------",
           "------------", "------------", "------------", "------------");
    print_metric_summary("BlockTotal(us)", total);
    print_metric_summary("BlockNoCompute(us)", no_compute_idle);
    print_metric_summary("BlockActive(us)", active_compute);
    print_metric_summary("BlockIdleRatio(%)", idle_ratio_pct);
    printf("=== End FlashANNS Block Timeline Breakdown ===\n\n");
}

inline void append_completed_io_latencies(
    std::vector<double>& out,
    const std::vector<int64_t>& submit_ns,
    const std::vector<int64_t>& complete_ns)
{
    const size_t n = std::min(submit_ns.size(), complete_ns.size());
    out.reserve(out.size() + n);
    for (size_t i = 0; i < n; ++i) {
        if (submit_ns[i] <= 0 || complete_ns[i] <= submit_ns[i]) continue;
        out.push_back(static_cast<double>(complete_ns[i] - submit_ns[i]) /
                      1000.0);
    }
}

inline size_t flashanns_cdf_point_limit() {
    const char* env = std::getenv("FLASHANNS_PROBE_CDF_POINTS");
    if (env == nullptr || env[0] == '\0') return 200;
    char* end = nullptr;
    unsigned long value = std::strtoul(env, &end, 10);
    if (end == env || value == 0) return 200;
    return static_cast<size_t>(value);
}

inline void write_cdf_points(std::ofstream& out, const char* metric,
                             std::vector<double> values, size_t max_points) {
    if (values.empty()) return;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    const size_t points = std::min(n, max_points);
    size_t last_idx = static_cast<size_t>(-1);
    for (size_t p = 0; p < points; ++p) {
        size_t idx = p * n / points;
        if (idx == last_idx) continue;
        last_idx = idx;
        double cdf = static_cast<double>(idx + 1) / static_cast<double>(n);
        out << metric << "," << values[idx] << "," << cdf << "," << n << "\n";
    }
}

inline void write_end_to_end_io_cdf_csv(
    const char* path, const std::vector<double>& end_to_end_io_us)
{
    if (path == nullptr || path[0] == '\0') return;
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "Failed to open FlashANNS probe CDF CSV: %s\n", path);
        return;
    }
    out << "metric,value_us,cdf,total_count\n";
    write_cdf_points(out, "EndToEndIO", end_to_end_io_us,
                     flashanns_cdf_point_limit());
}

// ================================================================
// Per-query GPU latency breakdown with percentile table
// ================================================================
inline void print_query_latency_breakdown(
    const FlashProbeQueryTrace* traces, int num_traces,
    double cpu_h2d_sec, double cpu_nav_sec, double cpu_d2h_sec,
    double total_search_sec)
{
    if (num_traces <= 0) return;

    int valid = 0;
    for (int i = 0; i < num_traces; i++) {
        if (traces[i].t_query_done > 0) valid++;
    }
    if (valid == 0) {
        printf("\n=== FlashANNS Per-Query Latency Breakdown: no valid traces ===\n");
        return;
    }

    struct StageVec {
        const char* name;
        std::vector<double> vals;
        StageVec(const char* n, int cap) : name(n) { vals.reserve(cap); }
    };

    StageVec stages[] = {
        {"Init-Query",  valid}, {"Boot-IO",     valid},
        {"Boot-PQ",     valid}, {"Boot-Merge",  valid},
        {"Boot-Sort",   valid}, {"Boot-Fill",   valid},
        {"Pipe-IO",     valid}, {"Pipe-PQ",     valid},
        {"Pipe-Merge",  valid}, {"Pipe-Sort",   valid},
        {"Pipe-Issue",  valid}, {"Conv-Spin",   valid},
        {"Epilogue",    valid}, {"Other",       valid},
        {"GPU-Total",   valid},
    };

    for (int i = 0; i < num_traces; i++) {
        if (traces[i].t_query_done == 0) continue;
        FlashQueryStageBreak b = compute_stage_break(traces[i]);
        double* vals[] = {
            &b.init_query, &b.boot_io, &b.boot_pq, &b.boot_merge,
            &b.boot_sort, &b.boot_fill,
            &b.pipe_io, &b.pipe_pq, &b.pipe_merge, &b.pipe_sort,
            &b.pipe_issue, &b.conv_spin, &b.epilogue, &b.other,
            &b.total,
        };
        for (int s = 0; s < FLASH_NUM_STAGES; s++)
            stages[s].vals.push_back(*vals[s]);
    }

    for (auto& s : stages)
        std::sort(s.vals.begin(), s.vals.end());

    const int total_idx = FLASH_NUM_STAGES - 1;
    double avg_total = 0;
    for (auto& v : stages[total_idx].vals) avg_total += v;
    avg_total /= stages[total_idx].vals.size();

    double N = static_cast<double>(valid);
    double avg_h2d = cpu_h2d_sec * 1e6 / N;
    double avg_nav = cpu_nav_sec * 1e6 / N;
    double avg_d2h = cpu_d2h_sec * 1e6 / N;
    double avg_query_lat = total_search_sec * 1e6 / N;

    printf("\n=== FlashANNS Per-Query Latency Breakdown (%d traces) ===\n", valid);
    printf("  Avg query latency: %.2f us  (= total_time / num_queries)\n", avg_query_lat);
    printf("  Avg GPU-side query time: %.2f us\n", avg_total);

    printf("\n  CPU Stages (avg per query):\n");
    printf("    %-14s %10.2f us\n", "H2D", avg_h2d);
    printf("    %-14s %10.2f us\n", "Nav", avg_nav);
    printf("    %-14s %10.2f us\n", "D2H", avg_d2h);

    printf("\n  GPU Stage Breakdown:\n");
    printf("  %-14s %9s %9s %9s %9s %9s %7s\n",
           "Stage", "avg(us)", "p50(us)", "p90(us)", "p99(us)", "max(us)", "%avg");
    printf("  %-14s %9s %9s %9s %9s %9s %7s\n",
           "--------------", "---------", "---------", "---------",
           "---------", "---------", "-------");

    for (int s = 0; s < FLASH_NUM_STAGES; s++) {
        auto& v = stages[s].vals;
        double avg = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double p50 = percentile(v, 50);
        double p90 = percentile(v, 90);
        double p99 = percentile(v, 99);
        double mx  = v.back();
        double pct = (avg_total > 0) ? 100.0 * avg / avg_total : 0.0;

        if (s == FLASH_NUM_STAGES - 1) {
            printf("  %-14s %9s %9s %9s %9s %9s %7s\n",
                   "--------------", "---------", "---------", "---------",
                   "---------", "---------", "-------");
        }
        printf("  %-14s %9.1f %9.1f %9.1f %9.1f %9.1f %6.1f%%\n",
               stages[s].name, avg, p50, p90, p99, mx, pct);
    }
    printf("=== End FlashANNS Per-Query Latency Breakdown ===\n\n");
}

} // namespace flashanns

#endif // FLASHANNS_LATENCY_PROBE
