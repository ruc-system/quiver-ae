#pragma once

#ifdef QUIVER_LATENCY_PROBE

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <utility>
#include <vector>
#include "io_control.cuh"

namespace quiver {

inline double ns_to_us(int64_t ns) {
    return static_cast<double>(ns) / 1000.0;
}

inline double percentile(std::vector<double>& v, double pct) {
    if (v.empty()) return 0;
    size_t idx = (size_t)(v.size() * pct / 100.0);
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

struct QuiverBlockTimelineBreak {
    int block_id;
    int query_count;
    double total;
    double no_compute_idle;
    double active_compute;
    double idle_ratio_pct;
};

inline void print_metric_summary(const char* name, std::vector<double>& values)
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

inline void add_probe_interval(std::vector<std::pair<int64_t, int64_t>>& intervals,
                               int64_t start, int64_t end)
{
    if (start <= 0 && end <= 0) return;
    if (end <= start) return;
    intervals.push_back({start, end});
}

inline double merged_interval_us(std::vector<std::pair<int64_t, int64_t>>& intervals,
                                 int64_t timeline_start,
                                 int64_t timeline_end)
{
    if (intervals.empty() || timeline_end <= timeline_start) return 0.0;
    std::sort(intervals.begin(), intervals.end());
    int64_t active_ns = 0;
    int64_t cur_start = 0;
    int64_t cur_end = 0;
    bool have_current = false;

    for (auto interval : intervals) {
        int64_t start = std::max(interval.first, timeline_start);
        int64_t end = std::min(interval.second, timeline_end);
        if (end <= start) continue;
        if (!have_current) {
            cur_start = start;
            cur_end = end;
            have_current = true;
            continue;
        }
        if (start <= cur_end) {
            cur_end = std::max(cur_end, end);
        } else {
            active_ns += cur_end - cur_start;
            cur_start = start;
            cur_end = end;
        }
    }

    if (have_current) active_ns += cur_end - cur_start;
    return ns_to_us(active_ns);
}

inline std::vector<QuiverBlockTimelineBreak> collect_block_timeline_breaks(
    const QuiverProbeQueryTrace* traces, int num_traces, int queries_per_block)
{
    std::vector<QuiverBlockTimelineBreak> out;
    if (traces == nullptr || num_traces <= 0 || queries_per_block <= 0) {
        return out;
    }

    int max_block_id = -1;
    for (int i = 0; i < num_traces; i++) {
        if (traces[i].t_query_done == 0 || traces[i].query_id < 0) continue;
        max_block_id = std::max(max_block_id,
                                traces[i].query_id / queries_per_block);
    }
    if (max_block_id < 0) return out;

    for (int block_id = 0; block_id <= max_block_id; block_id++) {
        int query_count = 0;
        int64_t block_start = 0;
        int64_t block_end = 0;
        bool has_block_start = false;
        std::vector<std::pair<int64_t, int64_t>> active_intervals;

        for (int i = 0; i < num_traces; i++) {
            const auto& tr = traces[i];
            if (tr.t_query_done == 0 || tr.query_id < 0) continue;
            if (tr.query_id / queries_per_block != block_id) continue;

            query_count++;
            if (!has_block_start || tr.t_query_start < block_start) {
                block_start = tr.t_query_start;
                has_block_start = true;
            }
            block_end = std::max(block_end, tr.t_query_done);

            add_probe_interval(active_intervals, tr.t_query_start,
                               tr.t_boot_issue);
            add_probe_interval(active_intervals, tr.t_boot_ready,
                               tr.t_boot_compute_done);
            const int hops = std::min(tr.num_hops, PROBE_MAX_HOPS);
            for (int h = 0; h < hops; h++) {
                add_probe_interval(active_intervals, tr.hops[h].compute_start_ts,
                                   tr.hops[h].compute_done_ts);
            }
            add_probe_interval(active_intervals, tr.t_pipe_done,
                               tr.t_query_done);
        }

        if (query_count == 0 || block_end <= block_start) continue;
        QuiverBlockTimelineBreak b{};
        b.block_id = block_id;
        b.query_count = query_count;
        b.total = ns_to_us(block_end - block_start);
        b.active_compute = merged_interval_us(active_intervals, block_start,
                                              block_end);
        if (b.active_compute > b.total) b.active_compute = b.total;
        b.no_compute_idle = b.total - b.active_compute;
        if (b.no_compute_idle < 0) b.no_compute_idle = 0;
        b.idle_ratio_pct = (b.total > 0)
                               ? 100.0 * b.no_compute_idle / b.total
                               : 0.0;
        out.push_back(b);
    }

    return out;
}

inline void print_block_timeline_breakdown(
    const QuiverProbeQueryTrace* traces, int num_traces, int queries_per_block)
{
    std::vector<QuiverBlockTimelineBreak> timelines =
        collect_block_timeline_breaks(traces, num_traces, queries_per_block);
    if (timelines.empty()) {
        printf("\n=== Quiver Block Timeline Breakdown: no valid traces ===\n");
        return;
    }

    std::vector<double> total;
    std::vector<double> no_compute_idle;
    std::vector<double> active_compute;
    std::vector<double> idle_ratio_pct;
    total.reserve(timelines.size());
    no_compute_idle.reserve(timelines.size());
    active_compute.reserve(timelines.size());
    idle_ratio_pct.reserve(timelines.size());

    for (const auto& b : timelines) {
        total.push_back(b.total);
        no_compute_idle.push_back(b.no_compute_idle);
        active_compute.push_back(b.active_compute);
        idle_ratio_pct.push_back(b.idle_ratio_pct);
    }

    printf("\n=== Quiver Block Timeline Breakdown (%zu blocks, Q=%d) ===\n",
           timelines.size(), queries_per_block);
    printf("  Definition: block idle = no query slot has active compute work\n");
    printf("  Active intervals from same block are merged before computing idle\n");
    printf("  %-22s %10s %12s %12s %12s %12s %12s\n", "Metric", "count",
           "avg", "p50", "p90", "p99", "max");
    printf("  %-22s %10s %12s %12s %12s %12s %12s\n",
           "----------------------", "----------", "------------",
           "------------", "------------", "------------", "------------");
    print_metric_summary("BlockTotal(us)", total);
    print_metric_summary("BlockNoCompute(us)", no_compute_idle);
    print_metric_summary("BlockActive(us)", active_compute);
    print_metric_summary("BlockIdleRatio(%)", idle_ratio_pct);
    printf("=== End Quiver Block Timeline Breakdown ===\n\n");
}

inline std::vector<QuiverBlockTimelineBreak> collect_native_block_timeline_breaks(
    const QuiverProbeBlockTrace* traces, int num_traces)
{
    std::vector<QuiverBlockTimelineBreak> out;
    if (traces == nullptr || num_traces <= 0) return out;
    out.reserve(num_traces);

    for (int i = 0; i < num_traces; i++) {
        const auto& tr = traces[i];
        if (tr.t_block_done <= tr.t_block_start || tr.query_count <= 0) {
            continue;
        }
        QuiverBlockTimelineBreak b{};
        b.block_id = tr.block_id;
        b.query_count = tr.query_count;
        b.total = ns_to_us(tr.t_block_done - tr.t_block_start);
        b.no_compute_idle = ns_to_us(tr.no_ready_wait_ns);
        if (b.no_compute_idle > b.total) b.no_compute_idle = b.total;
        b.active_compute = b.total - b.no_compute_idle;
        if (b.active_compute < 0) b.active_compute = 0;
        b.idle_ratio_pct = (b.total > 0)
                               ? 100.0 * b.no_compute_idle / b.total
                               : 0.0;
        out.push_back(b);
    }

    return out;
}

inline void print_native_block_timeline_breakdown(
    const QuiverProbeBlockTrace* traces, int num_traces, int queries_per_block)
{
    std::vector<QuiverBlockTimelineBreak> timelines =
        collect_native_block_timeline_breaks(traces, num_traces);
    if (timelines.empty()) {
        printf("\n=== Quiver Native Block Timeline Breakdown: no valid traces ===\n");
        return;
    }

    std::vector<double> total;
    std::vector<double> no_compute_idle;
    std::vector<double> active_compute;
    std::vector<double> idle_ratio_pct;
    total.reserve(timelines.size());
    no_compute_idle.reserve(timelines.size());
    active_compute.reserve(timelines.size());
    idle_ratio_pct.reserve(timelines.size());

    for (const auto& b : timelines) {
        total.push_back(b.total);
        no_compute_idle.push_back(b.no_compute_idle);
        active_compute.push_back(b.active_compute);
        idle_ratio_pct.push_back(b.idle_ratio_pct);
    }

    printf("\n=== Quiver Native Block Timeline Breakdown (%zu blocks, Q=%d) ===\n",
           timelines.size(), queries_per_block);
    printf("  Definition: block idle = scheduler found no ready slot/lane and spun waiting for IO\n");
    printf("  Metric                      count          avg          p50          p90          p99          max\n");
    printf("  ---------------------- ---------- ------------ ------------ ------------ ------------ ------------\n");
    print_metric_summary("BlockTotal(us)", total);
    print_metric_summary("BlockNoReady(us)", no_compute_idle);
    print_metric_summary("BlockBusy(us)", active_compute);
    print_metric_summary("BlockIdleRatio(%)", idle_ratio_pct);
    printf("=== End Quiver Native Block Timeline Breakdown ===\n\n");
}

struct QuiverGlobalHopMetrics {
    std::vector<double> schedule_gap;
    std::vector<double> inter_hop_wait;
    std::vector<double> compute;
    std::vector<double> lifecycle;
};

inline void collect_global_hop_metrics(const QuiverProbeQueryTrace* traces,
                                       int num_traces,
                                       QuiverGlobalHopMetrics& out)
{
    size_t reserve_count = 0;
    for (int i = 0; i < num_traces; i++) {
        if (traces[i].t_query_done == 0) continue;
        reserve_count += std::min(traces[i].num_hops, PROBE_MAX_HOPS);
    }
    out.schedule_gap.reserve(reserve_count);
    out.inter_hop_wait.reserve(reserve_count);
    out.compute.reserve(reserve_count);
    out.lifecycle.reserve(reserve_count);

    for (int i = 0; i < num_traces; i++) {
        const auto& tr = traces[i];
        if (tr.t_query_done == 0) continue;
        int64_t prev_compute_done = tr.t_boot_compute_done;
        int hops = std::min(tr.num_hops, PROBE_MAX_HOPS);
        for (int h = 0; h < hops; h++) {
            const auto& hop = tr.hops[h];
            if (prev_compute_done <= 0 || hop.ready_found_ts <= 0 ||
                hop.compute_start_ts <= 0 || hop.compute_done_ts <= 0) {
                if (hop.compute_done_ts > 0) prev_compute_done = hop.compute_done_ts;
                continue;
            }
            if (hop.compute_start_ts >= hop.ready_found_ts) {
                out.schedule_gap.push_back(
                    ns_to_us(hop.compute_start_ts - hop.ready_found_ts));
            }
            if (hop.compute_start_ts >= prev_compute_done) {
                out.inter_hop_wait.push_back(
                    ns_to_us(hop.compute_start_ts - prev_compute_done));
            }
            if (hop.compute_done_ts >= hop.compute_start_ts) {
                out.compute.push_back(
                    ns_to_us(hop.compute_done_ts - hop.compute_start_ts));
            }
            if (hop.compute_done_ts >= prev_compute_done) {
                out.lifecycle.push_back(
                    ns_to_us(hop.compute_done_ts - prev_compute_done));
            }
            prev_compute_done = hop.compute_done_ts;
        }
    }
}

inline void print_global_hop_probe_breakdown(
    const QuiverProbeQueryTrace* traces, int num_traces)
{
    QuiverGlobalHopMetrics metrics;
    collect_global_hop_metrics(traces, num_traces, metrics);
    size_t total = metrics.compute.size();
    if (total == 0) {
        printf("\n=== Quiver GPU Hop Probe: no valid samples ===\n");
        return;
    }
    printf("\n=== Quiver GPU Per-Query Probe (%zu samples) ===\n", total);
    printf("  ScheduleGap: IO ready found -> compute start in persistent kernel\n");
    printf("  InterHopWait: previous query compute done -> next compute start\n");
    printf("  Compute: compute start -> compute done\n");
    printf("  Lifecycle: previous query compute done -> current compute done\n\n");

    printf("=== Quiver GPU Probe Global Metric Breakdown ===\n");
    printf("  %-18s %10s %12s %12s %12s %12s %12s\n", "Metric", "count",
           "avg_us", "p50_us", "p90_us", "p99_us", "max_us");
    printf("  %-18s %10s %12s %12s %12s %12s %12s\n",
           "------------------", "----------", "------------",
           "------------", "------------", "------------", "------------");
    print_metric_summary("ScheduleGap", metrics.schedule_gap);
    print_metric_summary("InterHopWait", metrics.inter_hop_wait);
    print_metric_summary("Compute", metrics.compute);
    print_metric_summary("Lifecycle", metrics.lifecycle);
    printf("=== End Quiver GPU Probe Global Metric Breakdown ===\n\n");
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

inline size_t quiver_cdf_point_limit() {
    const char* env = std::getenv("QUIVER_PROBE_CDF_POINTS");
    if (env == nullptr || env[0] == '\0') return 200;
    char* end = nullptr;
    unsigned long value = std::strtoul(env, &end, 10);
    if (end == env || value == 0) return 200;
    return static_cast<size_t>(value);
}

inline void write_cdf_points(std::ofstream& out, const char* metric,
                             std::vector<double>& values, size_t max_points) {
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

inline void write_probe_cdf_csv(const char* path,
                                const std::vector<double>& end_to_end_io_us)
{
    if (path == nullptr || path[0] == '\0') return;
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "Failed to open Quiver probe CDF CSV: %s\n", path);
        return;
    }
    const size_t max_points = quiver_cdf_point_limit();
    std::vector<double> values = end_to_end_io_us;
    out << "metric,value_us,cdf,total_count\n";
    write_cdf_points(out, "EndToEndIO", values, max_points);
    std::printf("Quiver preprocessed CDF points written to %s "
                "(max_points_per_metric=%zu)\n", path, max_points);
}

inline void print_end_to_end_io_breakdown(
    const std::vector<double>& end_to_end_io_us)
{
    if (end_to_end_io_us.empty()) {
        printf("\n=== Quiver IO Latency: no samples ===\n");
        return;
    }

    std::vector<double> values = end_to_end_io_us;
    printf("\n=== Quiver IO Latency (%zu IO samples) ===\n",
           values.size());
    printf("  EndToEndIO: CPU submit_direct -> SPDK callback complete\n");
    printf("  %-18s %10s %12s %12s %12s %12s %12s\n", "Metric", "count",
           "avg_us", "p50_us", "p90_us", "p99_us", "max_us");
    printf("  %-18s %10s %12s %12s %12s %12s %12s\n",
           "------------------", "----------", "------------",
           "------------", "------------", "------------", "------------");
    print_metric_summary("EndToEndIO", values);
    printf("=== End Quiver IO Latency ===\n\n");
}

inline void print_query_latency_breakdown(
    const QuiverProbeQueryTrace*, int)
{
    printf("\n=== Per-Query Latency Breakdown disabled by default for hop probe ===\n\n");
}

} // namespace quiver

#endif // QUIVER_LATENCY_PROBE
