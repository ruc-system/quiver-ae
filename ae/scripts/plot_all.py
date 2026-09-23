#!/usr/bin/env python3
"""Redraw the six submitted figures from normalized CSVs or raw search logs.

Point the canonical plotter at the reviewer result tree or one completed
figure directory:

    ./ae/scripts/plot_all.py ae/results -o ae/figures
    ./ae/scripts/plot_all.py ae/results/e2e -o /tmp/paper-figs

`--mode paper` (default) applies the submitted selection rules; `--mode full`
uses every Q and pipe width present in the selected result directories.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

import paper_style as ps

logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib.tables._p_o_s_t").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_contract import (  # noqa: E402
    E2E_BUCKETS,
    E2E_DISPLAY,
    EF_RECALL,
    FIG8_BUCKETS,
    PAPER_E2E_BATCH_LAT,
    PAPER_E2E_QPB,
    PAPER_PW,
    figure_size,
)
from paper_logs import Point, best_under, load_tree, pareto  # noqa: E402

P99_CAP_MS = 10.0
QUIVER_PLOT_LAT_MS = 30.0
DATASET_ORDER = ("sift1b", "deep1b", "sift1m", "deep1m")

ABLATION_DISPLAY = {"FlashANNS": "FlashANNS", "+S": "+S", "Quiver": "+S+C"}
SYSTEM_DISPLAY = {"Quiver": "Quiver(ours)"}
TRADEOFF_CORNER = "upper left"
BAR_CORNER = "upper left"
# Paper-era host copies. Figure 7(b) reads Compute / kernel launch / Resume
# from this run; only H2D and D2H stay at these independently measured values.
PAPER_ABLATION_HOST_COPY_MS = {
    "GustANN": {"H2D": 0.240, "D2H": 0.240},
    "FlashANNS": {"H2D": 0.107, "D2H": 0.107},
    "+S": {"H2D": 0.107, "D2H": 0.107},
    "+S+C": {"H2D": 0.107, "D2H": 0.107},
}


class Corpus:
    def __init__(self, points: list[Point], mode: str,
                 quiver_grid: list[Point] | None = None,
                 root: Path | None = None, phase_debug: bool = False):
        self.points = points
        self.mode = mode
        self.quiver_grid = quiver_grid or []
        self.root = root
        self.phase_debug = phase_debug

    def select(self, system: str, dataset: str, ef: int,
               restrict_q: bool = False) -> list[Point]:
        """Sweep points for one (system, dataset, ef).

        `restrict_q` is for the e2e and latency_qps figures: SIFT keeps Q=1,2
        and DEEP keeps Q=1..4 (the two extra DEEP points at nb=864). The
        plotted curve is the Pareto front of whatever points are present.
        """
        out = [p for p in self.points
               if p.system == system and p.dataset == dataset and p.ef == ef
               and p.qps > 0 and p.blocks > 0 and not p.variant]
        if self.mode != "paper":
            return out
        pw = PAPER_PW.get(system)
        if pw is not None:
            out = [p for p in out if p.pipe_width == pw]
        if restrict_q and system == "Quiver":
            keep = PAPER_E2E_QPB.get(dataset[:4], (1, 2, 3, 4))
            out = [p for p in out if p.qpb in keep]
        return out

    def prefer_batch(self, dataset: str) -> bool:
        return self.mode == "paper" and PAPER_E2E_BATCH_LAT.get(dataset[:4], False)

    def ef_for(self, dataset: str, bucket: str) -> int | None:
        for ef, rec in EF_RECALL.get(dataset, {}).items():
            if rec == bucket:
                return ef
        return None

    def datasets(self) -> list[str]:
        have = {p.dataset for p in self.points}
        return [d for d in DATASET_ORDER if d in have]

    def sift(self) -> str | None:
        """The SIFT tree to draw. Every figure but e2e is SIFT-only."""
        have = {p.dataset for p in self.points}
        return next((d for d in ("sift1b", "sift1m") if d in have), None)


def draw_tradeoff(
    ax, corpus: Corpus, dataset: str, ef: int, systems, stat: str,
    *, drop_lowest_quiver: bool = False,
    latency_cap: float | None = None,
) -> bool:
    drawn = False
    xs: list[float] = []
    ys: list[float] = []
    for s in ps.order_systems(systems):
        pts = corpus.select(s, dataset, ef, restrict_q=True)
        if not pts:
            continue
        pb = corpus.prefer_batch(dataset) and s == "Quiver"
        front = pareto(pts, stat, pb)
        if drop_lowest_quiver and s == "Quiver" and len(front) > 1:
            lowest = min(front, key=lambda point: point.qps)
            front = [point for point in front if point is not lowest]
        if latency_cap is not None and s == "Quiver":
            front = [
                p for p in front if p.latency(stat, pb) <= latency_cap
            ]
        if not front:
            continue
        px = [p.latency(stat, pb) for p in front]
        py = [p.qps for p in front]
        ax.plot(px, py, label=SYSTEM_DISPLAY.get(s, s), **ps.line_style(s))
        xs.extend(px)
        ys.extend(py)
        drawn = True
    if drawn:
        ps.prepare_tradeoff(ax, xs, ys, TRADEOFF_CORNER)
    return drawn


def draw_quiver_grid(
    ax, corpus: Corpus, dataset: str, ef: int, stat: str,
    *, latency_cap: float | None = None,
) -> bool:
    """Overlay the full Q=1..4 occupancy grid on an e2e panel.

    Faint crosses show every measured point.  The dashed line is their Pareto
    front, so the overlay remains comparable to the submitted-system curves.
    """
    points = [
        point for point in corpus.quiver_grid
        if point.system == "Quiver"
        and point.dataset == dataset
        and point.ef == ef
        and point.qpb in (1, 2, 3, 4)
        and point.pipe_width == 2
        and point.qps > 0
        and point.blocks > 0
    ]
    if not points:
        return False

    prefer_batch = corpus.prefer_batch(dataset)
    if latency_cap is not None:
        points = [
            point for point in points
            if point.latency(stat, prefer_batch) <= latency_cap
        ]
        if not points:
            return False
    xs = [point.latency(stat, prefer_batch) for point in points]
    ys = [point.qps for point in points]
    color = ps.PALETTE[3]
    ax.scatter(xs, ys, marker="x", s=8, linewidths=0.5,
               color=color, alpha=0.35)

    front = pareto(points, stat, prefer_batch)
    ax.plot(
        [point.latency(stat, prefer_batch) for point in front],
        [point.qps for point in front],
        label="Quiver (Q=1–4 grid)",
        color=color,
        marker="D",
        linestyle="--",
        linewidth=1.0,
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.5,
    )

    all_x = list(xs)
    all_y = list(ys)
    for line in ax.lines:
        all_x.extend(line.get_xdata())
        all_y.extend(line.get_ydata())
    ps.prepare_tradeoff(ax, all_x, all_y, TRADEOFF_CORNER)
    return True


def panel_tag(index: int, text: str) -> str:
    return f"({chr(ord('a') + index)}) {text}"


def dataset_title(dataset: str) -> str:
    name = dataset.lower()
    family = "SIFT" if name.startswith("sift") else "DEEP"
    scale = "1B" if name.endswith("1b") else "1M"
    return f"{family}-{scale}"


def expand_ylim_above_ticks(ax, pad: float = 0.25, floor: float = 0.0) -> None:
    """Keep auto y ticks, then grow the axis so a top-left label is clear.

    If the current window does not start at `floor` (a tight auto-scale around
    a flat curve), recompute ticks on the 0-based range first. Otherwise the
    frozen ticks stay clustered at the top, which is what broke Figure 6(b).
    """
    lo, hi = ax.get_ylim()
    if lo > floor + 1e-9:
        ax.set_ylim(floor, hi)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, min_n_ticks=3))
        lo, hi = floor, hi
    ticks = [float(tick) for tick in ax.get_yticks() if math.isfinite(tick)]
    visible = [tick for tick in ticks if tick >= floor - 1e-9 and tick <= hi + 1e-9]
    if not visible:
        ax.set_ylim(floor, hi + pad * max(hi - lo, 1e-6))
        return
    last = max(visible)
    first = min(visible)
    span = max(last - first, abs(last) * 0.05, 1e-6)
    ax.set_ylim(floor, last + pad * span)
    ax.set_yticks(visible)


def complete_system_legend(systems) -> tuple[list, list[str]]:
    handles = []
    labels = []
    for system in systems:
        display = SYSTEM_DISPLAY.get(system, system)
        handles.append(Line2D([], [], label=display, **ps.line_style(system)))
        labels.append(display)
    return handles, labels


def complete_recall_legend() -> tuple[list, list[str]]:
    handles = []
    labels = []
    for index, bucket in enumerate(FIG8_BUCKETS):
        label = bucket
        handles.append(Line2D([], [], label=label, **ps.series_style(index)))
        labels.append(label)
    return handles, labels


def fig_latency_qps(corpus: Corpus, out: Path) -> bool:
    # Figure 1 is the SIFT1B/R@10=0.90/P99 panel from Figure 5 enlarged.
    ds = corpus.sift()
    if ds is None:
        return False
    ef = corpus.ef_for(ds, "0.90")
    if ef is None:
        return False
    fig, ax = plt.subplots(figsize=figure_size("latency_qps"))
    if not draw_tradeoff(
        ax, corpus, ds, ef, ("GustANN", "FlashANNS", "Quiver"), "p99",
        drop_lowest_quiver=True,
    ):
        plt.close(fig)
        return False
    # Put 10^1 left of center and 10^2 right of center, leaving a clear
    # lower-right band for the legend.
    ax.set_xlim(3.0, 300.0)
    bottom, _top = ax.get_ylim()
    ax.set_ylim(bottom, 160000)
    ax.set_yticks((50000, 100000))
    ax.text(
        0.03, 0.97, f"{dataset_title(ds)}, Recall@10=0.90",
        transform=ax.transAxes, ha="left", va="top",
        color="#8F3333", fontsize=9, fontproperties=ps.semibold_font(),
    )
    ax.text(0.14, 0.80, "Taming tradeoff", transform=ax.transAxes,
            color="#1F77B4", fontsize=9, fontproperties=ps.semibold_font())
    ax.annotate("", xy=(0.30, 0.65), xytext=(0.39, 0.76),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#1F77B4", "lw": 1.0})
    ax.text(0.31, 0.08, "Favor low latency", transform=ax.transAxes,
            color="#FF7F0E", fontsize=9, fontproperties=ps.semibold_font())
    ax.annotate("", xy=(0.30, 0.33), xytext=(0.42, 0.17),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#FF7F0E", "lw": 1.0})
    ax.text(0.66, 0.80, "Favor high QPS", transform=ax.transAxes,
            color="#2CA02C", fontsize=9, fontproperties=ps.semibold_font())
    ax.annotate("", xy=(0.70, 0.65), xytext=(0.77, 0.76),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#2CA02C", "lw": 1.0})
    handles, labels = complete_system_legend(
        ("GustANN", "FlashANNS", "Quiver")
    )
    ps.finish(
        fig, [[ax]], out,
        xlabel="P99 Latency (ms)",
        ylabel="Throughput",
        legends=[{"handles": handles, "labels": labels, "ax": ax,
                  "placement": "inside", "loc": "lower right", "frameon": True}],
    )
    return True


def fig_e2e(corpus: Corpus, out: Path) -> bool:
    datasets = corpus.datasets()
    cols = [(b, stat) for b in E2E_BUCKETS for stat in ("p99", "avg")]
    if not datasets:
        return False
    fig, axes = plt.subplots(
        len(datasets), len(cols), figsize=figure_size("e2e"), squeeze=False
    )
    drawn = False
    index = 0
    for i, ds in enumerate(datasets):
        for j, (bucket, stat) in enumerate(cols):
            ax = axes[i][j]
            ef = corpus.ef_for(ds, bucket)
            show_data = not corpus.phase_debug or (bucket == "0.90" and stat == "p99")
            if show_data and ef is not None and draw_tradeoff(
                    ax, corpus, ds, ef, ("FlashANNS", "GustANN", "Quiver"),
                    stat, latency_cap=QUIVER_PLOT_LAT_MS):
                drawn = True
            if show_data and ef is not None and draw_quiver_grid(
                    ax, corpus, ds, ef, stat, latency_cap=QUIVER_PLOT_LAT_MS):
                drawn = True
            disp = E2E_DISPLAY.get(bucket, bucket)
            title = dataset_title(ds).replace("-", "")
            ps.panel_label(ax, f"{title}, Recall={disp}", TRADEOFF_CORNER)
            if not show_data:
                ax.set_xscale("log")
                ax.set_xlim(3.0, 300.0)
                ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
                ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(
                    base=10, subs=tuple(range(2, 10))
                ))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax._paper_minor_ticks = True
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(ps.thousands)
            )
            if not (ax.lines or ax.collections):
                if ds.startswith("sift") and bucket == "0.90":
                    ax.set_ylim(0, 120000)
                    ax.set_yticks((0, 50000, 100000))
                elif ds.startswith("sift"):
                    ax.set_ylim(0, 70000)
                    ax.set_yticks((0, 20000, 40000, 60000))
                elif bucket == "0.90":
                    ax.set_ylim(0, 110000)
                    ax.set_yticks((0, 25000, 50000, 75000, 100000))
                else:
                    ax.set_ylim(0, 45000)
                    ax.set_yticks((0, 10000, 20000, 30000, 40000))
            expand_ylim_above_ticks(ax)
            index += 1
    if not drawn:
        plt.close(fig)
        return False
    handles, labels = complete_system_legend(
        ("FlashANNS", "GustANN", "Quiver")
    )
    ps.finish(
        fig, axes, out,
        xlabel=["P99 Latency (ms)" if stat == "p99" else "AVG Latency (ms)"
                for _bucket, stat in cols],
        ylabel="Throughput",
        legends=[{"handles": handles, "labels": labels, "ax": None}],
    )
    return True


def fig_ablation(corpus: Corpus, out: Path) -> bool:
    ds = corpus.sift()
    if ds is None:
        return False
    efs = sorted(EF_RECALL[ds])
    series = ("FlashANNS", "+S", "Quiver")
    bars: dict[str, list[float]] = {s: [] for s in series}
    for s in series:
        for ef in efs:
            if corpus.phase_debug and EF_RECALL[ds][ef] not in ("0.90", "0.96"):
                bars[s].append(0.0)
                continue
            candidates = [
                point for point in corpus.select(s, ds, ef)
                if not point.stem.startswith("b_")
            ]
            best = max(candidates, key=lambda point: point.qps) if candidates else None
            bars[s].append(best.qps if best else 0.0)
    if not any(any(v) for v in bars.values()):
        return False

    fig, axes = plt.subplots(
        1, 2, figsize=figure_size("ablation"), squeeze=False
    )
    axes = axes[0]
    width = 0.8 / len(series)
    base = list(range(len(efs)))
    heights: list[float] = []
    ablation_hatches = {
        "FlashANNS": "\\\\",
        "+S": "////",
    }
    for si, s in enumerate(series):
        offset = (si - (len(series) - 1) / 2) * width
        style = ps.bar_style(s)
        style["hatch"] = ablation_hatches.get(s, style["hatch"])
        axes[0].bar(
            [b + offset for b in base], bars[s], width,
            label=ABLATION_DISPLAY[s], **style,
        )
        heights.extend(bars[s])
    axes[0].set_xticks(base, [EF_RECALL[ds][e] for e in efs])
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].set_xlim(-0.55, len(efs) - 0.45)
    ps.prepare_bars(axes[0], heights, BAR_CORNER)
    ps.panel_label(axes[0], panel_tag(0, "Throughput"), BAR_CORNER)

    stacks, labels = breakdown_points(corpus)
    if stacks:
        parts = ("H2D", "kernel launch", "D2H", "Compute", "Resume")
        present = []
        for part in parts:
            vals = [s.get(part, 0.0) / 1000.0 for s in stacks]
            if not any(vals):
                continue
            present.append((part, vals))

        def draw_breakdown_stacks(ax, show_labels: bool) -> list[float]:
            bottom = [0.0] * len(labels)
            for pi, (part, vals) in enumerate(present):
                label, style = ps.part_style(part, pi)
                style["linewidth"] = 0.8
                ax.bar(
                    range(len(labels)), vals, width=0.52, bottom=bottom,
                    label=label if show_labels else "_nolegend_", **style,
                )
                bottom = [b + v for b, v in zip(bottom, vals)]
            return bottom

        bottom = draw_breakdown_stacks(axes[1], True)
        axes[1].set_xticks(range(len(labels)), labels)
        axes[1].tick_params(axis="x", labelrotation=90)
        axes[1].set_xlim(-0.55, len(labels) - 0.45)
        ps.prepare_bars(axes[1], bottom, BAR_CORNER)
        ps.panel_label(axes[1], panel_tag(1, "Latency"), BAR_CORNER)
        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")
        axes[1].set_ylabel("Latency (ms)", rotation=270, labelpad=10)

        zoom_ymax = 2.0
        axes[1].add_patch(Rectangle(
            (-0.38, 0.0), len(labels) - 0.24, zoom_ymax,
            fill=False, edgecolor="black", linewidth=0.8, zorder=5,
        ))
        zoom_ax = fig.add_axes([0.82, 0.08, 0.16, 0.31])
        draw_breakdown_stacks(zoom_ax, False)
        zoom_ax.set_xlim(-0.5, len(labels) - 0.5)
        zoom_ax.set_ylim(0.0, zoom_ymax)
        zoom_ax.set_xticks([])
        zoom_ax.set_yticks([0.0, zoom_ymax])
        zoom_ax.tick_params(axis="both", length=2.2, width=0.8, pad=1)
        ps.box_ax(zoom_ax)
        fig.add_artist(ConnectionPatch(
            xyA=(len(labels) - 0.24, zoom_ymax),
            coordsA=axes[1].transData,
            xyB=(-0.5, zoom_ymax),
            coordsB=zoom_ax.transData,
            color="black", linewidth=0.6,
        ))
        fig.add_artist(ConnectionPatch(
            xyA=(len(labels) - 0.24, 0.0),
            coordsA=axes[1].transData,
            xyB=(-0.5, 0.0),
            coordsB=zoom_ax.transData,
            color="black", linewidth=0.6,
        ))
    bar_handles = []
    for s in series:
        style = ps.bar_style(s)
        style["hatch"] = ablation_hatches.get(s, style["hatch"])
        style["facecolor"] = style.pop("color")
        bar_handles.append(Patch(label=ABLATION_DISPLAY[s], **style))
    part_handles = []
    part_labels = []
    for pi, part in enumerate(("H2D", "kernel launch", "D2H", "Compute", "Resume")):
        label, style = ps.part_style(part, pi)
        style["facecolor"] = style.pop("color")
        part_handles.append(Patch(label=label, **style))
        part_labels.append(label)
    fig.legend(
        part_handles, part_labels,
        loc="upper left", bbox_to_anchor=(0.83, 0.97),
        ncol=1, frameon=False, fontsize=9,
        borderaxespad=0.0, handlelength=1.25,
        handletextpad=0.35, labelspacing=0.15,
    )
    ps.finish(
        fig, [list(axes)], out,
        xlabel=["Recall@10", ""],
        ylabel=["Throughput", ""],
        legends=[
            {"handles": bar_handles,
             "labels": [ABLATION_DISPLAY[s] for s in series],
             "ax": axes[0], "placement": "inside", "loc": "upper right",
             "ncol": 1, "frameon": False, "fontsize": 9},
        ],
        widths=[2.1, 0.85],
        left_pad=0.08,
        right_pad=1.15,
        top_pad=0.08,
        bottom_pad=0.09,
        xlabel_down=[0.08, 0.0],
        ylabel_out=[0.07, 0.0],
    )
    return True


def breakdown_points(corpus: Corpus) -> tuple[list[dict], list[str]]:
    """The one operating point per system that the breakdown figures use.

    Compute, kernel launch, and Resume come from this run. Resume is the
    current IO-wait counter (the paper's stall stage). H2D / D2H stay at
    the paper-era host-copy measurements.
    """
    ds = corpus.sift()
    if ds is None:
        return [], []
    ef90 = corpus.ef_for(ds, "0.90")
    wanted = (
        ("GustANN", "GustANN", 48, 0),
        ("FlashANNS", "FlashANNS", 756, 0),
        ("+S", "+S", 324, 2),
        ("Quiver", "+S+C", 216, 2),
    )
    stacks, labels = [], []
    for system, label, blocks, qpb in wanted:
        cands = [p for p in corpus.points
                 if p.system == system and p.dataset == ds and p.ef == ef90
                 and p.stem.startswith("b_") and p.blocks == blocks
                 and p.qpb == qpb and "Compute" in p.tech]
        if not cands:
            continue
        chosen = max(cands, key=lambda point: point.qps)

        def tech_us(name: str) -> float:
            return float(chosen.tech.get(name, {}).get("avg", 0.0) or 0.0)

        host = PAPER_ABLATION_HOST_COPY_MS[label]
        stacks.append({
            "H2D": host["H2D"] * 1000.0,
            "D2H": host["D2H"] * 1000.0,
            "kernel launch": tech_us("kernel launch"),
            "Compute": tech_us("Compute"),
            "Resume": tech_us("IO wait") or tech_us("Resume"),
        })
        labels.append(label)
    return stacks, labels


def run_stamp(path: Path) -> str:
    m = re.search(r"\d{8}_\d{6}", str(path))
    return m.group(0) if m else ""


def fig_q_sensitivity(corpus: Corpus, out: Path) -> bool:
    ds = corpus.sift()
    if ds is None:
        return False
    fig, axes = plt.subplots(
        1, 2, figsize=figure_size("q_sensitivity"), squeeze=False
    )
    axes = axes[0]
    drawn = False
    # (a) Q on the x axis is queries_per_block * pipe_width. Q=1 is the
    # single-context case, which is exactly what FlashANNS runs. The
    # paper-selected occupancy is already the only point in the logs;
    # record that QPS as-is. Do not pick a max-QPS point or apply a 10 ms cap.
    q_axis = [1, 2, 4, 6, 8]
    pos = {q: i for i, q in enumerate(q_axis)}
    curve_y: list[float] = []
    for index, bucket in enumerate(FIG8_BUCKETS):
        if corpus.phase_debug and bucket != "0.90":
            continue
        ef = corpus.ef_for(ds, bucket)
        if ef is None:
            continue
        got: dict[int, float] = {}
        flash_pts = [p for p in corpus.select("FlashANNS", ds, ef) if p.qps > 0]
        if flash_pts:
            got[1] = flash_pts[0].qps
        by_q: dict[int, list[Point]] = defaultdict(list)
        for p in corpus.select("Quiver", ds, ef):
            if p.qps > 0 and p.q != 1:
                by_q[p.q].append(p)
        for q, pts in by_q.items():
            got[q] = pts[0].qps
        xs = [pos[q] for q in q_axis if q in got]
        ys = [got[q] for q in q_axis if q in got]
        # Recall 0.96 / Q=6,8 were left unmarked in the paper; keep those
        # two slots as crosses instead of inventing a latency-cap fallback.
        missing = [pos[q] for q in q_axis if q not in got]
        if len(xs) > 1:
            style = ps.series_style(index)
            line, = axes[0].plot(xs, ys, label=f"Recall@10= {bucket}", **style)
            if missing:
                mark_y = 4500 if bucket == "0.96" else min(ys) * 0.75
                axes[0].scatter(missing, [mark_y] * len(missing), marker="x",
                                s=18, color=line.get_color(), linewidths=0.6)
            curve_y.extend(ys)
            drawn = True
    axes[0].set_xticks(list(pos.values()), [str(q) for q in q_axis])
    axes[0].set_xlim(-0.45, len(q_axis) - 0.55)
    if curve_y:
        axes[0].set_ylim(0, 120000)
        axes[0].set_yticks((0, 40000, 80000, 120000))
        axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ps.thousands))
    ps.panel_label(axes[0], panel_tag(0, "Throughput"), BAR_CORNER)

    # (b) is a separate Quiver occupancy run: pipe-width=1, num-block=108,
    # Q = queries_per_block. The paper measured Q=1,2,4,6 this way and filled
    # Q=8; the selected AE path measures all five Q values the same way.
    ef90 = corpus.ef_for(ds, "0.90")
    util_labels: list[str] = []
    util_vals: list[float] = []
    for q in q_axis:
        qpts = [
            p for p in corpus.points
            if p.system == "Quiver" and p.dataset == ds and p.ef == ef90
            and p.q == q and p.pipe_width == 1 and "CTA active %" in p.tech
        ]
        if not qpts:
            qpts = [
                p for p in corpus.points
                if p.system == "Quiver" and p.dataset == ds and p.ef == ef90
                and p.q == q and "CTA active %" in p.tech
            ]
        if qpts:
            chosen = max(qpts, key=lambda p: p.tech["CTA active %"]["avg"])
            util_labels.append(str(q))
            util_vals.append(chosen.tech["CTA active %"]["avg"])
    if util_vals:
        axes[1].bar(
            range(len(util_vals)), util_vals,
            color="#FDD9A6", hatch="//", edgecolor="black", linewidth=1.2,
        )
        axes[1].set_xticks(range(len(util_labels)), util_labels)
        axes[1].set_xlim(-0.55, len(util_labels) - 0.45)
        axes[1].set_ylim(0, 105)
        axes[1].set_yticks((0, 50, 100))
        axes[1].yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}%")
        )
        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")
        drawn = True
    # The right panel is narrow and its 96--98% labels occupy the top edge;
    # the y-axis already names the metric, so keep only the panel tag here.
    ps.panel_label(axes[1], "(b)", BAR_CORNER)
    if not drawn:
        plt.close(fig)
        return False
    handles, labels = complete_recall_legend()
    prefix = Line2D([], [], linestyle="none", marker=None, color="none")
    ps.finish(
        fig, [list(axes)], out,
        xlabel="Q",
        ylabel=["Throughput", "Utilization"],
        ylabel_side=["left", "right"],
        legends=[{
            "handles": [prefix] + handles,
            "labels": ["Recall@10="] + labels,
            "ax": axes[0],
            "ncol": 5,
            "fontsize": 9,
            "handlelength": 0.45,
            "handletextpad": 0.12,
            "columnspacing": 0.25,
            "hide_first": True,
        }],
        widths=[2.15, 1.0],
        subplot_gap=0.02,
        top_pad=0.04,
    )
    return True


def fig_io_latency(corpus: Corpus, out: Path) -> bool:
    systems = ("FlashANNS", "GustANN", "Quiver")
    expected_blocks = {"FlashANNS": 756, "GustANN": 1120, "Quiver": 216}
    chosen = {}
    for s in systems:
        cands = [p for p in corpus.points
                 if p.system == s and "CTA active %" in p.tech and "Compute" in p.tech
                 and p.tech.get("IO wait", {}).get("avg", 0.0) > 0.0
                 and p.blocks == expected_blocks[s]]
        if cands:
            chosen[s] = max(cands, key=lambda p: (run_stamp(p.path), p.path.name))
    labels = [s for s in systems if s in chosen]
    util = [chosen[s].tech["CTA active %"]["avg"] for s in labels]
    if not util:
        return False
    fig, axes = plt.subplots(
        1, 3, figsize=figure_size("io_latency"), squeeze=False
    )
    axes = axes[0]
    for ax in axes:
        ax._paper_tick_labelsize = 7
    utilization_hatches = ("///", "xx", "..")
    for i, (label, value) in enumerate(zip(labels, util)):
        style = ps.bar_style(label, palette="utilization")
        style["hatch"] = utilization_hatches[i]
        axes[0].bar(i, value, **style)
        axes[0].text(i, value, f"{value:.0f}%", ha="center", va="bottom", fontsize=7,
                     color="#963735")
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].set_xlim(-0.55, len(labels) - 0.45)
    axes[0].set_ylim(0, 120)
    axes[0].set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ps.panel_label(axes[0], "(a)", "upper left")
    # Read only the samples beside each selected (latest complete) point.
    # Recursing from a persistent debug root mixes failed/older attempts into
    # the same CDF, which makes a rerun change the meaning of the figure.
    hop_rows: dict[str, list[dict[str, str]]] = {}
    for system, point in chosen.items():
        path = point.path.parent / "hop_samples.csv"
        if path.is_file():
            rows = _read_csv(path)
            hop_rows[system] = [
                row for row in rows
                if (not row.get("system") or row["system"] == system)
                and (not point.stem or not row.get("stem")
                     or row["stem"] == point.stem)
            ]

    def steady_samples(system: str, field: str) -> list[float]:
        values = []
        for row in hop_rows.get(system, []):
            try:
                hop = int(row.get("hop", "-1"))
            except (TypeError, ValueError):
                continue
            value = _fnum(row, field)
            if hop > 0 and math.isfinite(value) and value >= 0.0:
                values.append(value)
        return sorted(values)

    def per_hop_means(system: str, field: str) -> list[float]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in hop_rows.get(system, []):
            try:
                hop = int(row.get("hop", "-1"))
            except (TypeError, ValueError):
                continue
            value = _fnum(row, field)
            # Hop zero contains boot/entry-point work, not a steady graph hop.
            if hop > 0 and math.isfinite(value) and value >= 0.0:
                grouped[hop].append(value)
        if not grouped:
            return []
        max_count = max(len(values) for values in grouped.values())
        min_count = max(10, math.ceil(max_count * 0.1))
        return sorted(
            sum(values) / len(values)
            for values in grouped.values()
            if len(values) >= min_count
        )

    # Fresh AE runs provide empirical per-hop samples. Paper-era logs kept only
    # percentiles, so retain the summary fallback for the archived tree.
    for ax, metric, title, index in (
        (axes[1], "IO wait", "I/O stall", 1),
        (axes[2], "Compute", "Computing", 2),
    ):
        xs_all: list[float] = []
        # Figure 3(b) compares all schedulers. Figure 3(c) illustrates the
        # stability of Quiver's steady per-hop compute work with one curve.
        plot_systems = labels if metric == "IO wait" else ("Quiver",)
        for s in plot_systems:
            field = "wait_us" if metric == "IO wait" else "compute_us"
            # Stall variability is the phenomenon panel (b) compares, so keep
            # each steady query-hop observation. Panel (c) asks whether the
            # work of graph hop N is predictable; average query instances at
            # the same hop before constructing that single Quiver CDF.
            samples = (
                steady_samples(s, field)
                if metric == "IO wait"
                else per_hop_means(s, field)
            )
            if metric == "IO wait":
                samples = [value / 1000.0 for value in samples]
            if samples:
                xs = samples
                ys = [(i + 1) / len(samples) for i in range(len(samples))]
            else:
                t = chosen[s].tech.get(metric)
                if not t:
                    continue
                scale = 1000.0 if metric == "IO wait" else 1.0
                xs = [t["p50"] / scale, t["p90"] / scale,
                      t["p99"] / scale, t["max"] / scale]
                ys = [0.50, 0.90, 0.99, 1.0]
            ax.plot(
                xs, ys, label=SYSTEM_DISPLAY.get(s, s),
                markevery=max(1, len(xs) // 12), **ps.line_style(s)
            )
            xs_all.extend(xs)
        if xs_all:
            ordered = sorted(xs_all)
            visible_index = min(
                len(ordered) - 1, int(0.995 * len(ordered))
            )
            lo, hi = min(xs_all), ordered[visible_index]
            span = max(hi - lo, 1.0)
            left = 0.0 if metric == "Compute" else max(0.0, lo - 0.08 * span)
            ax.set_xlim(left, hi + 0.12 * span)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(True, linestyle=(0, (1, 5)), linewidth=0.6, color="black")
        ax._paper_keep_grid = True
        ps.panel_label(
            ax, f"({chr(ord('a') + index)})", "upper left",
            x_offset=0.07 if index == 1 else 0.0,
        )
    handles, legend_labels = ps.legend_entries(
        [axes[1], axes[2]], ("GustANN", "FlashANNS", "Quiver")
    )
    axes[1].set_yticklabels([])
    axes[2].yaxis.tick_right()
    axes[2].yaxis.set_label_position("right")
    axes[2].set_ylabel("CDF", rotation=270, labelpad=8)
    ps.finish(
        fig, [list(axes)], out,
        xlabel=["", "Per-hop I/O\nstall time (ms)",
                "Per-hop computing\ntime (µs)"],
        ylabel=["Utilization", "", ""],
        legends=[{"handles": handles, "labels": legend_labels, "ax": axes[1],
                  "placement": "inside", "loc": "lower right",
                  "bbox_to_anchor": (1.05, 0.02),
                  "frameon": False, "fontsize": 6}],
        widths=[1.0, 1.45, 0.72],
        right_pad=0.30,
        top_pad=0.08,
        bottom_pad=0.05,
        xlabel_fontsize=7,
        xlabel_down=[0.0, 0.05, 0.05],
    )
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _pair_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    front: list[tuple[float, float]] = []
    best_qps = -math.inf
    for latency, qps in sorted(points):
        if latency > 0 and qps > best_qps:
            front.append((latency, qps))
            best_qps = qps
    return front


def fig_fusion(corpus: Corpus, out: Path) -> bool:
    """Single-SSD Quiver 1B vs FusionANNS 100M; plot the Pareto front."""
    root = corpus.root
    if root is None:
        return False
    summaries: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in root.rglob("fusionanns_*_summary.csv"):
        name = path.name.lower()
        family = "deep" if "deep" in name else "sift"
        summaries[family].extend(_read_csv(path))

    fig, axes = plt.subplots(1, 2, figsize=figure_size("fusion"), squeeze=False)
    axes = axes[0]
    drawn = False
    for index, family in enumerate(("sift", "deep")):
        ax = axes[index]
        dataset = next(
            (name for name in (f"{family}1b", f"{family}1m")
             if any(p.dataset == name for p in corpus.points)),
            f"{family}1b",
        )
        qpoints = [
            p for p in corpus.points
            if p.system == "Quiver" and p.dataset == dataset
            and "_1ssd" in p.stem and p.qps > 0
        ]
        qpoints = [
            p for p in qpoints
            if p.latency("p99", corpus.prefer_batch(dataset)) <= QUIVER_PLOT_LAT_MS
        ]
        qfront = pareto(qpoints, "p99", corpus.prefer_batch(dataset))
        if qfront:
            ax.plot(
                [p.latency("p99", corpus.prefer_batch(dataset)) for p in qfront],
                [p.qps for p in qfront],
                label="Quiver(ours)",
                **ps.line_style("Quiver"),
            )
            drawn = True

        fpoints = []
        for row in summaries.get(family, []):
            latency = _fnum(row, "p99_ms")
            qps = _fnum(row, "qps_wall")
            if math.isfinite(latency) and math.isfinite(qps):
                fpoints.append((latency, qps))
        ffront = _pair_front(fpoints)
        if ffront:
            ax.plot(
                [point[0] for point in ffront],
                [point[1] for point in ffront],
                label="FusionANNS",
                **ps.line_style("FusionANNS"),
            )
            drawn = True
        all_x = [line.get_xdata() for line in ax.lines]
        all_y = [line.get_ydata() for line in ax.lines]
        flat_x = [value for values in all_x for value in values]
        flat_y = [value for values in all_y for value in values]
        if flat_x:
            ps.prepare_tradeoff(ax, flat_x, flat_y, "upper left")
            expand_ylim_above_ticks(ax)
        ps.panel_label(ax, f"{family.upper()}1B", "upper left")

    if not drawn:
        plt.close(fig)
        return False
    handles, labels = complete_system_legend(("FusionANNS", "Quiver"))
    ps.finish(
        fig, [list(axes)], out,
        xlabel="P99 Latency (ms)",
        ylabel="Throughput",
        legends=[{"handles": handles, "labels": labels, "ax": None}],
        left_pad=0.08,
        ylabel_out=0.06,
    )
    return True


# Values read off the submitted PDFs (recovered from the pre-AE plot scripts),
# so `--verify` can say which bars and points a tree actually reproduces.
SUBMITTED_ABLATION = {
    "FlashANNS": {40: 68529.50, 45: 58803.31, 55: 51479.12,
                  65: 39633.64, 80: 30674.68, 120: 14109.73},
    "+S": {40: 82876.13, 45: 73771.83, 55: 61087.22,
           65: 52977.83, 80: 35669.08, 120: 23293.84},
    "Quiver": {40: 98590.07, 45: 92716.63, 55: 75881.63,
               65: 59088.07, 80: 44834.81, 120: 27757.88},
}
SUBMITTED_Q = {
    "0.90": {1: 58803.31, 2: 80204.87, 4: 92716.63, 6: 89841.90, 8: 70600.18},
    "0.92": {1: 51479.12, 2: 70641.60, 4: 75881.63, 6: 63517.02, 8: 59402.58},
    "0.94": {1: 39633.64, 2: 57805.21, 4: 59088.07, 6: 51135.83, 8: 47736.10},
    "0.96": {1: 30674.68, 2: 44834.81, 4: 35450.38},
}


def verify(corpus: Corpus) -> None:
    ds = corpus.sift()
    if ds is None:
        return
    ok = bad = 0

    def cmp(tag: str, got: float | None, want: float | None) -> None:
        nonlocal ok, bad
        if got is not None and want is not None and abs(got - want) < 1.0:
            ok += 1
            return
        if got is None and want is None:
            ok += 1
            return
        bad += 1
        g = "-" if got is None else f"{got:.2f}"
        w = "-" if want is None else f"{want:.2f}"
        print(f"  MISMATCH {tag}: this tree={g}  submitted={w}")

    print("Figure 7(a) bars:")
    for system, wanted in SUBMITTED_ABLATION.items():
        for ef, want in wanted.items():
            best = best_under(corpus.select(system, ds, ef), P99_CAP_MS, "p99")
            cmp(f"7a {system} ef={ef}", best.qps if best else None, want)

    print("Figure 8(a) points:")
    for bucket, wanted in SUBMITTED_Q.items():
        ef = corpus.ef_for(ds, bucket)
        got: dict[int, float] = {}
        flash = best_under(corpus.select("FlashANNS", ds, ef), P99_CAP_MS, "p99")
        if flash and flash.latency("p99") < P99_CAP_MS:
            got[1] = flash.qps
        by_q: dict[int, list[Point]] = defaultdict(list)
        for p in corpus.select("Quiver", ds, ef):
            by_q[p.q].append(p)
        for q, pts in by_q.items():
            best = best_under(pts, P99_CAP_MS, "p99")
            if best and best.latency("p99") < P99_CAP_MS:
                got[q] = best.qps
        for q in (1, 2, 4, 6, 8):
            cmp(f"8a R@10={bucket} Q={q}", got.get(q), wanted.get(q))
    print(f"matched {ok}/{ok + bad} reference values")


FIGURES = (
    (1, "latency_qps", fig_latency_qps),
    (3, "io_latency", fig_io_latency),
    (5, "e2e", fig_e2e),
    (6, "fusion", fig_fusion),
    (7, "ablation", fig_ablation),
    (8, "q_sensitivity", fig_q_sensitivity),
)


def figure_source(root: Path, name: str) -> Path | None:
    if root.name == name:
        return root
    if root.parent.name == name and (root / "env.txt").is_file():
        return root
    direct = root / name
    if direct.is_dir():
        return direct
    if name == "latency_qps":
        e2e = root / "e2e"
        if e2e.is_dir():
            return e2e
    return None


def requested_figures(root: Path):
    by_name = {name: (number, name, fn) for number, name, fn in FIGURES}
    if root.name in by_name:
        return (by_name[root.name],)
    if root.parent.name in by_name and (root / "env.txt").is_file():
        return (by_name[root.parent.name],)
    return FIGURES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_src = os.environ.get("AE_DEBUG_ROOT")
    ap.add_argument(
        "src", type=Path, nargs="?",
        default=Path(default_src) if default_src else
        Path(__file__).resolve().parent.parent / "results",
        help="result root (defaults to AE_DEBUG_ROOT or ae/results)",
    )
    ap.add_argument("-o", "--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "figures")
    ap.add_argument("--mode", choices=("paper", "full"), default="paper")
    ap.add_argument("--quiver-grid", type=Path,
                    help="overlay a Q=1..4 Quiver grid on matching e2e panels")
    ap.add_argument("--verify", action="store_true",
                    help="diff Figure 7(a)/8(a) against the submitted values")
    ap.add_argument(
        "--phase-debug", action="store_true",
        help="keep the submitted layouts but draw only the current reduced data",
    )
    args = ap.parse_args()

    if not args.src.exists():
        print(f"missing {args.src}", file=sys.stderr)
        return 1

    jobs = []
    all_points: list[Point] = []
    for number, name, fn in requested_figures(args.src):
        source = figure_source(args.src, name)
        if source is None:
            print(f"skip Figure {number} ({name}): missing result directory")
            continue
        points = load_tree(source)
        if args.phase_debug and name == "ablation":
            e2e_source = source.parent / "e2e"
            if e2e_source.is_dir():
                points.extend(load_tree(e2e_source))
        jobs.append((number, name, fn, source, points))
        all_points.extend(points)
    if not jobs:
        print(f"no completed figure results under {args.src}", file=sys.stderr)
        return 1
    print(f"parsed {len(all_points)} sweep points from "
          f"{len(jobs)} completed figure result(s)  (mode={args.mode})")

    grid_points: list[Point] = []
    if args.quiver_grid is not None:
        if not args.quiver_grid.exists():
            print(f"missing {args.quiver_grid}", file=sys.stderr)
            return 1
        grid_points = load_tree(args.quiver_grid)
        print(f"overlaying {len(grid_points)} Quiver grid points from "
              f"{args.quiver_grid}")
    corpus = Corpus(
        all_points, args.mode, grid_points, args.src, args.phase_debug
    )
    if args.verify:
        verify(corpus)
        print()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for number, name, fn, source, points in jobs:
        ps.apply_style(submitted_gill=name == "latency_qps")
        figure_corpus = Corpus(
            points, args.mode, grid_points, source, args.phase_debug
        )
        pdf = args.out_dir / f"figure{number}_{name}.pdf"
        if fn(figure_corpus, pdf):
            print(f"paper Figure {number} ({name}) -> {pdf}")
            n += 1
        else:
            print(f"skip Figure {number} ({name}): no data for it under {args.src}")
    return 0 if n else 2


if __name__ == "__main__":
    raise SystemExit(main())
