#!/usr/bin/env python3
"""Parse Quiver/FlashANNS/GustANN/strawman search logs into sweep points.

Works on both the paper-era logs under `results/` and anything `fig_paper.sh`
writes, so the same plotter can draw the submitted figures and a fresh run.
"""
from __future__ import annotations

import re
import csv
from dataclasses import dataclass, field
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SWEEP = re.compile(r"^(\w+) Sweep:\s*(.*)$")
RESULT = re.compile(r"^=+ (.+?) Result =+$")
KV = re.compile(r"([A-Za-z_][\w.]*)=([\w.\-/]+)")
QPS = re.compile(r"^QPS:\s*([\d.]+)")
LAT = re.compile(r"^(Query|Batch)Latency\(ms\):\s*(.*)$")
QUIVER_BATCH = re.compile(r"^\[Quiver\] BatchLatency\(ms\):\s*(.*)$")
RECALL = re.compile(r"^Recall @ (\d+):\s*([\d.]+)")
# "  Compute      200    914.45   896.88  1017.36  1176.44  1374.37"
TECH = re.compile(r"^([A-Za-z][\w %./-]*?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$")

SYSTEM_ALIASES = {
    "Quiver": "Quiver",
    "FlashANNS": "FlashANNS",
    "GustANN": "GustANN",
    "PQInSmem": "+S",
    "PQInGmem": "+G",
    "Strawman": "+S",   # tech-breakdown logs title the block, not the sweep
    "Strawman PQ-in-shared-memory (PQ LUT in shared memory)": "+S",
}


@dataclass
class Point:
    path: Path
    system: str
    dataset: str
    ef: int
    qpb: int           # queries_per_block, 0 for the baselines
    pipe_width: int
    blocks: int        # num_blocks, or mini_batch for GustANN
    knob: str          # "num_blocks" | "mini_batch"
    qps: float
    recall: float
    query: dict = field(default_factory=dict)   # avg/p50/p90/p99/... from QueryLatency
    batch: dict = field(default_factory=dict)   # same from [Quiver] BatchLatency
    tech: dict = field(default_factory=dict)    # Tech-Analysis Samples rows
    stem: str = ""
    variant: str = ""

    def latency(self, stat: str, prefer_batch: bool = False) -> float:
        """One latency statistic, optionally preferring the BatchLatency table.

        Quiver logs carry both; the paper's e2e figure reads BatchLatency for
        SIFT and QueryLatency for DEEP, so the caller has to say which.
        """
        if prefer_batch and stat in self.batch:
            return self.batch[stat]
        if stat in self.query:
            return self.query[stat]
        return self.batch.get(stat, float("nan"))

    @property
    def q(self) -> int:
        """Query contexts per CTA, i.e. the Q the paper's figures label."""
        return self.qpb * self.pipe_width


def _floats(text: str) -> dict:
    return {k: float(v) for k, v in KV.findall(text) if _isnum(v)}


def _isnum(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _dataset(path: Path, data_type: str) -> str:
    low = str(path).lower()
    scale = "1m" if "1m" in low else "1b"
    return ("deep" if "deep" in low else "sift") + scale


def parse_log(path: Path) -> list[Point]:
    text = ANSI.sub("", path.read_text(errors="replace"))
    points: list[Point] = []
    sweep: dict = {}
    system = ""
    cur: dict | None = None
    tech_section = ""

    def flush(recall: float) -> None:
        if cur is None or "qps" not in cur:
            return
        knob = "mini_batch" if "mini_batch" in cur else "num_blocks"
        blocks = int(cur.get(knob, cur.get("batch", 0)))
        points.append(Point(
            path=path,
            system=SYSTEM_ALIASES.get(system, system),
            dataset=_dataset(path, cur.get("data_type", "")),
            ef=int(cur.get("ef_search", 0)),
            qpb=int(cur.get("queries_per_block", 0)),
            pipe_width=int(cur.get("pipe_width", 1)),
            blocks=blocks,
            knob=knob,
            qps=cur["qps"],
            recall=recall,
            query=cur.get("_query", {}),
            batch=cur.get("_batch", {}),
            tech=cur.get("_tech", {}),
        ))

    for raw in text.splitlines():
        line = raw.strip()
        m = SWEEP.match(line)
        if m:
            system = m.group(1)
            sweep = _floats(m.group(2))
            continue
        m = RESULT.match(line)
        if m:
            if not system:
                system = m.group(1).split()[0]
            cur = dict(sweep)
            tech_section = ""
            continue
        if line.startswith("========== Tech-Analysis Samples"):
            tech_section = line
            continue
        if cur is None:
            continue
        if line.startswith(("topk=", "num_blocks=", "batch=", "mini_batch=")):
            cur.update(_floats(line))
            if "data_type=" in line:
                cur["data_type"] = line.split("data_type=")[1].split()[0]
            continue
        m = QPS.match(line)
        if m:
            cur["qps"] = float(m.group(1))
            continue
        m = LAT.match(line)
        if m:
            cur["_batch" if m.group(1) == "Batch" else "_query"] = _floats(m.group(2))
            continue
        m = QUIVER_BATCH.match(line)
        if m:
            cur["_batch"] = _floats(m.group(1))
            continue
        if tech_section:
            m = TECH.match(line)
            if m:
                cur.setdefault("_tech", {})[m.group(1).strip()] = {
                    "count": int(m.group(2)), "avg": float(m.group(3)),
                    "p50": float(m.group(4)), "p90": float(m.group(5)),
                    "p99": float(m.group(6)), "max": float(m.group(7)),
                }
                continue
        m = RECALL.match(line)
        if m:
            flush(float(m.group(2)))
            cur = None

    # GustANN prints Recall once per sweep; a trailing block without one still
    # carries the Tech-Analysis table the breakdown figures need.
    if cur is not None and "qps" in cur:
        flush(float("nan"))
    return points


def load_tree(root: Path) -> list[Point]:
    metrics = [
        path for path in sorted(root.rglob("metrics.csv"))
        if _completed_group(path.parent)
    ]
    if metrics:
        return _dedupe(_load_csv_tree(metrics))
    breakdowns = sorted(root.rglob("breakdown.csv"))
    if breakdowns:
        return _load_breakdown_tree(breakdowns)
    out: list[Point] = []
    for log in sorted(root.rglob("*.log")):
        try:
            out.extend(parse_log(log))
        except Exception as exc:  # a truncated log must not kill the plot
            print(f"warn: {log}: {exc}")
    return out


def _completed_group(directory: Path) -> bool:
    """Ignore retained failed debug attempts; reviewer CSVs have no manifest."""
    manifest = directory / "manifest.env"
    if not manifest.is_file():
        return True
    values = {}
    for line in manifest.read_text(errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values.get("status") == "complete"


def _dedupe(points: list[Point]) -> list[Point]:
    """Keep the newest complete version of the same immutable sweep point."""
    latest: dict[tuple, Point] = {}
    for point in points:
        key = (
            point.stem, point.variant, point.system, point.dataset, point.ef,
            point.qpb, point.pipe_width, point.blocks, point.knob,
        )
        latest[key] = point
    return list(latest.values())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float((row.get(key) or "").strip())
    except ValueError:
        return default


def _load_breakdown_tree(paths: list[Path]) -> list[Point]:
    """Load instrumentation-only archives that predate metrics.csv."""
    grouped: dict[tuple[Path, str, str], dict] = {}
    metadata: dict[tuple[Path, str, str], dict[str, str]] = {}
    for path in paths:
        for row in _rows(path):
            stem = row.get("stem") or ""
            variant = row.get("variant") or ""
            metric = row.get("metric") or ""
            if not stem or not metric:
                continue
            key = (path, stem, variant)
            grouped.setdefault(key, {})[metric] = {
                "count": int(_number(row, "count")),
                "avg": _number(row, "avg_us"),
                "p50": _number(row, "p50_us"),
                "p90": _number(row, "p90_us"),
                "p99": _number(row, "p99_us"),
                "max": _number(row, "max_us"),
            }
            metadata[key] = row

    missing_latency = {
        stat: float("nan")
        for stat in ("avg", "p50", "p90", "p99", "p999", "max")
    }
    out: list[Point] = []
    for (path, stem, variant), tech in grouped.items():
        row = metadata[(path, stem, variant)]
        system = row.get("system") or ""
        out.append(Point(
            path=path,
            system=SYSTEM_ALIASES.get(system, system),
            dataset=row.get("dataset") or "",
            ef=0,
            qpb=0,
            pipe_width=1,
            blocks=0,
            knob="",
            qps=float("nan"),
            recall=float("nan"),
            query=dict(missing_latency),
            batch=dict(missing_latency),
            tech=tech,
            stem=stem,
            variant=variant,
        ))
    return out


def _load_csv_tree(metrics_paths: list[Path]) -> list[Point]:
    """Load the reviewer's normalized CSVs into the historical Point model."""
    out: list[Point] = []
    for metrics_path in metrics_paths:
        run = metrics_path.parent
        tech_by_key: dict[tuple[str, str], dict] = {}
        breakdown = run / "breakdown.csv"
        if breakdown.is_file():
            for row in _rows(breakdown):
                key = (row.get("stem") or "", row.get("variant") or "")
                metric = row.get("metric") or ""
                if not metric:
                    continue
                tech_by_key.setdefault(key, {})[metric] = {
                    "count": int(_number(row, "count")),
                    "avg": _number(row, "avg_us"),
                    "p50": _number(row, "p50_us"),
                    "p90": _number(row, "p90_us"),
                    "p99": _number(row, "p99_us"),
                    "max": _number(row, "max_us"),
                }
        cta = run / "cta_samples.csv"
        if cta.is_file():
            samples: dict[tuple[str, str], list[float]] = {}
            for row in _rows(cta):
                key = (row.get("stem") or "", row.get("variant") or "")
                samples.setdefault(key, []).append(_number(row, "cta_active_pct"))
            for key, values in samples.items():
                if values:
                    tech_by_key.setdefault(key, {})["CTA active %"] = {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "p50": 0.0,
                        "p90": 0.0,
                        "p99": 0.0,
                        "max": max(values),
                    }
        for row in _rows(metrics_path):
            stem = row.get("stem") or ""
            variant = row.get("variant") or ""
            num_blocks = int(_number(row, "num_blocks"))
            mini_batch = int(_number(row, "mini_batch"))
            latency = {
                stat: _number(row, f"{stat}_ms", float("nan"))
                for stat in ("avg", "p50", "p90", "p99", "p999", "max")
            }
            out.append(Point(
                path=metrics_path,
                system=SYSTEM_ALIASES.get(row.get("system") or "", row.get("system") or ""),
                dataset=row.get("dataset") or "",
                ef=int(_number(row, "ef")),
                qpb=int(_number(row, "queries_per_block")),
                pipe_width=int(_number(row, "pipe_width", 1.0)),
                blocks=num_blocks or mini_batch,
                knob="num_blocks" if num_blocks else "mini_batch",
                qps=_number(row, "qps"),
                recall=float("nan"),
                query=latency,
                batch=latency,
                tech=tech_by_key.get((stem, variant), {}),
                stem=stem,
                variant=variant,
            ))
    return out


def pareto(points: list[Point], stat: str, prefer_batch: bool = False) -> list[Point]:
    """Points no other point beats on both latency and throughput."""
    ordered = sorted(points, key=lambda p: (p.latency(stat, prefer_batch), -p.qps))
    front: list[Point] = []
    best = float("-inf")
    for p in ordered:
        lat = p.latency(stat, prefer_batch)
        if lat != lat or p.qps != p.qps:
            continue
        if p.qps > best:
            front.append(p)
            best = p.qps
    return front


def best_under(points: list[Point], cap_ms: float, stat: str = "p99",
               prefer_batch: bool = False) -> Point | None:
    """Highest throughput under a latency cap; the paper falls back to the
    lowest-latency point when nothing fits, so do the same."""
    if not points:
        return None
    ok = [p for p in points if p.latency(stat, prefer_batch) < cap_ms]
    if ok:
        return max(ok, key=lambda p: p.qps)
    return min(points, key=lambda p: p.latency(stat, prefer_batch))
