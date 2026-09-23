"""Canonical experiment and presentation contract for the six paper figures."""

from __future__ import annotations

EF_RECALL = {
    "sift1b": {40: "0.88", 45: "0.90", 55: "0.92", 65: "0.94", 80: "0.96", 120: "0.98"},
    "deep1b": {60: "0.88", 70: "0.90", 85: "0.92", 105: "0.94", 145: "0.96", 255: "0.98"},
}
EF_RECALL["sift1m"] = EF_RECALL["sift1b"]
EF_RECALL["deep1m"] = EF_RECALL["deep1b"]

E2E_BUCKETS = ("0.90", "0.96")
E2E_DISPLAY = {"0.96": "0.95"}
FIG8_BUCKETS = ("0.90", "0.92", "0.94", "0.96")

PAPER_PW = {"Quiver": 2, "FlashANNS": 2, "GustANN": 1, "+S": 2}
PAPER_E2E_QPB = {"sift": (1, 2), "deep": (1, 2, 3, 4)}
PAPER_E2E_BATCH_LAT = {"sift": True, "deep": False}

QUIVER_LOWLAT_NB = (108, 216, 324)
QUIVER_MID_NB = (216, 324, 432, 540, 648, 756, 864, 972)
QUIVER_DEEP_EXTRA_NB = (864, 972)
FIG8_NUM_BLOCKS = (108, 216, 324, 432, 540, 648, 756, 864, 972)

# Exact MediaBox sizes of the submitted vector PDFs, in PostScript points.
FIGURE_SIZE_PT = {
    "latency_qps": (252.0, 108.252),
    "io_latency": (252.0, 108.0),
    "e2e": (518.4, 169.423),
    "fusion": (252.0, 86.4),
    "ablation": (324.0, 115.2),
    "q_sensitivity": (252.0, 100.8),
}


def figure_size(name: str) -> tuple[float, float]:
    width, height = FIGURE_SIZE_PT[name]
    return width / 72.0, height / 72.0
