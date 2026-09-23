"""Paper-figure style shared by the ae plotters.

Vector PDF, 9pt Arial, boxed axes, legend above the panels, primary system
first. Layout is in inches so the legend gap, subplot gap, and margins stay
fixed instead of following tight_layout.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
GILL_REGULAR = FONT_DIR / "GillSans.ttc"
GILL_SEMIBOLD = FONT_DIR / "GillSans-SemiBold.ttf"

# Exact line colors used by the submitted PDFs.
PALETTE = ("#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD")
RECALL_PALETTE = ("#D95F02", "#1F78B4", "#33A02C", "#6A3D9A")
MARKERS = ("o", "^", "v", "s", "D")
HATCHES = ("---", "|||", "///", "\\\\\\")
SYSTEM_ORDER = ("Quiver", "FlashANNS", "GustANN", "+S", "FusionANNS")
SYSTEM_STYLE = {
    "Quiver": {"color": PALETTE[0], "marker": "o", "hatch": HATCHES[0]},
    "FlashANNS": {"color": PALETTE[1], "marker": "^", "hatch": HATCHES[1]},
    "GustANN": {"color": PALETTE[2], "marker": "v", "hatch": HATCHES[2]},
    "+S": {"color": PALETTE[3], "marker": "s", "hatch": HATCHES[3]},
    "FusionANNS": {"color": "#9467BD", "marker": "D", "hatch": HATCHES[3]},
}
# Latency-breakdown stacks. Hatches keep the parts distinct in grayscale.
PART_STYLE = (
    ("H2D", "#B3CDE3", "//"),
    ("kernel launch", "#FBB4AE", "||"),
    ("D2H", "#DECBE4", ".."),
    ("Compute", "#FDD9A6", "----"),
    ("IO wait", "#FDD9A6", "||"),
    ("Resume", "#CCEBC5", "xx"),
)
PART_LABEL = {
    "Compute": "Compute",
    "IO wait": "I/O wait",
    "Resume": "Resume",
    "kernel launch": "Launch",
    "H2D": "H2D",
    "D2H": "D2H",
}

SINGLE = (3.5, 1.5)
DOUBLE_W = 7.0
OUTER = 0.01
LEGEND_GAP = 0.0025
XLABEL_GAP = 0.005
SUBPLOT_GAP = 0.04

_CORNER = {
    "upper left": (0.03, 0.97, "left", "top"),
    "upper right": (0.97, 0.97, "right", "top"),
    "lower left": (0.03, 0.03, "left", "bottom"),
    "lower right": (0.97, 0.03, "right", "bottom"),
}


def size_for(ncols: int, nrows: int) -> tuple[float, float]:
    """Single column for one panel; double column when panels sit side by side.

    A second row cannot fit in 1.5in with 9pt ticks, so each extra row adds
    a plot band. One row stays at 1.5in.
    """
    width = SINGLE[0] if ncols <= 1 else DOUBLE_W
    height = SINGLE[1] if nrows <= 1 else SINGLE[1] + (nrows - 1) * 1.2
    return width, height


def require_fonts(need_cjk: bool = False) -> None:
    """Use Arial for Latin text. Do not substitute another family."""
    from matplotlib import font_manager

    arial = Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
    if not arial.is_file():
        found = list(Path("/usr/share/fonts").rglob("Arial.ttf"))
        arial = found[0] if found else None
    if arial is None:
        sys.exit(
            "Arial is not installed, so the figure was not written.\n"
            "On Ubuntu: sudo apt install ttf-mscorefonts-installer"
        )
    font_manager.fontManager.addfont(str(arial))
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" not in names:
        sys.exit(
            "Arial is not installed, so the figure was not written.\n"
            "On Ubuntu: sudo apt install ttf-mscorefonts-installer"
        )
    if need_cjk and "SimHei" not in names:
        sys.exit(
            "SimHei is not installed, so the figure was not written.\n"
            "Install SimHei (黑体) and retry."
        )


def apply_style(submitted_gill: bool = False) -> None:
    from matplotlib import font_manager

    require_fonts(need_cjk=False)
    families = ["Arial", "SimHei"]
    if submitted_gill:
        for font in (GILL_REGULAR, GILL_SEMIBOLD):
            if not font.is_file():
                sys.exit(f"missing bundled paper font: {font}")
            font_manager.fontManager.addfont(str(font))
        regular_name = font_manager.FontProperties(fname=GILL_REGULAR).get_name()
        families = [regular_name, "Arial", "SimHei"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": families,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    })


def semibold_font():
    """Return the bundled Gill Sans SemiBold face used by Figure 1."""
    from matplotlib import font_manager

    if not GILL_SEMIBOLD.is_file():
        sys.exit(f"missing bundled paper font: {GILL_SEMIBOLD}")
    return font_manager.FontProperties(fname=GILL_SEMIBOLD)


def thousands(value, _pos=None) -> str:
    if value >= 1000:
        return f"{value / 1000:g}K"
    return f"{value:g}"


def order_systems(names) -> list[str]:
    rank = {name: i for i, name in enumerate(SYSTEM_ORDER)}
    return sorted(names, key=lambda name: rank.get(name, len(SYSTEM_ORDER)))


def series_style(index: int) -> dict:
    color = RECALL_PALETTE[index % len(RECALL_PALETTE)]
    return {
        "color": color,
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": "-",
        "linewidth": 1.0,
        "markersize": 3.2,
        "markerfacecolor": color,
        "markeredgecolor": "black",
        "markeredgewidth": 0.35,
    }


def line_style(system: str) -> dict:
    style = SYSTEM_STYLE.get(system)
    if style is None:
        return series_style(0)
    return {
        "color": style["color"],
        "marker": style["marker"],
        "linestyle": "-",
        "linewidth": 1.0,
        "markersize": 3.2,
        "markerfacecolor": style["color"],
        "markeredgecolor": "black",
        "markeredgewidth": 0.35,
    }


def bar_style(system: str, index: int = 0, palette: str = "ablation") -> dict:
    colors = {
        "ablation": {
            "FlashANNS": "#FBB4AE", "+S": "#B3CDE3", "Quiver": "#CCEBC5",
        },
        "utilization": {
            "FlashANNS": "#FDD9A6", "GustANN": "#CCEBC5", "Quiver": "#B3CDE3",
        },
    }
    style = SYSTEM_STYLE.get(system)
    if style is None:
        color = PALETTE[index % len(PALETTE)]
        hatch = HATCHES[index % len(HATCHES)]
    else:
        color = colors.get(palette, {}).get(system, style["color"])
        hatch = style["hatch"]
    return {
        "color": color,
        "hatch": hatch,
        "edgecolor": "black",
        "linewidth": 0.4,
    }


def part_style(name: str, index: int) -> tuple[str, dict]:
    for part, color, hatch in PART_STYLE:
        if part == name:
            return PART_LABEL.get(part, part), {
                "color": color,
                "hatch": hatch,
                "edgecolor": "black",
                "linewidth": 0.4,
            }
    label = PART_LABEL.get(name, name)
    return label, {
        "color": PALETTE[index % len(PALETTE)],
        "hatch": HATCHES[index % len(HATCHES)],
        "edgecolor": "black",
        "linewidth": 0.4,
    }


def box_ax(ax) -> None:
    ax.set_frame_on(True)
    if not getattr(ax, "_paper_keep_grid", False):
        ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.tick_params(
        which="major", direction="out", width=0.8, length=2.4, pad=1,
        labelsize=getattr(ax, "_paper_tick_labelsize", 9),
    )
    ax.tick_params(which="minor", direction="out", width=0.5, length=1.4, pad=1)
    if not getattr(ax, "_paper_minor_ticks", False):
        ax.minorticks_off()


def panel_label(
    ax, text: str, corner: str, *, x_offset: float = 0.0
) -> None:
    x, y, ha, va = _CORNER[corner]
    ax.text(
        x + x_offset, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9,
        color="#8F3333",
        fontweight="bold",
        zorder=6,
        clip_on=True,
    )


def log_xlim(values, corner: str) -> tuple[float, float]:
    vals = [v for v in values if v is not None and math.isfinite(v) and v > 0]
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo, hi = lo / 1.5, hi * 1.5
    span = max(math.log10(hi) - math.log10(lo), 0.35)
    left = 0.28 * span if "left" in corner else 0.10 * span
    right = 0.22 * span if "right" in corner else 0.10 * span
    return 10 ** (math.log10(lo) - left), 10 ** (math.log10(hi) + right)


def linear_ylim(values, corner: str, floor: float | None = None) -> tuple[float, float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, abs(hi) * 0.05, 1e-6)
    bottom = 0.42 * span if "lower" in corner else 0.08 * span
    top = 0.48 * span if "upper" in corner else 0.16 * span
    y0, y1 = lo - bottom, hi + top
    if floor is not None:
        y0 = floor
        # Keep the upper corner empty so an in-axis label clears the bars.
        if "upper" in corner and hi > y0:
            y1 = max(y1, y0 + (hi - y0) / 0.60)
    return y0, y1


def prepare_tradeoff(ax, xs, ys, corner: str) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax._paper_minor_ticks = True
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlim(*log_xlim(xs, corner))
    ax.set_ylim(*linear_ylim(ys, corner))


def prepare_bars(ax, heights, corner: str, floor: float = 0.0) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylim(*linear_ylim(list(heights) + [floor], corner, floor=floor))


def _extent_in(artist, fig) -> tuple[float, float]:
    fig.canvas.draw()
    box = artist.get_window_extent(fig.canvas.get_renderer())
    return box.width / fig.dpi, box.height / fig.dpi


def _text_size(
    fig, text: str, rotation: float = 0.0, fontsize: float = 9,
) -> tuple[float, float]:
    # Measure away from the figure edge so the bbox is not clipped.
    artist = fig.text(
        0.5, 0.5, text, fontsize=fontsize, rotation=rotation,
        rotation_mode="anchor",
        ha="center", va="center", clip_on=False,
    )
    width, height = _extent_in(artist, fig)
    artist.remove()
    return width, height


def _tick_size(ax, axis: str) -> tuple[float, float]:
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    width = height = 0.0
    fig = ax.figure
    fig.canvas.draw()
    for label in labels:
        if not label.get_text():
            continue
        box = label.get_window_extent(fig.canvas.get_renderer())
        width = max(width, box.width / fig.dpi)
        height = max(height, box.height / fig.dpi)
    if axis == "x":
        return width, height or 0.12
    return width or 0.22, height


def finish(
    fig, axes, out, *, xlabel=None, ylabel=None, legends=None, widths=None,
    left_pad: float = 0.0, right_pad: float = 0.0, top_pad: float = 0.0,
    bottom_pad: float = 0.0, xlabel_fontsize: float = 9,
    xlabel_down=None, ylabel_out=None, ylabel_side=None,
    subplot_gap: float | None = None,
) -> None:
    """Place legends, shared labels, and panels, then write a vector PDF.

    `xlabel` / `ylabel` are either one shared string or one string per column.
    An empty string skips that column. `legends` is a list of
    {handles, labels, ax}; `ax` None centers the legend over the whole figure.
    """
    grid = _as_grid(axes)
    nrows = len(grid)
    ncols = len(grid[0])
    for row in grid:
        for ax in row:
            box_ax(ax)

    legend_specs = []
    for spec in legends or []:
        handles = list(spec.get("handles") or [])
        labels = list(spec.get("labels") or [])
        if not labels:
            continue
        target_ax = spec.get("ax")
        if spec.get("placement") == "inside":
            if target_ax is None:
                raise ValueError("inside legend requires ax")
            legend_kwargs = {}
            if spec.get("bbox_to_anchor") is not None:
                legend_kwargs["bbox_to_anchor"] = spec["bbox_to_anchor"]
            legend = target_ax.legend(
                handles,
                labels,
                loc=spec.get("loc", "lower right"),
                ncol=spec.get("ncol", 1),
                frameon=spec.get("frameon", True),
                fontsize=spec.get("fontsize", 9),
                borderaxespad=0.2,
                handlelength=1.25,
                handletextpad=0.35,
                columnspacing=0.75,
                labelspacing=0.15,
                **legend_kwargs,
            )
            legend_specs.append((legend, "inside", None))
        else:
            legend = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.5),
                bbox_transform=fig.transFigure,
                ncol=spec.get("ncol", len(labels)),
                frameon=False,
                fontsize=spec.get("fontsize", 9),
                borderaxespad=0.0,
                handlelength=spec.get("handlelength", 1.25),
                handletextpad=spec.get("handletextpad", 0.35),
                columnspacing=spec.get("columnspacing", 0.75),
                labelspacing=spec.get("labelspacing", 0.15),
            )
            if spec.get("hide_first"):
                handles_out = getattr(legend, "legend_handles", None) or getattr(
                    legend, "legendHandles", []
                )
                if handles_out:
                    handles_out[0].set_visible(False)
            legend_specs.append((legend, target_ax, spec.get("center")))

    fig.canvas.draw()
    legend_h = 0.0
    for legend, target, _center in legend_specs:
        if target == "inside":
            continue
        _w, h = _extent_in(legend, fig)
        legend_h = max(legend_h, h)

    x_labels = _per_column(xlabel, ncols)
    y_labels = _per_column(ylabel, ncols)
    shared_x = isinstance(xlabel, str)
    shared_y = isinstance(ylabel, str)
    if xlabel_down is None:
        x_label_down = [0.0] * ncols
    elif isinstance(xlabel_down, (int, float)):
        x_label_down = [float(xlabel_down)] * ncols
    else:
        x_label_down = [float(value) for value in xlabel_down]
        if len(x_label_down) != ncols:
            raise ValueError(
                f"expected {ncols} xlabel offsets, got {len(x_label_down)}"
            )
    if ylabel_out is None:
        y_label_out = [0.0] * ncols
    elif isinstance(ylabel_out, (int, float)):
        y_label_out = [float(ylabel_out)] * ncols
    else:
        y_label_out = [float(value) for value in ylabel_out]
        if len(y_label_out) != ncols:
            raise ValueError(
                f"expected {ncols} ylabel offsets, got {len(y_label_out)}"
            )
    if ylabel_side is None:
        y_sides = ["left"] * ncols
    elif isinstance(ylabel_side, str):
        y_sides = [ylabel_side] * ncols
    else:
        y_sides = list(ylabel_side)
        if len(y_sides) != ncols:
            raise ValueError(
                f"expected {ncols} ylabel sides, got {len(y_sides)}"
            )

    x_label_h_by_col = [0.0] * ncols
    for j, text in enumerate(x_labels):
        if text:
            _w, h = _text_size(fig, text, fontsize=xlabel_fontsize)
            x_label_h_by_col[j] = h
    y_label_w = 0.0
    for text in ([ylabel] if shared_y else y_labels):
        if text:
            w, _h = _text_size(fig, text, rotation=90)
            y_label_w = max(y_label_w, w)

    x_tick_w = x_tick_h = 0.0
    x_tick_h_by_col = [0.0] * ncols
    y_tick_w = [0.0] * ncols
    for row in grid:
        for j, ax in enumerate(row):
            tw, th = _tick_size(ax, "x")
            yw, _yh = _tick_size(ax, "y")
            x_tick_w = max(x_tick_w, tw)
            x_tick_h = max(x_tick_h, th)
            x_tick_h_by_col[j] = max(x_tick_h_by_col[j], th)
            y_tick_w[j] = max(y_tick_w[j], yw)

    width, height = fig.get_size_inches()
    top = (
        height - OUTER - top_pad
        - (legend_h + LEGEND_GAP if legend_specs else 0.0)
    )
    if shared_x and xlabel:
        bottom_stack = (
            x_tick_h + max(x_label_h_by_col) + XLABEL_GAP
            + max(x_label_down)
        )
    else:
        bottom_stack = max(
            x_tick_h_by_col[j]
            + (
                x_label_h_by_col[j] + XLABEL_GAP + x_label_down[j]
                if x_labels[j] else 0.0
            )
            for j in range(ncols)
        )
    bottom = OUTER + bottom_pad + bottom_stack + 0.02

    y_block = []
    right_stack = x_tick_w * 0.55
    for j in range(ncols):
        label_w = y_label_w if shared_y and j == 0 else (y_label_w if not shared_y and y_labels[j] else 0.0)
        if not shared_y and y_labels[j]:
            label_w, _h = _text_size(fig, y_labels[j], rotation=90)
        block = y_tick_w[j] + 0.02
        if label_w:
            block += label_w + 0.02
        if y_sides[j] == "right":
            right_stack = max(
                right_stack,
                y_tick_w[j] + 0.02
                + (label_w + 0.02 if y_labels[j] else 0.0)
                + y_label_out[j],
            )
            y_block.append(0.0)
        else:
            extra = y_label_out[j] if shared_y and label_w else 0.0
            y_block.append(block + extra)
    right = OUTER + right_stack + right_pad

    left = OUTER + left_pad + y_block[0]
    col_gap = SUBPLOT_GAP if subplot_gap is None else float(subplot_gap)
    gaps = []
    for j in range(1, ncols):
        gaps.append(max(col_gap, y_block[j] + 0.02))
    v_gap = col_gap if nrows == 1 else max(col_gap, x_tick_h + 0.03)

    avail_w = width - left - right - sum(gaps)
    avail_h = top - bottom - v_gap * (nrows - 1)
    ratios = [1.0] * ncols if widths is None else [float(w) for w in widths]
    if len(ratios) != ncols:
        raise ValueError(f"expected {ncols} widths, got {len(ratios)}")
    ax_ws = [avail_w * r / sum(ratios) for r in ratios]
    ax_h = avail_h / nrows
    if min(ax_ws) <= 0.2 or ax_h <= 0.2:
        sys.exit(f"figure is too small for {ncols}x{nrows} panels at {width:.2f}x{height:.2f}in")

    xs = []
    cursor = left
    for j in range(ncols):
        xs.append(cursor)
        cursor += ax_ws[j]
        if j < ncols - 1:
            cursor += gaps[j]
    ys = []
    for i in range(nrows):
        # row 0 is the top row
        y = bottom + (nrows - 1 - i) * (ax_h + v_gap)
        ys.append(y)

    for i, row in enumerate(grid):
        for j, ax in enumerate(row):
            ax.set_position([xs[j] / width, ys[i] / height, ax_ws[j] / width, ax_h / height])

    for legend, ax, requested_center in legend_specs:
        if ax == "inside":
            continue
        if ax is None:
            center = 0.5 if requested_center is None else float(requested_center)
        else:
            pos = ax.get_position()
            center = pos.x0 + pos.width / 2
        legend.set_bbox_to_anchor(
            (center, (height - OUTER) / height), transform=fig.transFigure
        )

    if shared_x and xlabel:
        fig.text(
            0.5, OUTER / height, xlabel, ha="center", va="bottom",
            fontsize=xlabel_fontsize, transform=fig.transFigure,
        )
    else:
        for j, text in enumerate(x_labels):
            if not text:
                continue
            center = (xs[j] + ax_ws[j] / 2) / width
            _label_w, label_h = _text_size(
                fig, text, fontsize=xlabel_fontsize,
            )
            # Axes share a baseline, but their tick labels need not have the
            # same height (Figure 3(a), for example, has rotated categories).
            # Place each column label below its own ticks instead of below the
            # tallest ticks in the entire row.
            label_y = max(
                OUTER,
                ys[-1] - x_tick_h_by_col[j] - XLABEL_GAP - 0.02
                - label_h - x_label_down[j],
            )
            fig.text(
                center, label_y / height, text,
                ha="center", va="bottom", fontsize=xlabel_fontsize,
                transform=fig.transFigure,
            )

    stack_bottom = ys[-1]
    stack_top = ys[0] + ax_h
    stack_center = (stack_bottom + stack_top) / 2
    if shared_y and ylabel:
        # Anchor at the center of the label's horizontal slot. rotation_mode
        # keeps ha/va attached to that point after the counter-clockwise turn.
        slot = y_label_w
        fig.text(
            (OUTER + left_pad + slot / 2) / width,
            stack_center / height,
            ylabel,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=9,
            transform=fig.transFigure,
        )
    else:
        for j, text in enumerate(y_labels):
            if not text:
                continue
            label_w, _h = _text_size(fig, text, rotation=90)
            if y_sides[j] == "right":
                x = (
                    xs[j] + ax_ws[j] + y_tick_w[j] + 0.02 + label_w / 2
                    + y_label_out[j]
                )
                rotation = 270
            else:
                x = (
                    xs[j] - y_tick_w[j] - 0.02 - label_w / 2
                    - y_label_out[j]
                )
                rotation = 90
            fig.text(
                x / width,
                stack_center / height,
                text,
                rotation=rotation,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontsize=9,
                transform=fig.transFigure,
            )

    fig.savefig(out, format="pdf")
    plt.close(fig)


def _as_grid(axes) -> list[list]:
    if hasattr(axes, "get_legend_handles_labels") and not hasattr(axes, "ravel"):
        return [[axes]]
    if hasattr(axes, "shape"):
        if len(axes.shape) == 1:
            return [list(axes)]
        rows, cols = axes.shape
        return [[axes[i, j] for j in range(cols)] for i in range(rows)]
    if axes and isinstance(axes[0], (list, tuple)):
        return [list(row) for row in axes]
    return [list(axes)]


def _per_column(value, ncols: int) -> list[str]:
    if value is None:
        return [""] * ncols
    if isinstance(value, str):
        return [value] * ncols
    labels = list(value)
    if len(labels) != ncols:
        raise ValueError(f"expected {ncols} labels, got {len(labels)}")
    return labels


def legend_entries(axes, order=None) -> tuple[list, list]:
    found = {}
    flat = []
    grid = _as_grid(axes)
    for row in grid:
        flat.extend(row)
    for ax in flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            found.setdefault(label, handle)
    if order:
        labels = [name for name in order if name in found]
        labels.extend(name for name in found if name not in labels)
    else:
        labels = list(found)
    return [found[name] for name in labels], labels
