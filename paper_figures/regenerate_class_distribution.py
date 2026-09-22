"""Regenerate the trainval class-distribution chart in the same house style
as regenerate_fig3_fig4.py (muted palette, Times font, Type 42 PDF fonts,
compressed height, framed legend below) -- but with no in-bar value labels.

DATA NOTE: dataset_class_distribution_trainval_color.webp was handed over as
a finished image with no backing script/data file in this repo (the closest
match, plot_class_distribution() in figures-creation.ipynb, is the held-out/
4-category version and produces a different chart). The counts below were
reconstructed by measuring the original image's bar-segment boundaries in
pixels, calibrated against its own y-axis gridlines (5000/10000/15000) --
they are close estimates of the real values, not read from a logged source,
per user's go-ahead to estimate from the image.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# This is a standalone figure (not paired tightly with another chart like
# fig3/fig4 are), so it keeps a taller, roughly-square aspect ratio matching
# the original reference chart instead of fig3/fig4's compressed strip shape.
FIG_H = 2.9
BAR_KW = dict(edgecolor="black", linewidth=1.0)

# Same muted Tol-derived class palette as fig3/fig4 (each hue blended 35%
# toward mid-gray) so this chart reads as part of the same figure family.
COLOR_CLASS = {
    "Clear": ("#597A9B", ""),
    "Person": ("#C86F7A", "///"),
    "Other Object": ("#43854E", "..."),
}
GRAY_CLASS = {
    "Clear": ("#FFFFFF", ""),
    "Person": ("#AAAAAA", "///"),
    "Other Object": ("#444444", "..."),
}

DIRECTIONS = ["Left", "Middle", "Right"]
# Pixel-measured estimates -- see module docstring. Each direction's raw
# measured total (15590 / 15499 / 15499) had ~0.6% pixel noise between them,
# but the three directions share the same recorded frame count by
# construction (synchronized cameras), so the class proportions below are
# rescaled to a single common total of 15500 rather than three slightly
# different ones.
COUNTS = {
    "Clear":        [7144, 4481, 9382],
    "Person":       [2137, 3712, 1008],
    "Other Object": [6219, 7307, 5110],
}


def _style_axes(ax):
    ax.set_yticks([0, 5000, 10000, 15000])
    ax.yaxis.grid(True, linestyle="--", color="#AAAAAA", alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.set_ylabel("Samples")
    ax.set_xlabel("Camera Direction")


def _legend_below(ax, handles, labels, ncol=3, anchor_y=0.05):
    # Centered on the *axes* (bbox_transform=ax.transAxes), not the figure --
    # fig.legend's (0.5, y) anchor centers on the whole canvas, which drifts
    # off from the bars whenever the axes itself isn't centered in the figure
    # (e.g. a wide left margin reserved for a rotated y-axis label).
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, anchor_y),
              bbox_transform=ax.transAxes, ncol=ncol, frameon=True,
              edgecolor="black", facecolor="white",
              columnspacing=0.9, handlelength=1.2, handletextpad=0.4)


def _save_fig(fig, name):
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.16, facecolor="white", **kw)
        print(f"saved -> {path}")
    plt.close(fig)


def draw_class_distribution(palette, suffix):
    x = np.arange(len(DIRECTIONS))
    w = 0.5

    fig, ax = plt.subplots(figsize=(4.0, FIG_H))
    bottoms = np.zeros(len(DIRECTIONS))
    bars = []
    for cls in ["Clear", "Person", "Other Object"]:
        color, hatch = palette[cls]
        vals = np.array(COUNTS[cls])
        b = ax.bar(x, vals, w, bottom=bottoms, label=cls, color=color, hatch=hatch, **BAR_KW)
        bars.append(b)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(DIRECTIONS)
    ax.set_xlim(-0.6, len(DIRECTIONS) - 0.4)
    # Just enough headroom above the tallest bar (~15500) to keep the top
    # spine from being clipped by the tight bbox crop, without leaving a
    # large empty band between the bars and the top of the chart.
    ax.set_ylim(0, 16200)
    _style_axes(ax)
    ax.set_title("Class Distribution by Camera Direction", fontsize=13, pad=8)
    handles, labels = ax.get_legend_handles_labels()
    _legend_below(ax, handles, labels, ncol=3, anchor_y=-0.20)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.22)
    _save_fig(fig, f"dataset_class_distribution_trainval_{suffix}")


if __name__ == "__main__":
    draw_class_distribution(COLOR_CLASS, "color")
    draw_class_distribution(GRAY_CLASS, "grayscale")
