"""Regenerate the trainval class-distribution chart in the same house style
as regenerate_fig3_fig4.py (muted palette, Times font, Type 42 PDF fonts,
compressed height, framed legend below) -- but with no in-bar value labels.

DATA NOTE: counts are exact, computed from the label cache of the 3-seed
retrain (kaggle_retrain/results/main_run/label_cache.pkl): every train+val
timestep with a usable RGB frame (15,646), counted per camera direction. They
replace the earlier pixel-measured estimates, whose figure files
(dataset_class_distribution_trainval_*) are left untouched; this script now
writes dataset_class_distribution_trainval_exact_*.
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
    # larger than the other figures: this one is printed at about half a column wide
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 150,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 0.6,
})

# This is a standalone figure (not paired tightly with another chart like
# fig3/fig4 are), so it keeps a taller, roughly-square aspect ratio matching
# the original reference chart instead of fig3/fig4's compressed strip shape.
FIG_H = 2.9
BAR_KW = dict(edgecolor="#3A3A3A", linewidth=0.8)   # Fig. 1's stroke gray

# Same muted Tol-derived class palette as fig3/fig4 (each hue blended 35%
# toward mid-gray) so this chart reads as part of the same figure family.
# Class colors: Fig. 1's badge hues (clear green / person red / object orange), muted for large bars.
COLOR_CLASS = {
    "Clear": ("#6E995D", ""),
    "Person": ("#B8554F", "///"),
    "Other Object": ("#D2924D", "..."),
}
GRAY_CLASS = {
    "Clear": ("#FFFFFF", ""),
    "Person": ("#AAAAAA", "///"),
    "Other Object": ("#444444", "..."),
}

DIRECTIONS = ["Left", "Middle", "Right"]
# Exact counts per direction (left, middle, right); each direction totals 15,646.
COUNTS = {
    "Clear":        [7138, 4529, 9431],
    "Person":       [2421, 3815, 1133],
    "Other Object": [6087, 7302, 5082],
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
    ax.set_title("Class Distribution by Camera Direction", fontsize=16, pad=8)
    handles, labels = ax.get_legend_handles_labels()
    _legend_below(ax, handles, labels, ncol=3, anchor_y=-0.30)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.22)
    _save_fig(fig, f"dataset_class_distribution_trainval_exact_{suffix}")


if __name__ == "__main__":
    draw_class_distribution(COLOR_CLASS, "color")
    draw_class_distribution(GRAY_CLASS, "grayscale")
