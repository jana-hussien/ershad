"""Regenerate fig1/fig2 from cached values (no full notebook run needed)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Keep DareDevil paper font (Times New Roman serif)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

OUT_DIR = Path(__file__).resolve().parent / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GROUPS = ["left", "middle", "right", "overall"]
GROUP_LABELS = ["Left", "Middle", "Right", "Overall"]
CLASS_KEYS = [0, 1, 2]
CLASS_PANEL_LABELS = ["Clear", "Person", "Other Object"]

FIG1_CAPTION = (
    "Per-class F1 by camera direction for six obstruction-detection architectures. "
    "Baselines 1–2 and Feature Fusion are evaluated on the validation set; "
    "Feature Fusion with shuffle correction (blind, classifier, oracle) is evaluated "
    "on the held-out test set."
)
FIG2_CAPTION = (
    "Per-class F1 of Feature Fusion with classifier-based shuffle correction on the "
    "held-out test set, broken down by camera direction (left, middle, right) and overall."
)


def _mk(c0, c1, c2):
    return {0: c0, 1: c1, 2: c2}


models_data_by_class = {
    "Baseline 1\n(Stitched)": {
        "left": _mk(0.786, 0.785, 0.739),
        "middle": _mk(0.706, 0.842, 0.769),
        "right": _mk(0.851, 0.591, 0.692),
        "overall": _mk(0.781, 0.739, 0.733),
    },
    "Baseline 2\n(Output Fusion)": {
        "left": _mk(0.821, 0.872, 0.768),
        "middle": _mk(0.560, 0.822, 0.767),
        "right": _mk(0.847, 0.612, 0.678),
        "overall": _mk(0.743, 0.769, 0.738),
    },
    "Feature Fusion\n(Best Combo)": {
        "left": _mk(0.835, 0.890, 0.812),
        "middle": _mk(0.723, 0.866, 0.807),
        "right": _mk(0.885, 0.724, 0.775),
        "overall": _mk(0.814, 0.827, 0.798),
    },
    "FF + Shuffle\n(Blind)": {
        "left": _mk(0.36, 0.35, 0.33),
        "middle": _mk(0.56, 0.55, 0.54),
        "right": _mk(0.34, 0.33, 0.32),
        "overall": _mk(0.42, 0.41, 0.40),
    },
    "FF + Shuffle\n(Classifier)": {
        "left": _mk(0.937, 0.941, 0.872),
        "middle": _mk(0.901, 0.948, 0.961),
        "right": _mk(0.922, 0.852, 0.881),
        "overall": _mk(0.920, 0.914, 0.905),
    },
    "FF + Shuffle\n(Oracle)": {
        "left": _mk(0.937, 0.941, 0.872),
        "middle": _mk(0.901, 0.948, 0.961),
        "right": _mk(0.922, 0.852, 0.881),
        "overall": _mk(0.920, 0.914, 0.905),
    },
}
COLOR_SHADES = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]
COLOR_HATCHES = ["", "///", "...", "xxx", "---", "|||"]

clear_f1 = [0.937, 0.901, 0.922, 0.920]
person_f1 = [0.941, 0.948, 0.852, 0.914]
other_f1 = [0.872, 0.961, 0.881, 0.905]
COLOR_CLASS = {
    "Clear": ("#4477AA", ""),
    "Person": ("#EE6677", "///"),
    "Other Object": ("#228833", "..."),
}


def _style_axes(ax, ylabel=None):
    """Match reference charts: y in [0,1], horizontal grid at 0.5 and 1.0 only."""
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.grid(True, linestyle="--", color="#AAAAAA", alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    if ylabel:
        ax.set_ylabel(ylabel)


def _trim_xaxis(ax, n_groups):
    ax.set_xlim(-0.52, n_groups - 0.48)
    ax.margins(x=0)


def _group_separators(ax, n_groups):
    for xi in range(1, n_groups):
        ax.axvline(xi - 0.5, color="#AAAAAA", linestyle=":", linewidth=0.8, alpha=0.7)


def _add_bar_value_labels(ax, bar_groups, val_groups, fmt="{:.2f}",
                          base_pad=0.015, min_sep=0.032, fontsize=8):
    """Bold value labels above bars; vertically stagger within each group to avoid overlap."""
    n_groups = len(val_groups[0])
    label_top = 0.0
    for gi in range(n_groups):
        entries = []
        for bars, vals in zip(bar_groups, val_groups):
            bar = bars[gi]
            cx = bar.get_x() + bar.get_width() / 2
            entries.append((cx, float(vals[gi])))
        entries.sort(key=lambda e: e[1], reverse=True)
        prev_top = -1.0
        for cx, val in entries:
            y = val + base_pad
            if y <= prev_top + min_sep:
                y = prev_top + min_sep
            prev_top = y
            label_top = max(label_top, y)
            ax.text(cx, y, fmt.format(val), ha="center", va="bottom",
                    fontsize=fontsize, fontweight="bold", clip_on=False)
    ymax = ax.get_ylim()[1]
    if label_top + 0.02 > ymax:
        ax.set_ylim(0, min(1.0, label_top + 0.04))


def _legend_below(fig, handles, labels, ncol=3, bottom=0.27):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01),
               ncol=ncol, frameon=True, edgecolor="black", facecolor="white",
               fontsize=8, columnspacing=0.9, handlelength=1.2, handletextpad=0.4)
    fig.subplots_adjust(bottom=bottom)


def draw_fig1():
    """Three-panel layout like the WER/SS reference; legend below in columns."""
    model_names = list(models_data_by_class.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.55), sharey=True)
    x = np.arange(len(GROUPS))
    w = 0.125
    offsets = np.linspace(-(n_models - 1) / 2 * w, (n_models - 1) / 2 * w, n_models)

    for ax, cls_key, panel_label in zip(axes, CLASS_KEYS, CLASS_PANEL_LABELS):
        for name, shade, hatch, offset in zip(model_names, COLOR_SHADES, COLOR_HATCHES, offsets):
            vals = [models_data_by_class[name][g][cls_key] for g in GROUPS]
            ax.bar(x + offset, vals, w, color=shade, hatch=hatch,
                   edgecolor="black", linewidth=1.0, label=name)
        _group_separators(ax, len(GROUPS))
        _trim_xaxis(ax, len(GROUPS))
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_LABELS)
        _style_axes(ax, panel_label)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=3, bottom=0.26)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.26, wspace=0.18)
    path = OUT_DIR / "fig1_macro_f1_color.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close()
    print(f"saved -> {path}")


def draw_fig2():
    """Single grouped chart like the T2M/IPE reference; legend upper-left."""
    x = np.arange(len(GROUPS))
    w = 0.28
    fig, ax = plt.subplots(figsize=(4.9, 2.55))
    kw = dict(edgecolor="black", linewidth=1.0)
    b1 = ax.bar(x - w, clear_f1, w, label="Clear",
                color=COLOR_CLASS["Clear"][0], hatch=COLOR_CLASS["Clear"][1], **kw)
    b2 = ax.bar(x, person_f1, w, label="Person",
                color=COLOR_CLASS["Person"][0], hatch=COLOR_CLASS["Person"][1], **kw)
    b3 = ax.bar(x + w, other_f1, w, label="Other Object",
                color=COLOR_CLASS["Other Object"][0], hatch=COLOR_CLASS["Other Object"][1], **kw)
    _add_bar_value_labels(ax, [b1, b2, b3], [clear_f1, person_f1, other_f1])
    _group_separators(ax, len(GROUPS))
    _trim_xaxis(ax, len(GROUPS))
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS)
    _style_axes(ax, "F1 Score")
    ax.legend(loc="upper left", frameon=True, edgecolor="black", facecolor="white",
              fontsize=8, handlelength=1.2, borderpad=0.3)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.14)
    path = OUT_DIR / "fig2_ff_per_class_color.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close()
    print(f"saved -> {path}")


if __name__ == "__main__":
    draw_fig1()
    draw_fig2()
    print("\n--- Figure captions (for paper, not rendered in plots) ---")
    print(f"Fig. 1: {FIG1_CAPTION}")
    print(f"Fig. 2: {FIG2_CAPTION}")
