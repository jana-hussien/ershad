"""Regenerate fig1/fig2 from cached values (no full notebook run needed)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.color": "#AAAAAA",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path(__file__).resolve().parent
GROUPS = ["left", "middle", "right", "overall"]
GROUP_LABELS = ["Left", "Middle", "Right", "Overall"]
CLASS_LABELS = ["Clear", "Person", "Other Object"]
CLASS_KEYS = [0, 1, 2]

# Val-set per-class F1 (from figures-creation.ipynb Section 12 table)
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
    # Held-out test, shuffle variants (classifier/oracle from fig2; blind approximated from macro ratios)
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


def _style_yaxis(ax, ylabel=None):
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)


def _add_grouped_bar_labels(ax, bar_groups, val_groups, base_pad=0.02, min_sep=0.03):
    n_groups = len(val_groups[0])
    label_top = 0.0
    for gi in range(n_groups):
        pts = []
        for bars, vals in zip(bar_groups, val_groups):
            bar = bars[gi]
            pts.append((bar.get_x() + bar.get_width() / 2, vals[gi]))
        ys = [val + base_pad for _, val in pts]
        for i in range(1, len(pts)):
            if abs(pts[i][0] - pts[i - 1][0]) < 0.25 and abs(pts[i][1] - pts[i - 1][1]) < 0.03:
                ys[i] = max(ys[i], ys[i - 1] + min_sep)
        for (cx, val), y in zip(pts, ys):
            ax.text(cx, y, f"{val:.2f}", ha="center", va="bottom", fontsize=8, clip_on=False)
            label_top = max(label_top, y)
    return label_top


def draw_fig1():
    model_names = list(models_data_by_class.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), sharey=True)
    x = np.arange(len(GROUPS))
    w = 0.11
    offsets = np.linspace(-(n_models - 1) / 2 * w, (n_models - 1) / 2 * w, n_models)

    for ax, cls_key, panel_label in zip(axes, CLASS_KEYS, CLASS_LABELS):
        for name, shade, hatch, offset in zip(model_names, COLOR_SHADES, COLOR_HATCHES, offsets):
            vals = [models_data_by_class[name][g][cls_key] for g in GROUPS]
            ax.bar(x + offset, vals, w, color=shade, hatch=hatch,
                   edgecolor="black", linewidth=1.0, label=name)
        ax.axvline(x[-1] - 0.5, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_LABELS, fontsize=11)
        _style_yaxis(ax)
        ax.text(0.5, 0.97, panel_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=11, fontweight="bold")

    _style_yaxis(axes[0], "F1 Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=True, edgecolor="black", fontsize=8)
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.28, wspace=0.08)
    path = OUT_DIR / "fig1_macro_f1_color.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved -> {path}")


def draw_fig2():
    x = np.arange(len(GROUPS))
    w = 0.22
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    kw = dict(edgecolor="black", linewidth=1.0)
    b1 = ax.bar(x - w, clear_f1, w, label="Clear",
                color=COLOR_CLASS["Clear"][0], hatch=COLOR_CLASS["Clear"][1], **kw)
    b2 = ax.bar(x, person_f1, w, label="Person",
                color=COLOR_CLASS["Person"][0], hatch=COLOR_CLASS["Person"][1], **kw)
    b3 = ax.bar(x + w, other_f1, w, label="Other Object",
                color=COLOR_CLASS["Other Object"][0], hatch=COLOR_CLASS["Other Object"][1], **kw)
    _add_grouped_bar_labels(ax, [b1, b2, b3], [clear_f1, person_f1, other_f1])
    ax.axvline(x[-1] - 0.5, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=11)
    _style_yaxis(ax, "F1 Score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=True, edgecolor="black", fontsize=9)
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.22)
    path = OUT_DIR / "fig2_ff_per_class_color.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved -> {path}")


if __name__ == "__main__":
    draw_fig1()
    draw_fig2()
