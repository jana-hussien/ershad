"""Regenerate fig1/fig2 from cached values (no full notebook run needed)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.color": "#AAAAAA",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path(__file__).resolve().parent
groups = ["left", "middle", "right", "overall"]
group_labels = ["Left", "Middle", "Right", "Overall"]

models_data = {
    "Baseline 1\n(Stitched)":       {"left": 0.762, "middle": 0.805, "right": 0.641, "overall": 0.736},
    "Baseline 2\n(Output Fusion)":  {"left": 0.820, "middle": 0.794, "right": 0.645, "overall": 0.753},
    "Feature Fusion\n(Best Combo)": {"left": 0.851, "middle": 0.836, "right": 0.750, "overall": 0.812},
    "FF + Shuffle\n(Blind)":        {"left": 0.348, "middle": 0.548, "right": 0.338, "overall": 0.411},
    "FF + Shuffle\n(Classifier)":  {"left": 0.909, "middle": 0.955, "right": 0.874, "overall": 0.909},
    "FF + Shuffle\n(Oracle)":      {"left": 0.911, "middle": 0.955, "right": 0.876, "overall": 0.911},
}
COLOR_SHADES = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]
COLOR_HATCHES = ["", "///", "...", "xxx", "---", "|||"]

noobj_f1 = [0.937, 0.901, 0.922, 0.920]
person_f1 = [0.941, 0.948, 0.852, 0.914]
other_f1 = [0.872, 0.961, 0.881, 0.905]
COLOR_CLASS = {
    "No Obstacle": ("#4477AA", ""),
    "Person": ("#EE6677", "///"),
    "Other Object": ("#228833", "..."),
}


def _add_grouped_bar_labels(ax, bar_groups, val_groups, base_pad=0.008, min_sep=0.022):
    """Bump labels only when neighboring bars have similar heights."""
    n_groups = len(val_groups[0])
    label_top = 0.0
    for gi in range(n_groups):
        pts = []
        for bars, vals in zip(bar_groups, val_groups):
            bar = bars[gi]
            pts.append((bar.get_x() + bar.get_width() / 2, vals[gi]))
        ys = [val + base_pad for _, val in pts]
        for i in range(1, len(pts)):
            close_x = abs(pts[i][0] - pts[i - 1][0]) < 0.25
            close_val = abs(pts[i][1] - pts[i - 1][1]) < 0.02
            if close_x and close_val:
                ys[i] = max(ys[i], ys[i - 1] + min_sep)
        for (cx, val), y in zip(pts, ys):
            ax.text(cx, y, f"{val:.3f}", ha="center", va="bottom", fontsize=8, clip_on=False)
            label_top = max(label_top, y)
    return label_top


def draw_fig1():
    model_names = list(models_data.keys())
    n_models = len(model_names)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(groups))
    w = 0.11
    offsets = np.linspace(-(n_models - 1) / 2 * w, (n_models - 1) / 2 * w, n_models)
    for name, shade, hatch, offset in zip(model_names, COLOR_SHADES, COLOR_HATCHES, offsets):
        vals = [models_data[name].get(g, 0.0) for g in groups]
        ax.bar(x + offset, vals, w, color=shade, hatch=hatch,
               edgecolor="black", linewidth=1.2, label=name)
    ax.axvline(x[-1] - 0.5, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=13)
    ax.set_ylabel("F1 Score (person + other object)", fontsize=13)
    ymax = max(models_data[n].get(g, 0.0) for n in model_names for g in groups)
    ax.set_ylim(0, ymax + 0.05)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95, ncol=2)
    plt.tight_layout()
    path = OUT_DIR / "fig1_macro_f1_color.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved -> {path}")


def draw_fig2():
    x = np.arange(len(groups))
    w = 0.22
    fig, ax = plt.subplots(figsize=(7, 4))
    kw = dict(edgecolor="black", linewidth=1.3)
    b1 = ax.bar(x - w, noobj_f1, w, label="No Obstacle",
                color=COLOR_CLASS["No Obstacle"][0], hatch=COLOR_CLASS["No Obstacle"][1], **kw)
    b2 = ax.bar(x, person_f1, w, label="Person",
                color=COLOR_CLASS["Person"][0], hatch=COLOR_CLASS["Person"][1], **kw)
    b3 = ax.bar(x + w, other_f1, w, label="Other Object",
                color=COLOR_CLASS["Other Object"][0], hatch=COLOR_CLASS["Other Object"][1], **kw)
    y_floor = max(0.0, round(min(noobj_f1 + person_f1 + other_f1) - 0.03, 2))
    ax.set_ylim(y_floor, 1.0)
    label_top = _add_grouped_bar_labels(ax, [b1, b2, b3], [noobj_f1, person_f1, other_f1])
    ax.set_ylim(y_floor, min(1.0, label_top + 0.012))
    ax.axvline(x[-1] - 0.5, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=13)
    ax.set_ylabel("F1 Score", fontsize=13)
    ax.legend(fontsize=11, framealpha=0.9, loc="lower left")
    plt.tight_layout()
    path = OUT_DIR / "fig2_ff_per_class_color.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved -> {path}")


if __name__ == "__main__":
    draw_fig1()
    draw_fig2()
