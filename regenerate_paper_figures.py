"""Regenerate fig1/fig2 from metrics exported by figures-creation.ipynb."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_figure_metrics import load_figure_metrics

# Keep DareDevil paper font (Times New Roman serif)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "paper_figures"
METRICS_PATH = OUT_DIR / "figure_metrics.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["left", "middle", "right", "overall"]
GROUP_LABELS = ["Left", "Middle", "Right", "Overall"]
CLASS_KEYS = [0, 1, 2]
CLASS_PANEL_LABELS = ["Clear", "Person", "Other Object"]

FIG_H = 2.75
BAR_KW = dict(edgecolor="black", linewidth=1.0)
LABEL_FMT = "{:.2f}"
LABEL_FONTSIZE = 6
LABEL_PAD = 0.018
LABEL_COLLISION_DY = 0.028
YLIM_HEADROOM = 0.03

COLOR_SHADES = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]
COLOR_HATCHES = ["", "///", "...", "xxx", "---", "|||"]
COLOR_CLASS = {
    "Clear": ("#4477AA", ""),
    "Person": ("#EE6677", "///"),
    "Other Object": ("#228833", "..."),
}

_metrics = load_figure_metrics(METRICS_PATH)
models_data_by_class = _metrics["models_data_by_class"]
clear_f1 = _metrics["clear_f1"]
person_f1 = _metrics["person_f1"]
other_f1 = _metrics["other_f1"]
FIG1_CAPTION = _metrics["fig1_caption"]
FIG2_CAPTION = _metrics["fig2_caption"]


def _style_axes(ax, ylabel=None):
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


def _add_bar_top_labels(ax, bar_groups, val_groups):
    bar_w = bar_groups[0][0].get_width()
    max_label_y = 0.0

    for gi in range(len(val_groups[0])):
        entries = []
        for bars, vals in zip(bar_groups, val_groups):
            bar = bars[gi]
            val = float(vals[gi])
            cx = bar.get_x() + bar.get_width() / 2
            entries.append([cx, val, val + LABEL_PAD])

        entries.sort(key=lambda e: e[0])
        for i in range(1, len(entries)):
            cx_prev, _, y_prev = entries[i - 1]
            cx, val, y = entries[i]
            if abs(cx - cx_prev) < bar_w * 1.05 and abs(y - y_prev) < LABEL_COLLISION_DY:
                entries[i][2] = y_prev + LABEL_COLLISION_DY

        for cx, val, y in entries:
            max_label_y = max(max_label_y, y)
            ax.text(cx, y, LABEL_FMT.format(val), fontsize=LABEL_FONTSIZE,
                    fontweight="bold", ha="center", va="bottom", clip_on=False)

    return max_label_y


def _set_ylim_with_labels(ax, max_label_y):
    ax.set_ylim(0, max(1.0, max_label_y + YLIM_HEADROOM))


def _legend_below(fig, handles, labels, ncol=3):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01),
               ncol=ncol, frameon=True, edgecolor="black", facecolor="white",
               columnspacing=0.9, handlelength=1.2, handletextpad=0.4)


def _finalize_figure(fig, bottom=0.28):
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=bottom, wspace=0.18)


def _save_fig(name):
    for directory in (OUT_DIR, REPO_ROOT):
        path = directory / name
        plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        print(f"saved -> {path}")


def draw_fig1():
    model_names = list(models_data_by_class.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, FIG_H), sharey=True)
    x = np.arange(len(GROUPS))
    w = 0.125
    offsets = np.linspace(-(n_models - 1) / 2 * w, (n_models - 1) / 2 * w, n_models)

    global_max_label_y = 0.0
    for ax, cls_key, panel_label in zip(axes, CLASS_KEYS, CLASS_PANEL_LABELS):
        bar_groups, val_groups = [], []
        for name, shade, hatch, offset in zip(model_names, COLOR_SHADES, COLOR_HATCHES, offsets):
            vals = [models_data_by_class[name][g][cls_key] for g in GROUPS]
            bars = ax.bar(x + offset, vals, w, color=shade, hatch=hatch, label=name, **BAR_KW)
            bar_groups.append(bars)
            val_groups.append(vals)
        global_max_label_y = max(global_max_label_y,
                                 _add_bar_top_labels(ax, bar_groups, val_groups))
        _group_separators(ax, len(GROUPS))
        _trim_xaxis(ax, len(GROUPS))
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_LABELS)
        _style_axes(ax, panel_label)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)

    for ax in axes:
        _set_ylim_with_labels(ax, global_max_label_y)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=3)
    _finalize_figure(fig, bottom=0.28)
    _save_fig("fig1_macro_f1_color.png")
    plt.close()


def draw_fig2():
    x = np.arange(len(GROUPS))
    w = 0.28
    fig, ax = plt.subplots(figsize=(4.9, FIG_H))
    b1 = ax.bar(x - w, clear_f1, w, label="Clear",
                color=COLOR_CLASS["Clear"][0], hatch=COLOR_CLASS["Clear"][1], **BAR_KW)
    b2 = ax.bar(x, person_f1, w, label="Person",
                color=COLOR_CLASS["Person"][0], hatch=COLOR_CLASS["Person"][1], **BAR_KW)
    b3 = ax.bar(x + w, other_f1, w, label="Other Object",
                color=COLOR_CLASS["Other Object"][0], hatch=COLOR_CLASS["Other Object"][1], **BAR_KW)
    max_label_y = _add_bar_top_labels(ax, [b1, b2, b3], [clear_f1, person_f1, other_f1])
    _set_ylim_with_labels(ax, max_label_y)
    _group_separators(ax, len(GROUPS))
    _trim_xaxis(ax, len(GROUPS))
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS)
    _style_axes(ax, "F1 Score")
    handles, labels = ax.get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=3)
    _finalize_figure(fig, bottom=0.22)
    _save_fig("fig2_ff_per_class_color.png")
    plt.close()


if __name__ == "__main__":
    print(f"loaded metrics from {_metrics['source']} ({METRICS_PATH})")
    draw_fig1()
    draw_fig2()
    print("\n--- Figure captions (for paper, not rendered in plots) ---")
    print(f"Fig. 1: {FIG1_CAPTION}")
    print(f"Fig. 2: {FIG2_CAPTION}")
