"""Regenerate fig3/fig4 from paper_figures/figure_metrics_correct.json.

Companion to regenerate_paper_figures.py (which handles fig1/fig2). Reuses
the same house style (Times New Roman, hatched bars, framed legend below).

NOTE on data source: this reads figure_metrics_correct.json, not the
figure_metrics.json used by regenerate_paper_figures.py. The latter is a
stale bootstrap file (see its own "source" field) with incorrect numbers
for the "FF + Shuffle (Blind)" row; figure_metrics_correct.json has the
verified numbers that actually match the published fig1/fig2 PDFs (cross-
checked against ershad/workspace_output/figure_metrics.json and
ershad/workspace_output/table_*.csv). The two files are intentionally kept
separate -- figure_metrics.json is left as-is here rather than overwritten.

Fig. 3: Overall-only per-class F1 for FF Blind / FF Classifier / FF Oracle
        (3 groups on the x-axis, 3 class-colored bars per group).
Fig. 4: Per-class F1 by direction (Left/Middle/Right only, no Overall) for
        the three base fusion strategies, relabeled as Pixel-/Decision-/
        Feature-Level Fusion.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "paper_figures"
METRICS_PATH = OUT_DIR / "figure_metrics_correct.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- exact house style from regenerate_paper_figures.py -------------------
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

FIG_H = 2.15
BAR_KW = dict(edgecolor="black", linewidth=1.0)
LABEL_FMT = "{:.2f}"
LABEL_FONTSIZE = 8.5
LABEL_PAD = 0.014

COLOR_SHADES = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]
COLOR_HATCHES = ["", "///", "...", "xxx", "---", "|||"]
GRAY_SHADES = ["#FFFFFF", "#CCCCCC", "#888888", "#FFFFFF", "#AAAAAA", "#333333"]
GRAY_HATCHES = ["", "///", "...", "xxx", "---", "|||"]

COLOR_CLASS = {
    "Clear": ("#4477AA", ""),
    "Person": ("#EE6677", "///"),
    "Other Object": ("#228833", "..."),
}
GRAY_CLASS = {
    "Clear": ("#FFFFFF", ""),
    "Person": ("#AAAAAA", "///"),
    "Other Object": ("#444444", "..."),
}

CLASS_KEYS_JSON = ["clear", "person", "other"]
CLASS_PANEL_LABELS = ["Clear", "Person", "Other Object"]


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


def _add_bar_top_labels(ax, bar_groups, val_groups, rotation=0, fontsize=None, pad=None):
    # Every label sits exactly LABEL_PAD above its own bar -- no per-label
    # collision bumping, so the gap is visually identical bar-to-bar.
    fontsize = fontsize or LABEL_FONTSIZE
    pad = LABEL_PAD if pad is None else pad
    max_label_y = 0.0
    for gi in range(len(val_groups[0])):
        for bars, vals in zip(bar_groups, val_groups):
            bar = bars[gi]
            val = float(vals[gi])
            cx = bar.get_x() + bar.get_width() / 2
            y = val + pad
            max_label_y = max(max_label_y, y)
            ax.text(cx, y, LABEL_FMT.format(val), fontsize=fontsize, rotation=rotation,
                    fontweight="bold", ha="center", va="bottom", clip_on=False)
    return max_label_y


def _add_bar_inside_labels(ax, bar_groups, val_groups, fontsize=7.5, pad=0.03):
    """Vertical value labels anchored just *inside* the top of each bar --
    can never collide with the axes border no matter how tall the bar is,
    since nothing is drawn above the bar's own value."""
    outline = [pe.withStroke(linewidth=2.2, foreground="white")]
    for gi in range(len(val_groups[0])):
        for bars, vals in zip(bar_groups, val_groups):
            bar = bars[gi]
            val = float(vals[gi])
            cx = bar.get_x() + bar.get_width() / 2
            ax.text(cx, val - pad, LABEL_FMT.format(val), fontsize=fontsize, rotation=90,
                    fontweight="bold", ha="center", va="top", color="black",
                    path_effects=outline, clip_on=False, zorder=6)


def _set_ylim_fixed(ax):
    # Fixed at exactly 1.0 -- F1 is bounded to [0, 1] so the axis top is
    # always the true ceiling, never stretched to make room for labels.
    ax.set_ylim(0, 1.0)


def _legend_below(fig, handles, labels, ncol=3, fontsize=None, anchor_y=0.10):
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, anchor_y),
                      ncol=ncol, frameon=True, edgecolor="black", facecolor="white",
                      columnspacing=0.9, handlelength=1.2, handletextpad=0.4)
    if fontsize:
        for t in leg.get_texts():
            t.set_fontsize(fontsize)


def _finalize_figure(fig, bottom=0.28, top=0.94):
    fig.subplots_adjust(left=0.07, right=0.99, top=top, bottom=bottom, wspace=0.18)


def _save_fig(fig, name):
    for directory in (OUT_DIR, REPO_ROOT):
        for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
            path = directory / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white", **kw)
            print(f"saved -> {path}")
    plt.close(fig)


# ---- load the verified-correct metrics ------------------------------------
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
M = metrics["models_data_by_class"]


def draw_fig3(shades, suffix):
    """Overall-only per-class F1: FF Blind / FF Classifier / FF Oracle."""
    model_keys = ["FF + Shuffle\n(Blind)", "FF + Shuffle\n(Classifier)", "FF + Shuffle\n(Oracle)"]
    group_labels = model_keys
    x = np.arange(len(model_keys))
    w = 0.26

    class_vals = {cls: [M[mk]["overall"][ck] for mk in model_keys]
                  for cls, ck in zip(CLASS_PANEL_LABELS, CLASS_KEYS_JSON)}

    fig, ax = plt.subplots(figsize=(4.9, FIG_H))
    offsets = [-w, 0, w]
    bars, vals = [], []
    for cls, off in zip(CLASS_PANEL_LABELS, offsets):
        color, hatch = shades[cls]
        b = ax.bar(x + off, class_vals[cls], w, label=cls, color=color, hatch=hatch, **BAR_KW)
        bars.append(b)
        vals.append(class_vals[cls])

    _add_bar_top_labels(ax, bars, vals)
    _set_ylim_fixed(ax)
    _group_separators(ax, len(model_keys))
    _trim_xaxis(ax, len(model_keys))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    _style_axes(ax, "F1 Score")
    handles, labels = ax.get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=3, anchor_y=0.085)
    _finalize_figure(fig, bottom=0.25, top=0.90)
    _save_fig(fig, f"fig3_shuffle_overall_{suffix}")


def draw_fig4(shades, hatches, suffix):
    """Per-class F1 by direction (Left/Middle/Right only) for the three
    base fusion strategies, relabeled by fusion level."""
    model_map = [
        ("Pixel-Level Fusion", "Baseline 1\n(Stitched)"),
        ("Decision-Level Fusion", "Baseline 2\n(Output Fusion)"),
        ("Feature-Level Fusion", "Feature Fusion\n(Best Combo)"),
    ]
    groups = ["left", "middle", "right"]
    group_labels = ["Left", "Middle", "Right"]
    n_models = len(model_map)
    x = np.arange(len(groups))
    w = 0.28
    offsets = np.linspace(-(n_models - 1) / 2 * w, (n_models - 1) / 2 * w, n_models)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, FIG_H), sharey=True)
    for ax, ck, panel_label in zip(axes, CLASS_KEYS_JSON, CLASS_PANEL_LABELS):
        bar_groups, val_groups = [], []
        for (new_name, orig_key), shade, hatch, offset in zip(model_map, shades, hatches, offsets):
            vals = [M[orig_key][g][ck] for g in groups]
            bars = ax.bar(x + offset, vals, w, color=shade, hatch=hatch, label=new_name, **BAR_KW)
            bar_groups.append(bars)
            val_groups.append(vals)
        # labels sit *inside* the bars (anchored to each bar's own top edge)
        # so they can never collide with the axes border up at F1 = 1.0
        _add_bar_inside_labels(ax, bar_groups, val_groups)
        _group_separators(ax, len(groups))
        _trim_xaxis(ax, len(groups))
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_title(panel_label, fontsize=13, pad=8)
        _style_axes(ax)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)

    axes[0].set_ylabel("F1 Score")
    for ax in axes:
        _set_ylim_fixed(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=3, fontsize=12, anchor_y=0.15)
    _finalize_figure(fig, bottom=0.27, top=0.90)
    _save_fig(fig, f"fig4_fusion_by_direction_{suffix}")


if __name__ == "__main__":
    print(f"loaded metrics from {METRICS_PATH}")
    draw_fig3(COLOR_CLASS, "color")
    draw_fig3(GRAY_CLASS, "grayscale")
    draw_fig4(COLOR_SHADES[:3], COLOR_HATCHES[:3], "color")
    draw_fig4(GRAY_SHADES[:3], GRAY_HATCHES[:3], "grayscale")
