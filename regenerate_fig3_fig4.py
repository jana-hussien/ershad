"""Regenerate fig3/fig4 (the paper's actual Fig. 4 and Fig. 2) from
paper_figures/figure_metrics_correct.json.

NOTE on data source: figure_metrics_correct.json holds verified numbers that
match the published PDFs. It is hand-verified and checked in as the source
of truth; no script in this repo regenerates it automatically.

Fig. 3 (-> paper's Fig. 4): Overall-only per-class F1 for FF Blind / FF
        Classifier / FF Oracle (3 groups on the x-axis, 3 class-colored bars
        per group).
Fig. 4 (-> paper's Fig. 2): Per-class F1 by direction (Left/Middle/Right
        only, no Overall) for the three base fusion strategies, relabeled as
        Pixel-/Decision-/Feature-Level Fusion.
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

# ---- house style ----------------------------------------------------------
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
    # Matplotlib defaults to Type 3 (bitmap) fonts in PDF/PS output, which
    # IEEE's own author instructions explicitly warn against ("ensure ...
    # PDF output ... uses only Type 1 fonts"). Type 42 embeds scalable
    # TrueType outlines instead, which is what PDF compliance checkers
    # (e.g. IEEE PDF eXpress) expect.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_H = 0.92
BAR_KW = dict(edgecolor="black", linewidth=1.0)
LABEL_FMT = "{:.2f}"
LABEL_FONTSIZE = 8.5
LABEL_PAD = 0.014

# Muted variants of the Tol "bright" qualitative palette (each color blended
# 35% toward mid-gray) -- same hues, less neon/saturated for print.
COLOR_SHADES = ["#597A9B", "#6FB1C8", "#43854E", "#B1A659", "#C86F7A", "#9B4E7A"]
COLOR_HATCHES = ["", "///", "...", "xxx", "---", "|||"]
GRAY_SHADES = ["#FFFFFF", "#CCCCCC", "#888888", "#FFFFFF", "#AAAAAA", "#333333"]
GRAY_HATCHES = ["", "///", "...", "xxx", "---", "|||"]

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


def _add_bar_inside_labels(ax, bar_groups, val_groups, fontsize=11, pad=0.03, rotation=90):
    """Value labels anchored just *inside* the top of each bar -- can never
    collide with the axes border no matter how tall the bar is, since
    nothing is drawn above the bar's own value. Always rotated the same way
    (default vertical) so every label across every panel/figure reads
    consistently; a short bar's label is free to overflow past the bar's own
    bottom edge into empty axes space, which is harmless -- unlike overflowing
    past the axes border, nothing else is drawn there to collide with."""
    outline = [pe.withStroke(linewidth=2.2, foreground="white")]
    for gi in range(len(val_groups[0])):
        for bars, vals in zip(bar_groups, val_groups):
            bar = bars[gi]
            val = float(vals[gi])
            cx = bar.get_x() + bar.get_width() / 2
            ax.text(cx, val - pad, LABEL_FMT.format(val), fontsize=fontsize,
                    rotation=rotation, fontweight="bold", ha="center", va="top",
                    color="black", path_effects=outline, clip_on=False, zorder=6)


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


FIG3_CAPTION = (
    "Overall per-class F1 of Feature Fusion with shuffle correction (FF + "
    "Shuffle) on the held-out test set, under blind, classifier, and oracle "
    "re-sorting."
)


def draw_fig3(shades, suffix):
    """Overall-only per-class F1: FF Blind / FF Classifier / FF Oracle."""
    model_keys = ["FF + Shuffle\n(Blind)", "FF + Shuffle\n(Classifier)", "FF + Shuffle\n(Oracle)"]
    # "FF + Shuffle" is named once in the caption instead of on every tick,
    # so the tick labels themselves can be short, single-line, and roomier.
    group_labels = ["Blind", "Classifier", "Oracle"]
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

    # Fig. 3 has only 3 wide, touching bars per group (w=0.26 spans the same
    # width as the offset between them), so a horizontal label fits inside
    # each bar without touching its neighbor -- no need for fig. 4's rotation,
    # which exists there because 9 much-narrower bars share each panel.
    _add_bar_inside_labels(ax, bars, vals, rotation=0)
    _set_ylim_fixed(ax)
    _group_separators(ax, len(model_keys))
    _trim_xaxis(ax, len(model_keys))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    _style_axes(ax, "F1 Score")
    handles, labels = ax.get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=3, anchor_y=0.10, fontsize=11)
    _finalize_figure(fig, bottom=0.28, top=0.90)
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
    _legend_below(fig, handles, labels, ncol=3, fontsize=12, anchor_y=0.10)
    _finalize_figure(fig, bottom=0.30, top=0.90)
    _save_fig(fig, f"fig4_fusion_by_direction_{suffix}")


if __name__ == "__main__":
    print(f"loaded metrics from {METRICS_PATH}")
    draw_fig3(COLOR_CLASS, "color")
    draw_fig3(GRAY_CLASS, "grayscale")
    # Fig. 4's bars encode fusion strategy (a different variable from fig. 3's
    # class colors), so it draws from the *other half* of the same six-color/
    # six-hatch palette -- sharing none of fig. 3's (color, hatch) pairs avoids
    # implying a correlation between the two unrelated groupings.
    fig4_color_idx = [1, 3, 5]
    fig4_gray_idx = [3, 4, 5]
    draw_fig4([COLOR_SHADES[i] for i in fig4_color_idx],
              [COLOR_HATCHES[i] for i in fig4_color_idx], "color")
    draw_fig4([GRAY_SHADES[i] for i in fig4_gray_idx],
              [GRAY_HATCHES[i] for i in fig4_gray_idx], "grayscale")
