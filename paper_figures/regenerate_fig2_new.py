"""New Fig. 2: per-class F1 by camera direction for pixel-, decision- and
feature-level fusion, all from the same Kaggle retrain (same split, labels and
recipe; mean over 3 seeds). The no-fusion (per-camera) ablation is reported in
the text, not the figure; set SHOW_NO_FUSION = True to draw it as a fourth bar.

Reads kaggle_retrain/results/{main_run,stitched_run}/baselines_per_direction_per_seed.csv
and reuses the house style from regenerate_fig3_fig4.py. The original Fig. 2
(fig4_fusion_by_direction_*, cluster-run numbers) is left untouched.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regenerate_fig3_fig4 import (BAR_KW, COLOR_HATCHES, COLOR_SHADES, FIG_H, GRAY_HATCHES, GRAY_SHADES,
                                  _add_bar_inside_labels, _finalize_figure, _group_separators, _legend_below,
                                  _save_fig, _set_ylim_fixed, _style_axes, _trim_xaxis)

RESULTS = Path(__file__).resolve().parent.parent / "kaggle_retrain" / "results"
per_seed = pd.concat([pd.read_csv(RESULTS / run / "baselines_per_direction_per_seed.csv")
                      for run in ("main_run", "stitched_run")])
MEAN = per_seed.groupby(["model", "direction", "class"]).f1.mean()

SHOW_NO_FUSION = False

# (legend name, model name in the results files); the no-fusion ablation would sit next to feature-level fusion
MODELS = [
    ("Pixel-Level Fusion", "Stitched"),
    ("Decision-Level Fusion", "Output Fusion"),
    ("No Fusion", "Single-camera"),
    ("Feature-Level Fusion", "Feature Fusion"),
]
CLASSES = [("clear", "Clear"), ("person", "Person"), ("other", "Other Object")]
GROUPS = [("left", "Left"), ("middle", "Middle"), ("right", "Right")]

# Models use one muted blue family (lightest to darkest, same tone as the class colors), with
# Feature Fusion darkest, so model colors never collide with the class colors (green / red / orange).
COLOR_STYLE = [("#C1CFDC", "///"), ("#829EBF", "xxx"), ("#D9D9D9", "\\\\\\"), ("#365F89", "|||")]
GRAY_STYLE = [(GRAY_SHADES[3], GRAY_HATCHES[3]), (GRAY_SHADES[4], GRAY_HATCHES[4]),
              ("#DDDDDD", "\\\\\\"), (GRAY_SHADES[5], GRAY_HATCHES[5])]


# Mark the largest feature-fusion gain over the single-camera baseline (class, direction).
ANNOTATE = ("clear", "middle")


def _annotate_gain(ax, bar_groups, val_groups, gi):
    """Bracket from the single-camera bar to the feature-fusion bar with the F1 gain above it."""
    sc, ff = bar_groups[2][gi], bar_groups[3][gi]
    gain = val_groups[3][gi] - val_groups[2][gi]
    x0, x1 = sc.get_x() + sc.get_width() / 2, ff.get_x() + ff.get_width() / 2
    top = max(val_groups[2][gi], val_groups[3][gi]) + 0.05
    ax.plot([x0, x0, x1, x1], [val_groups[2][gi] + 0.02, top, top, val_groups[3][gi] + 0.02],
            color="black", linewidth=0.9, clip_on=False, zorder=7)
    ax.text((x0 + x1) / 2, top + 0.015, f"+{gain:.3f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold", clip_on=False, zorder=7)


def draw(styles, suffix):
    models, styles = MODELS, styles
    if not SHOW_NO_FUSION:
        keep = [i for i, (_, key) in enumerate(MODELS) if key != "Single-camera"]
        models, styles = [MODELS[i] for i in keep], [styles[i] for i in keep]
    x = np.arange(len(GROUPS))
    w = 0.22 if len(models) == 4 else 0.28
    offsets = np.linspace(-(len(models) - 1) / 2 * w, (len(models) - 1) / 2 * w, len(models))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, FIG_H), sharey=True)
    for ax, (ck, panel_label) in zip(axes, CLASSES):
        bar_groups, val_groups = [], []
        for (name, key), (shade, hatch), offset in zip(models, styles, offsets):
            vals = [MEAN[(key, g, ck)] for g, _ in GROUPS]
            bar_groups.append(ax.bar(x + offset, vals, w, color=shade, hatch=hatch, label=name, **BAR_KW))
            val_groups.append(vals)
        _add_bar_inside_labels(ax, bar_groups, val_groups, fontsize=9 if len(models) == 4 else 11)
        if SHOW_NO_FUSION and ck == ANNOTATE[0]:
            _annotate_gain(ax, bar_groups, val_groups, [g for g, _ in GROUPS].index(ANNOTATE[1]))
        _group_separators(ax, len(GROUPS))
        _trim_xaxis(ax, len(GROUPS))
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in GROUPS])
        ax.set_title(panel_label, fontsize=13, pad=8)
        _style_axes(ax)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("F1 Score")
    for ax in axes:
        _set_ylim_fixed(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=len(models), fontsize=12, anchor_y=0.10)
    _finalize_figure(fig, bottom=0.30, top=0.90)
    _save_fig(fig, f"fig2_new_fusion_by_direction_{suffix}")


if __name__ == "__main__":
    for key in [k for _, k in MODELS]:
        n = per_seed[per_seed.model == key].seed.nunique()
        print(f"{key:15s} seeds={n}  overall={[round(MEAN[(key, 'overall', c)], 3) for c, _ in CLASSES]}")
    draw(COLOR_STYLE, "color")
    draw(GRAY_STYLE, "grayscale")
