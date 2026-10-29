# Paper figures

Scripts that draw the paper's figures. They run on a laptop with no GPU, and each one writes color
and grayscale PNG and PDF files into this folder.

## Quickstart

```bash
cd paper_figures
python3 -m venv .venv && source .venv/bin/activate
pip install matplotlib numpy pandas
python3 regenerate_fig2_new.py
python3 regenerate_class_distribution.py
python3 regenerate_fig3_fig4.py
```

Then copy the color version into `paper/figs/` under the name the paper uses (table below).

## Figure map

File names are historical and don't follow the paper's numbering, so use this table.

| Paper | Shows | Output file (here) | In `paper/figs/` as | Script | Data |
|---|---|---|---|---|---|
| Fig. 1 | Pipeline diagram | `pipeline_overview.pdf` | `fig1.pdf` | none, drawn by hand | sources: `pipeline_v2_*.svg` |
| Fig. 2 | Per-class F1 by direction for pixel, decision and feature fusion | `fig2_new_fusion_by_direction_color.*` | `fig2_new.pdf` | `regenerate_fig2_new.py` | `kaggle_retrain/results/` |
| Fig. 3 | Class counts per direction (train + val) | `dataset_class_distribution_trainval_exact_color.png` | `fig3_new.png` | `regenerate_class_distribution.py` | counts typed into the script (from the label cache) |
| Fig. 4 | Feature Fusion under camera shuffle: blind, classifier, oracle | `fig3_shuffle_overall_color.png` | `fig4_new.png` | `regenerate_fig3_fig4.py` | `figure_metrics_correct.json` |

`captions.txt` has the caption text.

## Older versions

These are kept for reference but are no longer in the paper:

- `fig4_fusion_by_direction_*`: the first Fig. 2, made from the cluster run. `regenerate_fig3_fig4.py`
  still writes it.
- `dataset_class_distribution_trainval_{color,grayscale}.*` and the `.webp` files: the first Fig. 3,
  with counts estimated from an image instead of counted from the labels.
- `paper/figs/fig2.png`, `fig3.png`, `fig4.png`: the matching old files on the paper side.

## Other files

- `figure_metrics_correct.json`: hand-checked numbers behind Fig. 4. No script rewrites it; if the
  results change, edit it by hand.
- `figures-creation.ipynb`: the Kaggle notebook that evaluated the cluster-trained models on the
  held-out session. `paper_figure_metrics.py` is a helper it imports. Keep the two together.
- `pipeline_v2_overview.svg`, `pipeline_v2_fig1_fusion.svg`: editable sources for Fig. 1. Edit in
  Inkscape, Illustrator or Figma and export to PDF.
