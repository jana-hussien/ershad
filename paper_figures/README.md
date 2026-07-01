# Paper figures

Latest evaluation bar charts for the DareDevil obstruction-detection paper.

| File | Description |
|------|-------------|
| `figure_metrics.json` | **Source of truth** — metrics exported by `figures-creation.ipynb` Section 13 |
| `fig1_macro_f1_color.png` | Architecture comparison — per-class F1 by camera direction (6 models, 3 panels) |
| `fig2_ff_per_class_color.png` | Feature Fusion + classifier shuffle — per-class F1 on held-out test set |
| `captions.txt` | LaTeX-ready figure captions (not rendered in the plots) |

Copies of the PNGs are also written to the **repo root** when you regenerate.

## Data pipeline

1. Run **`figures-creation.ipynb`** through **Section 13** on Kaggle (models + held-out data).
   - Computes `models_data_by_class` via `get_per_dir_f1_by_class()` and `shuffle_per_dir_by_class()`.
   - Computes fig2 arrays from held-out FF + classifier evaluation.
   - Exports **`figure_metrics.json`** to `OUT_DIR` (on Kaggle: `/kaggle/working/paper_figures/`).

2. Copy `figure_metrics.json` into this folder (commit to git).

3. Regenerate plots locally (no GPU / no Kaggle needed):

```bash
python3 regenerate_paper_figures.py
```

Requires: `matplotlib`, `numpy`

The regenerate script only reads `figure_metrics.json` and draws — it does not run models.
