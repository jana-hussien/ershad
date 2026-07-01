# Paper figures

Latest evaluation bar charts for the DareDevil obstruction-detection paper.

| File | Description |
|------|-------------|
| `fig1_macro_f1_color.png` | Architecture comparison — per-class F1 by camera direction (6 models, 3 panels) |
| `fig2_ff_per_class_color.png` | Feature Fusion + classifier shuffle — per-class F1 on held-out test set |
| `captions.txt` | LaTeX-ready figure captions (not rendered in the plots) |

Copies of the PNGs are also written to the **repo root** for quick access.

## Regenerate

From the repo root:

```bash
python3 regenerate_paper_figures.py
```

Requires: `matplotlib`, `numpy`
