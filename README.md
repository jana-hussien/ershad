# Directional navigability from low-resolution thermal cameras

This repo holds the code, results and paper for *Directional Navigability Estimation from
Ultra-Low-Resolution Multi-View Thermal Sensing* (Moazam, Hussien, Al-Taweel, Ramalho, Ansari,
Feo-Flushing; CMU Qatar).

**The project in short.** A wearable rig carries three 32×24 thermal cameras (left, middle, right)
and one RGB camera. For each direction we predict whether the path is clear, blocked by a person,
or blocked by another object. The RGB camera is only used while collecting data: a YOLOv8 detector
on the RGB frames produces the training labels automatically, so nothing is labeled by hand. At
run time the models see thermal images only.

**Dataset:** [TartanThermals Navigability Dataset on Hugging Face](https://huggingface.co/datasets/Faiz3005/TartanThermals_Navigability_Dataset)
(thermal frames, labels and homographies; RGB frames are withheld for privacy).

## Where to find things

| Folder | What's in it | Start here |
|---|---|---|
| [`paper/`](paper/) | LaTeX source of the paper, kept in sync with Overleaf | `main.tex` |
| [`paper_figures/`](paper_figures/) | Scripts that draw the paper's figures | [README](paper_figures/README.md) |
| [`cluster_pipeline/`](cluster_pipeline/) | The original training pipeline (10 Slurm jobs on CMU-Q's deepnet cluster): baselines, Feature Fusion ablation, camera-shuffle tests | [README](cluster_pipeline/README.md) |
| [`kaggle_retrain/`](kaggle_retrain/) | Camera-ready retrain on Kaggle: all models on one split, 3 seeds, plus the extra baselines reviewers asked for | [README](kaggle_retrain/README.md) |
| [`human_validation/`](human_validation/) | Check of the auto-labels against two human annotators | [README](human_validation/README.md) |
| [`tools/`](tools/) | Small helpers: export labels for the dataset release, sync `paper/` with Overleaf | see below |
| [`poster/`](poster/) | SURA poster (PDF) | |

## Which results come from where

The paper draws on two sets of training runs, so it helps to know which is which.

| In the paper | Comes from |
|---|---|
| Fig. 2 (fusion strategies by direction) and the comparison with extra baselines (single-camera, permutation-equivariant, threshold) | `kaggle_retrain/` |
| Feature Fusion settings (depth, width, activation, head, dropout), chosen by ablation | `cluster_pipeline/` stages 3a to 4 |
| Fig. 4 (camera shuffle and identity classifier, held-out session) | models trained by `cluster_pipeline/`, evaluated with `paper_figures/figures-creation.ipynb`; numbers saved in `paper_figures/figure_metrics_correct.json` |
| Fig. 3 (class distribution) | label cache in `kaggle_retrain/results/main_run/` |
| Human validation numbers (agreement, kappa) | `human_validation/analyze_labels.py` |

## Common tasks

**Edit the paper.** Edit on Overleaf or in `paper/`, then sync:

```bash
git overleaf-pull           # bring Overleaf edits into paper/, review, commit
git overleaf-push ["msg"]   # send the committed paper/ to Overleaf
```

These are git aliases for `tools/overleaf_sync.sh`. They are stored in your local git config, so
on a fresh clone set them up once (the Overleaf remote needs your Overleaf git token):

```bash
git remote add overleaf https://git.overleaf.com/6a26c865b931c1d62bd86efe
git fetch overleaf
git config alias.overleaf-pull '!tools/overleaf_sync.sh pull'
git config alias.overleaf-push '!f() { tools/overleaf_sync.sh push "$@"; }; f'
git update-ref refs/sync/overleaf overleaf/main   # only if paper/ currently matches Overleaf
```

**Redraw a figure.** See [`paper_figures/README.md`](paper_figures/README.md). It runs on a laptop,
no GPU needed. Copy the output into `paper/figs/` under the name the paper uses.

**Train models again.** Use `kaggle_retrain/train_baselines.ipynb` (Kaggle, one T4 GPU) for the
models in the main comparison. Use `cluster_pipeline/` for the full ablation on the cluster.

**Export labels for the dataset release.**

```bash
python tools/export_release_labels.py          # writes release/labels.csv
```

## Data

The raw recordings and homographies are on Kaggle (`janahussien90/rgb3thermals-full`,
`janahussien90/homography-mats`, `janahussien90/label-cache`). The notebooks expect these attached
as Kaggle inputs. The cluster pipeline downloads its own copy with
`cluster_pipeline/download_dataset_from_kaggle.sh`.

Private files stay out of git: annotator labels, the human-validation answer key, and RGB samples
(see `.gitignore`).
