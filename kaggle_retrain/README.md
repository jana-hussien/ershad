# Kaggle retrain and extra baselines

For the camera-ready version, reviewers asked for more baselines. To compare everything fairly,
`train_baselines.ipynb` trains all models on the same split, labels and training recipe as the
cluster pipeline, with 3 seeds each and a camera-shuffle test for every model. Fig. 2 and the
baseline comparison in the paper come from these runs.

## Running it

Open `train_baselines.ipynb` on Kaggle with a T4 GPU and attach these datasets:
`janahussien90/rgb3thermals-full`, `janahussien90/homography-mats`, `janahussien90/label-cache`.

`MODELS_TO_TRAIN` in the first code cell picks which models run. Set `SMOKE_TEST = True` for a quick
end-to-end check first. Download the output files into a new folder under `results/`.

## Models

| Model | Role |
|---|---|
| Stitched panorama | pixel-level fusion baseline (Model 1) |
| Output Fusion | decision-level fusion baseline (Model 2) |
| Feature Fusion | the paper's model (best settings from the cluster ablation) |
| Single-camera | each direction from its own tile only (the "no fusion" comparison) |
| Permutation-equivariant | shared encoder, pooled across cameras; ignores camera order by design |
| Threshold | no learning; hand-set temperature rules tuned on the training set |

## Results

Each folder in `results/` is one run:

| Folder | Models |
|---|---|
| `main_run/` | Feature Fusion, Output Fusion, single-camera, permutation-equivariant, threshold |
| `stitched_run/` | Stitched panorama |
| `of_ffconfig_run/` | Output Fusion given Feature Fusion's backbone and head settings |
| `threshold_rerun/` | Threshold tuned on all three classes, plus a per-direction version. Both are in the paper; this replaces the threshold row in `main_run/` |

Files in each folder:

- `baselines_table.csv`: overall scores per model (mean over seeds), shuffled scores, parameter count, CPU latency
- `baselines_per_direction_per_seed.csv`: per-class F1 for each direction and seed (input to Fig. 2)
- `baselines_raw.json`: everything the notebook recorded (settings, per-seed results, shuffle test)
- `split_info.json`: which sessions and frames went to train, val and test

`main_run/` also has `label_cache.pkl` (the YOLO auto-labels, used for Fig. 3 and the dataset
release) and `ff_val_predictions.csv` (Feature Fusion's validation predictions, used by the human
validation analysis).
