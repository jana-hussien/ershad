# Cluster training pipeline

The original training pipeline: 10 Slurm jobs on CMU-Q's deepnet cluster
(`gpujobs.qatar.cmu.edu`). It trains the two baselines, Feature Fusion with a 15-variant ablation,
and Feature Fusion with the camera-identity shuffle correction.

## What's here

| Path | What it is |
|---|---|
| `scripts/` | One `.py` and one `.ipynb` per stage. Slurm runs the `.py`; the `.ipynb` has the same code for reading or running interactively. |
| `slurm/` | One `.sbatch` per stage, and `submit_pipeline.sh` to submit them all. |
| `setup_environment.sh` | One-time setup: conda env, packages, folders. |
| `download_dataset_from_kaggle.sh` | Downloads the dataset straight onto the cluster. |
| `output/` | Saved output of a past full run (figures, tables). A record, not something to regenerate. |

## Stages

| Stage | Script | What it does | Time |
|---|---|---|---|
| 0 | `00_setup_and_labels` | Builds YOLO labels, the split, and a first Feature Fusion model | ~2 h |
| 1 | `01_baseline1` | Model 1: stitched panorama (pixel-level fusion) | |
| 2 | `02_baseline2` | Model 2: output fusion (decision-level fusion) | |
| 3a to 3e | `03a_ff_depth` ... `03e_ff_dropout` | Feature Fusion ablation: depth, width, activation, head, dropout | 1 to 3e together: ~5 to 6 h |
| 4 | `04_ff_best_combo` | Feature Fusion with the best setting from each ablation | ~1.3 h |
| 5 | `05_shuffle_and_final` | Camera-identity classifier, shuffle test, final tables | ~1.2 h |

The whole pipeline takes about 10 hours.

## Before you start

- An account on `gpujobs.qatar.cmu.edu`.
- `setup_conda_env.sh` and `DeepNet_instructions.md`. These come from the course and are not in
  this repo; `setup_environment.sh` calls the first one.
- A Kaggle API token (`kaggle.json`) from [kaggle.com/settings](https://www.kaggle.com/settings),
  under "Create New Token".

## Cluster limits the pipeline works around

- `jupyter nbconvert --execute` doesn't work there, which is why every stage also exists as a
  `.py`.
- At most 4 jobs run at once per user. `submit_pipeline.sh` submits everything anyway and Slurm
  queues the rest.
- Every job needs `--mcs-label` and a time limit (already in each `.sbatch`). Max job time is 48 h.

## Setup (once)

```bash
ssh ANDREWID@gpujobs.qatar.cmu.edu
bash setup_environment.sh                 # after uploading it with scp

scp kaggle.json ANDREWID@gpujobs.qatar.cmu.edu:~/    # from your own machine
bash download_dataset_from_kaggle.sh
```

Upload `scripts/` and `slurm/` to `/data1/ANDREWID/daredevil_run/scripts/` and
`/data1/ANDREWID/daredevil_run/slurm/`.

Then replace the previous user's values with yours (each spot is marked `EDIT`):

- In every `slurm/*.sbatch`: `jhussien` (Andrew ID), `sura` (`--mcs-label`) and `daredevil`
  (conda env name).
- In `setup_environment.sh` and `download_dataset_from_kaggle.sh`: `ANDREWID` and `ENV_NAME`.
- In every `scripts/*.py` and the matching `.ipynb`: `SHARED_DIR`, `DATA_ROOT`, `HOMOGRAPHY_DIR`,
  `RGB_HOMOGRAPHY_DIR`. Keep the two in sync, because the executed notebook is rebuilt from the
  `.ipynb` cell source.
- `SMOKE_TEST` is `True` in the committed scripts (a short test run). Set it to `False` in every
  script once Stage 0 has run cleanly.

## Running and monitoring

```bash
cd /data1/ANDREWID/daredevil_run/slurm
bash submit_pipeline.sh

sacct -u ANDREWID --format=JobID,JobName,State,Elapsed,Partition
tail -f /data1/ANDREWID/daredevil_run/logs/<stage>-<jobid>.out
```

## What each stage produces

All under `/data1/ANDREWID/daredevil_run/`:

- `executed_notebooks/<stage>_executed.ipynb`: the stage's code plus its full log and every figure,
  rebuilt at the end of the run (a copy also goes next to the script).
- `figures/`: every figure as a separate PNG.
- `saved_models/`: one `.pt` per trained model, including every ablation variant.
- `results/`: pickled predictions, F1 scores, parameter counts and latency. Later stages read
  these, so they never retrain earlier models.
- `logs/`: the Slurm `.out` and `.err` files.

Copy them back with `scp -r ANDREWID@gpujobs.qatar.cmu.edu:/data1/ANDREWID/daredevil_run/<folder> .`

## If a stage fails

Labels, the split, results and checkpoints are all cached under `daredevil_run/`, and each stage
skips work it has already finished. Resubmit just that stage's `.sbatch`.
