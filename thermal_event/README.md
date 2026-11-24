# Thermal + event code

Code for the thermal + event zone experiments: M2DGR recordings shrunk to our sensor sizes (32×24 thermal at 4 Hz, GenX320-like events at 320×240), labelled automatically from RGB, then small CNNs that classify each zone (left / center / right).

## Numbering

The file names use the numbering from the initial experiment (25/10 slides), not the new experimental plan:

| In the code | Means | New plan |
|---|---|---|
| E0 (`e0_inspect.py`) | Check a recording is usable, make labels | Dataset step |
| E1 (`e1_e2.py`) | Thermal alone, events alone | Baselines |
| E2 (`e1_e2.py`) | Late blend: average the two sensors' predictions | "Late average" baseline, and late fusion in E3 |

The new E1 (event representation) is `e1_representations.py`; see [E1](#e1-event-representation) below. Nothing for the new E2 (resolution) or E3 (fusion method) exists yet. The plan itself is in [`../experimental_plan/main.tex`](../experimental_plan/main.tex); the E1 list there is older than the 15/11 slides, which this code follows.

## What's in this folder

| Path | What |
|---|---|
| `cache/` | The three processed recordings, ready to train on (see below) |
| `results/` | Outputs of every earlier `e1_e2.py` run, with logs |
| `*.py` | The code; `e1_e2.py` is the one that trains and evaluates |
| `e0_kaggle.ipynb` | Builds caches from raw bags on Kaggle |

## Setup

Python 3.10+.

```bash
pip install torch numpy          # enough to train on the caches (e1_e2.py)
pip install rosbags opencv-python ultralytics   # only to build new caches from the raw bags
```

## The data

You don't need the raw bags (10–15 GB each). Each recording has already been turned into a small **cache** (60–90 MB). There are three recordings from M2DGR, 1,103 samples in total:

| Recording | Place | Samples |
|---|---|---|
| `room_01` | room | 290 |
| `hall_02` | hall | 514 |
| `door_02` | atrium / door | 299 (of 508) |

`door_02` starts outdoors. Its `keep.json` (`{"start": 52}`) makes the loader keep only the indoor part, after the robot passes the door at ~51 s.

One cache folder looks like this:

```
cache/room_01/
  meta.json          settings, sensor sizes, sample count, class counts
  samples.npz        per 4 Hz sample: t (time), thermal (n×24×32, no noise), gyro, ev_bin
  labels.csv         per sample and zone: label (clear/person/furniture/other) and near (0/1)
  boxes.csv          every YOLO box on the RGB frames, so label rules can change without the bag
  imu.npz            full IMU stream (time, gyro, accel)
  events/chunk_*.npz event counts in continuous 40 ms bins, shape (bins, 2, 240, 320), uint8,
                     channel 0 = got darker, channel 1 = got brighter; 10 s per file
```

`ev_bin` in `samples.npz` is the index of the 40 ms bin that ends at (or just before) that sample.

### Loading it in Python

```python
import dataset
recs = dataset.load_all("cache")   # run from thermal_event/; list, one dict per recording
r = recs[0]
r["thermal"]   # (n, 24, 32) float32
r["events"]    # (n, 2, 240, 320) uint8, the 40 ms bin before each sample
r["y"]         # (n, 3) int, class index per zone; classes = labels.CLASSES
r["near"]      # (n, 3) bool
r["gyro"]      # (n,) rad/s, used for the still/moving split
r["t"]         # (n,) seconds
```

`load_recording` only keeps the one 40 ms bin per sample. For longer windows (80, 120, 240 ms) add up consecutive bins from `events/chunk_*.npz` (`first_bin` + `bins` in each file) ending at `ev_bin`. The cache has counts, not individual event timestamps, so a time surface or a "fixed number of events" input would need a new cache built from the bags.

## Running the experiments

Run from this folder:

```bash
cd thermal_event
python e1_e2.py --cache cache --out results/new_place_binary --task binary
```

| Flag | Default | Meaning |
|---|---|---|
| `--task` | `classes` | `classes` = clear/person/furniture/other; `binary` = free vs blocked |
| `--split` | `recording` | `recording` = leave one recording out ("new place"); `blocks` = hold out time blocks inside each recording ("seen place") |
| `--seeds` | 3 | training runs per model |
| `--epochs` | 30 | |

Uses the Mac GPU (`mps`) if there is one, else CPU. Writes `summary.md` (tables with 95% CIs, plus parameters / MFLOPs / CPU ms), `metrics.csv` and `predictions.csv`.

The numbers on the 25/10 slides (thermal 75% seen place, 62% new place; "Hamming acc." column) came from:

```bash
python e1_e2.py --cache cache --out results/e1_e2_seen_binary_3rec --task binary --split blocks --seeds 1 --epochs 20
python e1_e2.py --cache cache --out results/e1_e2_binary --task binary
```

Those outputs are already in `results/`. Running a command again overwrites its `--out` folder, so give new runs a new name.

## Where to change things

- **Event input** (E1): `representations.py`, and the `SPECS` list in `e1_representations.py`.
- **Networks**: `ZoneNet` and `make_model()` in `e1_e2.py`. Both sensors use 5 conv layers pooled to a 2×3 grid; the event one downsamples harder because its input is ~100× larger.
- **Fusion** (new E3): right now the only fusion is the late average in `main()`. Early or mid fusion needs a new model that takes both inputs.
- **Labels**: `labels.py` (zone overlap, which YOLO classes count as furniture, near threshold).
- **Sensor simulation**: `simulate.py` (thermal size and noise, event window).

## E1: event representation

Which input should the event branch get? Every input below goes through the same small network, training recipe and leave-one-recording-out folds; only the input changes.

| Input | What the network gets | Built from |
|---|---|---|
| `count_40/80/120/240` | darker / brighter counts per pixel over 40-240 ms (2 ch) | old 40 ms cache (exact) |
| `stacked_6x40` | the last six 40 ms count images side by side (12 ch) | old 40 ms cache (exact) |
| `voxel_6` | 6 slices of brighter minus darker; each event split between its two nearest slices by its exact time | bag |
| `time_surface` | per pixel and polarity, exp(-age / 50 ms) since it last changed (2 ch) | bag |
| `fixed_n_50k/200k/800k` | counts of the most recent N events, however long that took (max 1 s) | bag |
| `snn_12x20` | spiking version of the same CNN (Li et al.'s LIF neurons), fed 12 steps of 20 ms | bag |
| `graph` | 2,048 sampled events as graph nodes, linked to their 16 nearest in (x, y, t) | bag |

The same inputs with each paper's own settings (checked against the official code where public):

| Input | Setting | Source |
|---|---|---|
| `count_40_raw` | plain counts, no log scale | count-image papers |
| `stacked_rvt_10x5` | last 50 ms in 10 bins, counts clipped at 10 (20 ch) | RVT code (`stacked_hist.yaml`) |
| `voxel5_norm` | 5 slices, non-zero values normalised to mean 0 / std 1 | E2VID code, Li et al. |
| `time_surface_tau25/100` | the time surface with a 25 or 100 ms fade (ours is 50 ms) | sweep |
| `time_surface_zubic` | 5 ms fade, sampled at 6 moments × 2 polarities (12 ch) | Zubic et al. |
| `fixed_n_27k` | 26,880 events = 0.35 × width × height | E2VID code default |
| `snn_voxel5` | spiking network fed the normalised 5-slice voxel, one slice per step | Li et al. |
| `time_surface_tevio` | TEVIO's adaptive time surface (eq. 4); the paper gives no values and its threshold needs depth, so bounds 100 / 10 ms, 5 × 5 neighbourhood and a constant threshold are ours | TEVIO, our settings |
| `aegnn` | AEGNN's graph network from its official code, N-Cars settings (10,000 events, radius 3, kernel 2, 8-32 channels; 32 k parameters). Its Gen1 setting rebuilt the same way has 20.2 M parameters and ~49 G multiply-adds per sample, too big to train here | AEGNN code |
| `eventmg_style` | EventMG's structure from the paper (20,000 events, δst 0.05, components ≥ 100, APN, 6-direction Mamba, component graph, gated fusion); its code is not released and the paper gives no sizes, so one stage of width 64 (~93 k parameters) is ours | EventMG paper, our sizes |

`aegnn` and `eventmg_style` are heavy, so they run in a second notebook, `e1_graphs_kaggle.ipynb`, on the inputs the first one saves (AEGNN needs `torch_spline_conv` and `torch_cluster`; EventMG-style needs `mamba_ssm` on a CUDA GPU).
| `thermal`, `thermal_x2` | references: thermal only, and a twice-as-wide thermal model | old cache |
| `thermal+snn`, `thermal+stacked_cnn` | action item from 15/11: the spiking branch with thermal in place of RGB, and the same fusion with a CNN branch for comparison | both |

The old cache only has counts per 40 ms bin, so the inputs that need exact event times are built from the raw bags by `build_e1_cache.py`. That runs on Kaggle (the bags are 10-15 GB each): `e1_kaggle.ipynb`, generated by `make_e1_kaggle_notebook.py`, fetches the old cache from this repo, downloads the bags, runs a smoke test, builds everything and trains E1 unattended. With your Kaggle API key added as notebook secrets it also saves each raw bag and each recording's built inputs as private Kaggle datasets, so later runs need no OneDrive download. Instructions are at the top of the notebook.

| File | What |
|---|---|
| `representations.py` | how every input is computed (build side and training side) |
| `build_e1_cache.py` | bag → `e1_cache/<recording>/` (~1 MB per sample), checked against the old cache's 40 ms bins |
| `e1_representations.py` | trains and evaluates all E1 models; resumable; `--smoke` checks everything runs in minutes; `--models` / `--skip` pick models |
| `e1_kaggle.ipynb`, `e1_graphs_kaggle.ipynb` | part 1 (download, build, core models) and part 2 (heavy graph models), both generated by `make_e1_kaggle_notebook.py` |
| `e1_visualize.py` | one figure showing every input for a real sample |

Running locally instead (needs the e1 cache; the spiking network needs a GPU with ~6 GB):

```bash
python e1_representations.py --cache cache --e1-cache e1_cache --out results/e1
```

Main metric is per-zone macro F1; `summary.md` also reports the paired difference against `count_40` (the input used so far), % zones correct, missed obstacles (zone called clear when it isn't), per-class and per-place F1, parameters, MFLOPs and CPU ms.

## Building a cache from a raw bag (only if needed)

```bash
python e0_inspect.py BAG --summary-only               # what's in the bag
python build_cache.py BAG --out cache/room_01 --yolo yolov8m.pt
```

`yolov8m.pt` downloads automatically through `ultralytics` if it isn't there. `e0_kaggle.ipynb` runs this on Kaggle so you don't need the disk space locally; it's generated by `make_kaggle_notebook.py`, so edit the .py files rather than the notebook. `bag_io.py` is the only file that knows the bag format; data from the real device would need a new loader with the same functions.
