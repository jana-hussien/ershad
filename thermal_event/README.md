# Thermal + event code

Code for the thermal + event zone experiments: M2DGR recordings shrunk to our sensor sizes (32×24 thermal at 4 Hz, GenX320-like events at 320×240), labelled automatically from RGB, then small CNNs that classify each zone (left / center / right).

## Numbering

The file names use the numbering from the initial experiment (25/10 slides), not the new experimental plan:

| In the code | Means | New plan |
|---|---|---|
| E0 (`e0_inspect.py`) | Check a recording is usable, make labels | Dataset step |
| E1 (`e1_e2.py`) | Thermal alone, events alone | Baselines |
| E2 (`e1_e2.py`) | Late blend: average the two sensors' predictions | "Late average" baseline, and late fusion in E3 |

Nothing for the new E1 (event representation), E2 (resolution) or E3 (fusion method) exists yet. The new plan itself is in [`../experimental_plan/main.tex`](../experimental_plan/main.tex).

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

- **Event input** (new E1): `inputs()` in `e1_e2.py` and `event_input()` in `dataset.py`. The network's first layer takes `in_ch` channels, so a stacked or voxel input only needs a different channel count in `make_model()`.
- **Networks**: `ZoneNet` and `make_model()` in `e1_e2.py`. Both sensors use 5 conv layers pooled to a 2×3 grid; the event one downsamples harder because its input is ~100× larger.
- **Fusion** (new E3): right now the only fusion is the late average in `main()`. Early or mid fusion needs a new model that takes both inputs.
- **Labels**: `labels.py` (zone overlap, which YOLO classes count as furniture, near threshold).
- **Sensor simulation**: `simulate.py` (thermal size and noise, event window).

## Building a cache from a raw bag (only if needed)

```bash
python e0_inspect.py BAG --summary-only               # what's in the bag
python build_cache.py BAG --out cache/room_01 --yolo yolov8m.pt
```

`yolov8m.pt` downloads automatically through `ultralytics` if it isn't there. `e0_kaggle.ipynb` runs this on Kaggle so you don't need the disk space locally; it's generated by `make_kaggle_notebook.py`, so edit the .py files rather than the notebook. `bag_io.py` is the only file that knows the bag format; data from the real device would need a new loader with the same functions.
