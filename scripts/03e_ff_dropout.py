
import copy
import pickle
import sys
import json
import random
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless deepnet/Slurm jobs have no display -- savefig_show() still
                         # saves every figure to disk; plt.show() is just a safe no-op with Agg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


STAGE_NAME = "03e_ff_dropout"
print(f"=== Running stage: {STAGE_NAME} ===")


# ---- paths (EDIT THESE) ----
# DATA_ROOT contains part1, part2, part3, ... partN (any number of capture sessions) --
# auto-discovered below by globbing "part*", not hardcoded to 3.
DATA_ROOT = Path("/data1/jhussien/dataset/all_runs - Copy")
HOMOGRAPHY_DIR = Path("/data1/jhussien/dataset/thermal_homography_mats 2/thermal_homography_mats")
RGB_HOMOGRAPHY_DIR = Path("/data1/jhussien/dataset/rgb homography final/rgb homography final")
# ^ These mirror the CONFIRMED correct Kaggle source paths:
#   DATA_ROOT           <- .../faizmoazam/full-stitched-and-unstitched/all_runs - Copy
#   HOMOGRAPHY_DIR       <- .../faizmoazam/thermal-homography-mats/thermal_homography_mats 2/thermal_homography_mats
#   RGB_HOMOGRAPHY_DIR   <- .../faizmoazam/rgb-homography/rgb homography final/rgb homography final
# After `kaggle datasets download <slug> -p DEST --unzip`, DEST will contain whatever the zip's
# internal folder structure is -- run `ls -R /data1/jhussien/dataset | head -50` after
# downloading and confirm these three paths actually exist before running Stage 0 for real.

YOLO_WEIGHTS = "yolov8x.pt"
MAX_RGB_GAP_MS = 150          # beyond this, treat a "matched" RGB frame as not actually usable
COCO_PERSON_CLASS_ID = 0      # YOLOv8 COCO class id for "person" -- everything else -> "other object"

# ---- rig geometry ----
SENSOR_W, SENSOR_H = 32, 24                 # native per-camera thermal resolution

# Full physical rig (left -> right, as in the rig photo). Only the first 3 (the RGB-supervised,
# RGB-overlapping middle cameras) are trained/evaluated on; the other two are inference-only.
ALL_CAMS_LEFT_TO_RIGHT = [3, 2, 5, 4, 6]
ACTIVE_CAMERAS = [2, 5, 4]                  # left, middle, right -- the 3 cameras used for
                                             # training AND the main evaluation
SIDE_CAMERAS = [3, 6]                       # inference-only, qualitative, no ground truth
CAM_ORDER_LEFT_TO_RIGHT = ACTIVE_CAMERAS    # kept under this name for drop-in compatibility with
                                             # code ported from the 3 source notebooks
DIRECTIONS = ["left", "middle", "right"]    # 1:1 with CAM_ORDER_LEFT_TO_RIGHT
CAM_TO_DIRECTION = dict(zip(CAM_ORDER_LEFT_TO_RIGHT, DIRECTIONS))
DIRECTION_TO_CAM = {v: k for k, v in CAM_TO_DIRECTION.items()}

CLUSTER_BUILD_ORDER = [4, 5, 2]   # BFS root/reference for the panorama -- re-verify against a
                                   # real frame from your 3-cam stitched .npy if this changes
PANORAMA_REFERENCE_CAM = CLUSTER_BUILD_ORDER[0]

# 3-class object taxonomy
CLASS_NAMES = ["no_object", "person", "other_object"]
NUM_CLASSES = len(CLASS_NAMES)

# ---- !!! sensor mirroring fix -- see baseline notebooks for full rationale !!! ----
FLIP_THERMAL_HORIZONTALLY = True
FLIP_CAM_FOOTPRINTS_HORIZONTALLY = False

NUM_CAMS = len(CAM_ORDER_LEFT_TO_RIGHT)

# ---- training config ----
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
SEED = 42
NUM_WORKERS = 2
VAL_FRAC = 0.2
N_SPLIT_BLOCKS = 10

# ---- shared output dirs (deepnet edit) ----
# Every stage notebook/script in this pipeline points at the SAME paths so they can hand off
# work to each other -- one stage's saved labels/split/checkpoints/results are the next stage's
# inputs. EDIT SHARED_DIR below to match your actual /data1/<andrewID>/ path before submitting
# any sbatch job -- every stage script must use the identical path.
SHARED_DIR = Path("/data1/jhussien/daredevil_run")   # <-- EDIT jhussien
MODEL_DIR = SHARED_DIR / "saved_models"
FIGURES_DIR = SHARED_DIR / "figures"
RESULTS_DIR = SHARED_DIR / "results"        # pickled per-model result dicts (preds/trues/masks/
                                             # macro_f1/clear_f1/params/latency), so a later stage
                                             # can rebuild ff_results etc. without re-training
CACHE_DIR = SHARED_DIR / "cache"            # label cache + train/val split, computed once in
                                             # Stage 0 and reused (not recomputed) by every later
                                             # stage -- this is what avoids re-paying the ~20min
                                             # YOLO labeling cost in every single sbatch job
EXECUTED_NB_DIR = SHARED_DIR / "executed_notebooks"  # see _assemble_executed_notebook below
for _d in (MODEL_DIR, FIGURES_DIR, RESULTS_DIR, CACHE_DIR, EXECUTED_NB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# which pipeline stage this notebook is (purely informational -- printed at the top of every run
# so a stdout/slurm log immediately tells you which stage produced it)
STAGE_NAME = "UNSET -- override in each stage notebook"


def savefig_show(name):
    '''Saves the CURRENT figure to FIGURES_DIR/{name}.png (always, for the directory of saved
    figures) AND displays it inline if running in a live notebook kernel. Also registers the
    figure with _FIGURES_THIS_RUN so _assemble_executed_notebook can embed it back into a real
    .ipynb at the end of the run (see below -- needed because `jupyter nbconvert --execute`
    doesn't work on deepnet, so a plain `python script.py` run has no other way to produce a
    notebook with figures showing).'''
    path = FIGURES_DIR / f"{name}.png"
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [figure saved] {path}")
    _FIGURES_THIS_RUN.append((name, path))


def pickle_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "rb") as f:
        return pickle.load(f)


# =====================================================================================
# Executed-notebook assembly: deepnet can't run `jupyter nbconvert --execute` (it's broken on
# that system -- see DeepNet_instructions.md), so a stage runs as a plain `python script.py`
# instead, which on its own would leave you with nothing but a .out text log -- no notebook with
# figures to look at. This rebuilds one anyway:
#   1. stdout is "teed" to a string buffer as well as the real terminal (so the Slurm .out file
#      is unaffected, but the full run log is also captured in-process).
#   2. every figure saved via savefig_show() is tracked in _FIGURES_THIS_RUN.
#   3. at normal process exit (atexit), if this is actually running as `python <stage>.py` (not
#      inside a live Jupyter kernel), it loads the ORIGINAL <stage>.ipynb (same directory, same
#      filename stem -- this is the .ipynb you uploaded, sitting right next to the .py script
#      Step 6 generated from it) and writes out <stage>_executed.ipynb: every original cell's
#      SOURCE preserved exactly, plus a final "Run Output" section with every figure embedded
#      as a real inline image (base64 PNG, viewable in Jupyter with no external file needed) and
#      the complete captured stdout log as a single text output cell.
# =====================================================================================
import atexit
import base64
import io as _io


_FIGURES_THIS_RUN = []


class _StdoutTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


_stdout_buffer = _io.StringIO()
_is_script_run = (
    len(sys.argv) > 0 and sys.argv[0].endswith(".py") and Path(sys.argv[0]).exists()
)
if _is_script_run:
    sys.stdout = _StdoutTee(sys.__stdout__, _stdout_buffer)


def _assemble_executed_notebook():
    if not _is_script_run:
        return
    try:
        script_path = Path(sys.argv[0]).resolve()
        source_nb_path = script_path.with_suffix(".ipynb")
        if source_nb_path.exists():
            with open(source_nb_path, "r", encoding="utf-8") as f:
                original = json.load(f)
            out_cells = []
            for c in original["cells"]:
                c = dict(c)
                if c["cell_type"] == "code":
                    c["outputs"] = []
                    c["execution_count"] = None
                out_cells.append(c)
        else:
            out_cells = [{
                "cell_type": "markdown", "metadata": {},
                "source": [f"_(original {source_nb_path.name} not found alongside the script -- "
                           "only run output is shown below)_"],
            }]

        out_cells.append({
            "cell_type": "markdown", "metadata": {},
            "source": [f"## Run output — {STAGE_NAME}\n",
                       f"Executed {datetime.now().isoformat(timespec='seconds')}"],
        })
        for fig_name, fig_path in _FIGURES_THIS_RUN:
            out_cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"### {fig_name}"]})
            with open(fig_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            out_cells.append({
                "cell_type": "code", "execution_count": None, "metadata": {}, "source": [],
                "outputs": [{
                    "output_type": "display_data",
                    "data": {"image/png": b64, "text/plain": [f"<Figure: {fig_name}>"]},
                    "metadata": {},
                }],
            })

        out_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### Full captured stdout log"]})
        out_cells.append({
            "cell_type": "code", "execution_count": None, "metadata": {}, "source": [],
            "outputs": [{
                "output_type": "stream", "name": "stdout",
                "text": _stdout_buffer.getvalue().splitlines(keepends=True),
            }],
        })

        notebook = {
            "cells": out_cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10"},
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        out_name = f"{STAGE_NAME}_executed.ipynb"
        for dest in (script_path.parent / out_name, EXECUTED_NB_DIR / out_name):
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1)
            sys.__stdout__.write(f"[executed notebook saved] {dest}\n")
    except Exception as e:
        sys.__stdout__.write(f"[warning] could not assemble executed notebook: {e}\n")


atexit.register(_assemble_executed_notebook)


# ---- smoke test mode ----
SMOKE_TEST = True
SMOKE_TEST_N_SAMPLES = 60
SMOKE_TEST_EPOCHS = 1
SMOKE_TEST_BATCH_SIZE = 4

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
RANDOM_SEED = SEED


def _part_sort_key(path):
    '''Natural sort: part2 < part10, not lexicographic ('part10' < 'part2' as plain strings).'''
    m = re.search(r"(\d+)$", path.name)
    return int(m.group(1)) if m else 0


def discover_capture_roots(data_root):
    '''Scans data_root for every subfolder named part<N> (any N, any digit count -- part1..part10+),
    sorted numerically. Returns {part_name: path}.'''
    roots = sorted(Path(data_root).glob("part*"), key=_part_sort_key)
    if not roots:
        raise FileNotFoundError(f"No 'part*' folders found under {data_root}")
    return {p.name: p for p in roots}


CAPTURE_ROOTS = discover_capture_roots(DATA_ROOT)
print(f"Discovered {len(CAPTURE_ROOTS)} capture part(s): {list(CAPTURE_ROOTS.keys())}")


def build_capture_parts(capture_roots):
    '''Auto-discovers each part's thermal/rgb/sync subfolders by prefix (thermal-g*, rgb-g*,
    sync-g*), since the exact timestamp suffix differs per recording session.'''

    def _find_one(root, prefix):
        matches = sorted(Path(root).glob(f"{prefix}-g*"))
        if not matches:
            raise FileNotFoundError(f"No '{prefix}-g*' folder found under {root}")
        if len(matches) > 1:
            warnings.warn(f"Multiple '{prefix}-g*' folders under {root}, using {matches[0]}")
        return matches[0]

    parts = {}
    for name, root in capture_roots.items():
        thermal_dir = _find_one(root, "thermal")
        rgb_dir = _find_one(root, "rgb")
        sync_dir = _find_one(root, "sync")
        parts[name] = {
            "thermal_dir": thermal_dir,
            "rgb_dir": rgb_dir,
            "sync_csv": sync_dir / "sync.csv",
        }
    return parts


CAPTURE_PARTS = build_capture_parts(CAPTURE_ROOTS)
for _name, _paths in CAPTURE_PARTS.items():
    print(f"{_name}: thermal={_paths['thermal_dir'].name}  rgb={_paths['rgb_dir'].name}  "
          f"sync_csv exists={_paths['sync_csv'].exists()}")


CAPTURE_RE = re.compile(r"^data_capture_(\d+)_(\d+)_(\d+)\.json$")
CAM_NAME_RE = re.compile(r"cam(\d+)")


def discover_camera_ids(folder):
    folder = Path(folder)
    return sorted({int(m.group(1)) for f in folder.iterdir() if (m := CAPTURE_RE.match(f.name))})


def list_captures(folder, cam_id):
    folder = Path(folder)
    captures = []
    for f in folder.iterdir():
        m = CAPTURE_RE.match(f.name)
        if m and int(m.group(1)) == cam_id:
            captures.append((int(m.group(2)), f))
    captures.sort(key=lambda item: item[0])
    return [path for _seq, path in captures]


def load_capture(path):
    with open(path) as f:
        text = f.read()
    text = re.sub(r"-nan\b", "NaN", text)
    return json.loads(text)


def build_camera_sequence(folder, cam_id):
    '''(frame_index, unix_time_ms, grid) per frame, in capture order. Any 0/NaN pixel is
    filled forward from the previous frame's value at that pixel (sticky fill-forward).
    FLIP_THERMAL_HORIZONTALLY corrects the raw sensor's left-right mirror vs RGB/real-world.'''
    paths = list_captures(folder, cam_id)
    if not paths:
        raise ValueError(f"No captures found for camera {cam_id} in {folder}")
    sequence = []
    buffer = np.zeros(SENSOR_W * SENSOR_H, dtype=np.float64)
    for path in paths:
        data = load_capture(path)
        raw = np.array(data["data"], dtype=np.float64)
        mask = (raw != 0) & ~np.isnan(raw)
        buffer[mask] = raw[mask]
        grid = buffer.copy().reshape(SENSOR_H, SENSOR_W)
        if FLIP_THERMAL_HORIZONTALLY:
            grid = np.ascontiguousarray(grid[:, ::-1])
        sequence.append((data["frame_index"], data["unix_time_ms"], grid))
    return sequence


def compute_normalization_range(grids, low_pct=1, high_pct=99):
    flat = np.concatenate([g.ravel() for g in grids])
    return np.percentile(flat, low_pct), np.percentile(flat, high_pct)


def temps_to_uint8(grid, vmin, vmax):
    clipped = np.clip(grid, vmin, vmax)
    return ((clipped - vmin) / max(vmax - vmin, 1e-6) * 255).astype(np.uint8)


def load_homography_json(path):
    with open(path) as f:
        return json.load(f)


def discover_homography_files(root):
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*_homography.json")):
        data = load_homography_json(path)
        a_m, b_m = CAM_NAME_RE.match(data.get("cam_a", "")), CAM_NAME_RE.match(data.get("cam_b", ""))
        if not (a_m and b_m):
            continue
        entries.append({"cam_a": int(a_m.group(1)), "cam_b": int(b_m.group(1)),
                         "H": np.array(data["H"], dtype=np.float64)})
    return entries


def load_sync_table(csv_path):
    return pd.read_csv(csv_path).to_dict("records")


def rgb_gap_ok(row, max_gap_ms=MAX_RGB_GAP_MS):
    gap = row.get("rgb_gap_ms", "-1")
    files = row.get("rgb_files", "")
    if not files:
        return False
    try:
        return 0 <= float(gap) <= max_gap_ms
    except ValueError:
        return False


def load_all_sync_rows(capture_parts=None, seed=RANDOM_SEED):
    '''Combines every discovered part's sync.csv -- (part, anchor_index) is the unique sample
    identity (anchor_index alone restarts at 0 for every part).'''
    capture_parts = capture_parts or CAPTURE_PARTS
    all_rows = []
    for part_name, paths in capture_parts.items():
        rows = load_sync_table(paths["sync_csv"])
        for i, row in enumerate(rows):
            row["_part"] = part_name
            row.setdefault("anchor_index", i)
        all_rows.extend(rows)
    random.Random(seed).shuffle(all_rows)
    print(f"total: {len(all_rows)} rows across {len(capture_parts)} part(s)")
    return all_rows


def _transform_corners(width, height, T):
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4, 1))])
    transformed = (T @ corners_h.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def compute_canvas_bounds(frame_shapes, transforms):
    all_points = np.vstack([_transform_corners(w, h, T) for (h, w), T in zip(frame_shapes, transforms)])
    min_x, min_y = np.floor(all_points.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_points.max(axis=0)).astype(int)
    return min_x, min_y, max_x, max_y


def build_clusters(homography_entries, cam_ids):
    graph = {cam: [] for cam in cam_ids}
    for entry in homography_entries:
        a, b, H = entry["cam_a"], entry["cam_b"], entry["H"]
        if a not in graph or b not in graph:
            continue
        graph[a].append((b, np.linalg.inv(H)))
        graph[b].append((a, H))

    visited, clusters = set(), []
    for root_cam in cam_ids:
        if root_cam in visited:
            continue
        visited.add(root_cam)
        cams, transforms = [root_cam], [np.eye(3)]
        transform_to_root = {root_cam: np.eye(3)}
        queue = [root_cam]
        while queue:
            current = queue.pop(0)
            for neighbor, T_neighbor_to_current in graph[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                transform_to_root[neighbor] = transform_to_root[current] @ T_neighbor_to_current
                cams.append(neighbor)
                transforms.append(transform_to_root[neighbor])
                queue.append(neighbor)
        clusters.append({"cams": cams, "transforms": transforms})
    return clusters


def warp_image(img, H, output_size):
    return cv2.warpPerspective(img, H, output_size, flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def stitch_frames(frames, transforms):
    shapes = [f.shape for f in frames]
    min_x, min_y, max_x, max_y = compute_canvas_bounds(shapes, transforms)
    canvas_w, canvas_h = max_x - min_x, max_y - min_y
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    canvas = np.zeros((canvas_h, canvas_w), dtype=frames[0].dtype)
    for frame, T in zip(frames, transforms):
        warped = warp_image(frame, translation @ T, (canvas_w, canvas_h))
        mask = warped > 0
        canvas[mask] = warped[mask]
    return canvas


# Build geometry over ONLY the 3 active cameras (cam2/cam5/cam4) -- this is the cluster that's
# actually guaranteed to be fully connected by HOMOGRAPHY_DIR's thermal-thermal pairs (cam4<->cam5,
# cam5<->cam2). Side cameras (cam3/cam6) are NOT included here: their homography edges to this
# cluster aren't guaranteed to exist (see baseline1's open-item notes -- "any leftover cam3/cam6
# homography files are harmless and ignored"), and nothing downstream actually needs side-camera
# footprint polygons -- Section 16's side-camera swap-in check works directly on raw tiles, never
# on CAM_FOOTPRINTS. Earlier attempt to also build side-camera footprints here caused a KeyError
# when cam2 ended up isolated in its own disconnected cluster -- this is the fix.
homography_entries = discover_homography_files(HOMOGRAPHY_DIR)
clusters = build_clusters(homography_entries, CLUSTER_BUILD_ORDER)
main_cluster = next((c for c in clusters if len(c["cams"]) == NUM_CAMS and set(c["cams"]) == set(ACTIVE_CAMERAS)), None)
if main_cluster is None:
    raise RuntimeError(
        f"Could not find a single connected cluster containing exactly {ACTIVE_CAMERAS} in "
        f"HOMOGRAPHY_DIR's thermal-thermal pairs. Clusters found: {[c['cams'] for c in clusters]} "
        "-- check that cam4<->cam5 and cam5<->cam2 homography files are present."
    )
print("cluster cams (BFS order from reference):", main_cluster["cams"])

PANORAMA_MIN_X, PANORAMA_MIN_Y, PANORAMA_MAX_X, PANORAMA_MAX_Y = compute_canvas_bounds(
    [(SENSOR_H, SENSOR_W)] * len(main_cluster["cams"]), main_cluster["transforms"]
)
PANORAMA_W = PANORAMA_MAX_X - PANORAMA_MIN_X
PANORAMA_H = PANORAMA_MAX_Y - PANORAMA_MIN_Y
print(f"panorama canvas: {PANORAMA_H} x {PANORAMA_W}  -- verify visually against a real frame")

PANORAMA_TRANSLATION = np.array([
    [1, 0, -PANORAMA_MIN_X], [0, 1, -PANORAMA_MIN_Y], [0, 0, 1],
], dtype=np.float64)

CAM_FOOTPRINTS = {}
CAM_TRANSFORMS = {}
for cam_id, T in zip(main_cluster["cams"], main_cluster["transforms"]):
    CAM_TRANSFORMS[cam_id] = T
    full_T = PANORAMA_TRANSLATION @ T
    corners = _transform_corners(SENSOR_W, SENSOR_H, full_T)
    x1, y1 = corners[:, 0].min(), corners[:, 1].min()
    x2, y2 = corners[:, 0].max(), corners[:, 1].max()
    CAM_FOOTPRINTS[cam_id] = {"polygon": corners, "bbox": (x1, y1, x2, y2)}
    tag = "ACTIVE" if cam_id in ACTIVE_CAMERAS else "side (inference-only)"
    print(f"cam{cam_id} [{tag}]: bbox = ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")


_RAW_SEQUENCES = {}  # part_name -> {cam_id: [grid, grid, ...]}, cached on first access


def _get_raw_sequences(part):
    if part not in _RAW_SEQUENCES:
        thermal_dir = CAPTURE_PARTS[part]["thermal_dir"]
        cams_needed = list(set(ALL_CAMS_LEFT_TO_RIGHT))
        _RAW_SEQUENCES[part] = {
            cam_id: [grid for _idx, _ts, grid in build_camera_sequence(thermal_dir, cam_id)]
            for cam_id in cams_needed
        }
    return _RAW_SEQUENCES[part]


def load_camera_tiles(part, anchor_index, cam_order=None):
    '''Raw (SENSOR_H, SENSOR_W) grids for this (part, anchor_index), one per camera, in
    cam_order (defaults to the 3 ACTIVE_CAMERAS).'''
    cam_order = cam_order or CAM_ORDER_LEFT_TO_RIGHT
    sequences = _get_raw_sequences(part)
    return [sequences[cam_id][anchor_index].astype(np.float32) for cam_id in cam_order]


def build_input_tensor(tiles):
    '''tiles: list of NUM_CAMS (H, W) arrays -> [NUM_CAMS, 1, H, W] -- used by Baselines 2-4
    (unstitched per-camera input).'''
    return np.stack([t[None, :, :] for t in tiles], axis=0)


def pad_or_crop_to_canvas(panorama, target_h=None, target_w=None):
    target_h = target_h or PANORAMA_H
    target_w = target_w or PANORAMA_W
    h, w = panorama.shape
    out = np.zeros((target_h, target_w), dtype=panorama.dtype)
    copy_h, copy_w = min(h, target_h), min(w, target_w)
    out[:copy_h, :copy_w] = panorama[:copy_h, :copy_w]
    return out


def load_panorama(part, anchor_index, cam_order=None):
    '''Stitched panorama -- used only by Baseline 1. Always recomputed from raw tiles +
    homographies (no pre-saved .npy dependency), consistent with FLIP_THERMAL_HORIZONTALLY.'''
    cam_order = cam_order or CAM_ORDER_LEFT_TO_RIGHT
    sequences = _get_raw_sequences(part)
    frames = [sequences[c][anchor_index] for c in main_cluster["cams"] if c in cam_order or c in ALL_CAMS_LEFT_TO_RIGHT]
    # stitch over exactly the cams in main_cluster (so transforms line up); cam_order restriction
    # is applied later via CAM_FOOTPRINTS bbox crops, not by omitting cameras from the canvas.
    frames_all = [sequences[c][anchor_index] for c in main_cluster["cams"]]
    transforms_all = main_cluster["transforms"]
    panorama = stitch_frames(frames_all, [PANORAMA_TRANSLATION @ T for T in transforms_all])
    return pad_or_crop_to_canvas(panorama)


def build_stitched_input_tensor(panorama):
    '''[2, H, W]: temperature channel + binary validity mask channel (1 = real pixel, 0 = black
    padding) -- used only by Baseline 1 (stitched).'''
    mask = (panorama > 0).astype(np.float32)
    return np.stack([panorama.astype(np.float32), mask], axis=0)


def discover_rgb_homographies(root):
    root = Path(root)
    calibrations = {}
    for path in sorted(root.rglob("rgb*_homography.json")):
        data = load_homography_json(path)
        if data["cam_a"] != "rgb":
            continue
        cam_id = int(CAM_NAME_RE.match(data["cam_b"]).group(1))
        quad_a = np.array(data["quad_a"], dtype=np.float64)
        calibrations[cam_id] = {
            "H": np.array(data["H"], dtype=np.float64),
            "quad_a": quad_a,
            "quad_a_centroid": quad_a.mean(axis=0),
        }
    return calibrations


RGB_CALIBRATIONS = discover_rgb_homographies(RGB_HOMOGRAPHY_DIR)
CAM_IDS_WITH_RGB_OVERLAP = set(RGB_CALIBRATIONS.keys()) & set(ALL_CAMS_LEFT_TO_RIGHT)
print("cameras with RGB overlap (calibrated):", sorted(CAM_IDS_WITH_RGB_OVERLAP))
print("ACTIVE cameras with NO RGB overlap (would be permanently unsupervised, shouldn't happen):",
      sorted(set(ACTIVE_CAMERAS) - CAM_IDS_WITH_RGB_OVERLAP))

USE_REAL_LABELS = len(RGB_CALIBRATIONS) > 0
if not USE_REAL_LABELS:
    warnings.warn(
        "No rgb*_homography.json files found -- falling back to NAIVE intensity-threshold "
        "pseudo-labels (2-class: object/no-object, since 'person vs other' can't be told apart "
        "without RGB+YOLO). Treat results from this run as a pipeline smoke test only."
    )

if USE_REAL_LABELS:
    from ultralytics import YOLO
    from matplotlib.path import Path as MplPath

    _yolo_model = YOLO(YOLO_WEIGHTS)
    PANORAMA_T_FOR_CAM = {cam_id: PANORAMA_TRANSLATION @ T for cam_id, T in CAM_TRANSFORMS.items()}

    def _nearest_calibration(box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        best_cam, best_dist = None, np.inf
        for cam_id, calib in RGB_CALIBRATIONS.items():
            dist = np.linalg.norm(box_center - calib["quad_a_centroid"])
            if dist < best_dist:
                best_cam, best_dist = cam_id, dist
        return best_cam

    def _project_rgb_box_to_panorama(box_xyxy, via_cam_id):
        x1, y1, x2, y2 = box_xyxy
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        corners_h = np.hstack([corners, np.ones((4, 1))])
        combined = PANORAMA_T_FOR_CAM[via_cam_id] @ RGB_CALIBRATIONS[via_cam_id]["H"]
        out = (combined @ corners_h.T).T
        return out[:, :2] / out[:, 2:3]

    def generate_real_labels(rgb_path, cam_ids=None):
        '''Returns {cam_id: (class_id, list_of_boxes_for_viz)} for every cam_id in cam_ids
        (defaults to every RGB-overlapping camera). class_id in {0, 1, 2}.'''
        cam_ids = cam_ids or sorted(CAM_IDS_WITH_RGB_OVERLAP)
        img = cv2.imread(str(rgb_path))
        if img is None:
            return None
        results = _yolo_model(img, verbose=False)[0]
        has_person = {cam_id: False for cam_id in cam_ids}
        has_other = {cam_id: False for cam_id in cam_ids}
        boxes_by_cam = {cam_id: [] for cam_id in cam_ids}
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            via_cam = _nearest_calibration(box)
            panorama_corners = _project_rgb_box_to_panorama(box, via_cam)
            cx, cy = panorama_corners[:, 0].mean(), panorama_corners[:, 1].mean()
            is_person = int(cls) == COCO_PERSON_CLASS_ID
            for cam_id in cam_ids:
                poly_path = MplPath(CAM_FOOTPRINTS[cam_id]["polygon"])
                pts_to_check = list(map(tuple, panorama_corners)) + [(cx, cy)]
                if any(poly_path.contains_point(p) for p in pts_to_check):
                    if is_person:
                        has_person[cam_id] = True
                    else:
                        has_other[cam_id] = True
                    boxes_by_cam[cam_id].append((box, int(cls), is_person))
        labels = {}
        for cam_id in cam_ids:
            if has_person[cam_id]:
                labels[cam_id] = 1
            elif has_other[cam_id]:
                labels[cam_id] = 2
            else:
                labels[cam_id] = 0
        return labels, boxes_by_cam


def generate_naive_labels(tiles_by_cam, threshold_std=1.5):
    '''Fallback-mode placeholder, UNSTITCHED: 2-class only (no_object=0 vs other_object=2 --
    'person' can't be distinguished without RGB+YOLO, so naive mode never emits class 1).'''
    return {cam_id: (2 if (grid.size > 0 and grid.std() > threshold_std) else 0)
            for cam_id, grid in tiles_by_cam.items()}


LABEL_CACHE_PATH = CACHE_DIR / "label_cache.pkl"

# (part, anchor_index) -> (class_vec[int], mask_vec[float]) -- loaded from disk if a previous
# stage (Stage 0, normally) already computed it, so this notebook never re-runs YOLO at all.
_label_cache = pickle_load(LABEL_CACHE_PATH, default={})
_label_boxes_cache = {}  # (part, anchor_index) -> boxes_by_cam, kept ONLY for the preview cells
                          # -- NOT persisted to disk (boxes are only needed for the live preview
                          # plot in the same run that generated them, not across stages)
print(f"Loaded {len(_label_cache)} cached label(s) from {LABEL_CACHE_PATH}"
      if _label_cache else f"No label cache found at {LABEL_CACHE_PATH} yet -- will compute fresh.")


def _sample_key(row):
    return (row["_part"], int(row["anchor_index"]))


def get_or_compute_label(row, tiles=None, keep_boxes=False):
    key = _sample_key(row)
    if key in _label_cache:
        return _label_cache[key]

    part, anchor_index = key
    if USE_REAL_LABELS:
        has_rgb = rgb_gap_ok(row)
        result = generate_real_labels(CAPTURE_PARTS[part]["rgb_dir"] / row["rgb_files"]) if has_rgb else None
        class_vec = np.zeros(NUM_CAMS, dtype=np.int64)
        mask_vec = np.zeros(NUM_CAMS, dtype=np.float32)
        boxes_by_cam = {}
        if result is not None:
            rgb_labels, boxes_by_cam = result
            for i, cam_id in enumerate(CAM_ORDER_LEFT_TO_RIGHT):
                if cam_id not in CAM_IDS_WITH_RGB_OVERLAP:
                    continue
                class_vec[i] = rgb_labels[cam_id]
                mask_vec[i] = 1.0
        if keep_boxes:
            _label_boxes_cache[key] = boxes_by_cam
    else:
        if tiles is None:
            tiles = load_camera_tiles(part, anchor_index)
        tiles_by_cam = dict(zip(CAM_ORDER_LEFT_TO_RIGHT, tiles))
        naive_labels = generate_naive_labels(tiles_by_cam)
        class_vec = np.array([naive_labels[c] for c in CAM_ORDER_LEFT_TO_RIGHT], dtype=np.int64)
        mask_vec = np.ones(NUM_CAMS, dtype=np.float32)

    _label_cache[key] = (class_vec, mask_vec)
    return class_vec, mask_vec


def precompute_labels(sync_rows_to_cache):
    '''Computes (and caches) labels up front, sequentially, in the main process -- required so
    DataLoader(num_workers>0) workers never run YOLO/CUDA inline (forked CUDA crash). Persists
    the FULL cache to LABEL_CACHE_PATH after computing any new labels, so the next stage
    notebook that imports this same SHARED_DIR loads the cache hit-only and skips YOLO
    entirely.'''
    n_new = sum(1 for r in sync_rows_to_cache if _sample_key(r) not in _label_cache)
    if n_new == 0:
        print(f"All {len(sync_rows_to_cache)} sample(s) already in the cache -- skipping YOLO entirely.")
        return
    print(f"Precomputing labels for {n_new} new sample(s) (of {len(sync_rows_to_cache)} requested)"
          f"{' (running YOLO once per frame)' if USE_REAL_LABELS else ' (naive thresholds)'}...")
    for row in sync_rows_to_cache:
        if _sample_key(row) in _label_cache:
            continue
        get_or_compute_label(row)
    pickle_save(_label_cache, LABEL_CACHE_PATH)
    print(f"done. Cache now has {len(_label_cache)} label(s), saved -> {LABEL_CACHE_PATH}")


_CLASS_COLOR = {0: (150, 150, 150), 1: (0, 0, 255), 2: (0, 165, 255)}   # gray / red / orange, BGR
_CLASS_LABEL_NAME = {0: "no object", 1: "person", 2: "other object"}


def preview_random_sample(row=None, seed=RANDOM_SEED):
    '''row: a sync row (defaults to a random one with a usable RGB match). Shows:
      - RGB frame with raw YOLO boxes (red=person, orange=other), thermal footprints outlined
      - each active camera's raw thermal tile, titled with its derived class label
    '''
    if row is None:
        candidates = [r for r in load_all_sync_rows() if rgb_gap_ok(r)]
        row = random.Random(seed).choice(candidates)

    part = row["_part"]
    anchor_index = int(row["anchor_index"])
    class_vec, mask_vec = get_or_compute_label(row, keep_boxes=True)
    boxes_by_cam = _label_boxes_cache.get(_sample_key(row), {})

    rgb_img = cv2.imread(str(CAPTURE_PARTS[part]["rgb_dir"] / row["rgb_files"])) if rgb_gap_ok(row) else None
    tiles = load_camera_tiles(part, anchor_index)

    n_panels = 1 + len(CAM_ORDER_LEFT_TO_RIGHT)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))

    if rgb_img is not None:
        annotated = rgb_img.copy()
        seen_boxes = set()
        for cam_id, boxes in boxes_by_cam.items():
            for box, cls, is_person in boxes:
                box_key = tuple(np.round(box, 1))
                if box_key in seen_boxes:
                    continue
                seen_boxes.add(box_key)
                x1, y1, x2, y2 = (int(v) for v in box)
                color = (0, 0, 255) if is_person else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, "person" if is_person else "object", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"{part}  anchor={anchor_index}\nRGB + YOLO boxes")
    else:
        axes[0].text(0.5, 0.5, "no synced RGB frame", ha="center", va="center")
    axes[0].axis("off")

    for i, cam_id in enumerate(CAM_ORDER_LEFT_TO_RIGHT):
        ax = axes[i + 1]
        ax.imshow(tiles[i], cmap="inferno")
        cls = int(class_vec[i])
        masked = mask_vec[i] > 0
        title = f"cam{cam_id} ({CAM_TO_DIRECTION[cam_id]})\n"
        title += f"label: {_CLASS_LABEL_NAME[cls]}" if masked else "label: -- (unsupervised)"
    plt.tight_layout()
    savefig_show(f"sample_preview_{part}_{anchor_index}")


if USE_REAL_LABELS:
    for _ in range(3):
        preview_random_sample(seed=random.randint(0, 99999))
else:
    print("Skipping rich preview -- naive fallback mode (no RGB calibration files found yet).")


def show_direction_class_distribution(rows, label="all data"):
    '''Stacked bar chart: per-direction (left/middle/right) counts of no_object/person/other,
    plus a 'no RGB label' bucket for samples where that camera's mask is 0. Requires labels to
    be precomputed already.'''
    counts = {d: {c: 0 for c in range(NUM_CLASSES)} | {"unsupervised": 0} for d in DIRECTIONS}
    for row in rows:
        class_vec, mask_vec = get_or_compute_label(row)
        for i, cam_id in enumerate(CAM_ORDER_LEFT_TO_RIGHT):
            d = CAM_TO_DIRECTION[cam_id]
            if mask_vec[i] > 0:
                counts[d][int(class_vec[i])] += 1
            else:
                counts[d]["unsupervised"] += 1

    df = pd.DataFrame(counts).T
    df = df.rename(columns={0: "no_object", 1: "person", 2: "other_object"})
    df = df[["no_object", "person", "other_object", "unsupervised"]]
    print(f"[{label}] per-direction class counts:\n{df}\n")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#9E9E9E", "#E53935", "#FB8C00", "#455A64"]
    df.plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_title(f"{label} — class distribution by camera direction")
    ax.set_ylabel("samples")
    ax.set_xlabel("direction")
    ax.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    savefig_show(f"class_distribution_by_direction__{label.replace(' ', '_').replace('(', '').replace(')', '')}")
    return df


def show_person_vs_other_rates(rows, label="all data"):
    '''Simple per-direction person-rate / other-object-rate bar chart, normalized by the
    number of supervised (masked) samples for that direction -- easier to compare directions
    with different total supervised counts at a glance.'''
    df = show_direction_class_distribution(rows, label=label + " (raw counts)")
    supervised = df[["no_object", "person", "other_object"]].sum(axis=1)
    rates = df[["person", "other_object"]].div(supervised.replace(0, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    rates.plot(kind="bar", ax=ax, color=["#E53935", "#FB8C00"])
    ax.set_ylabel("rate (of supervised samples)")
    ax.set_title(f"{label} — person vs. other-object rate by direction")
    plt.tight_layout()
    savefig_show(f"person_vs_other_rate__{label.replace(' ', '_').replace('(', '').replace(')', '')}")
    return rates


class StitchedPanoramaDataset(Dataset):
    '''Baseline 1 input: [2, H, W] stitched panorama + validity mask.'''

    def __init__(self, sync_rows):
        self.sync_rows = sync_rows

    def __len__(self):
        return len(self.sync_rows)

    def __getitem__(self, idx):
        row = self.sync_rows[idx]
        part, anchor_index = row["_part"], int(row["anchor_index"])
        panorama = load_panorama(part, anchor_index)
        tensor = build_stitched_input_tensor(panorama)
        class_vec, mask_vec = get_or_compute_label(row)
        return (torch.from_numpy(tensor), torch.from_numpy(class_vec), torch.from_numpy(mask_vec))


class TileDataset(Dataset):
    '''Baselines 2/3/4 input: [NUM_CAMS, 1, H, W] raw per-camera tiles, unstitched.'''

    def __init__(self, sync_rows, cam_order=None):
        self.sync_rows = sync_rows
        self.cam_order = cam_order or CAM_ORDER_LEFT_TO_RIGHT

    def __len__(self):
        return len(self.sync_rows)

    def __getitem__(self, idx):
        row = self.sync_rows[idx]
        part, anchor_index = row["_part"], int(row["anchor_index"])
        tiles = load_camera_tiles(part, anchor_index, cam_order=self.cam_order)
        tensor = build_input_tensor(tiles)
        class_vec, mask_vec = get_or_compute_label(row)
        return (torch.from_numpy(tensor), torch.from_numpy(class_vec), torch.from_numpy(mask_vec))


sync_rows = load_all_sync_rows()  # combines every discovered part, already shuffled

if SMOKE_TEST:
    print(f"*** SMOKE TEST MODE: using {SMOKE_TEST_N_SAMPLES} samples, "
          f"{SMOKE_TEST_EPOCHS} epoch(s), batch_size={SMOKE_TEST_BATCH_SIZE}. "
          f"Set SMOKE_TEST=False for a real run. ***")
    sync_rows = sync_rows[:SMOKE_TEST_N_SAMPLES]
    EFFECTIVE_EPOCHS = SMOKE_TEST_EPOCHS
    EFFECTIVE_BATCH_SIZE = SMOKE_TEST_BATCH_SIZE
    EFFECTIVE_NUM_WORKERS = 0
else:
    EFFECTIVE_EPOCHS = EPOCHS
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE
    EFFECTIVE_NUM_WORKERS = NUM_WORKERS


def blocked_train_val_split(rows, val_frac=VAL_FRAC, n_blocks=N_SPLIT_BLOCKS):
    n = len(rows)
    block_size = max(1, n // n_blocks)
    train_rows, val_rows = [], []
    for start in range(0, n, block_size):
        block = rows[start:start + block_size]
        n_val = max(1, int(len(block) * val_frac)) if len(block) > 1 else 0
        if n_val == 0:
            train_rows.extend(block)
        else:
            train_rows.extend(block[:-n_val])
            val_rows.extend(block[-n_val:])
    if not val_rows:
        val_rows = train_rows[-1:]
    return train_rows, val_rows


SPLIT_CACHE_PATH = CACHE_DIR / "train_val_split.pkl"

if SMOKE_TEST:
    n_total = len(sync_rows)
    n_train = max(1, int(0.8 * n_total))
    train_rows, val_rows = sync_rows[:n_train], sync_rows[n_train:]
    if not val_rows:
        val_rows = train_rows[-1:]
else:
    # real (non-smoke) runs: reuse the SAME split across every stage notebook so Baseline 1,
    # Baseline 2, every Feature Fusion variant, and the shuffle/identity stage are all trained
    # and scored on identical train/val rows -- whichever stage notebook runs FIRST computes and
    # saves it, every later stage notebook just loads it back.
    _cached_split = pickle_load(SPLIT_CACHE_PATH)
    if _cached_split is not None:
        train_rows, val_rows = _cached_split
        print(f"Loaded cached train/val split from {SPLIT_CACHE_PATH}")
    else:
        train_rows, val_rows = blocked_train_val_split(sync_rows)
        pickle_save((train_rows, val_rows), SPLIT_CACHE_PATH)
        print(f"Computed fresh train/val split, saved -> {SPLIT_CACHE_PATH}")

print(f"train samples: {len(train_rows)}  val samples: {len(val_rows)}")

# CRITICAL: must happen before any DataLoader(num_workers>0) below.
precompute_labels(train_rows + val_rows)

# ---- visualizations from Sections 6/7, now that labels are cached ----
show_direction_class_distribution(train_rows + val_rows, label="train + val (combined)")
show_person_vs_other_rates(train_rows + val_rows, label="train + val (combined)")

stitched_train_ds = StitchedPanoramaDataset(train_rows)
stitched_val_ds = StitchedPanoramaDataset(val_rows)
tile_train_ds = TileDataset(train_rows)
tile_val_ds = TileDataset(val_rows)

dl_kw = dict(batch_size=EFFECTIVE_BATCH_SIZE, num_workers=EFFECTIVE_NUM_WORKERS, pin_memory=True)
stitched_tr_ld = DataLoader(stitched_train_ds, shuffle=True, **dl_kw)
stitched_va_ld = DataLoader(stitched_val_ds, shuffle=False, **dl_kw)
tile_tr_ld = DataLoader(tile_train_ds, shuffle=True, **dl_kw)
tile_va_ld = DataLoader(tile_val_ds, shuffle=False, **dl_kw)
print(f"Loaders ready -- train batches: {len(tile_tr_ld)}  val batches: {len(tile_va_ld)}")

# ---- class balance check -- pos_weight style, but for 3-class CE. Computed HERE (not inside
# Baseline 1's cell) because EVERY model (Baseline 1, Baseline 2, every Feature Fusion variant)
# uses class_weight in its training call -- and on the split deepnet pipeline, several stages
# train Feature Fusion variants WITHOUT including Baseline 1's cell at all (e.g. Stage 0 trains
# the FF "baseline" variant by itself), so class_weight has to live somewhere every one of them
# actually has access to. ----
class_counts = torch.zeros(NUM_CLASSES)
for row in train_rows + val_rows:
    class_vec, mask_vec = get_or_compute_label(row)
    for i in range(NUM_CAMS):
        if mask_vec[i] > 0:
            class_counts[int(class_vec[i])] += 1
print("class counts (train+val, masked only):",
      {CLASS_NAMES[c]: int(class_counts[c]) for c in range(NUM_CLASSES)})

# inverse-frequency class weight, capped so a rare class doesn't blow up the gradient
CLASS_WEIGHT_CAP = 20.0
total = class_counts.sum().clamp_min(1.0)
class_weight = (total / (class_counts.clamp_min(1.0) * NUM_CLASSES)).clamp(max=CLASS_WEIGHT_CAP)
print("class_weight:", {CLASS_NAMES[c]: round(class_weight[c].item(), 2) for c in range(NUM_CLASSES)})


def masked_ce_loss(logits, class_labels, mask, class_weight=None):
    '''logits: [B, NUM_CAMS, NUM_CLASSES]. class_labels: [B, NUM_CAMS] (int64). mask: [B, NUM_CAMS].
    Cross-entropy per camera-slot, masked by RGB-overlap/sync validity, averaged only over
    valid slots.'''
    B, C, K = logits.shape
    per_element = F.cross_entropy(
        logits.reshape(B * C, K), class_labels.reshape(B * C).long(),
        weight=class_weight, reduction="none",
    ).reshape(B, C)
    masked = per_element * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def masked_macro_f1_multiclass(preds, trues, masks, positive_classes=(1, 2)):
    '''preds/trues/masks: [N, NUM_CAMS] numpy int arrays (masks float). Per-camera macro F1 over
    `positive_classes` only (skips a camera entirely if it has zero valid rows), then averaged
    across cameras that had at least one -- mirrors the source notebooks' masked_macro_f1, now
    multiclass via sklearn's per-class F1 with average=None, restricted to the positive classes.'''
    from sklearn.metrics import f1_score as _f1
    cam_f1s = []
    for i in range(preds.shape[1]):
        valid = masks[:, i] > 0
        if valid.sum() == 0:
            continue
        per_class = _f1(trues[valid, i], preds[valid, i], labels=list(positive_classes),
                         average=None, zero_division=0)
        cam_f1s.append(float(np.mean(per_class)))
    return float(np.mean(cam_f1s)) if cam_f1s else 0.0


def derive_no_object(class_logits):
    '''class_logits: [B, NUM_CAMS, NUM_CLASSES]. Hard rule (consistent with the source
    notebooks' derive_clear): the per-camera argmax class, with 'Clear' meaning every camera's
    argmax is 0 (no_object).'''
    pred_class = class_logits.argmax(dim=-1)
    return (pred_class.sum(dim=1) == 0).float(), pred_class


@torch.no_grad()
def evaluate_model(model, loader, model_input_kind="tile"):
    '''model_input_kind: 'tile' ([B,NUM_CAMS,1,H,W]) or 'stitched' ([B,2,H,W], cropped inside
    the model). Returns a dict with macro_f1, per-class predictions/trues/masks, and a 'clear'
    (no_object-everywhere) F1 where well-defined.'''
    model.eval()
    all_logits, all_labels, all_masks = [], [], []
    for x, y, m in loader:
        x = x.to(device)
        logits = model(x).cpu()  # [B, NUM_CAMS, NUM_CLASSES]
        all_logits.append(logits)
        all_labels.append(y)
        all_masks.append(m)
    if not all_labels:
        return {"macro_f1": 0., "preds": np.zeros((0, NUM_CAMS)), "trues": np.zeros((0, NUM_CAMS)),
                "masks": np.zeros((0, NUM_CAMS)), "clear_f1": None}
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks)
    pred_class = logits.argmax(dim=-1)

    p_np, t_np, m_np = pred_class.numpy(), labels.numpy(), masks.numpy()
    macro_f1 = masked_macro_f1_multiclass(p_np, t_np, m_np)

    no_object_pred, _ = derive_no_object(logits)
    full_coverage = masks.all(dim=1)
    if full_coverage.sum().item() > 0:
        no_object_true = (labels[full_coverage].sum(dim=1) == 0).float()
        clear_f1 = f1_score(no_object_true.numpy(), no_object_pred[full_coverage].numpy(), zero_division=0)
    else:
        clear_f1 = None

    return {"macro_f1": macro_f1, "clear_f1": clear_f1, "preds": p_np, "trues": t_np, "masks": m_np}


def train_classifier_model(model, name, tr, va, epochs=EPOCHS, lr=LR, class_weight=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_f1, best_w = -1., copy.deepcopy(model.state_dict())
    cw = class_weight.to(device) if class_weight is not None else None
    for ep in range(epochs):
        model.train(); running = 0.; n_seen = 0
        for x, y, mask in tr:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            loss = masked_ce_loss(model(x), y, mask, class_weight=cw)
            loss.backward(); opt.step()
            running += loss.item() * len(y); n_seen += len(y)
        sch.step()
        vm = evaluate_model(model, va)
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_w = copy.deepcopy(model.state_dict())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [{name}] ep {ep+1:02d}/{epochs}  "
                  f"train_loss={running/max(n_seen,1):.4f}  val_macro_f1={vm['macro_f1']:.3f}")
    model.load_state_dict(best_w)
    return model


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def measure_latency(model, dummy_input, n_runs=200):
    m = model.cpu().eval()
    with torch.no_grad():
        for _ in range(20):
            m(dummy_input)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            m(dummy_input)
    model.to(device)
    return (time.perf_counter() - t0) / n_runs * 1000


def save_model(model, tag):
    path = MODEL_DIR / f"{tag}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Saved -> {path}")


def load_model_weights(model, tag):
    '''Loads a checkpoint saved by save_model(..., tag) into an already-instantiated model
    (architecture must match) -- used by later-stage notebooks to bring back a model trained in
    an earlier stage (e.g. Stage 5 reloading the Feature Fusion best-combo model trained in
    Stage 4, for the shuffle-correction inference pipeline).'''
    path = MODEL_DIR / f"{tag}.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"  Loaded <- {path}")
    return model


def save_eval_result(result, model, tag, latency, extra=None):
    '''Pickles everything a LATER stage needs about a trained model's evaluation -- without the
    model object itself (that's saved separately via save_model/load_model_weights) -- to
    RESULTS_DIR/{tag}_result.pkl. This is what lets Stage 5's final comparison tables/plots be
    built without re-training or even re-running inference for Baseline 1/2/Feature Fusion.'''
    payload = {
        "macro_f1": result["macro_f1"], "clear_f1": result["clear_f1"],
        "preds": result["preds"], "trues": result["trues"], "masks": result["masks"],
        "params": count_params(model), "latency_ms": latency,
    }
    if extra:
        payload.update(extra)
    pickle_save(payload, RESULTS_DIR / f"{tag}_result.pkl")
    print(f"  Result saved -> {RESULTS_DIR / f'{tag}_result.pkl'}")
    return payload


def load_eval_result(tag):
    return pickle_load(RESULTS_DIR / f"{tag}_result.pkl")


def per_camera_report(result, label, class_names=CLASS_NAMES):
    p, t, msk = result["preds"], result["trues"], result["masks"]
    print(f"\nPer-camera report — {label}")
    for i, cam_id in enumerate(CAM_ORDER_LEFT_TO_RIGHT):
        valid = msk[:, i] > 0
        n = int(valid.sum())
        if n == 0:
            print(f"  cam{cam_id} ({CAM_TO_DIRECTION[cam_id]}):  -- (no real labels in this val set)")
            continue
        per_class_f1 = f1_score(t[valid, i], p[valid, i], labels=[0, 1, 2], average=None, zero_division=0)
        pos_macro = float(np.mean([per_class_f1[1], per_class_f1[2]]))
        print(f"  cam{cam_id} ({CAM_TO_DIRECTION[cam_id]}, n={n}):  "
              f"F1[no_object]={per_class_f1[0]:.3f}  F1[person]={per_class_f1[1]:.3f}  "
              f"F1[other]={per_class_f1[2]:.3f}  macro(person+other)={pos_macro:.3f}")
    print(f"  {'Overall macro F1':<28} {result['macro_f1']:.3f}")


def plot_confusion(result, label):
    p, t, msk = result["preds"], result["trues"], result["masks"]
    valid = msk.reshape(-1) > 0
    cm = confusion_matrix(t.reshape(-1)[valid], p.reshape(-1)[valid], labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — {label}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    savefig_show(f"confusion_matrix__{label.replace(' ', '_').replace('(', '').replace(')', '')}")


ACT_MAP = {
    "relu":      nn.ReLU,
    "gelu":      nn.GELU,
    "leakyrelu": lambda: nn.LeakyReLU(0.1),
}


def make_conv_block(in_ch, out_ch, act_fn):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), act_fn())


class FlexBackbone(nn.Module):
    '''depth controls how many downsampling blocks, width_mult scales all channel counts.
    in_channels is always 1 (single raw thermal tile, no validity-mask channel).'''
    BASE_CHANNELS = [16, 32, 64, 128, 256]

    def __init__(self, depth=3, width_mult=1.0, act="relu"):
        super().__init__()
        assert 2 <= depth <= 5, "depth must be 2-5"
        act_fn = ACT_MAP[act]
        channels = [max(4, int(c * width_mult)) for c in self.BASE_CHANNELS[:depth]]
        self.blocks = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(channels):
            pool = (i < depth - 1)
            block_layers = [make_conv_block(in_ch, out_ch, act_fn)]
            if pool:
                block_layers.append(nn.MaxPool2d(2))
            self.blocks.append(nn.Sequential(*block_layers))
            in_ch = out_ch
        self.out_channels = channels

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


def backbone_feature_dims(depth=3, width_mult=1.0, act="relu"):
    b = FlexBackbone(depth, width_mult, act)
    with torch.no_grad():
        feats = b(torch.zeros(1, 1, SENSOR_H, SENSOR_W))
    return [f.shape[1] for f in feats], [tuple(f.shape[-2:]) for f in feats]


ch, sz = backbone_feature_dims(3, 1.0)
print(f"Baseline backbone: channels={ch}, spatial={sz}")

# Defined HERE (not inside Baseline 2's cell below) because Baseline 2, Feature Fusion, AND the
# camera-identity classifier ALL need a [B, NUM_CAMS, 1, H, W] dummy tensor for their model
# smoke-test print and for measure_latency() -- and on the split deepnet pipeline, several
# stages (e.g. Stage 0, the FF ablation axis stages) include Feature Fusion's model definition
# WITHOUT including Baseline 2's cell, so dummy_tile must live somewhere every one of them
# actually has access to.
dummy_tile = torch.zeros(2, NUM_CAMS, 1, SENSOR_H, SENSOR_W)


class FlexClassHead3Class(nn.Module):
    def __init__(self, feature_dims, hidden=128, dropout=0.3, n_out_per_cam=NUM_CLASSES):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.n_out_per_cam = n_out_per_cam
        self.fc = nn.Sequential(
            nn.Linear(sum(feature_dims), hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CAMS * n_out_per_cam))

    def forward(self, features):
        out = self.fc(torch.cat([self.pool(f).flatten(1) for f in features], dim=1))
        return out.view(-1, NUM_CAMS, self.n_out_per_cam)


class FeatureFusionModel(nn.Module):
    '''Shared-weight FlexBackbone runs on each of the NUM_CAMS camera tiles independently.
    Feature maps at each level are concatenated across cameras, then projected back to the
    original channel count via a 1x1 conv before the shared head. Output:
    [B, NUM_CAMS, NUM_CLASSES].'''

    def __init__(self, depth=3, width_mult=1.0, act="relu", head_hidden=128, head_dropout=0.3):
        super().__init__()
        self.backbone = FlexBackbone(depth, width_mult, act)
        per_dims = self.backbone.out_channels
        self.proj = nn.ModuleList([nn.Conv2d(d * NUM_CAMS, d, 1) for d in per_dims])
        self.head = FlexClassHead3Class(per_dims, hidden=head_hidden, dropout=head_dropout)

    def forward(self, x):
        # x: [B, NUM_CAMS, 1, H, W]
        n_levels = len(self.backbone.out_channels)
        all_feats = [[] for _ in range(n_levels)]
        for i in range(NUM_CAMS):
            for lvl, f in enumerate(self.backbone(x[:, i])):
                all_feats[lvl].append(f)
        fused = [proj(torch.cat(lf, dim=1)) for proj, lf in zip(self.proj, all_feats)]
        return self.head(fused)


m3 = FeatureFusionModel(depth=3, width_mult=1.0)
out3 = m3(dummy_tile)
print(f"Feature Fusion baseline output shape: {out3.shape}  | params: {count_params(m3):,}")


ff_results = {}

def run_ff_variant(key, label, force_retrain=False, **model_kwargs):
    '''Trains (or, if force_retrain=False and a cached result+checkpoint already exist on disk
    from an earlier run of THIS key, just reloads) a Feature Fusion variant. Always persists its
    result to RESULTS_DIR/ff_{key}_result.pkl and its weights to MODEL_DIR/ff_{key}.pt -- this
    is what lets a later stage notebook (e.g. the best-combo stage, run in its own sbatch job)
    pull in results from variants trained by an earlier, separate job without retraining them.'''
    cached = None if force_retrain else load_eval_result(f"ff_{key}")
    model = FeatureFusionModel(**model_kwargs)
    if cached is not None and (MODEL_DIR / f"ff_{key}.pt").exists():
        print(f"[{key}] found cached result + checkpoint -- loading instead of retraining.")
        load_model_weights(model, f"ff_{key}")
        ff_results[key] = {**cached, "label": label, "model": model}
        return model

    print(f"\n{'='*60}\nFEATURE FUSION -- {label}\n{'='*60}")
    print(f"Params: {count_params(model):,}")
    train_classifier_model(model, key, tile_tr_ld, tile_va_ld, epochs=EFFECTIVE_EPOCHS,
                            class_weight=class_weight)
    save_model(model, f"ff_{key}")
    test_m = evaluate_model(model, tile_va_ld)
    lat = measure_latency(model, dummy_tile[:1])
    save_eval_result(test_m, model, f"ff_{key}", lat, extra={"label": label})
    ff_results[key] = {
        "label": label, "macro_f1": test_m["macro_f1"], "clear_f1": test_m["clear_f1"],
        "preds": test_m["preds"], "trues": test_m["trues"], "masks": test_m["masks"],
        "params": count_params(model), "latency_ms": lat, "model": model,
    }
    print(f"  VAL  macro_f1={test_m['macro_f1']:.4f}  clear_f1={test_m['clear_f1']}  latency={lat:.2f}ms")
    return model


def load_ff_results_from_disk(keys):
    '''Pulls already-trained Feature Fusion variant results (no model objects, no retraining)
    straight from RESULTS_DIR into ff_results -- used by the best-combo stage to read back
    results from the depth/width/activation/head/dropout stages, which may have run as entirely
    separate, parallel sbatch jobs. ALSO used by Stage 5, which needs this function (and the
    ff_results dict above) WITHOUT the baseline-training call below -- that's why this cell is
    split from the "train baseline" cell that follows it.'''
    n_loaded = 0
    for key in keys:
        if key in ff_results:
            continue
        cached = load_eval_result(f"ff_{key}")
        if cached is not None:
            ff_results[key] = {**cached, "label": cached.get("label", key)}
            n_loaded += 1
        else:
            print(f"  [warning] no cached result found for ff_{key} -- that axis's job may not "
                  f"have finished yet, or hasn't run in this SHARED_DIR.")
    print(f"Loaded {n_loaded} Feature Fusion result(s) from disk.")


# ---- baseline ----
run_ff_variant("baseline", "Baseline (d=3, w=1x, relu, h=128, dr=0.3)",
                depth=3, width_mult=1.0, act="relu", head_hidden=128, head_dropout=0.3)


# ---- ablation 5: dropout ----
for dr, tag in [(0.1, "0p1"), (0.5, "0p5")]:
    run_ff_variant(f"d3_w1_relu_h128_dr{tag}", f"dropout={dr} (d=3, w=1x, relu, h=128)",
                    depth=3, width_mult=1.0, act="relu", head_hidden=128, head_dropout=dr)


