"""Load caches written by build_cache.py into arrays the models train on."""
import csv
import json
from pathlib import Path

import numpy as np

from labels import CLASSES, ZONES

EVENT_SCALE = float(np.log1p(255.0))  # counts are stored capped at 255; a Python float keeps float32


def load_recording(cache_dir):
    """One recording: simulated thermal, the 40 ms event frame before each sample, labels, motion."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    s = np.load(cache_dir / "samples.npz")
    n = meta["samples"]
    y = np.zeros((n, len(ZONES)), np.int64)
    near = np.zeros((n, len(ZONES)), bool)
    with open(cache_dir / "labels.csv") as f:
        for r in csv.DictReader(f):
            i, z = int(r["sample"]), ZONES.index(r["zone"])
            y[i, z], near[i, z] = CLASSES.index(r["label"]), r["near"] == "1"

    # pick the event bin that ends at each sample, reading one 10 s chunk at a time
    want = np.clip(s["ev_bin"], 0, None)
    h, w = meta["event_shape"]
    events = np.zeros((n, 2, h, w), np.uint8)
    for f in sorted((cache_dir / "events").glob("chunk_*.npz")):
        c = np.load(f)
        first, bins = int(c["first_bin"]), c["bins"]
        hit = (want >= first) & (want < first + len(bins))
        events[hit] = bins[want[hit] - first]

    rec = {
        "name": cache_dir.name,
        "t": s["t"],
        "thermal": s["thermal"].astype(np.float32),  # (n, 24, 32), noise-free; noise is added in training
        "events": events,                             # (n, 2, h, w) counts, darker/brighter
        "y": y, "near": near,
        "gyro": np.linalg.norm(s["gyro"], axis=1),    # rad/s, for the still/moving split
    }

    # optional keep.json {"start": s, "end": s}, seconds from the first sample: keep only that part
    # (e.g. the indoor part of a recording that starts outdoors)
    keep_file = cache_dir / "keep.json"
    if keep_file.exists():
        k = json.loads(keep_file.read_text())
        rel = rec["t"] - rec["t"][0]
        m = (rel >= k.get("start", 0)) & (rel <= k.get("end", np.inf))
        rec = {key: (v[m] if isinstance(v, np.ndarray) else v) for key, v in rec.items()}
    return rec


def load_all(cache_root):
    dirs = sorted(d for d in Path(cache_root).expanduser().iterdir() if (d / "meta.json").exists())
    if not dirs:
        raise SystemExit(f"no complete caches in {cache_root}")
    return [load_recording(d) for d in dirs]


def event_input(counts):
    """uint8 counts -> float in [0, 1] on a log scale (dense flicker would swamp a linear scale)."""
    return np.log1p(counts.astype(np.float32)) / EVENT_SCALE
