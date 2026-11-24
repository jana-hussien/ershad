"""Load caches written by build_cache.py into arrays the models train on."""
import csv
import json
from pathlib import Path

import numpy as np

from labels import CLASSES, ZONES

EVENT_SCALE = float(np.log1p(255.0))  # counts are stored capped at 255; a Python float keeps float32


def load_recording(cache_dir, history=1, every=1):
    """One recording: simulated thermal, the 40 ms event frame(s) before each sample, labels, motion.

    history > 1 also returns "hist": the last `history` 40 ms bins before each sample, oldest first,
    shape (n, history, 2, h, w); bins before the recording started are zero.
    every > 1 keeps only samples whose index is a multiple of `every` (every=2: 2 per second instead of 4;
    neighbouring samples 0.25 s apart are nearly identical, and this halves memory)."""
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

    # which samples to keep: every Nth, and only the part keep.json names (e.g. the indoor part of a
    # recording that starts outdoors: {"start": s, "end": s}, seconds from the first sample)
    sel = np.arange(n) % every == 0
    keep_file = cache_dir / "keep.json"
    if keep_file.exists():
        k = json.loads(keep_file.read_text())
        rel = s["t"] - s["t"][0]
        sel &= (rel >= k.get("start", 0)) & (rel <= k.get("end", np.inf))
    idx = np.flatnonzero(sel)

    # pick the event bin(s) that end at each kept sample, reading one 10 s chunk at a time
    want = s["ev_bin"][idx, None] - np.arange(history - 1, -1, -1)[None]   # (n, history), oldest first
    want = np.where(want >= 0, want, -1)                                   # -1: before the recording
    h, w = meta["event_shape"]
    hist = np.zeros((len(idx), history, 2, h, w), np.uint8)
    for f in sorted((cache_dir / "events").glob("chunk_*.npz")):
        c = np.load(f)
        first, bins = int(c["first_bin"]), c["bins"]
        hit = (want >= first) & (want < first + len(bins))
        hist[hit] = bins[want[hit] - first]

    rec = {
        "name": cache_dir.name,
        "t": s["t"][idx],
        "dir": cache_dir,
        "sample": idx,                                    # index in samples.npz
        "thermal": s["thermal"][idx].astype(np.float32),  # (n, 24, 32), noise-free; noise is added in training
        "hist": hist,                                     # (n, history, 2, h, w) counts, darker/brighter
        "y": y[idx], "near": near[idx],
        "gyro": np.linalg.norm(s["gyro"][idx], axis=1),   # rad/s, for the still/moving split
    }
    rec["events"] = rec["hist"][:, -1]                # (n, 2, h, w): the 40 ms bin before each sample
    if history == 1:
        del rec["hist"]
    return rec


def load_all(cache_root, history=1, every=1):
    dirs = sorted(d for d in Path(cache_root).expanduser().iterdir() if (d / "meta.json").exists())
    if not dirs:
        raise SystemExit(f"no complete caches in {cache_root}")
    return [load_recording(d, history, every) for d in dirs]


def load_e1(e1_dir, rec, keys):
    """Arrays from build_e1_cache.py for the samples in `rec`, in rec's order. Returns {key: array}."""
    e1_dir = Path(e1_dir).expanduser()
    if not (e1_dir / "meta.json").exists():
        raise SystemExit(f"{e1_dir} is not a complete E1 cache; run build_e1_cache.py first")
    pos = {int(s): j for j, s in enumerate(rec["sample"])}
    out, filled = {}, np.zeros(len(pos), bool)
    for f in sorted(e1_dir.glob("part_*.npz")):
        c = np.load(f)
        rows = [(i, pos[int(s)]) for i, s in enumerate(c["sample"]) if int(s) in pos]
        if not rows:
            continue
        src, dst = map(np.array, zip(*rows))
        for k in keys:
            if k not in out:
                out[k] = np.zeros((len(pos),) + c[k].shape[1:], c[k].dtype)
            out[k][dst] = c[k][src]
        filled[dst] = True
    if not filled.all():
        raise SystemExit(f"{e1_dir}: {int((~filled).sum())} samples of {rec['name']} are missing")
    return out


def event_input(counts):
    """uint8 counts -> float in [0, 1] on a log scale (dense flicker would swamp a linear scale)."""
    return np.log1p(counts.astype(np.float32)) / EVENT_SCALE
