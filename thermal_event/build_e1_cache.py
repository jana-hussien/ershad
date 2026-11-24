"""Build the exact E1 inputs for one recording from its raw bag (needs per-event timestamps).

Usage:
    python build_e1_cache.py BAG --cache cache/room_01 --out e1_cache/room_01

--cache is the recording's existing cache (from build_cache.py). It is only read: its samples,
event grid and keep.json say which moments to build inputs for, so labels, thermal and the old
40 ms count bins all stay lined up with the new inputs.

For each kept sample, the window ends exactly where the old 40 ms bin before that sample ends.
Stored per sample (formulas in representations.window_inputs):
    fixed_n        count images of the most recent 26,880 / 50k / 200k / 800k events (uint16)
    ts_age_ms      time surface: ms since each pixel last changed, per polarity (uint16, 1000 = none)
    ts_zubic       the same at 6 moments, in 0.1 ms steps (uint8, 255 = none in 25.5 ms)
    voxel, voxel5  6- and 5-slice voxel grids over 240 ms with exact time weighting (float16)
    rvt            RVT stacked histogram: last 50 ms, 10 bins x 2 polarities, clipped at 10 (uint8)
    snn            12 x 20 ms count images for the spiking network (uint8)
    graph          up to 10,000 events sampled at random from the 240 ms window (AEGNN's N-Cars count), exact times
    eventmg        the most recent 20,000 events (EventMG's window), exact times
Written in parts of 64 samples to <out>.partial, renamed to <out> when complete.

Alignment check: for every sample, the events of the last 40 ms are recounted from the bag and
compared with the old cache's bin. meta.json reports the share of pixels that differ; it should be
~0 (a few late-arriving events at most).
"""
import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

import bag_io
import representations as R
import storage

PART = 64      # samples per output file
LATE_S = 0.2   # wait until the stream is this far past a sample's end before building it


def load_targets(cache):
    """Sample indices to build, their window ends, and the event grid of the existing cache."""
    meta = json.loads((cache / "meta.json").read_text())
    s = np.load(cache / "samples.npz")
    t, ev_bin = s["t"], s["ev_bin"]
    keep = np.ones(len(t), bool)
    if (cache / "keep.json").exists():
        k = json.loads((cache / "keep.json").read_text())
        rel = t - t[0]
        keep = (rel >= k.get("start", 0)) & (rel <= k.get("end", np.inf))
    keep &= ev_bin >= 0
    grid0, b = meta["event_grid_start"], meta["event_bin_s"]
    ends = grid0 + (ev_bin + 1) * b
    return meta, np.flatnonzero(keep), ends, ev_bin


def old_bin(cache, ev_bin, cache_chunks={}):
    """The old cache's 40 ms count bin, for the alignment check."""
    c = ev_bin // 250
    if c not in cache_chunks:
        cache_chunks.clear()
        cache_chunks[c] = np.load(cache / "events" / f"chunk_{c:04d}.npz")["bins"]
    return cache_chunks[c][ev_bin - c * 250]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--cache", required=True, help="existing cache of this recording, e.g. cache/room_01")
    ap.add_argument("--out", required=True, help="where to write, e.g. e1_cache/room_01")
    ap.add_argument("--dataset", choices=sorted(bag_io.DATASETS), default="m2dgr")
    ap.add_argument("--seed", type=int, default=0, help="for the graph network's event sample")
    ap.add_argument("--max-samples", type=int, default=0, help="only the first N kept samples (smoke test)")
    ap.add_argument("--every", type=int, default=1, help="only samples whose index is a multiple of N (2 = 2 Hz)")
    args = ap.parse_args()

    cache, out = Path(args.cache).expanduser(), Path(args.out).expanduser()
    if (out / "meta.json").exists():
        print(f"{out} is already complete; nothing to do. Delete it first to rebuild.")
        return
    part = out.with_name(out.name + ".partial")
    if part.exists():
        shutil.rmtree(part)

    meta, todo, ends, ev_bin = load_targets(cache)
    todo = todo[todo % args.every == 0]
    kept = len(todo)
    if args.max_samples:
        todo = todo[:args.max_samples]
    h, w = meta["event_shape"]
    factor, grid0, binw = meta["event_binning_factor"], meta["event_grid_start"], meta["event_bin_s"]
    per_sample = h * w * (len(R.FIXED_N) * 2 * 2 + 2 * 2 + R.ZUBIC_SAMPLES * 2 + R.SNN_STEPS * 2
                          + (R.VOXEL_BINS + R.VOXEL5_BINS) * 2 + R.RVT_BINS * 2) + (R.GRAPH_STORE + R.EVENTMG_N) * 9
    storage.require_free(out.parent, len(todo) * per_sample, f"E1 inputs for {len(todo)} samples")
    part.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    print(f"{cache.name}: {len(todo)} samples to build"
          + (f" (first {len(todo)} of {kept}, --max-samples)" if len(todo) < kept else "")
          + (f"; {len(ends) - kept} outside keep.json" if len(ends) > kept else ""))

    try:
        buf, stats, rows, done = [], Counter(), [], 0
        last_end = -np.inf
        t_start = time.time()

        def build(i):
            """Build sample i from the buffered events."""
            nonlocal last_end
            end = ends[i]
            t = np.concatenate([b[0] for b in buf]) if buf else np.empty(0)
            m = (t >= end - R.LOOKBACK_S) & (t < end)
            x, y, p = (np.concatenate([b[k] for b in buf])[m] if buf else np.empty(0, np.uint16)
                       for k in (1, 2, 3))
            r = R.window_inputs(t[m], x, y, p, end, h, w, rng)
            # alignment check against the old cache's 40 ms bin (same binning formula as build_cache.py)
            in_bin = np.floor((t[m] - grid0) / binw).astype(np.int64) == ev_bin[i]
            mine = np.zeros((2, h, w), np.int64)
            np.add.at(mine, (p[in_bin].astype(np.int64), y[in_bin], x[in_bin]), 1)
            stats["check_pixels"] += mine.size
            stats["check_pixels_differ"] += int((np.minimum(mine, 255) != old_bin(cache, ev_bin[i])).sum())
            stats["fixed_n_capped"] += int((r["fixed_n_events"] < np.array(R.FIXED_N)).sum())
            stats["saturated_pixels"] += int((r["snn"] == 255).sum())  # 20 ms steps stored as uint8
            r["sample"] = i
            rows.append(r)
            last_end = end

        def flush():
            nonlocal rows
            if not rows:
                return
            keys = [k for k in rows[0] if k != "sample"]
            np.savez_compressed(part / f"part_{rows[0]['sample']:05d}.npz",
                                sample=np.array([r["sample"] for r in rows]),
                                **{k: np.stack([np.asarray(r[k]) for r in rows]) for k in keys})
            rows = []

        nxt = 0
        for ev in bag_io.iter_event_packets(args.bag, bag_io.DATASETS[args.dataset]["events"]):
            # events older than a sample already built: that sample missed them (later samples still get them)
            stats["late_events"] += int((ev.t < last_end).sum())
            buf.append((ev.t, ev.x // factor, ev.y // factor, ev.p))
            while nxt < len(todo) and ev.t.max() > ends[todo[nxt]] + LATE_S:
                build(todo[nxt])
                nxt += 1
                if len(rows) == PART:
                    flush()
                done = nxt
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)} samples, {time.time() - t_start:.0f} s")
            # drop packets too old for any remaining sample
            if nxt < len(todo):
                oldest = ends[todo[nxt]] - R.LOOKBACK_S
                while buf and buf[0][0].max() < oldest:
                    buf.pop(0)
            elif buf:
                break  # every sample built
        while nxt < len(todo):  # stream ended
            build(todo[nxt])
            nxt += 1
        flush()

        stats = dict(stats)
        stats["check_share_differ"] = stats.pop("check_pixels_differ") / max(stats.pop("check_pixels"), 1)
        info = {
            "recording": cache.name, "bag": Path(args.bag).name, "samples_built": int(len(todo)),
            "event_shape": [h, w], "window_s": R.WINDOW_S, "voxel_bins": R.VOXEL_BINS,
            "snn_steps": R.SNN_STEPS, "ts_max_ms": R.TS_MAX_MS, "fixed_n": list(R.FIXED_N),
            "fixed_max_s": R.FIXED_MAX_S, "graph_store": R.GRAPH_STORE, "seed": args.seed,
            "stats": stats, "build_s": round(time.time() - t_start),
        }
        (part / "meta.json").write_text(json.dumps(info, indent=1))
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        print("failed; removed the partial output. The bag and the old cache were not touched.")
        raise
    part.rename(out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(json.dumps(info["stats"], indent=1))
    if stats["check_share_differ"] > 1e-3:
        print("WARNING: the recounted 40 ms bins differ from the old cache; the windows may be misaligned.")
    print(f"E1 cache complete: {out} ({size / 1024 ** 2:.0f} MB, {info['build_s']} s). "
          f"You can now delete {args.bag}.")


if __name__ == "__main__":
    main()
