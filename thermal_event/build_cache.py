"""Turn one bag into a small cache with everything E1 to E5 need, so the bag can be deleted.

Usage:
    python build_cache.py BAG --out cache/room_01 [--dataset m2dgr] [--yolo yolov8m.pt]

Checks free space before starting, writes into <out>.partial, and only renames it to <out> once
everything is written. Delete the bag only after it prints "cache complete".

Cache contents:
    meta.json          settings, sensor sizes, sample count, class distribution (written last)
    samples.npz        per 4 Hz sample: time, simulated 32x24 thermal (no noise), RGB offset, gyro,
                       index of the event bin that ends at or just before the sample
    labels.csv         per sample and zone: label and near/far (rules in labels.py)
    boxes.csv          every YOLO box (conf >= --conf), so the label rules can change without the bag
    imu.npz            full IMU stream: time, gyro, accel
    events/chunk_XXXX.npz   continuous 40 ms event-count bins (2 x H x W, uint8, darker/brighter),
                       10 s per file; E1-E4 use the bin before each sample, E5 uses all of them
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

import bag_io
import simulate
import storage
from labels import CLASSES, LABEL_CONF, ZONES, nearest, zone_labels

BIN = simulate.EVENT_WINDOW  # 40 ms, the "event clock" for E5
CHUNK_BINS = 250             # 10 s of bins per file
MAX_BIN_BYTES = 2 * 240 * 320  # upper bound per bin (uint8, after binning to <= 320 wide)


def build_event_bins(bag, topic, grid0, n_bins, outdir):
    """Stream events into 40 ms count bins and write them chunk by chunk. Returns (H, W, stats)."""
    outdir.mkdir()
    chunks, flushed, shape, factor = {}, set(), None, 1
    stats = Counter()

    def flush(c):
        acc = chunks.pop(c)
        flushed.add(c)
        stats["saturated_pixels"] += int((acc > 255).sum())
        if storage.free_bytes(outdir) < storage.MARGIN:
            raise OSError("disk nearly full while writing event bins")
        np.savez_compressed(outdir / f"chunk_{c:04d}.npz", bins=np.minimum(acc, 255).astype(np.uint8),
                            first_bin=c * CHUNK_BINS)

    for ev in bag_io.iter_event_packets(bag, topic):
        if shape is None:
            factor = max(1, ev.width // simulate.EVENT_WIDTH)
            shape = (ev.height // factor, ev.width // factor)
        h, w = shape
        b = np.floor((ev.t - grid0) / BIN).astype(np.int64)
        ok = (b >= 0) & (b < n_bins)
        stats["events_outside_samples"] += int((~ok).sum())
        b, x, y, p = b[ok], ev.x[ok] // factor, ev.y[ok] // factor, ev.p[ok].astype(np.int64)
        if len(b) == 0:
            continue
        # chunks more than one chunk behind this packet are complete
        for c in [c for c in chunks if c < b.min() // CHUNK_BINS - 1]:
            flush(c)
        for c in np.unique(b // CHUNK_BINS):
            m = b // CHUNK_BINS == c
            if c in flushed:  # arrived after its chunk was written; rare, so drop and count
                stats["late_events_dropped"] += int(m.sum())
                continue
            lb = b[m] - c * CHUNK_BINS
            lo, hi = lb.min(), lb.max() + 1
            flat = ((lb - lo) * 2 + p[m]) * (h * w) + y[m].astype(np.int64) * w + x[m]
            counts = np.bincount(flat, minlength=(hi - lo) * 2 * h * w).reshape(hi - lo, 2, h, w)
            if c not in chunks:
                chunks[c] = np.zeros((min(CHUNK_BINS, n_bins - c * CHUNK_BINS), 2, h, w), np.uint16)
            chunks[c][lo:hi] += counts.astype(np.uint16)
        stats["events"] += len(b)
    for c in sorted(chunks):
        flush(c)
    # chunks with no events at all still get a file, so every bin exists
    if shape:
        for c in range((n_bins + CHUNK_BINS - 1) // CHUNK_BINS):
            f = outdir / f"chunk_{c:04d}.npz"
            if not f.exists():
                n = min(CHUNK_BINS, n_bins - c * CHUNK_BINS)
                np.savez_compressed(f, bins=np.zeros((n, 2) + shape, np.uint8), first_bin=c * CHUNK_BINS)
    return shape, factor, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--out", required=True, help="cache folder, e.g. cache/room_01")
    ap.add_argument("--dataset", choices=sorted(bag_io.DATASETS), default="m2dgr")
    ap.add_argument("--yolo", default="yolov8m.pt", help="YOLO weights (needs ultralytics)")
    ap.add_argument("--conf", type=float, default=0.1, help="lowest box confidence to store")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    if (out / "meta.json").exists():
        print(f"{out} is already a complete cache; nothing to do. Delete it first to rebuild.")
        return
    part = out.with_name(out.name + ".partial")
    if part.exists():
        shutil.rmtree(part)  # left over from an interrupted run

    _, dur = bag_io.summarize(args.bag)
    estimate = int(dur / BIN * MAX_BIN_BYTES) + 100 * 1024 ** 2
    storage.require_free(out.parent, estimate, f"the cache of {Path(args.bag).name} ({dur:.0f} s)")
    part.mkdir(parents=True)

    try:
        T = bag_io.DATASETS[args.dataset]
        thermal_t, _ = bag_io.read_images(args.bag, T["thermal"], keep=set())
        rgb_t, _ = bag_io.read_images(args.bag, T["rgb"], keep=set())
        imu_t, gyro, accel = bag_io.read_imu(args.bag, T["imu"])
        np.savez_compressed(part / "imu.npz", t=imu_t, gyro=gyro, accel=accel)

        a_idx = simulate.anchor_times(thermal_t)
        a_t = thermal_t[a_idx]
        ok = (a_t - BIN >= max(rgb_t[0], imu_t[0])) & (a_t <= min(rgb_t[-1], imu_t[-1]))
        a_idx, a_t = a_idx[ok], a_t[ok]
        rgb_idx, rgb_dt = nearest(rgb_t, a_t)
        imu_idx, _ = nearest(imu_t, a_t)

        # event grid: bins of 40 ms from just before the first sample to the last sample
        grid0 = a_t[0] - BIN
        n_bins = int(np.ceil((a_t[-1] - grid0) / BIN))
        ev_bin = np.floor((a_t - grid0) / BIN + 1e-6).astype(int) - 1  # last bin ending at or before the sample
        print(f"{len(a_t)} samples; binning events into {n_bins} bins of {BIN * 1e3:.0f} ms ...")
        shape, factor, ev_stats = build_event_bins(args.bag, T["events"], grid0, n_bins, part / "events")

        print("simulating thermal ...")
        thermal = np.zeros((len(a_t),) + simulate.THERMAL_SIZE[::-1], np.float16)
        pos = {f: i for i, f in enumerate(a_idx)}
        raw_dtype = None
        for f, _, img in bag_io.iter_images(args.bag, T["thermal"], keep=set(pos)):
            if img is not None:
                raw_dtype = str(img.dtype)
                thermal[pos[f]] = simulate.thermal_to_mlx(img, noise_std=0)

        print("auto-labelling RGB with YOLO ...")
        from ultralytics import YOLO
        model = YOLO(args.yolo)
        want = {}
        for i, r in enumerate(rgb_idx):
            want.setdefault(r, []).append(i)
        box_rows, label_rows, counts = [], [], Counter()
        for r, _, img in bag_io.iter_images(args.bag, T["rgb"], keep=set(want)):
            if img is None:
                continue
            h, w = img.shape[:2]
            res = model(np.ascontiguousarray(img[:, :, ::-1]), conf=args.conf, verbose=False)[0]
            boxes = [(*b, model.names[int(c)], float(s)) for b, c, s in
                     zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]
            labels = zone_labels([b[:5] for b in boxes if b[5] >= LABEL_CONF], w, h)
            for i in want[r]:
                box_rows += [[i, *[f"{v:.1f}" for v in b[:4]], b[4], f"{b[5]:.3f}", w, h] for b in boxes]
                for zone, (label, near) in labels.items():
                    label_rows.append([i, zone, label, int(near)])
                    counts[f"{zone}/{label}/{'near' if near else 'far'}"] += 1

        with open(part / "boxes.csv", "w", newline="") as f:
            csv.writer(f).writerows([["sample", "x1", "y1", "x2", "y2", "class", "conf", "img_w", "img_h"],
                                     *sorted(box_rows, key=lambda r: r[0])])
        with open(part / "labels.csv", "w", newline="") as f:
            csv.writer(f).writerows([["sample", "zone", "label", "near"], *sorted(label_rows)])
        np.savez_compressed(part / "samples.npz", t=a_t, thermal=thermal, rgb_offset=rgb_dt,
                            gyro=gyro[imu_idx], ev_bin=ev_bin)

        meta = {
            "bag": Path(args.bag).name, "dataset": args.dataset, "duration_s": dur, "samples": len(a_t),
            "thermal_hz": simulate.THERMAL_HZ, "thermal_size": simulate.THERMAL_SIZE,
            "thermal_raw_dtype": raw_dtype, "event_bin_s": BIN, "event_grid_start": grid0,
            "event_bins": n_bins, "event_shape": shape, "event_binning_factor": factor,
            "event_stats": dict(ev_stats), "yolo": args.yolo, "label_conf": LABEL_CONF,
            "rgb_offset_ms_median": float(np.median(np.abs(rgb_dt)) * 1e3),
            "class_counts": dict(sorted(counts.items())),
        }
        (part / "meta.json").write_text(json.dumps(meta, indent=1))
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        print("failed; removed the partial cache. The bag was not touched.")
        raise
    part.rename(out)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nclass distribution (samples per zone; near in brackets):\n{'':8s}" +
          "".join(f"{c:>14s}" for c in CLASSES))
    for zone in ZONES:
        cells = [f"{counts[f'{zone}/{c}/near'] + counts[f'{zone}/{c}/far']} ({counts[f'{zone}/{c}/near']})"
                 for c in CLASSES]
        print(f"{zone:8s}" + "".join(f"{c:>14s}" for c in cells))
    print(f"\ncache complete: {out} ({size / 1024 ** 2:.0f} MB). You can now delete {args.bag}.")


if __name__ == "__main__":
    main()
