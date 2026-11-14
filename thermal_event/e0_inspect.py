"""E0: is this bag usable? Check sync, look at the data, and count auto-labels per zone.

Usage:
    python e0_inspect.py BAG --summary-only          # list topics, types and rates
    python e0_inspect.py BAG --dataset vivid ...     # ViViD++ topic names (default: m2dgr)
    python e0_inspect.py BAG --out e0_out            # sync check + sample images
    python e0_inspect.py BAG --out e0_out --yolo yolov8x.pt   # + auto-labels and class distribution

Writes to --out:
    summary.txt             topics, sync statistics, class distribution
    sync.csv                one row per 4 Hz sample: matched RGB frame, time offset, event count, gyro
    samples/sample_XX.png   thermal | simulated 32x24 | 40 ms events | RGB (with boxes and zones)
    labels.csv              (--yolo) label and near/far per sample and zone
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import bag_io
import simulate
import storage
from labels import PRIORITY, ZONES, nearest, zone_labels

PANEL_H = 240


def stretch(img):
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def panel(img, interp=cv2.INTER_AREA):
    w = round(img.shape[1] * PANEL_H / img.shape[0])
    img = cv2.resize(img, (w, PANEL_H), interpolation=interp)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def caption(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def sample_image(thermal, events, t, rgb, boxes, labels):
    th = caption(panel(cv2.applyColorMap(stretch(thermal), cv2.COLORMAP_INFERNO)), "thermal")
    mlx = simulate.thermal_to_mlx(thermal, noise_std=0)
    mlx = caption(panel(cv2.applyColorMap(stretch(mlx), cv2.COLORMAP_INFERNO), cv2.INTER_NEAREST),
                  "simulated 32x24")
    ev = simulate.event_frame(events, t)
    vis = np.full(ev.shape[1:] + (3,), 255, np.uint8)
    vis[ev[1] > 0] = (0, 0, 255)  # brighter: red (BGR)
    vis[ev[0] > 0] = (255, 0, 0)  # darker: blue
    ev = caption(panel(vis, cv2.INTER_NEAREST), f"events 40 ms ({int(ev.sum())})")
    rgb = np.ascontiguousarray(rgb[:, :, ::-1]) if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    h, w = rgb.shape[:2]
    for x1, y1, x2, y2, name in boxes:
        cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(rgb, name, (int(x1), int(y1) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for k in (1, 2):
        cv2.line(rgb, (k * w // 3, 0), (k * w // 3, h), (255, 255, 255), 1)
    text = " | ".join(f"{z[0].upper()}: {l}{' (near)' if n else ''}" for z, (l, n) in labels.items())
    rgb = caption(panel(rgb), text or "RGB")
    return np.hstack([th, mlx, ev, rgb])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--out", default="e0_out")
    ap.add_argument("--dataset", choices=sorted(bag_io.DATASETS), default="m2dgr")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--samples", type=int, default=12, help="number of sample images")
    ap.add_argument("--yolo", help="YOLO weights, e.g. yolov8x.pt (needs ultralytics)")
    args = ap.parse_args()

    rows, dur = bag_io.summarize(args.bag)
    topic_lines = [f"{t:45s} {ty:35s} {n:8d} msgs {hz:8.1f} Hz" for t, ty, n, hz in rows]
    print(f"{args.bag}: {dur:.1f} s\n" + "\n".join(topic_lines))
    if args.summary_only:
        return
    out = Path(args.out)
    storage.require_free(out, 0.2 * storage.GB, "E0 results and YOLO weights")
    (out / "samples").mkdir(parents=True, exist_ok=True)

    T = bag_io.DATASETS[args.dataset]
    thermal_t, _ = bag_io.read_images(args.bag, T["thermal"], keep=set())
    rgb_t, _ = bag_io.read_images(args.bag, T["rgb"], keep=set())
    imu_t, gyro, _ = bag_io.read_imu(args.bag, T["imu"])

    # keep 4 Hz samples where every stream has data
    a_idx = simulate.anchor_times(thermal_t)
    a_t = thermal_t[a_idx]
    ok = ((a_t - simulate.EVENT_WINDOW >= max(rgb_t[0], imu_t[0])) & (a_t <= min(rgb_t[-1], imu_t[-1])))
    a_idx, a_t = a_idx[ok], a_t[ok]
    rgb_idx, rgb_dt = nearest(rgb_t, a_t)
    imu_idx, _ = nearest(imu_t, a_t)
    gyro_norm = np.linalg.norm(gyro[imu_idx], axis=1)

    events = bag_io.read_events(args.bag, [(t - simulate.EVENT_WINDOW, t) for t in a_t], T["events"])
    order = np.argsort(events.t, kind="stable")
    events = bag_io.Events(events.t[order], events.x[order], events.y[order], events.p[order],
                           events.width, events.height)
    ev_count = np.searchsorted(events.t, a_t) - np.searchsorted(events.t, a_t - simulate.EVENT_WINDOW)

    with open(out / "sync.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "time", "thermal_frame", "rgb_frame", "rgb_offset_ms", "events_40ms", "gyro_rad_s"])
        for i in range(len(a_t)):
            w.writerow([i, f"{a_t[i]:.4f}", a_idx[i], rgb_idx[i], f"{rgb_dt[i] * 1e3:.1f}", ev_count[i],
                        f"{gyro_norm[i]:.3f}"])

    off = np.abs(rgb_dt) * 1e3
    report = [
        f"bag: {args.bag} ({dur:.1f} s)", "", "topics:", *topic_lines, "",
        f"stream start/end (s): thermal {thermal_t[0]:.2f}-{thermal_t[-1]:.2f}, "
        f"rgb {rgb_t[0]:.2f}-{rgb_t[-1]:.2f}, imu {imu_t[0]:.2f}-{imu_t[-1]:.2f}",
        f"4 Hz samples with all streams: {len(a_t)}",
        f"thermal to nearest RGB offset (ms): median {np.median(off):.1f}, 95% {np.percentile(off, 95):.1f}, "
        f"max {off.max():.1f}",
        f"events per 40 ms window: median {np.median(ev_count):.0f}, "
        f"windows with no events {np.mean(ev_count == 0):.1%}",
        f"event sensor size: {events.width}x{events.height}",
        f"gyro (rad/s): median {np.median(gyro_norm):.2f}, 95% {np.percentile(gyro_norm, 95):.2f}",
    ]

    # sample images (and auto-labels for every sample if --yolo)
    pick = set(np.linspace(0, len(a_t) - 1, min(args.samples, len(a_t))).astype(int))
    _, thermal_frames = bag_io.read_images(args.bag, T["thermal"], keep={a_idx[i] for i in pick})
    model = None
    if args.yolo:
        from ultralytics import YOLO
        model = YOLO(args.yolo)
    want = {}  # RGB frame -> samples that use it
    for i in (range(len(a_t)) if model else sorted(pick)):
        want.setdefault(rgb_idx[i], []).append(i)

    labels_rows, counts = [], Counter()
    for r, _, img in bag_io.iter_images(args.bag, T["rgb"], keep=set(want)):
        if img is None:
            continue
        boxes = []
        if model:
            res = model(np.ascontiguousarray(img[:, :, ::-1]), verbose=False)[0]
            boxes = [(*b, model.names[int(c)]) for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist())]
        labels = zone_labels(boxes, img.shape[1], img.shape[0]) if model else {}
        for i in want[r]:
            for zone, (label, near) in labels.items():
                labels_rows.append([i, f"{a_t[i]:.4f}", zone, label, int(near)])
                counts[(zone, label, near)] += 1
            if i in pick:
                n = sorted(pick).index(i)
                cv2.imwrite(str(out / "samples" / f"sample_{n:02d}.png"),
                            sample_image(thermal_frames[a_idx[i]], events, a_t[i], img, boxes, labels))

    if model:
        with open(out / "labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample", "time", "zone", "label", "near"])
            w.writerows(sorted(labels_rows))
        classes = ("clear",) + PRIORITY
        report += ["", "class distribution (samples per zone; near count in brackets):",
                   f"{'':8s}" + "".join(f"{c:>16s}" for c in classes)]
        for zone in ZONES:
            cells = [f"{counts[(zone, c, False)] + counts[(zone, c, True)]} ({counts[(zone, c, True)]})"
                     for c in classes]
            report.append(f"{zone:8s}" + "".join(f"{c:>16s}" for c in cells))

    (out / "summary.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report[len(topic_lines) + 4:]))
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
