"""Draw every E1 input for one real sample, as the network receives it.

Usage:
    python e1_visualize.py --cache cache --e1-cache e1_cache --recording hall_02 --out e1_hall_02.png
    [--sample N]   index among the recording's kept samples; default: a sample with a person in it
    [--first N]    only look at the first N kept samples (for the smoke test)

Colours: count-type inputs show brighter minus darker (red = got brighter, blue = got darker, white =
nothing). Grey lines are the zone borders; green boxes are the YOLO boxes from the RGB camera, which
is a different camera, so they only roughly line up.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset
import representations as R
from labels import CLASSES, LABEL_CONF, ZONES


def signed(img2):
    """(2, H, W) log-scaled darker/brighter -> brighter minus darker."""
    return img2[1] - img2[0]


def pick(rec, first):
    n = min(len(rec["y"]), first or len(rec["y"]))
    person = np.flatnonzero((rec["y"][:n] == CLASSES.index("person")).any(1))
    pool = person if len(person) else np.arange(n)
    busy = rec["hist"][pool, -1].reshape(len(pool), -1).sum(1, dtype=np.int64)
    return int(pool[busy.argmax()])


def boxes_for(cache_dir, sample, w, h):
    out = []
    with open(Path(cache_dir) / "boxes.csv") as f:
        for r in csv.DictReader(f):
            if int(r["sample"]) == sample and float(r["conf"]) >= LABEL_CONF:
                sx, sy = w / float(r["img_w"]), h / float(r["img_h"])
                out.append((float(r["x1"]) * sx, float(r["y1"]) * sy, float(r["x2"]) * sx, float(r["y2"]) * sy, r["class"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--e1-cache", required=True)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--first", type=int, default=0)
    args = ap.parse_args()

    rec = dataset.load_recording(Path(args.cache) / args.recording, history=6)
    i = args.sample if args.sample is not None else pick(rec, args.first)
    one = {k: (v[i:i + 1] if isinstance(v, np.ndarray) else v) for k, v in rec.items()}
    e = dataset.load_e1(Path(args.e1_cache) / args.recording, one,
                        ["voxel", "ts_age_ms", "fixed_n", "fixed_n_window_s", "snn", "graph", "graph_n",
                         "rvt", "voxel5", "ts_zubic"])
    hist = one["hist"]
    h, w = hist.shape[-2:]
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    boxes = boxes_for(Path(args.cache) / args.recording, int(one["sample"][0]), w, h)

    fig = plt.figure(figsize=(19, 22))
    labels = ", ".join(f"{z}: {CLASSES[c]}" for z, c in zip(ZONES, one["y"][0]))
    fig.suptitle(f"{args.recording}, sample {int(one['sample'][0])} (t = {one['t'][0] - rec['t'][0]:.2f} s)   |   "
                 f"labels: {labels}", fontsize=14)
    rows = fig.subfigures(7, 1, hspace=0.02)
    for sub, t in zip(rows, ["Count images (2 channels each: darker, brighter)",
                             "Stacked count images: 6 × 40 ms side by side → 12 channels",
                             "Voxel grid: 6 slices of brighter − darker, each event split between its 2 nearest slices",
                             "Time surface  ·  fixed number of events  ·  graph",
                             "Spiking network input: 12 steps × 20 ms (steps 1-6)",
                             "Spiking network input (steps 7-12)",
                             "Paper settings: raw counts · RVT (50 ms, 10 bins) · E2VID voxel (5, normalised) · "
                             "time-surface fades · Zubic"]):
        sub.suptitle(t, fontsize=11, weight="bold", x=0.02, ha="left")
    axes = [sub.subplots(1, 6) for sub in rows]
    for ax in np.ravel(axes):
        ax.set_axis_off()
    grab = lambda r, c: (axes[r][c].set_axis_on(), axes[r][c])[1]

    def show(ax, img, title, kind="signed"):
        if kind == "signed":
            nz = np.abs(img[img != 0])  # colour scale from the 99.5th percentile, so a few hot pixels don't wash it out
            lim = max(float(np.percentile(nz, 99.5)) if nz.size else 0.0, 1e-6)
            ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        else:
            ax.imshow(img, cmap=kind, interpolation="nearest")
        for x in (w / 3, 2 * w / 3):
            ax.axvline(x, color="0.6", lw=0.6)
        for x1, y1, x2, y2, _ in boxes:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#1a9850", lw=1))
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(0, w), ax.set_ylim(h, 0)

    for c, n in enumerate((1, 2, 3, 6)):
        x = R.log_counts(T(R.count_window(hist, n)))[0].numpy()
        show(grab(0, c), signed(x), f"count_{40 * n}: 2×{h}×{w}, last {40 * n} ms")
    ax = grab(0, 4)
    ax.imshow(one["thermal"][0], cmap="inferno", interpolation="nearest")
    ax.set_title("thermal (reference): 1×24×32", fontsize=8.5), ax.set_xticks([]), ax.set_yticks([])

    st = R.log_counts(T(R.stack_bins(hist)))[0].numpy().reshape(6, 2, h, w)
    for c in range(6):
        show(grab(1, c), signed(st[c]), f"ch {2 * c}-{2 * c + 1}: {-240 + 40 * c} to {-200 + 40 * c} ms")
    vx = R.signed_log(T(e["voxel"]))[0].numpy()
    for c in range(6):
        show(grab(2, c), vx[c], f"slice {c}: centred at {-240 + 48 * c} ms")

    ts = R.time_surface(T(e["ts_age_ms"]))[0].numpy()
    show(grab(3, 0), ts.max(0), "time surface: exp(−age/50 ms), 2 ch\n(shown: newer polarity)", "gray")
    for c, n in enumerate(R.FIXED_N[1:], 1):  # 26,880 (E2VID's rule) is drawn in the paper row
        x = R.log_counts(T(e["fixed_n"][:, c]))[0].numpy()
        show(grab(3, c), signed(x),
             f"fixed_n_{n // 1000}k: last {n:,} events\n= last {1e3 * e['fixed_n_window_s'][0, c]:.0f} ms here")
    ax = grab(3, 4)
    pts = R.graph_points(e["graph"], e["graph_n"], np.random.default_rng(0))[0]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2] * 1e3, s=2, cmap="viridis_r")
    for x in (w / 3, 2 * w / 3):
        ax.axvline(x, color="0.6", lw=0.6)
    ax.set_xlim(0, w), ax.set_ylim(h, 0), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"graph: {R.GRAPH_POINTS} sampled events (dots),\neach linked to its 16 nearest in (x, y, t)",
                 fontsize=8.5)
    fig.colorbar(sc, ax=axes[3][5], fraction=0.5, location="left", label="graph: ms before now")

    raw = R.raw_counts(T(R.count_window(hist, 1)))[0].numpy()
    show(grab(6, 0), signed(raw), "count_40_raw: plain counts (no log)")
    rv = R.rvt_input(T(e["rvt"]))[0].numpy().reshape(R.RVT_BINS, 2, h, w)
    show(grab(6, 1), signed(rv.sum(0)), "stacked_rvt_10x5: 20 ch, 50 ms\n(shown: all 10 bins added)")
    v5 = R.voxel_norm(T(e["voxel5"]))[0].numpy()
    show(grab(6, 2), v5[-1], "voxel5_norm: 5 slices, mean 0 / std 1\n(shown: newest slice)")
    for c, tau in enumerate((25, 100)):
        show(grab(6, 3 + c), R.time_surface(T(e["ts_age_ms"]), tau)[0].numpy().max(0), f"time surface, fade {tau} ms",
             "gray")
    z = R.time_surface_zubic(T(e["ts_zubic"]))[0].numpy().reshape(R.ZUBIC_SAMPLES, 2, h, w)
    show(grab(6, 5), z[-1].max(0), "time_surface_zubic: 12 ch, fade 5 ms\n(shown: newest moment)", "gray")

    sn = R.log_counts(T(e["snn"]))[0].numpy()
    for s in range(R.SNN_STEPS):
        show(grab(4 + s // 6, s % 6), signed(sn[s]),
             f"step {s + 1}: {-240 + 20 * s} to {-220 + 20 * s} ms")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=90, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
