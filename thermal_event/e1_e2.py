"""E1 (thermal-only, event-only) and E2 (late blend of their per-zone predictions).

Usage:
    python e1_e2.py --cache ~/Downloads/thermal_event_data/cache --out ~/Downloads/thermal_event_data/results/e1_e2

Train/test splits are by recording (leave one recording out), never by frame. With a single recording
it falls back to five contiguous time blocks, which is weaker and is flagged in the report.

Options:
    --task binary     free vs blocked per zone instead of clear/person/furniture/other
    --split blocks    "seen places": hold out time blocks inside each recording (5 per recording, 1 s gap),
                      to compare with leave-one-recording-out ("new places")

Writes:
    summary.md        tables: accuracy (all / still / moving) with 95% bootstrap CIs, and cost
    metrics.csv       the same numbers, machine-readable
    predictions.csv   per sample and zone: truth and each model's prediction
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import dataset
import storage
from labels import CLASSES, ZONES

N_Z = len(ZONES)
NAMES = CLASSES          # class names; main() switches to ("free", "blocked") for --task binary
N_C = len(NAMES)
THERMAL_NOISE = 0.01  # MLX90640-like noise, as a fraction of the thermal range (see simulate.py)
BOOT = 1000


# ---------------------------------------------------------------- models

def grid_pool(x, rows=2, cols=3):
    """Average over a rows x cols grid (works for any size, unlike adaptive pooling on Apple GPUs)."""
    cells = [c.mean((2, 3)) for r in torch.tensor_split(x, rows, 2) for c in torch.tensor_split(r, cols, 3)]
    return torch.stack(cells, -1).flatten(1)


def conv_block(i, o, stride):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ZoneNet(nn.Module):
    """Small CNN; pooled to a 2x3 grid so left/center/right stay spatially separate, then 3 zone heads."""

    def __init__(self, in_ch, widths, strides):
        super().__init__()
        layers, c = [], in_ch
        for w, s in zip(widths, strides):
            layers.append(conv_block(c, w, s))
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c * 6, N_Z * N_C)

    def forward(self, x):
        return self.head(grid_pool(self.features(x))).view(-1, N_Z, N_C)


def make_model(kind):
    if kind == "thermal":  # 1 x 24 x 32 -> 6 x 8
        return ZoneNet(1, (16, 32, 64, 64, 64), (1, 2, 1, 2, 1))
    # events: 2 x 240 x 320; strided convs are the learned downsampling (MSAD/MCFNet-style) -> 15 x 20
    return ZoneNet(2, (16, 32, 64, 64, 64), (2, 2, 2, 2, 1))


def cost(model, shape):
    """Parameters, MFLOPs (2 x multiply-adds of conv/linear layers) and CPU ms per single update."""
    macs = []

    def hook(m, _, out):
        if isinstance(m, nn.Conv2d):
            macs.append(out.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, nn.Linear):
            macs.append(m.in_features * m.out_features)

    model = model.cpu().eval()
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    x = torch.zeros((1,) + shape)
    with torch.no_grad():
        model(x)
        for h in hs:
            h.remove()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        for _ in range(20):
            model(x)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1e3)
        torch.set_num_threads(threads)
    return sum(p.numel() for p in model.parameters()), 2 * sum(macs) / 1e6, float(np.median(times))


# ---------------------------------------------------------------- data

def inputs(rec, kind, idx, rng):
    if kind == "thermal":
        x = rec["thermal"][idx][:, None]
        return x + rng.normal(0, THERMAL_NOISE, x.shape).astype(np.float32)
    return dataset.event_input(rec["events"][idx])


def batches(recs, kind, rows, batch, rng, train):
    """rows: list of (recording index, sample index). Yields (x, y) tensors."""
    order = rng.permutation(len(rows)) if train else np.arange(len(rows))
    for s in range(0, len(rows), batch):
        part = [rows[i] for i in order[s:s + batch]]
        xs, ys = [], []
        for r in sorted({r for r, _ in part}):
            idx = np.array([i for rr, i in part if rr == r])
            xs.append(inputs(recs[r], kind, idx, rng))
            ys.append(recs[r]["y"][idx])
        x, y = np.concatenate(xs), np.concatenate(ys)
        if train:  # mirror left/right: flip the image and swap the left and right zone labels
            flip = rng.random(len(x)) < 0.5
            x[flip] = x[flip][..., ::-1]
            y[flip] = y[flip][:, ::-1]
        yield torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def block_folds(recs, n_blocks=5, gap=4):
    """Hold out one contiguous time block of one recording at a time; train on everything else.

    gap drops 1 s either side of the test block so neighbouring frames don't leak."""
    for k, rec in enumerate(recs):
        n = len(rec["y"])
        edges = np.linspace(0, n, n_blocks + 1).astype(int)
        for b in range(n_blocks):
            lo, hi = edges[b], edges[b + 1]
            train = [(r, i) for r in range(len(recs)) for i in range(len(recs[r]["y"]))
                     if r != k or i < lo - gap or i >= hi + gap]
            yield f"{rec['name']}_block{b}", train, [(k, i) for i in range(lo, hi)]


def folds(recs, split="recording"):
    """(name, train rows, test rows). Leave one recording out; time blocks if there is only one."""
    if split == "blocks":
        yield from block_folds(recs)
        return
    if len(recs) > 1:
        for k, rec in enumerate(recs):
            train = [(r, i) for r in range(len(recs)) if r != k for i in range(len(recs[r]["y"]))]
            yield rec["name"], train, [(k, i) for i in range(len(rec["y"]))]
        return
    n = len(recs[0]["y"])
    edges = np.linspace(0, n, 6).astype(int)
    gap = 4  # drop 1 s either side of the test block so neighbouring frames don't leak
    for b in range(5):
        lo, hi = edges[b], edges[b + 1]
        train = [(0, i) for i in range(n) if i < lo - gap or i >= hi + gap]
        yield f"{recs[0]['name']}_block{b}", train, [(0, i) for i in range(lo, hi)]


# ---------------------------------------------------------------- training

def train_predict(recs, kind, train_rows, test_rows, seed, device, epochs, batch=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(kind).to(device)
    # inverse-frequency class weights from the training rows
    ytr = np.concatenate([recs[r]["y"][[i]] for r, i in train_rows])
    freq = np.bincount(ytr.ravel(), minlength=N_C) + 1
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(freq.sum() / (N_C * freq), dtype=torch.float32).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        for x, y in batches(recs, kind, train_rows, batch, rng, train=True):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x).reshape(-1, N_C), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in batches(recs, kind, test_rows, 128, np.random.default_rng(10_000 + seed), train=False):
            probs.append(torch.softmax(model(x.to(device)), -1).cpu().numpy())
    return np.concatenate(probs)  # (n_test, zones, classes)


# ---------------------------------------------------------------- metrics

def macro_f1(t, p):
    f1s = []
    for c in np.union1d(t, p):
        tp, fp, fn = np.sum((p == c) & (t == c)), np.sum((p == c) & (t != c)), np.sum((p != c) & (t == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


def metrics(t, p):
    """t, p: (n, zones). Per-zone macro F1 averaged over zones, exact match, Hamming accuracy."""
    if len(t) == 0:
        return dict(macro_f1=np.nan, exact=np.nan, hamming=np.nan)
    return dict(macro_f1=float(np.mean([macro_f1(t[:, z], p[:, z]) for z in range(t.shape[1])])),
                exact=float(np.mean((t == p).all(1))), hamming=float(np.mean(t == p)))


def with_ci(t, preds, groups, mask, rng):
    """Point estimate (mean over seeds) and 95% CI from resampling whole test groups (recordings/blocks)."""
    names = np.unique(groups[mask])
    point = {k: np.mean([metrics(t[mask], p[mask])[k] for p in preds]) for k in ("macro_f1", "exact", "hamming")}
    draws = {k: [] for k in point}
    for _ in range(BOOT):
        pick = rng.choice(names, len(names))
        sel = np.concatenate([np.flatnonzero(mask & (groups == g)) for g in pick])
        for k in point:
            draws[k].append(np.mean([metrics(t[sel], p[sel])[k] for p in preds]))
    return {k: (point[k], *np.nanpercentile(draws[k], [2.5, 97.5])) for k in point}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--task", choices=("classes", "binary"), default="classes")
    ap.add_argument("--split", choices=("recording", "blocks"), default="recording")
    args = ap.parse_args()
    global NAMES, N_C
    if args.task == "binary":
        NAMES = ("free", "blocked")
        N_C = 2
    out = Path(args.out).expanduser()
    storage.require_free(out, 0.05 * storage.GB, "E1/E2 results")
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    recs = dataset.load_all(args.cache)
    if args.task == "binary":
        for r in recs:
            r["y"] = (r["y"] > 0).astype(np.int64)  # anything but "clear" blocks the zone
    print(f"{len(recs)} recordings: " + ", ".join(f"{r['name']} ({len(r['y'])})" for r in recs) + f"; device {device}")
    split_note = ("seen places: 5 held-out time blocks per recording, 1 s gap" if args.split == "blocks" else
                  "leave-one-recording-out" if len(recs) > 1 else
                  "ONE recording only: five contiguous time blocks (weaker than a split by recording)")

    t_all, groups, gyro, rows_all, base = [], [], [], [], []
    probs = {k: [[] for _ in range(args.seeds)] for k in ("thermal", "event")}
    for name, train_rows, test_rows in folds(recs, args.split):
        print(f"fold {name}: train {len(train_rows)}, test {len(test_rows)}")
        for seed in range(args.seeds):
            for kind in probs:
                t0 = time.time()
                probs[kind][seed].append(train_predict(recs, kind, train_rows, test_rows, seed, device, args.epochs))
                print(f"  {kind} seed {seed}: {time.time() - t0:.0f} s")
        # reference: always predict each zone's most common class in the training recordings
        y_train = np.stack([recs[r]["y"][i] for r, i in train_rows])
        major = [np.bincount(y_train[:, z], minlength=N_C).argmax() for z in range(len(ZONES))]
        base.append(np.tile(major, (len(test_rows), 1)))
        t_all.append(np.stack([recs[r]["y"][i] for r, i in test_rows]))
        gyro.append(np.array([recs[r]["gyro"][i] for r, i in test_rows]))
        groups += [name] * len(test_rows)
        rows_all += [(recs[r]["name"], i) for r, i in test_rows]

    t = np.concatenate(t_all)
    groups, gyro = np.array(groups), np.concatenate(gyro)
    P = {k: [np.concatenate(s) for s in v] for k, v in probs.items()}
    P["late blend (E2)"] = [(a + b) / 2 for a, b in zip(P["thermal"], P["event"])]
    preds = {"majority class (reference)": [np.concatenate(base)] * args.seeds}
    preds.update({k: [p.argmax(-1) for p in v] for k, v in P.items()})

    thr = float(np.median(np.concatenate([r["gyro"] for r in recs])))
    masks = {"all": np.ones(len(t), bool), "still": gyro < thr, "moving": gyro >= thr}
    rng = np.random.default_rng(0)
    rows = []
    for model, pr in preds.items():
        for state, m in masks.items():
            res = with_ci(t, pr, groups, m, rng)
            rows.append([model, state, int(m.sum())] + [v for k in ("macro_f1", "exact", "hamming") for v in res[k]])

    costs = {k: cost(make_model(k), s) for k, s in (("thermal", (1, 24, 32)),
                                                    ("event", (2,) + recs[0]["events"].shape[2:]))}
    costs["late blend (E2)"] = tuple(a + b for a, b in zip(costs["thermal"], costs["event"]))

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "motion", "n", "macro_f1", "f1_lo", "f1_hi", "exact", "exact_lo", "exact_hi",
                    "hamming", "ham_lo", "ham_hi"])
        w.writerows([[r[0], r[1], r[2]] + [f"{v:.4f}" for v in r[3:]] for r in rows])
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "sample", "zone", "truth"] + [f"pred_{k}" for k in preds])
        for j, (rec, i) in enumerate(rows_all):
            for z, zone in enumerate(ZONES):
                w.writerow([rec, i, zone, NAMES[t[j, z]]] + [NAMES[preds[k][0][j, z]] for k in preds])

    fmt = lambda v, lo, hi: f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
    md = [f"# E1 / E2 results", "",
          f"Recordings: {', '.join(r['name'] for r in recs)} ({len(t)} test samples in total). "
          f"Task: {'free vs blocked' if args.task == 'binary' else 'clear/person/furniture/other'}. "
          f"Split: {split_note}. {args.seeds} seeds, {args.epochs} epochs, device {device}.",
          f"Still/moving split at the median gyro rate, {thr:.3f} rad/s. "
          f"Brackets: 95% bootstrap CI over test {'recordings' if len(recs) > 1 and args.split == 'recording' else 'blocks'}.", "",
          "| Model | Motion | n | Per-zone macro F1 | Exact match | Hamming acc. |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {fmt(*r[3:6])} | {fmt(*r[6:9])} | {fmt(*r[9:12])} |")
    md += ["", "| Model | Parameters | MFLOPs | CPU ms per update (1 thread) |", "|---|---|---|---|"]
    md += [f"| {k} | {p:,} | {fl:.1f} | {ms:.2f} |" for k, (p, fl, ms) in costs.items()]
    counts = np.bincount(t.ravel(), minlength=N_C)
    md += ["", "Test-label distribution (all zones): " +
           ", ".join(f"{c} {n} ({n / counts.sum():.0%})" for c, n in zip(NAMES, counts))]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
