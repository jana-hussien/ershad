"""Score the auto-labels and Feature Fusion against both annotators.

Usage (from the repo root):  python human_validation/analyze_labels.py
Reads human_validation/labels_{jana,Faiz}.csv, human_validation/hv_key/ and
kaggle_retrain/results/main_run/ff_val_predictions.csv.
CIs are 95% bootstrap intervals that resample whole frames (2000 draws, seed 0).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HV = ROOT / "human_validation"
ANNOTATORS = ["jana", "Faiz"]
CLASSES = ["clear", "person", "other"]
B = 2000

key = pd.read_csv(HV / "hv_key/auto_labels_key.csv")
val = pd.read_csv(HV / "hv_key/val_auto_labels_all.csv")
ff = pd.read_csv(ROOT / "kaggle_retrain/results/main_run/ff_val_predictions.csv")
ff["direction"] = ff["direction"].map({"left": "L", "middle": "M", "right": "R"})

df = key.merge(ff[["part", "anchor_index", "direction", "pred"]], on=["part", "anchor_index", "direction"], how="left")
assert df["pred"].notna().all(), "sample frames missing from ff_val_predictions.csv"
for a in ANNOTATORS:
    lab = pd.read_csv(HV / f"labels_{a}.csv")
    assert len(lab) == len(key), f"{a}: {len(lab)} rows, expected {len(key)}"
    df = df.merge(lab[["frame_id", "direction", "label", "blocks_path"]]
                  .rename(columns={"label": a, "blocks_path": f"{a}_blk"}), on=["frame_id", "direction"])
J, F = ANNOTATORS

# val-set class mix, pooled over directions, for prevalence-weighted agreement
prev = pd.concat([val[f"auto_{d}"] for d in "LMR"]).map(dict(enumerate(CLASSES))).value_counts(normalize=True)


def kappa(x, y):
    po = (x == y).mean()
    pe = sum((x == c).mean() * (y == c).mean() for c in set(x) | set(y))
    return (po - pe) / (1 - pe)


def weighted_agreement(d, h):
    return sum(prev[c] * (d.loc[d.auto_label == c, h] == c).mean() for c in CLASSES)


def blocked_scores(y, yhat):
    tp, fp, fn = (y & yhat).sum(), (~y & yhat).sum(), (y & ~yhat).sum()
    p, r = tp / (tp + fp), tp / (tp + fn)
    return {"F1": 2 * p * r / (p + r), "Acc": (y == yhat).mean(), "Prec": p, "Rec": r, "FP": fp}


rng = np.random.default_rng(0)


def ci(fn, d=df):
    by_frame = {f: g for f, g in d.groupby("frame_id")}
    fr = np.array(list(by_frame))
    draws = [fn(pd.concat([by_frame[f] for f in rng.choice(fr, len(fr))])) for _ in range(B)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return f"{fn(d):.3f} [{lo:.3f}, {hi:.3f}]"


print(f"{len(df)} directions, {df.frame_id.nunique()} frames")
print("val-set class mix:", prev.round(3).to_dict())

print("\n== Inter-annotator ==")
print("class agreement", ci(lambda d: (d[J] == d[F]).mean()))
print("class kappa    ", ci(lambda d: kappa(d[J].values, d[F].values)))
print("blocks agreement", ci(lambda d: (d[f"{J}_blk"] == d[f"{F}_blk"]).mean()))
print("blocks kappa    ", ci(lambda d: kappa(d[f"{J}_blk"].values, d[f"{F}_blk"].values)))
print(pd.crosstab(df[J], df[F], rownames=[J], colnames=[F]))

cls = df[df[J] == df[F]].assign(consensus=lambda d: d[J])
for h, d in [(J, df), (F, df), ("consensus", cls)]:
    print(f"\n== Auto-labels vs {h} (n={len(d)}) ==")
    print(pd.crosstab(d.auto_label, d[h]))
    print("agreement         ", ci(lambda x: (x.auto_label == x[h]).mean(), d))
    print("weighted agreement", ci(lambda x: weighted_agreement(x, h), d))
    for c in CLASSES:
        print(f"  {c:6s} precision {(d.loc[d.auto_label == c, h] == c).mean():.3f} (n={(d.auto_label == c).sum()})"
              f"  recall {(d.loc[d[h] == c, 'auto_label'] == c).mean():.3f} (n={(d[h] == c).sum()})")

blk = df[df[f"{J}_blk"] == df[f"{F}_blk"]].assign(consensus_blk=lambda d: d[f"{J}_blk"])
for h, d in [(J, df), (F, df), ("consensus", blk)]:
    y = d[f"{h}_blk"] == "yes"
    print(f"\n== Blocked-direction detection vs {h} (n={len(d)}, {y.sum()} blocked) ==")
    for name, yhat in [("RGB detector", d.auto_label != "clear"), ("Feature Fusion", d.pred != "clear")]:
        print(f"  {name:15s}", {k: round(float(v), 3) for k, v in blocked_scores(y, yhat).items()})
    other = d.auto_label == "other"
    print(f"  detector 'other' directions judged open: {(~y[other]).mean():.3f} (n={other.sum()})")
