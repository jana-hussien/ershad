'''Exports the YOLO auto-labels (label_cache.pkl) to a plain CSV for the public dataset release.

One row per synced frame, keyed by (part, anchor_index). Labels are per direction (L/M/R =
cameras 2/5/4). valid_* = 0 means the frame had no usable RGB match (or the camera has no RGB
overlap), so there is no label for that direction -- label_* is then left empty.

If --data-root is given (the folder holding part1/, part2/, ...), each row also gets the thermal
capture filename for every camera at that anchor_index. anchor_index is the position in the
camera's captures sorted by sequence number, same as list_captures() in cluster_pipeline/scripts/00_setup_and_labels.py.

Usage (from the repo root):
    python tools/export_release_labels.py
    python tools/export_release_labels.py --data-root "/data1/jhussien/dataset/all_runs - Copy"
'''
import argparse
import csv
import pickle
import re
from collections import Counter
from pathlib import Path

# kept in sync with 00_setup_and_labels.py
CLASS_NAMES = ["no_object", "person", "other_object"]
ACTIVE_CAMERAS = [2, 5, 4]                  # left, middle, right
ALL_CAMS_LEFT_TO_RIGHT = [3, 2, 5, 4, 6]
DIRECTIONS = ["L", "M", "R"]
CAPTURE_RE = re.compile(r"^data_capture_(\d+)_(\d+)_(\d+)\.json$")


def _part_sort_key(part):
    return int(re.sub(r"\D", "", part) or 0)


def list_captures(folder, cam_id):
    captures = []
    for f in Path(folder).iterdir():
        m = CAPTURE_RE.match(f.name)
        if m and int(m.group(1)) == cam_id:
            captures.append((int(m.group(2)), f.name))
    captures.sort(key=lambda item: item[0])
    return [name for _seq, name in captures]


def find_thermal_dir(data_root, part):
    matches = sorted((Path(data_root) / part).glob("thermal-g*"))
    if not matches:
        raise FileNotFoundError(f"No 'thermal-g*' folder under {Path(data_root) / part}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="kaggle_retrain/results/main_run/label_cache.pkl")
    ap.add_argument("--data-root", default=None, help="folder with part*/thermal-g*/ (optional)")
    ap.add_argument("--out", default="release/labels.csv")
    args = ap.parse_args()

    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    keys = sorted(cache, key=lambda k: (_part_sort_key(k[0]), k[1]))

    # part -> cam_id -> [thermal filename per anchor_index]
    thermal_files = {}
    if args.data_root:
        for part in sorted({p for p, _ in keys}, key=_part_sort_key):
            thermal_dir = find_thermal_dir(args.data_root, part)
            thermal_files[part] = {c: list_captures(thermal_dir, c) for c in ALL_CAMS_LEFT_TO_RIGHT}

    header = ["part", "anchor_index"]
    if thermal_files:
        header += ["thermal_dir"] + [f"thermal_cam{c}" for c in ALL_CAMS_LEFT_TO_RIGHT]
    header += [f"label_{d}" for d in DIRECTIONS] + [f"valid_{d}" for d in DIRECTIONS]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for part, anchor_index in keys:
            class_vec, mask_vec = cache[(part, anchor_index)]
            row = [part, anchor_index]
            if thermal_files:
                row.append(find_thermal_dir(args.data_root, part).name)
                for c in ALL_CAMS_LEFT_TO_RIGHT:
                    files = thermal_files[part][c]
                    if anchor_index >= len(files):
                        raise IndexError(f"{part} cam{c}: anchor_index {anchor_index} but only {len(files)} captures")
                    row.append(files[anchor_index])
            labels = []
            for cls, valid in zip(class_vec, mask_vec):
                if valid:
                    labels.append(CLASS_NAMES[int(cls)])
                    counts[CLASS_NAMES[int(cls)]] += 1
                else:
                    labels.append("")
                    counts["invalid"] += 1
            row += labels + [int(v) for v in mask_vec]
            w.writerow(row)

    print(f"{len(keys)} frames -> {out}")
    print("per-direction counts:", dict(counts))


if __name__ == "__main__":
    main()
