# Human validation of the auto-labels

Two annotators each label the same ~100 sampled val frames blind, meaning they never see
the YOLO boxes or the auto-labels. We then compare those labels against the auto-labels
(per-class precision, overall agreement weighted back to the val-set class mix) and against
each other (Cohen's κ).

## Building the sample

Run `build_sample.ipynb` on Kaggle with a T4 GPU, with `janahussien90/rgb3thermals-full` and
`janahussien90/homography-mats` attached. It rebuilds the pipeline's val split (seed 42, blocked
80/20), auto-labels the val frames with YOLOv8x, samples 100 frames, and writes two files:

- `hv_sample.zip` goes to the annotators. It holds the images, the footprints and **no labels**.
- `hv_key.zip` is for the analysis only. Don't open it until both annotators have exported.

Set `SMOKE_TEST = True` in the first cell for a quick end-to-end check before the real run.

## Labeling

1. Open `label_tool.html` in a browser (double-click it).
2. Type your name, click **Open sample folder…**, and pick the sample folder.
   **Try with demo frames** lets you practise the hotkeys first.
3. For each footprint (L / M / R), answer two questions: what's inside it, and does it block the path?
4. Export with ⌘S / Ctrl+S and send back `labels_<name>.csv`.

Labels autosave in the browser, so you can close the tab and pick up where you left off.
Don't compare answers until both of you have exported.

**Seeing the auto-labels** (`hv_key/auto_labels_key.csv`) can be done in two ways. Either way, a frame
locks once you've seen its auto-labels, and the export records when (`auto_label_seen_at`).

- **Show full key…**: reveals everything at once. Use this after you've finished and exported. Your labels stay fully blind.
- **Key per frame…**: after finishing a frame, press `K` to see that frame's auto-labels. This is handy for
  learning, but seeing YOLO's answers as you go influences your later frames. For the paper, at least
  one annotator should label fully blind.

| Key | Action |
|---|---|
| `1`/`C` `2`/`P` `3`/`O` `4`/`U` | clear / person / other / unsure (clear also sets "not blocking") |
| `Y` / `N` | blocks the path? |
| `Q` `W` `E`, `↑` `↓` | pick direction |
| `←` `→`, `Space` | prev / next frame, next unfinished frame |
| `⌫`, `H`, `/` | reset direction, toggle footprints, note |

An all-clear frame takes three keystrokes: `1 1 1`.

## Analysis

Put each annotator's export in this folder as `labels_<name>.csv`, then run
`python human_validation/analyze_labels.py` from the repo root. It also needs `hv_key/` (from the
sample build) and `kaggle_retrain/results/main_run/ff_val_predictions.csv`. The labels and the key
are private and kept out of git. It reports inter-annotator κ, how well the
auto-labels agree with each annotator and with their consensus, and blocked-direction scores for the RGB detector
and Feature Fusion. CIs come from bootstrapping over frames.

The output of the run reported in the paper is saved in `results.txt`.

## Sample folder format

```
sample/
  manifest.json
  images/000.jpg ...
```

```json
{
  "dataset_id": "hv_v1",
  "frames": [
    {
      "frame_id": "part3_01234",
      "image": "images/000.jpg",
      "footprints": { "L": [[x, y], ...], "M": [[x, y], ...], "R": [[x, y], ...] },
      "directions": ["L", "M", "R"]
    }
  ]
}
```

- `footprints` are the camera footprint polygons in the RGB image's pixel coordinates.
- `directions` is optional. It lists only the directions with a valid label mask.
- The manifest must **not** contain the auto-labels. Those go in a separate key file that
  only the analysis script reads.
