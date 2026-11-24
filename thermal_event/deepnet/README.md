# E1 on DeepNet: all models, 13 indoor recordings, 1 run each

Runs every E1 model (27, including AEGNN and EventMG-style) on all M2DGR recordings that are indoors
with normal light: room ×3, hall ×5, lift ×4, door ×1 (door_01 is left out: it goes outdoor → indoor →
outdoor). 2 samples per second. The test holds out one **place type** at a time (room / hall / door /
lift), so every score is on a kind of place the model never saw in training. One training run per model.

## Steps

**1. Get the bags into your OneDrive.** Open the M2DGR "Browse all rosbags" OneDrive link (M2DGR GitHub
README), and use **Add shortcut to My files** on the folder. It must show up as `sequences/` in My files.

**2. On DeepNet** (`ssh <andrewid>@gpujobs.qatar.cmu.edu`, CMU VPN off campus):
```bash
git clone -b thermal-event-fusion https://github.com/jana-hussien/ershad.git
cd ershad/thermal_event/deepnet
nano config.sh             # set MCS_LABEL and ENV_NAME (your conda env in /data1/<andrewid>/.conda/envs)
bash setup.sh              # stops at the rclone step the first time if there is no OneDrive login yet
```
OneDrive login, once: `~/bin/rclone config` → `n` (new) → name `od` → storage `onedrive` → Enter
through the rest. When it asks "Use web browser to automatically authenticate?", answer `n` and
follow what it prints (run `rclone authorize "onedrive"` on your laptop and paste back the result).
Choose "OneDrive Personal or Business". Check that `~/bin/rclone lsf od:sequences | head` lists `.bag` files.
Then run `bash setup.sh` again until it prints `SETUP DONE`.

**3. Start it** (keeps going after you log out):
```bash
tmux new -s e1
bash /data1/$USER/e1_run/run_all.sh
```
Detach with `Ctrl-b d`. Come back with `tmux attach -t e1`.

**4. Watch it:**
```bash
squeue -u $USER                                  # running / waiting jobs
tail /data1/$USER/e1_run/logs/smoke-*.out        # should say "SMOKE TEST PASSED"
ls /data1/$USER/e1_run/logs                      # one .out/.err per job
```

**5. Results:** `/data1/$USER/e1_run/results/e1_all/summary.md` (plus the csv files in that folder).

## What runs
| Step | Where | What | Time |
|---|---|---|---|
| `setup.sh` | login node, once | packages, rclone, code, labels of the 3 old recordings | ~15 min |
| `run_all.sh` | login node, in tmux | downloads the bags one by one (≤3 on disk), submits a build job per bag, then training and merge | ~1-8 h of downloads |
| `build_<rec>` (≤13) | GPU | labels (YOLO), thermal, event bins, E1 inputs; deletes the bag | ~1 h each |
| `smoke` | GPU | every model, 24 samples, 1 epoch, after the 3 old recordings are built | ~20 min |
| `train_<group>` (5) | GPU | after smoke + all builds succeed | ~3-6 h each, 4 at a time |
| `merge` | GPU | one `summary.md` with all models | ~1 h |

## If something fails
- Fix the cause and run `bash /data1/$USER/e1_run/run_all.sh` again: built recordings and finished
  training runs are skipped.
- A failed smoke test or build cancels the training jobs (they show `DependencyNeverSatisfied` in
  `squeue`); cancel them with `scancel -u $USER` before running again.
- If AEGNN's or EventMG's packages don't install, `setup.sh` warns and those two models are skipped.
- Disk: ~140 GB at peak (3 bags) on `/data1`. Lower `MAX_BAGS` in `config.sh` if that is too much.
