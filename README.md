# DareDevil deepnet pipeline (matches `gpujobs.qatar.cmu.edu` / Slurm exactly)

Same 10-stage split as before (Baseline 1, Baseline 2, Feature Fusion + full 15-variant
ablation, Feature Fusion + camera-identity shuffle correction), rebuilt to match the **actual**
constraints of CMU-Q's deepnet cluster:

- **`jupyter nbconvert --execute` doesn't work on this system.** Every stage ships as both an
  `.ipynb` (for reference / to re-run interactively) and a `.py` script (converted via the exact
  JSON method from `DeepNet_instructions.md` Step 6) — the `.py` is what Slurm actually runs.
- **Max 4 concurrent jobs per user.** `submit_pipeline.sh` still submits all 7 of Stage 1/2/3a-3e
  at once, but Slurm itself queues anything past 4 until a slot frees up — automatic, no script
  changes needed, just a longer wall-clock for that block (see time estimate below).
- **`--mcs-label` and a time limit are required** on every job — every `.sbatch` file has both.
- **48-hour max job time** — every stage here is well under that individually.

## What you get back, per stage

Because `nbconvert` can't execute notebooks here, a plain `python script.py` run on its own
would only give you a `.out` text log — no notebook, no embedded figures. This pipeline works
around that: every stage script captures its full run log AND every figure it produces, and at
exit rebuilds a real notebook from scratch. Concretely, after a stage finishes you get:

1. **`<stage>_executed.ipynb`** — written to both `scripts/` (next to the original) and
   `daredevil_run/executed_notebooks/`. Contains every original cell's source code, PLUS a
   "Run output" section at the end with every figure embedded as a real inline image (no
   external file needed to view them) and the complete captured stdout log as a text cell. Open
   it in Jupyter and you see the whole run, figures included, exactly like a normal executed
   notebook.
2. **Standalone figure PNGs** in `daredevil_run/figures/` — every chart, individually, for
   pulling into a paper/poster directly without opening a notebook.
3. **Saved model checkpoints** in `daredevil_run/saved_models/` — one `.pt` per trained model
   (every Feature Fusion ablation variant included, not just the winner).
4. **Pickled result dicts** in `daredevil_run/results/` (predictions, F1 scores, params,
   latency) — what lets later stages rebuild comparison tables without re-running anything.
5. The normal Slurm `.out`/`.err` log in `daredevil_run/logs/`.

## Setup (run once)

```bash
ssh ANDREWID@gpujobs.qatar.cmu.edu
# upload setup_environment.sh and download_dataset_from_kaggle.sh first (scp from PowerShell),
# or paste their contents directly via a heredoc on the SSH session
bash setup_environment.sh                  # creates conda env, installs packages, makes dirs
bash download_dataset_from_kaggle.sh       # pulls the dataset straight from Kaggle, no local hop
```

Then upload everything in `scripts/` and `slurm/` to `/data1/ANDREWID/daredevil_run/scripts/`
and `/data1/ANDREWID/daredevil_run/slurm/` respectively (scp from PowerShell, or `git clone` /
`rsync` if deepnet has access to wherever you're hosting these files).

**Before submitting anything, edit these placeholders:**
- `ANDREWID` and `COURSE_LABEL` in every file under `slurm/*.sbatch`
- `ENV_NAME` in every `.sbatch` (must match what you used in `setup_environment.sh`)
- `SHARED_DIR`, `DATA_ROOT`, `HOMOGRAPHY_DIR`, `RGB_HOMOGRAPHY_DIR` in every `scripts/*.py` AND
  the matching `scripts/*.ipynb` (both must agree, since the executed-notebook assembler reads
  the `.ipynb` for its cell source)
- Set `SMOKE_TEST = False` in every script once Stage 0 has been smoke-tested successfully

## Running it

```bash
cd /data1/ANDREWID/daredevil_run/slurm
bash submit_pipeline.sh
```

## Monitoring

```bash
sacct -u ANDREWID --format=JobID,JobName,State,Elapsed,Partition
tail -f /data1/ANDREWID/daredevil_run/logs/<stage>-<jobid>.out
```

## Time estimate, accounting for the 4-concurrent-job cap

| Block | Time |
|---|---|
| Stage 0 (setup + labels + FF baseline) | ~2 hr |
| Stage 1 + 2 + 3a-3e (7 jobs, only 4 run at once) | ~5-6 hr (was ~4 hr with unlimited concurrency) |
| Stage 4 (best combo) | ~1.3 hr |
| Stage 5 (shuffle + final) | ~1.2 hr |
| **Total** | **~9.5-10.5 hours** |

This comfortably fits inside the 48-hour-per-job cap (each individual stage is far under that),
and total wall-clock across the whole pipeline stays well under a day even with the
4-concurrent-job ceiling.

## Downloading results back to your machine

From **Windows PowerShell**:

```powershell
mkdir C:\Users\<username>\daredevil_results

scp -r ANDREWID@gpujobs.qatar.cmu.edu:/data1/ANDREWID/daredevil_run/executed_notebooks C:\Users\<username>\daredevil_results\
scp -r ANDREWID@gpujobs.qatar.cmu.edu:/data1/ANDREWID/daredevil_run/figures C:\Users\<username>\daredevil_results\
scp -r ANDREWID@gpujobs.qatar.cmu.edu:/data1/ANDREWID/daredevil_run/saved_models C:\Users\<username>\daredevil_results\
scp -r ANDREWID@gpujobs.qatar.cmu.edu:/data1/ANDREWID/daredevil_run/logs C:\Users\<username>\daredevil_results\
```

## Re-running / resuming a failed stage

Every cache (labels, split, per-variant results, model checkpoints) is keyed by name and
persisted under `daredevil_run/`. `run_ff_variant` checks for an existing result + checkpoint
before retraining anything. If a stage fails partway through, just resubmit that one `.sbatch`
file — it picks up from whatever it already finished.
