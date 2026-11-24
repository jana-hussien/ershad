# E1 on DeepNet: all models, 14 indoor recordings, 1 run each

## Before you start: what you need
| # | What | How to check / get it |
|---|---|---|
| 1 | CMU-Q Andrew ID that can run DeepNet jobs | `srun -p gpu2 --mcs-label=sura --time=2 --gres=gpu:nvidia_h200_1g.18gb:1 hostname` prints `deepnet2` |
| 2 | CMU VPN (only when off campus) | `ssh <andrewid>@gpujobs.qatar.cmu.edu` works |
| 3 | Access to the GitHub repo `jana-hussien/ershad` | ask Jana to add you as a collaborator |
| 4 | A conda env on DeepNet (the course one is fine) | `ls /data1/$USER/.conda/envs` shows a folder. If empty, run the course's `setup_conda_env.sh` (Python 3.10) |
| 5 | The M2DGR bags in **your** OneDrive | open the [M2DGR OneDrive folder](https://1drv.ms/f/c/3e825cd7abd6dff7/IgCW90ge__sjSrGsFdArjCYMATtrrylG05J8lD1npWwmEAQ) → **Add shortcut to My files**. Your OneDrive must then show a folder called `sequences` |
| 6 | ~150 GB free on `/data1` | `df -h /data1` |

You do **not** need to edit any file: your Andrew ID, conda env and mcs-label (`sura`) are found
automatically. (If your env or label is different, set them at the top of `config.sh`.)

**The only things you type yourself** (everything else is copy-paste):
- `<andrewid>` in the commands below: replace it with your Andrew ID, without the `< >`
  (e.g. `ssh abc123@gpujobs.qatar.cmu.edu`).
- The answers to the OneDrive questions in step 2 (listed in the table there).
- Your CMU login in the browser, once, in step 2.

## Steps (copy-paste)

**1. Log in and get the code.** Log in with this exact command (the `-L 53682:...` part is needed for
the OneDrive login in step 2):
```bash
ssh -L 53682:localhost:53682 <andrewid>@gpujobs.qatar.cmu.edu
git clone -b thermal-event-fusion https://github.com/jana-hussien/ershad.git
cd ershad/thermal_event/deepnet
bash setup.sh
```
The first time, it stops with `no rclone remote 'od'`. That is expected: do step 2.

**2. Connect your OneDrive (once).** In the same DeepNet window, run `~/bin/rclone config`. It asks
questions one at a time; type the answer in the right column and press Enter:

| It asks (the line ends with) | You type |
|---|---|
| `n/s/q>` | `n` |
| `name>` | `od` |
| `Storage>` | `onedrive` |
| `client_id>` | (nothing, just Enter) |
| `client_secret>` | (nothing, just Enter) |
| `region>` | `1` (Microsoft Cloud Global) |
| `Edit advanced config?` `y/n>` | `n` |
| `Use web browser to automatically authenticate rclone with remote?` `y/n>` | `y` |
| it prints a link `http://127.0.0.1:53682/auth?state=...` and waits | open that link in the browser **on your laptop**, sign in with your CMU account, wait for "Success" |
| `config_type>` | `1` (OneDrive Personal or Business) |
| `config_driveid>` | the number next to `OneDrive` (usually `1`) |
| `Is that okay?` `y/n>` | `y` |
| `Keep this "od" remote?` `y/e/d>` | `y` |
| `e/n/d/r/c/s/q>` | `q` |

If the link says "can't connect", you logged in without `-L 53682:localhost:53682`: type `q`, log out,
log in with the command from step 1, and run `~/bin/rclone config` again.

Check it: `~/bin/rclone lsf od:sequences | head` lists files ending in `.bag`.

**3. Finish setup**
```bash
bash setup.sh          # run again; it ends with "SETUP DONE"
```

**4. Start the run** (keeps going after you log out)
```bash
tmux new -s e1
bash /data1/$USER/e1_run/run_all.sh
```
Wait until it prints `downloading ...`, then press `Ctrl-b` then `d` to detach. You can log out.
To look at it again: `tmux attach -t e1`.

**5. Check on it**
```bash
squeue -u $USER                                  # your running / waiting jobs
tail /data1/$USER/e1_run/logs/smoke-*.out        # after a few hours: should say "SMOKE TEST PASSED"
```
The whole run takes about 2-3 days.

**6. Results:** send Jana the folder `/data1/$USER/e1_run/results/e1_all/` (`summary.md` and the csv
files). From your laptop:
```bash
scp -r <andrewid>@gpujobs.qatar.cmu.edu:/data1/<andrewid>/e1_run/results/e1_all ./
```

## If something fails
- Send Jana the `.out` and `.err` files of the failed job from `/data1/$USER/e1_run/logs/`.
- After fixing it: `scancel -u $USER`, then run `bash /data1/$USER/e1_run/run_all.sh` again in tmux.
  Recordings already built and models already trained are skipped.
- A warning in `setup.sh` that AEGNN or EventMG packages did not install is OK: those two models are
  skipped, everything else runs.

## What runs (for reference)
| Step | Where | What | Time |
|---|---|---|---|
| `setup.sh` | login node, once | packages, rclone, code, labels of the 3 old recordings | ~15 min |
| `run_all.sh` | login node, in tmux | downloads the bags one by one (≤3 on disk), submits a build job per bag, then training and merge | ~1-8 h of downloads |
| `build_<rec>` (14) | GPU | labels (YOLO), thermal, event bins, E1 inputs; deletes the bag | ~1 h each |
| `smoke` | GPU | every model, 24 samples, 1 epoch, after the 3 old recordings are built | ~20 min |
| `train_<group>` (6) | GPU | after smoke + all builds succeed: images ×2 (~10 models each), spiking ×2, AEGNN, EventMG-style | ~6-15 h each, 4 at a time |
| `merge` | GPU | one `summary.md` with all models | ~1 h |

Data: M2DGR recordings indoors with normal light: room ×3, hall ×5, lift ×4, door ×2 (only the
indoor part of the two door recordings: door_01 62-360 s, door_02 from 52 s). 4 samples per second (every frame). The test holds out one place type at a
time (room / hall / door / lift). One training run per model.
