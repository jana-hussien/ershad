#!/bin/bash
# Runs all of E1 on DeepNet. Start it on the gpujobs login node inside tmux (or with nohup), see README:
#   1. rebuilds any of the 3 old recordings whose E1 inputs are missing, then submits a smoke test
#      (every model, 24 samples, the 3 old recordings);
#   2. downloads the other bags one by one (the login node has internet), keeping at most MAX_BAGS on disk,
#      and submits a build job for each as soon as it has arrived (the job deletes the bag);
#   3. submits the training jobs (after the smoke test and every build succeeded) and the merge job.
# Safe to run again: finished recordings are skipped, so only what is missing is redone.
set -uo pipefail
export E1_HOME="$(cd "$(dirname "$0")" && pwd)"
source "$E1_HOME/config.sh"
mkdir -p "$RUN"/{bags,logs,results}
export NAME="" MODELS="" REC="" SMOKE=0   # passed to the jobs through the environment (--export=ALL)

sub() {  # sub LOGNAME SBATCH_FILE [sbatch options...] -> job id
  local name=$1 file=$2; shift 2
  sbatch --parsable --export=ALL --mcs-label="$MCS_LABEL" --job-name="e1_$name" \
    --output="$RUN/logs/$name-%j.out" --error="$RUN/logs/$name-%j.err" "$@" "$E1_HOME/$file"
}

BUILDS=(); OLD_BUILDS=(); LAST=""
build() {  # download one bag (at most MAX_BAGS on disk) and submit its build job; sets LAST to the job id
  local rec=$1 bag="$RUN/bags/$1.bag"; LAST=""
  if [ -f "$RUN/e1/$rec/meta.json" ]; then echo "$rec: already built"; return; fi
  while [ "$(ls "$RUN"/bags/*.bag 2>/dev/null | wc -l)" -ge "$MAX_BAGS" ]; do sleep 60; done
  echo "$(date) downloading $rec ... (progress: $RUN/logs/download_$rec.log)"
  if ! "$RCLONE" copyto "$REMOTE:sequences/$rec.bag" "$bag.part" --multi-thread-streams 8 \
        --stats 5m --stats-one-line -v > "$RUN/logs/download_$rec.log" 2>&1; then
    echo "download of $rec FAILED (see its log); skipping it, run this script again later"; rm -f "$bag.part"; return
  fi
  mv "$bag.part" "$bag"
  if [ "$(head -c 12 "$bag")" != "#ROSBAG V2.0" ]; then echo "$rec is not a rosbag; skipping"; rm -f "$bag"; return; fi
  REC=$rec
  LAST=$(sub "build_$rec" build.sbatch) || { echo "could not submit the build of $rec"; LAST=""; return; }
  BUILDS+=("$LAST"); echo "$(date) $rec: $(du -h "$bag" | cut -f1) downloaded, build job $LAST"
}

# the 3 old recordings first: the smoke test uses them (no-op for those that came from Kaggle)
for rec in $OLD; do build "$rec"; [ -n "$LAST" ] && OLD_BUILDS+=("$LAST"); done
SDEP=(); if [ ${#OLD_BUILDS[@]} -gt 0 ]; then SDEP=(--dependency="afterok:$(IFS=:; echo "${OLD_BUILDS[*]}")"); fi
NAME=smoke MODELS=all SMOKE=1
SMOKE_ID=$(sub smoke train.sbatch --time=02:00:00 "${SDEP[@]}") || { echo "could not submit the smoke test"; exit 1; }
SMOKE=0 NAME="" MODELS=""
echo "$(date) smoke test: job $SMOKE_ID  (log: $RUN/logs/smoke-$SMOKE_ID.out)"

for rec in $NEW; do build "$rec"; done

DEP="afterok:$SMOKE_ID"; for id in "${BUILDS[@]}"; do DEP="$DEP:$id"; done
TRAINS=()
for g in "${TRAIN_GROUPS[@]}"; do
  NAME=${g%%:*} MODELS=${g#*:}
  id=$(sub "train_$NAME" train.sbatch --dependency="$DEP") || { echo "could not submit training $NAME"; continue; }
  TRAINS+=("$id"); echo "training $NAME: job $id"
done
MDEP="afterany"; for id in "${TRAINS[@]}"; do MDEP="$MDEP:$id"; done
MERGE=$(sub merge merge.sbatch --dependency="$MDEP")
echo "merge: job $MERGE"
echo
echo "$(date) ALL SUBMITTED. Watch with:  squeue -u $ANDREWID   and   ls $RUN/logs"
echo "Final results: $RUN/results/e1_all/summary.md"
