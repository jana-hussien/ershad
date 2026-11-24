#!/bin/bash
# One recording: labels (YOLO on the RGB camera), thermal and 40 ms event bins (build_cache.py), then the
# exact E1 inputs at 2 samples per second (build_e1_cache.py). Deletes the bag when it ends.
# Submitted by run_all.sh with REC=<recording>; output and --mcs-label are set there.
#SBATCH --job-name=e1_build
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:nvidia_h200_1g.18gb:1
#SBATCH --nodelist=deepnet2
#SBATCH --time=06:00:00
#SBATCH --mem=32G
set -euo pipefail
source "$E1_HOME/config.sh"
BAG="$RUN/bags/$REC.bag"
trap 'rm -f "$BAG"' EXIT          # free the disk even if it fails (run_all.sh downloads it again)
cd "$RUN/code"
echo "$(date) building $REC on $(hostname)"
if [ ! -f "$RUN/cache/$REC/meta.json" ]; then
  $PY build_cache.py "$BAG" --out "$RUN/cache/$REC" --yolo "$RUN/code/yolov8m.pt"
fi
$PY build_e1_cache.py "$BAG" --cache "$RUN/cache/$REC" --out "$RUN/e1/$REC" --every "$EVERY"
echo "$(date) BUILD OK $REC"
