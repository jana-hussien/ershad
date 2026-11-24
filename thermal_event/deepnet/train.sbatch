#!/bin/bash
# Trains one group of E1 models on all recordings (1 run, hold out one place type at a time).
# Submitted by run_all.sh with NAME=<group> MODELS=<comma list>, and SMOKE=1 for the quick check.
# Finished (model, place) runs are saved in results/<group>/runs and skipped if resubmitted.
#SBATCH --job-name=e1_train
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --nodelist=deepnet2
#SBATCH --time=47:00:00
#SBATCH --mem=64G
set -euo pipefail
source "$E1_HOME/config.sh"
cd "$RUN/code"
EXTRA=""
if [ "${SMOKE:-0}" = 1 ]; then          # 24 samples of the 3 old recordings, 1 epoch, every model
  EXTRA="--smoke --recordings ${OLD// /,}"
fi
echo "$(date) training $NAME on $(hostname): $MODELS"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
$PY e1_representations.py --cache "$RUN/cache" --e1-cache "$RUN/e1" --out "$RUN/results/$NAME" \
  --split place --every "$EVERY" --seeds 1 --models "$MODELS" $EXTRA
echo "$(date) TRAIN OK $NAME"
