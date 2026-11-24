#!/bin/bash
# Collects every group's finished runs into results/e1_all and writes ONE summary.md with all models.
# Runs after the training jobs, even if one failed: a model with missing runs is trained here instead.
#SBATCH --job-name=e1_merge
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:nvidia_h200_1g.18gb:1
#SBATCH --nodelist=deepnet2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
set -euo pipefail
source "$E1_HOME/config.sh"
OUT="$RUN/results/e1_all"
mkdir -p "$OUT/runs"
for d in "$RUN"/results/*/runs; do
  case "$d" in */smoke/*|*/e1_all/*) continue ;; esac
  cp -n "$d"/*.npz "$OUT/runs/" 2>/dev/null || true
done
echo "$(ls "$OUT/runs" | wc -l) finished runs collected"
cd "$RUN/code"
$PY e1_representations.py --cache "$RUN/cache" --e1-cache "$RUN/e1" --out "$OUT" \
  --split place --every "$EVERY" --seeds 1 --models all
echo "$(date) MERGE OK: $OUT/summary.md"
