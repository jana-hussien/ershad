#!/bin/bash
# Submits all 10 stages with the correct --dependency=afterok chain so Stages 1, 2, and 3a-3e
# all become eligible to run as soon as Stage 0 finishes.
#
# IMPORTANT -- read before running: your account has a hard cap of 4 CONCURRENT jobs (per the
# GPU Resources Guide). This script still submits all 7 of Stage 1/2/3a-3e at once -- Slurm
# itself will run the first 4 immediately and leave the rest PENDING (reason
# QOSMaxJobsPerUserLimit) until a slot frees up. You do NOT need to do anything about this --
# it's normal, automatic, and just means the wall-clock time for that parallel block is closer
# to ~5-6 hours than the ~4 hours it would be with unlimited concurrency. See README.md for the
# updated time estimate under this constraint.
#
# Usage: cd into this directory and run:  bash submit_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "Submitting Stage 0 (setup + labels)..."
JOB0=$(sbatch --parsable 00_setup_and_labels.sbatch)
echo "  -> job $JOB0"

echo "Submitting Stage 1, Stage 2, and Stages 3a-3e (depend only on Stage 0 -- Slurm will run"
echo "  at most 4 of these at once due to the per-user concurrent-job limit, queueing the rest)..."
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 01_baseline1.sbatch)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB0 02_baseline2.sbatch)
JOB3A=$(sbatch --parsable --dependency=afterok:$JOB0 03a_ff_depth.sbatch)
JOB3B=$(sbatch --parsable --dependency=afterok:$JOB0 03b_ff_width.sbatch)
JOB3C=$(sbatch --parsable --dependency=afterok:$JOB0 03c_ff_activation.sbatch)
JOB3D=$(sbatch --parsable --dependency=afterok:$JOB0 03d_ff_head.sbatch)
JOB3E=$(sbatch --parsable --dependency=afterok:$JOB0 03e_ff_dropout.sbatch)
echo "  -> baseline1=$JOB1  baseline2=$JOB2  depth=$JOB3A  width=$JOB3B  act=$JOB3C  head=$JOB3D  dropout=$JOB3E"

echo "Submitting Stage 4 (best combo -- depends on all 5 ablation axes finishing)..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3A:$JOB3B:$JOB3C:$JOB3D:$JOB3E 04_ff_best_combo.sbatch)
echo "  -> job $JOB4"

echo "Submitting Stage 5 (shuffle + final comparison -- depends on Stage 1, 2, and 4)..."
JOB5=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2:$JOB4 05_shuffle_and_final.sbatch)
echo "  -> job $JOB5"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  sacct -u \$USER --format=JobID,JobName,State,Elapsed,Partition"
echo "If ANY job fails, every job depending on it (directly or transitively) is auto-cancelled"
echo "by Slurm rather than running on incomplete inputs."
