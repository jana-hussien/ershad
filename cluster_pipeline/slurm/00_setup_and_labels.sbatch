#!/bin/bash
#SBATCH --job-name=ddv_00_setup_and_labels
#SBATCH --output=/data1/jhussien/daredevil_run/logs/00_setup_and_labels-%j.out
#SBATCH --error=/data1/jhussien/daredevil_run/logs/00_setup_and_labels-%j.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --nodelist=deepnet2
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --mcs-label=sura

# ---- EDIT jhussien and sura above, and daredevil below, before submitting ----
PYTHON_BIN=/data1/jhussien/.conda/envs/daredevil/bin/python

$PYTHON_BIN /data1/jhussien/daredevil_run/scripts/00_setup_and_labels.py
