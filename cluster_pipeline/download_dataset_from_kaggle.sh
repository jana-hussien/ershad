#!/bin/bash
# Downloads the dataset DIRECTLY onto deepnet from Kaggle -- no local download/scp detour.
# This works because the gpujobs login node has outbound internet access (confirmed by
# DeepNet_instructions.md Step 3, which pip-installs PyTorch from an external index URL).
#
# Prerequisite: a Kaggle API token (kaggle.json), downloaded once from
# https://www.kaggle.com/settings -> "Create New Token". Upload just that one small file first:
#   scp kaggle.json <andrewID>@gpujobs.qatar.cmu.edu:~/
# then run this script.
#
# Usage: edit ANDREWID below, then:  bash download_dataset_from_kaggle.sh
set -euo pipefail

ANDREWID="ANDREWID"     # <-- EDIT
ENV_NAME="daredevil"    # <-- EDIT to match setup_environment.sh

PIP_BIN="/data1/${ANDREWID}/.conda/envs/${ENV_NAME}/bin/pip"
PYTHON_BIN="/data1/${ANDREWID}/.conda/envs/${ENV_NAME}/bin/python"
KAGGLE_BIN="/data1/${ANDREWID}/.conda/envs/${ENV_NAME}/bin/kaggle"

$PIP_BIN install kaggle

mkdir -p ~/.kaggle
cp ~/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

DEST="/data1/${ANDREWID}/dataset"
mkdir -p "$DEST"

echo "Downloading datasets from Kaggle directly to ${DEST}..."
$KAGGLE_BIN datasets download faizmoazam/full-stitched-and-unstitched -p "$DEST" --unzip
$KAGGLE_BIN datasets download faizmoazam/thermal-homography-mats -p "$DEST" --unzip
$KAGGLE_BIN datasets download faizmoazam/rgb-homography -p "$DEST" --unzip

echo ""
echo "Done. Verify the structure:"
echo "  ls ${DEST}"
echo "Then update DATA_ROOT / HOMOGRAPHY_DIR / RGB_HOMOGRAPHY_DIR in every stage script's"
echo "config cell to point at the actual folder names under ${DEST}."
