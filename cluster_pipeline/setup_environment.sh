#!/bin/bash
# Run ONCE on the gpujobs login node (never on deepnet2/mcore directly) to set up everything
# this pipeline needs: the conda env, all required packages, and the shared directory structure.
# Matches the package-install method from DeepNet_instructions.md Step 2-3 exactly (full binary
# paths, never bare pip/conda activate) -- with ultralytics/YOLO added, since that's not in the
# instructions' default package list but every stage notebook needs it for label generation.
#
# Usage: edit ANDREWID and ENV_NAME below, then:  bash setup_environment.sh
set -euo pipefail

ANDREWID="ANDREWID"     # <-- EDIT
ENV_NAME="daredevil"    # <-- EDIT if you want a different env name

# ---- 1. create the conda env (only if it doesn't already exist) ----
if [ ! -d "/data1/${ANDREWID}/.conda/envs/${ENV_NAME}" ]; then
    echo "Creating conda env '${ENV_NAME}'..."
    echo "When prompted: env name = ${ENV_NAME}, Python version = 3.10"
    ./setup_conda_env.sh
else
    echo "Conda env '${ENV_NAME}' already exists, skipping creation."
fi

# ---- 2. install packages, ALWAYS via full binary paths (never bare pip/conda activate) ----
PYTHON_BIN="/data1/${ANDREWID}/.conda/envs/${ENV_NAME}/bin/python"
PIP_BIN="/data1/${ANDREWID}/.conda/envs/${ENV_NAME}/bin/pip"

$PIP_BIN install --upgrade pip
$PIP_BIN install numpy pandas scipy scikit-learn matplotlib seaborn pillow pytz
$PIP_BIN install torch torchvision --index-url https://download.pytorch.org/whl/cu121
$PIP_BIN install opencv-python ultralytics

echo "Verifying imports..."
$PYTHON_BIN -c "import torch, torchvision, pandas, numpy, cv2, sklearn, matplotlib, seaborn, PIL, pytz, ultralytics; print('ALL IMPORTS OK')"

# ---- 3. shared directory structure every stage script writes to/reads from ----
mkdir -p "/data1/${ANDREWID}/daredevil_run/logs"
mkdir -p "/data1/${ANDREWID}/daredevil_run/scripts"
mkdir -p "/data1/${ANDREWID}/daredevil_run/saved_models"
mkdir -p "/data1/${ANDREWID}/daredevil_run/figures"
mkdir -p "/data1/${ANDREWID}/daredevil_run/results"
mkdir -p "/data1/${ANDREWID}/daredevil_run/cache"
mkdir -p "/data1/${ANDREWID}/daredevil_run/executed_notebooks"
mkdir -p "/data1/${ANDREWID}/dataset"

echo ""
echo "Done. Directory structure created under /data1/${ANDREWID}/daredevil_run/"
echo "Next: upload your dataset (see download_dataset_from_kaggle.sh for a no-local-download"
echo "option), upload scripts/*.ipynb + scripts/*.py into daredevil_run/scripts/, edit"
echo "ANDREWID/ENV_NAME/COURSE_LABEL in every .sbatch file, then run slurm/submit_pipeline.sh."
