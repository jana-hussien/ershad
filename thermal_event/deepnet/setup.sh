#!/bin/bash
# Run ONCE on the gpujobs login node, from this folder (thermal_event/deepnet in the repo):  bash setup.sh
# Installs the extra packages into your conda env, installs rclone, copies the code and the 3 old
# recordings' labels into $RUN, gets their E1 inputs from Kaggle if it can (else run_all.sh rebuilds them
# from the bags), and checks everything.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
SRC="$HERE/code"; [ -d "$SRC" ] || SRC="$HERE/.."          # bundle layout or repo layout
CACHE="$HERE/cache"; [ -d "$CACHE" ] || CACHE="$HERE/../cache"
[ -x "$PY" ] || { echo "no conda env found in /data1/$ANDREWID/.conda/envs: create one first (README, 'Before you start')"; exit 1; }
echo "Andrew ID: $ANDREWID   conda env: $ENV_NAME   mcs-label: $MCS_LABEL"

mkdir -p "$RUN"/{code,cache,e1,bags,results,logs}
cp "$SRC"/*.py "$RUN/code/"
[ -f "$SRC/yolov8m.pt" ] && cp "$SRC/yolov8m.pt" "$RUN/code/"
cp "$HERE"/*.sh "$HERE"/*.sbatch "$RUN/"
for rec in $OLD; do
  [ -f "$RUN/cache/$rec/meta.json" ] || cp -r "$CACHE/$rec" "$RUN/cache/"
done

echo "== packages =="
$PY -c "import torch" 2>/dev/null || $PIP install -q numpy torch torchvision --index-url https://download.pytorch.org/whl/cu121
$PIP install -q rosbags ultralytics kaggle
TV=$($PY -c "import torch; print(torch.__version__.split('+')[0])")
CU=$($PY -c "import torch; print((torch.version.cuda or '').replace('.', ''))")
echo "torch $TV, CUDA $CU"
$PIP install -q torch_spline_conv torch_cluster -f "https://data.pyg.org/whl/torch-${TV}+cu${CU}.html" \
  --only-binary :all: || echo "WARNING: no AEGNN packages (torch_spline_conv/torch_cluster); aegnn will be skipped"
$PIP install -q mamba-ssm --no-build-isolation || echo "WARNING: mamba-ssm did not install; eventmg_style will be skipped"
$PY - <<'PYEOF'
import importlib
for m in ("torch", "numpy", "rosbags", "ultralytics", "torch_spline_conv", "torch_cluster", "mamba_ssm"):
    try:
        importlib.import_module(m); print(f"  {m}: OK")
    except Exception as e:
        print(f"  {m}: NOT AVAILABLE ({type(e).__name__})")
PYEOF
if [ ! -f "$RUN/code/yolov8m.pt" ]; then
  (cd "$RUN/code" && $PY -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" >/dev/null)
fi
echo "  yolov8m.pt: $([ -f "$RUN/code/yolov8m.pt" ] && echo OK || echo MISSING)"

echo "== E1 inputs of the 3 old recordings (Kaggle, optional) =="
KAGGLE=/data1/$ANDREWID/.conda/envs/$ENV_NAME/bin/kaggle
for rec in $OLD; do
  if [ ! -f "$RUN/e1/$rec/meta.json" ] && [ -f "$HOME/.kaggle/kaggle.json" ]; then
    chmod 600 "$HOME/.kaggle/kaggle.json"
    "$KAGGLE" datasets download "$KAGGLE_USER/dalilsense-e1-${rec//_/-}" -p "$RUN/e1/$rec" --unzip \
      >/dev/null 2>&1 || rm -rf "$RUN/e1/$rec"
  fi
  if [ -f "$RUN/e1/$rec/meta.json" ]; then echo "  $rec: from Kaggle"; else echo "  $rec: will be rebuilt from its bag"; fi
done

echo "== rclone (OneDrive) =="
if [ ! -x "$RCLONE" ]; then
  mkdir -p "$HOME/bin" && cd /tmp
  curl -sSLO https://downloads.rclone.org/rclone-current-linux-amd64.zip
  unzip -oq rclone-current-linux-amd64.zip && cp rclone-*-linux-amd64/rclone "$RCLONE" && chmod +x "$RCLONE"
  cd "$HERE"
fi
"$RCLONE" listremotes | grep -qx "$REMOTE:" || { echo "no rclone remote '$REMOTE': run  $RCLONE config  (see README)"; exit 1; }
n=0
for rec in $OLD $NEW; do
  [ -f "$RUN/e1/$rec/meta.json" ] && continue
  "$RCLONE" lsf "$REMOTE:sequences/$rec.bag" >/dev/null 2>&1 || { echo "not found in your OneDrive: sequences/$rec.bag"; exit 1; }
  n=$((n+1))
done
echo "  all $n bags to build found in your OneDrive"

echo
echo "SETUP DONE. Next: bash $RUN/run_all.sh   (inside tmux, see README)"
