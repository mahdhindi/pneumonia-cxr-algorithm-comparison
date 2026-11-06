#!/usr/bin/env bash
# External test on RSNA Pneumonia (adults, other hospitals). From the repo root, after the Kermany data cache exists.
#   bash scripts/run_external.sh                 # needs rsna-pneumonia-detection-challenge.zip in the repo root (see notebook cell 5)
# Trains any missing checkpoint (same CNN, small CNN, ResNet-50; seeds 42 43; results go to results/retrain/,
# so the A1 result files are never overwritten), builds the RSNA cache once, then scores every model on it.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
SEEDS="${SEEDS:-42 43}"
ZIP="${ZIP:-rsna-pneumonia-detection-challenge.zip}"
pip install -q pydicom
for NAME in writtencnn smallcnn resnet50; do
  for SEED in $SEEDS; do
    CK="runs/${NAME}__abl-none__frac-1__seed-${SEED}/best.pt"
    [ -f "$CK" ] && { echo "   have $CK"; continue; }
    python -m cxr.train --model "$NAME" --seed "$SEED" --epochs 12 --img_size 224 --results results/retrain
  done
done
[ -f data_ext/rsna/cache/images_224.npy ] || python -m cxr.external prepare --zip "$ZIP" --n_per_class 500
python -m cxr.external evaluate --seeds $SEEDS
