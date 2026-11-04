#!/usr/bin/env bash
# Assignment 2 (paper critique) reality check - Singh et al., Sci. Rep. 14:2487 (2024).
# Re-runs the paper's model (DeiT-B/16, ImageNet-pretrained, fully fine-tuned) inside the
# A1 pipeline, and re-runs the paper's *protocol* (random 80/10/10 split by image) on our
# models, so the two effects (model vs split) can be separated.  From the repo root:
#   bash scripts/run_a2.sh                       # seeds 42 43, ~1.5-2 h on a Colab T4
#   SEEDS="42" bash scripts/run_a2.sh            # one seed first (~50 min)
#   STAGE=core bash scripts/run_a2.sh            # only the essential runs
# Finished runs are skipped, so an interrupted session can simply be re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

SEEDS="${SEEDS:-42 43}"
EPOCHS="${EPOCHS:-12}"
RAW="${RAW:-data/raw}"
IMG_SIZE="${IMG_SIZE:-224}"
STAGE="${STAGE:-all}"      # core | all
EXTRA="${EXTRA:-}"

run() {  # run <name-of-results-json-without-ext> <train args...>
  local NAME="$1"; shift
  if [ -f "results/${NAME}.json" ]; then echo "   skip ${NAME} (done)"; return; fi
  python -m cxr.train --epochs "$EPOCHS" --img_size "$IMG_SIZE" "$@" $EXTRA
}

echo "== 1/3 prepare data (same audit, split and cache as A1)"
[ -f "data/cache/images_${IMG_SIZE}.npy" ] || python -m cxr.prepare_data --source "$RAW" --out data --seed 42 --img_size "$IMG_SIZE"

for SEED in $SEEDS; do
  echo "== 2/3 seed $SEED: the paper's model under the A1 protocol (patient split, paper LR 1e-5)"
  run "vit_deit_b__abl-none__frac-1__seed-${SEED}"              --model vit_deit_b --seed "$SEED"
  echo "== 2/3 seed $SEED: the paper's protocol (random split by image) on the paper's model and on our winner"
  run "vit_deit_b__abl-none__frac-1__seed-${SEED}__imagesplit"  --model vit_deit_b --seed "$SEED" --split image
  run "resnet50__abl-none__frac-1__seed-${SEED}__imagesplit"    --model resnet50   --seed "$SEED" --split image
  if [ "$STAGE" = "all" ]; then
    echo "== 2/3 seed $SEED: ViT at the A1 pretrained LR (1e-4), to rule out an LR artefact"
    run "vit_deit_b__abl-none__frac-1__seed-${SEED}__lr1e-4"    --model vit_deit_b --seed "$SEED" --lr 1e-4 --tag lr1e-4
    echo "== 2/3 seed $SEED: image split on the remaining A1 models"
    run "densenet121__abl-none__frac-1__seed-${SEED}__imagesplit" --model densenet121 --seed "$SEED" --split image
    run "smallcnn__abl-none__frac-1__seed-${SEED}__imagesplit"    --model smallcnn    --seed "$SEED" --split image
    run "logreg__abl-none__frac-1__seed-${SEED}__imagesplit"      --model logreg      --seed "$SEED" --split image
  fi
done

echo "== 3/3 reports"
python -m cxr.report --results results --data data
python -m cxr.report_a2 --results results
