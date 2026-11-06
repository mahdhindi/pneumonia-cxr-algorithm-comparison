#!/usr/bin/env bash
# Project experiment: LLM-written weights vs trained weights for WrittenCNN (5,569 params).
# From the repo root, after the data cache exists (same as A1/A2):
#   bash scripts/run_written.sh            # baselines only (trained + untrained); scores nothing written by Astra
#   FINAL=1 bash scripts/run_written.sh    # once, after conditions A and B: scores Astra's weights on the test set
# Trains the same architecture by gradient descent (seeds 42, 43; ~3 min each on a T4),
# scores the random-init floor, evaluates any Astra weights already in astra/, and writes
# results/written_table.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
SEEDS="${SEEDS:-42 43}"
RAW="${RAW:-data/raw}"
[ -f data/cache/images_224.npy ] || python -m cxr.prepare_data --source "$RAW" --out data --seed 42 --img_size 224

for SEED in $SEEDS; do
  NAME="writtencnn__abl-none__frac-1__seed-${SEED}"
  [ -f "results/${NAME}.json" ] && { echo "   skip ${NAME} (done)"; continue; }
  python -m cxr.train --model writtencnn --seed "$SEED" --epochs 12 --img_size 224
done
python -m cxr.written untrained --seeds $SEEDS

# Astra's weights are scored ONLY when FINAL=1 (after both conditions are finished), never by accident.
if [ "${FINAL:-0}" = "1" ] && [ -f astra/weights_zeroshot.py ] && [ ! -f results/written__astra-zeroshot.json ]; then
  python -m cxr.written evaluate --weights astra/weights_zeroshot.py --label astra-zeroshot
fi
if [ "${FINAL:-0}" = "1" ] && [ -f astra/weights_practice_final.py ] && [ ! -f results/written__astra-practice.json ]; then
  python -m cxr.written evaluate --weights astra/weights_practice_final.py --label astra-practice --log astra/practice_log.jsonl
fi
[ "${FINAL:-0}" = "1" ] && python -m cxr.written report || echo "baselines done; Astra weights NOT scored (run with FINAL=1 after both conditions)"
