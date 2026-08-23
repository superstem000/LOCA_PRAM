#!/usr/bin/env bash
# Sweeps --tanh-scale across {1.0, 2.0, 4.0, 6.0} sequentially, with everything
# else at train.py defaults (2000 epochs, seed=2, batch_size=16, lr=3e-5,
# defect BG folder). Same seed and schedule across all four runs so the
# only varying axis is tanh_scale. 6.0 is included so we get a same-budget
# comparison against the current inference default rather than relying on
# the 5000-epoch checkpoint we already have.
#
# Usage: ./ablate_tanh.sh
# The script exits on the first failed run; check runs/tanh_<s>_<ts>/ for
# TB logs (`tensorboard --logdir runs/`), config.json, checkpoints, and
# the final loss curve PNG.

set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for TS_VAL in 1.0 2.0 4.0 6.0; do
    OUT="runs/tanh_${TS_VAL}_${TS}/"
    echo "=========================================================="
    echo "[$(date +'%H:%M:%S')] START tanh_scale=${TS_VAL}  ->  ${OUT}"
    echo "=========================================================="
    python train.py \
        --tanh-scale "${TS_VAL}" \
        --output-dir "${OUT}" \
        --seed 2 \
        --max-epoch 2000
    echo "[$(date +'%H:%M:%S')] DONE  tanh_scale=${TS_VAL}"
done

echo
echo "All three runs finished. Compare with:"
echo "  tensorboard --logdir runs/"
