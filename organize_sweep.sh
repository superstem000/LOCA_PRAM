#!/usr/bin/env bash
# Copies the LAST-epoch checkpoint from every runs/tanh_*/ folder into
# runs/sweep/tanh_<value>.pth, ready for synthetic_eval.py.
#
# "Last" = the highest LOCA_PRAM_checkpoint_epoch_XXXX.pth in that folder.
# Copies (not moves) so the source dirs stay intact. Overwrite is enabled
# so re-running picks up new final-epoch checkpoints.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUNS_DIR="runs"
SWEEP_DIR="${RUNS_DIR}/sweep"
mkdir -p "${SWEEP_DIR}"

shopt -s nullglob
found_any=0
for run_dir in "${RUNS_DIR}"/tanh_*/; do
    # skip the sweep dir itself (defensive — its name doesn't match tanh_* but just in case)
    [[ "${run_dir}" == *"/sweep/"* ]] && continue

    base="$(basename "${run_dir}")"                     # e.g. tanh_1.0_20260822_232937
    # Extract the tanh value: everything between "tanh_" and the next "_"
    ts_val="$(echo "${base}" | sed -E 's/^tanh_([0-9]+\.[0-9]+).*/\1/')"
    if [[ "${ts_val}" == "${base}" ]]; then
        echo "  [skip] can't parse tanh_scale from ${base}"
        continue
    fi

    # Find the highest-epoch checkpoint
    ckpt="$(ls "${run_dir}"LOCA_PRAM_checkpoint_epoch_*.pth 2>/dev/null \
              | sort -V | tail -n1 || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "  [skip] no LOCA_PRAM_checkpoint_epoch_*.pth in ${run_dir}"
        continue
    fi

    dest="${SWEEP_DIR}/tanh_${ts_val}.pth"
    echo "  ${base}"
    echo "    highest ckpt : $(basename "${ckpt}")"
    echo "    -> ${dest}"
    cp -f "${ckpt}" "${dest}"
    found_any=1
done

if [[ ${found_any} -eq 0 ]]; then
    echo "no runs/tanh_*/ folders with checkpoints found."
    exit 1
fi

echo
echo "Done. Files in ${SWEEP_DIR}/:"
ls -la "${SWEEP_DIR}"
echo
echo "Next: python synthetic_eval.py"
