"""Run eval_set.npy through both emulator contexts and dump raw outputs.

RUN IN: ~/venv  (DFC env, Python 3.10, no torch)
FROM:   ~/loca

Reads:  hailo/eval_set.npy, hailo/loca_pram_quantized.har
Writes: hailo/preds_fp.npy    (float-optimized emulation)
        hailo/preds_quant.npy (quantized emulation)

Each file is a np.save'd dict keyed by SEMANTIC name:
    {'p':     (N, 128, 128, 1) float32,
     'mu_xy': (N, 128, 128, 2) float32}

Semantic keys (not Hailo's `loca_pram/output_layerN`) so downstream code
never has to know Hailo's numbering, and channel-count assertions catch a
future DFC version that renumbers or reshapes the outputs.

Detection logic lives in the analysis venv (compare_quant.py); this script
only produces raw model outputs so it can stay torch-free.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the same semantic-resolution logic build.py uses at parse time so
# there is exactly one source of truth for output naming.
from build import resolve_outputs


HERE = _HERE
HAR_QUANTIZED = os.path.join(HERE, 'loca_pram_quantized.har')
EVAL_PATH     = os.path.join(HERE, 'eval_set.npy')
PREDS_FP      = os.path.join(HERE, 'preds_fp.npy')
PREDS_QUANT   = os.path.join(HERE, 'preds_quant.npy')


def _load_eval():
    obj = np.load(EVAL_PATH, allow_pickle=True).item()
    tiles = obj['tiles']
    assert tiles.ndim == 4 and tiles.shape[1:] == (128, 128, 1), \
        f"eval tiles shape wrong: {tiles.shape}"
    assert tiles.dtype == np.float32
    return tiles, obj


def _run_context(runner, tiles, context_enum, sem_to_full):
    """Run tiles through the specified InferenceContext, return dict keyed by
    SEMANTIC name ('p', 'mu_xy'):
        {'p':     (N, 128, 128, 1) float32,
         'mu_xy': (N, 128, 128, 2) float32}

    sem_to_full is {semantic_name: fully_qualified_hailo_name} from
    resolve_outputs(runner). Asserts channel counts at the point of use —
    a future DFC version that renumbers the outputs will fail here loudly
    rather than silently swap p and mu_xy.
    """
    from hailo_sdk_client import InferenceContext
    ctx_name = context_enum.name if hasattr(context_enum, 'name') else str(context_enum)
    print(f"[eval] running InferenceContext.{ctx_name} on {len(tiles)} tiles...")

    with runner.infer_context(context_enum) as ctx:
        raw = runner.infer(ctx, tiles)

    # Normalize the return: DFC returns either a dict {full_name: (N,H,W,C)}
    # or a list of arrays in output-layer order. Convert to dict of full-name
    # -> array, then re-key by semantic name below.
    if isinstance(raw, dict):
        full = {k: np.asarray(v) for k, v in raw.items()}
    else:
        # List. Do NOT trust list order — key by channel count instead so a
        # DFC-version reorder can't silently swap p and mu_xy.
        arrs = [np.asarray(a) for a in raw]
        assert len(arrs) == len(sem_to_full), \
            f"{len(arrs)} output arrays, {len(sem_to_full)} expected outputs"
        # Deterministically match array -> full name by last-dim channel count.
        # This works because our two outputs have distinct channel counts (1 vs 2);
        # if that ever stops being true, this assertion catches it.
        channels_full = {full_name: (1 if sem == 'p' else 2)
                         for sem, full_name in sem_to_full.items()}
        full = {}
        for full_name, expected_c in channels_full.items():
            matches = [a for a in arrs if a.shape[-1] == expected_c
                       and id(a) not in {id(v) for v in full.values()}]
            assert len(matches) == 1, \
                f"[eval] cannot uniquely match array for {full_name} "\
                f"(expected channels={expected_c}); array shapes: "\
                f"{[a.shape for a in arrs]}"
            full[full_name] = matches[0]

    # Re-key by semantic name and assert channel count at the point of use.
    out = {}
    for sem, full_name in sem_to_full.items():
        assert full_name in full, \
            f"[eval] runner returned no array for {full_name!r}; got keys {list(full.keys())}"
        arr = full[full_name]
        expected_c = 2 if sem == 'mu_xy' else 1
        assert arr.ndim == 4 and arr.shape[1:3] == (128, 128), \
            f"[eval] {sem} ({full_name}) shape wrong: {arr.shape}"
        assert arr.shape[-1] == expected_c, \
            f"[eval] {sem} ({full_name}) channels={arr.shape[-1]}, expected {expected_c}"
        out[sem] = arr

    for sem, arr in out.items():
        print(f"[eval]   {ctx_name}: {sem:6s} (<- {sem_to_full[sem]})  "
              f"shape={arr.shape}  dtype={arr.dtype}  "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")
    return out


def main():
    from hailo_sdk_client import ClientRunner, InferenceContext

    assert os.path.isfile(HAR_QUANTIZED), \
        f"missing {HAR_QUANTIZED} — run hailo/build.py optimize first"
    assert os.path.isfile(EVAL_PATH), \
        f"missing {EVAL_PATH} — run hailo/make_calib.py first"

    tiles, _ = _load_eval()
    runner = ClientRunner(har=HAR_QUANTIZED, hw_arch='hailo8')

    # Resolve Hailo full-qualified output names -> semantic ('p', 'mu_xy').
    # Same channel-count check as build.py parse — will fail loudly if the
    # HAR's output naming/order ever changes.
    sem_to_full = resolve_outputs(runner)
    print(f"[eval] semantic mapping: {sem_to_full}")

    preds_fp = _run_context(runner, tiles, InferenceContext.SDK_FP_OPTIMIZED, sem_to_full)
    np.save(PREDS_FP, preds_fp)
    print(f"[eval] wrote {PREDS_FP}")

    preds_q = _run_context(runner, tiles, InferenceContext.SDK_QUANTIZED, sem_to_full)
    np.save(PREDS_QUANT, preds_q)
    print(f"[eval] wrote {PREDS_QUANT}")


if __name__ == '__main__':
    main()
