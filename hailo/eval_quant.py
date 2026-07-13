"""Run eval_set.npy through both emulator contexts and dump raw outputs.

RUN IN: ~/venv  (DFC env, Python 3.10, no torch)
FROM:   ~/loca

Reads:  hailo/eval_set.npy, hailo/loca_pram_quantized.har
Writes: hailo/preds_fp.npy    (float-optimized emulation)
        hailo/preds_quant.npy (quantized emulation)

Each file is a np.save'd dict of {output_layer_name: array}. Detection logic
lives in the analysis venv (compare_quant.py); this script only produces raw
model outputs so it can stay torch-free.
"""
import os
import sys
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
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


def _run_context(runner, tiles, context_enum):
    """Run tiles through the specified InferenceContext, return dict of
    {output_layer_name: (N, H, W, C) float32}."""
    from hailo_sdk_client import InferenceContext
    ctx_name = context_enum.name if hasattr(context_enum, 'name') else str(context_enum)
    print(f"[eval] running InferenceContext.{ctx_name} on {len(tiles)} tiles...")

    with runner.infer_context(context_enum) as ctx:
        raw = runner.infer(ctx, tiles)

    # Normalize the return: DFC returns either a dict {name: (N,H,W,C)} or a
    # list of arrays in output-layer order. Convert to dict either way.
    if isinstance(raw, dict):
        out = {k: np.asarray(v) for k, v in raw.items()}
    else:
        # List — use runner metadata to map to names.
        try:
            names = runner.get_output_names()
        except AttributeError:
            hn = runner.get_hn_dict()
            names = [n for n, spec in hn.get('layers', {}).items()
                     if spec.get('type') == 'output_layer']
        assert len(raw) == len(names), f"{len(raw)} outputs, {len(names)} names"
        out = {name: np.asarray(arr) for name, arr in zip(names, raw)}

    for name, arr in out.items():
        print(f"[eval]   {ctx_name}: {name}  shape={arr.shape}  dtype={arr.dtype}  "
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

    preds_fp = _run_context(runner, tiles, InferenceContext.SDK_FP_OPTIMIZED)
    np.save(PREDS_FP, preds_fp)
    print(f"[eval] wrote {PREDS_FP}")

    preds_q = _run_context(runner, tiles, InferenceContext.SDK_QUANTIZED)
    np.save(PREDS_QUANT, preds_q)
    print(f"[eval] wrote {PREDS_QUANT}")


if __name__ == '__main__':
    main()
