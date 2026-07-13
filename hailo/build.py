"""Hailo-8 build pipeline: parse -> optimize -> compile.

RUN IN: ~/venv  (Hailo DFC 3.31.0, Python 3.10, TF 2.12, numpy 1.23.3).
        Do NOT pip install anything into it.
FROM:   ~/loca

Subcommands so a failure late in the pipeline doesn't force redoing the
expensive optimize step:
    python hailo/build.py parse
    python hailo/build.py optimize
    python hailo/build.py analyze
    python hailo/build.py compile
    python hailo/build.py all         (parse + optimize + analyze + compile)

Artifacts produced (all under hailo/):
    loca_pram.har                  after parse
    loca_pram_quantized.har        after optimize
    loca_pram_noise.txt            after analyze (per-layer SNR)
    loca_pram.hef                  after compile

Notes:
- `hailo parser onnx` (CLI) SEGFAULTS on this model. Python API works.
- Hailo renames output layers, NOT in export order. Actual mapping (from the
  profiler) — build.py PRINTS and ASSERTS this:
      output_layer1  -> mu_xy  (128, 128, 2)
      output_layer2  -> p      (128, 128, 1)
- Calibration data must be NHWC (N, 128, 128, 1) float32.

NO TORCH imports here — DFC env doesn't have it and installing it would break
the TF 2.12 / numpy 1.23 pin.
"""
import argparse
import os
import sys
import time
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ONNX_PATH       = os.path.join(HERE, 'loca_pram_export.onnx')
HAR_PARSED      = os.path.join(HERE, 'loca_pram.har')
HAR_QUANTIZED   = os.path.join(HERE, 'loca_pram_quantized.har')
HEF_PATH        = os.path.join(HERE, 'loca_pram.hef')
ALLS_PATH       = os.path.join(HERE, 'model_script.alls')
CALIB_PATH      = os.path.join(HERE, 'calib_set.npy')
NOISE_TXT       = os.path.join(HERE, 'loca_pram_noise.txt')

MODEL_NAME = 'loca_pram'
HW_ARCH    = 'hailo8'

# ---- expected output name → semantic mapping (verified at parse) ----
EXPECTED_OUTPUT_MAP = {
    'output_layer1': ('mu_xy', (128, 128, 2)),
    'output_layer2': ('p',     (128, 128, 1)),
}


def _log_tf_devices():
    """Level-4 optimize runs Adaround for 320 epochs; if it silently lands on
    CPU it takes hours. Log GPU visibility at startup."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"[env] TF version: {tf.__version__}")
        print(f"[env] Visible GPUs: {gpus if gpus else '(none — optimize will be VERY slow)'}")
    except Exception as e:
        print(f"[env] Could not query TF GPUs: {e}")


def _load_runner_from_har(har_path):
    from hailo_sdk_client import ClientRunner
    assert os.path.isfile(har_path), f"missing HAR: {har_path}"
    runner = ClientRunner(har=har_path, hw_arch=HW_ARCH)
    print(f"[load] restored runner from {har_path}")
    return runner


def cmd_parse():
    from hailo_sdk_client import ClientRunner
    assert os.path.isfile(ONNX_PATH), f"missing ONNX: {ONNX_PATH}"
    print(f"[parse] hw_arch={HW_ARCH}   onnx={ONNX_PATH}")

    runner = ClientRunner(hw_arch=HW_ARCH)
    hn, npz = runner.translate_onnx_model(ONNX_PATH, MODEL_NAME)
    runner.save_har(HAR_PARSED)
    print(f"[parse] wrote {HAR_PARSED}")

    # Introspect I/O
    hn_dict = runner.get_hn_dict() if hasattr(runner, 'get_hn_dict') else {}
    net = hn_dict.get('layers', {}) if hn_dict else {}
    inputs  = [name for name, spec in net.items() if spec.get('type') == 'input_layer']
    outputs = [name for name, spec in net.items() if spec.get('type') == 'output_layer']
    print(f"[parse] input layers:  {inputs}")
    print(f"[parse] output layers: {outputs}")

    # Assert output mapping. The order is: HN gives the shape metadata for each
    # output_layer; we assert against EXPECTED_OUTPUT_MAP. If Hailo ever
    # renames these differently, this fails LOUDLY.
    for out_name in outputs:
        spec = net[out_name]
        shape = tuple(spec.get('output_shapes', [[None]])[0][1:])   # drop batch
        expected = EXPECTED_OUTPUT_MAP.get(out_name)
        if expected is None:
            raise AssertionError(
                f"[parse] unexpected output layer {out_name!r}. "
                f"Update EXPECTED_OUTPUT_MAP if the parser's naming changed."
            )
        sem, exp_shape = expected
        assert shape == exp_shape, \
            f"[parse] {out_name} ({sem}) shape mismatch: got {shape}, expected {exp_shape}"
        print(f"[parse]   {out_name} -> {sem}  shape={shape}  ✓")

    return runner


def cmd_optimize():
    runner = _load_runner_from_har(HAR_PARSED)
    runner.load_model_script(ALLS_PATH)
    print(f"[optimize] loaded {ALLS_PATH}")

    assert os.path.isfile(CALIB_PATH), \
        f"missing calibration set: {CALIB_PATH} — run hailo/make_calib.py first"
    calib = np.load(CALIB_PATH)
    print(f"[optimize] calib shape={calib.shape}  dtype={calib.dtype}")
    assert calib.ndim == 4, f"calib must be NHWC 4-D, got shape {calib.shape}"
    assert calib.shape[1:] == (128, 128, 1), \
        f"calib shape mismatch: expected (N, 128, 128, 1), got {calib.shape}"
    assert calib.dtype == np.float32, f"calib dtype must be float32, got {calib.dtype}"
    assert calib.shape[0] >= 1024, \
        f"optimization_level=4 requires >=1024 calib samples, got {calib.shape[0]}"

    t0 = time.time()
    runner.optimize(calib)
    dt = time.time() - t0
    print(f"[optimize] wall time: {dt:.1f} s  ({dt/60:.1f} min)")
    if dt > 30 * 60:
        print(f"[optimize] WARNING: >30 min. If TF was on CPU, expect several hours.")

    runner.save_har(HAR_QUANTIZED)
    print(f"[optimize] wrote {HAR_QUANTIZED}")
    return runner


def cmd_analyze():
    runner = _load_runner_from_har(HAR_QUANTIZED)
    calib = np.load(CALIB_PATH)

    # analyze_noise returns per-layer SNR (dB). API surface varies slightly
    # across DFC versions; try common signatures.
    print(f"[analyze] running analyze_noise on {len(calib)} calib samples...")
    try:
        result = runner.analyze_noise(calib)
    except TypeError:
        # older API — no arg
        result = runner.analyze_noise()

    # Emit as a text file for reproducibility. Support multiple return shapes:
    # dict-of-{layer: snr_db}, list of tuples, or a summary object.
    lines = []
    if isinstance(result, dict):
        items = sorted(result.items(), key=lambda kv: kv[1])
        for name, snr in items:
            lines.append(f"{snr:8.2f} dB   {name}")
    elif hasattr(result, '__iter__'):
        for entry in result:
            lines.append(str(entry))
    else:
        lines.append(repr(result))

    with open(NOISE_TXT, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[analyze] wrote {NOISE_TXT}")

    print("[analyze] worst 10 layers (SNR ascending, docs: >10 dB is fine):")
    for ln in lines[:10]:
        print(f"    {ln}")

    return runner


def cmd_compile():
    runner = _load_runner_from_har(HAR_QUANTIZED)
    print(f"[compile] compiling to {HEF_PATH} ...")
    t0 = time.time()
    hef = runner.compile()
    dt = time.time() - t0
    with open(HEF_PATH, 'wb') as f:
        f.write(hef)
    print(f"[compile] wrote {HEF_PATH}   ({dt:.1f} s)")
    return runner


def cmd_all():
    cmd_parse()
    cmd_optimize()
    cmd_analyze()
    cmd_compile()


def main():
    _log_tf_devices()

    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['parse', 'optimize', 'analyze', 'compile', 'all'])
    args = ap.parse_args()

    {
        'parse':    cmd_parse,
        'optimize': cmd_optimize,
        'analyze':  cmd_analyze,
        'compile':  cmd_compile,
        'all':      cmd_all,
    }[args.stage]()


if __name__ == '__main__':
    main()
