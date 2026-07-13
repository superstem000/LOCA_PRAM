# Hailo-8 quantization pipeline

Everything needed to take `loca_pram_export.onnx` and produce a compiled `.hef`
plus an honest quant-vs-FP accuracy report.

## Environments (hard constraint)

Two venvs, both in WSL, both reading this same checkout:

| venv               | purpose            | python | key libs                                             |
| ------------------ | ------------------ | ------ | ---------------------------------------------------- |
| `~/venv`           | DFC (Hailo build)  | 3.10   | `hailo_sdk_client` (DFC 3.31.0), TF 2.12, numpy 1.23.3 — **no torch** |
| `~/analysis_venv`  | Analysis / data    | 3.10   | torch 2.13 (CPU), numpy 2.2.6, cv2 5.0, scipy, skimage, sklearn, pandas, matplotlib |

**Do not `pip install` anything into `~/venv`** — it's a pinned working install.
Installing torch would drag numpy 2.x in and break TF 2.12.

Scripts communicate only through `.npy` files in this directory.
**No DFC-env script imports torch.**

## Run order

Run all commands from the repo root (`~/loca`). The scripts resolve paths
relative to `hailo/`, but the notebook config uses `demo_0001/background/`
relative to the repo root.

| step | command                                              | venv              | outputs                                          |
| ---- | ---------------------------------------------------- | ----------------- | ------------------------------------------------ |
| 1    | `python hailo/make_calib.py`                         | `~/analysis_venv` | `hailo/calib_set.npy`, `hailo/eval_set.npy`      |
| 2    | `source ~/venv/bin/activate; python hailo/build.py parse`     | `~/venv`  | `hailo/loca_pram.har`                            |
| 3    | `python hailo/build.py optimize`                     | `~/venv`          | `hailo/loca_pram_quantized.har`                  |
| 4    | `python hailo/build.py analyze`                      | `~/venv`          | `hailo/loca_pram_noise.txt`  (per-layer SNR)     |
| 5    | `python hailo/build.py compile`                      | `~/venv`          | `hailo/loca_pram.hef`                            |
| 6    | `python hailo/eval_quant.py`                         | `~/venv`          | `hailo/preds_fp.npy`, `hailo/preds_quant.npy`    |
| 7    | `source ~/analysis_venv/bin/activate; python hailo/compare_quant.py` | `~/analysis_venv` | prints FP vs quant metrics  |

Or run 2-5 in one shot: `python hailo/build.py all`.

Every DFC-env step reads from `hailo/*.npy` produced by the analysis venv (or
by a previous DFC step) — no cross-import, no cross-venv Python calls.

## Files

- `loca_core.py`         — extracted from the training notebook (verbatim
  copy of synthesis, preprocessing, and detection functions). The notebook
  can't be imported directly because it has `import napari` at top level.
  **Keep in sync manually** when the notebook cells change.
- `make_calib.py`        — synthesize calibration and eval tiles from the
  held-out FOV pool. NHWC, float32, `(N, 128, 128, 1)`.
- `model_script.alls`    — Hailo model script with the exact optimize flags.
- `build.py`             — parse / optimize / analyze / compile subcommands.
  Uses the Python API (`ClientRunner.translate_onnx_model`) — the CLI
  `hailo parser onnx` segfaults on this model.
- `eval_quant.py`        — runs the eval set through both emulator contexts
  and dumps raw NHWC outputs. Torch-free.
- `compare_quant.py`     — loads preds + eval_set, runs the notebook's
  detection path, prints FP-vs-quant precision/recall/RMSE.

## Known facts (don't re-derive these)

- ONNX: input `input` (1, 1, 128, 128); outputs `p` (1ch Sigmoid),
  `mu_xy` (2ch Tanh). Parsed model: 4.2M params, 50 layers, 9.4 GOPs,
  hw_arch=hailo8.
- Hailo renames outputs, not in export order:
  ```
  output_layer1 -> mu_xy (128, 128, 2)
  output_layer2 -> p     (128, 128, 1)
  ```
  `build.py parse` prints the actual names and asserts this mapping. If Hailo
  ever renames differently, that step fails loudly.
- The `hailo parser onnx` **CLI segfaults** on this model. `build.py` uses
  the Python API instead — it works.
- Hailo tensors are **NHWC**. `(N, 128, 128, 1)` for calibration. NCHW will
  run and quantize to garbage without erroring.
- `postprocess.json` holds `tanh_scale` — apply host-side:
  `mu_pixels = mu_xy_raw * tanh_scale`.

## Useful Hailo CLI commands (already working)

- `hailo profiler hailo/loca_pram.har`
  → produces `loca_pram_profiler.html` with the per-layer breakdown, output
  layer names, and I/O shapes. Use this to confirm the output-name mapping
  above if it ever changes.
- `hailo parser onnx …`
  → segfaults on this model. Do not use. `build.py parse` handles it via the
  Python API.

## Model-script flag rationale

`hailo/model_script.alls`:

```
model_optimization_flavor(optimization_level=4, compression_level=0)
```

- `optimization_level=4` is set **explicitly** because it silently drops to 0
  (equalization only) when no GPU is detected. That fallback would look like
  "this model doesn't quantize well." `build.py` prints TF's visible GPU
  devices at startup so this can't be missed.
- `compression_level=0` is set **explicitly** because we have huge FPS
  headroom and want accuracy. At 4.2M params it's under Hailo's 20M
  threshold and would auto-revert to 0 anyway.
- **No `normalization()` command.** `normalize_tile` computes mean/std from
  each tile individually — data-dependent, can't be baked into fixed arrays.
  Preprocessing stays on the host.
