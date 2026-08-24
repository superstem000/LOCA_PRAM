"""Synthetic eval across density ranges for the tanh_scale sweep.

Discovers every runs/sweep/tanh_*.pth, parses tanh_scale from the filename,
runs a validation test at multiple density ranges (num_gaus_high in
[10, 15, 20, 25, 30] by default) on the same defect BG folder used for
training, and writes runs/sweep/eval_results.csv + eval_curves.png.

Reuses train.py's model class + config constants + BG loader; copies the
eval-only helpers from cell 26 of LOCA_PRAM_main_downsampled.ipynb, stripped
of the visualization / threshold-sweep / amplitude-analysis paths.

Usage:
    python organize_sweep.ps1        # or .sh — populates runs/sweep/
    python synthetic_eval.py         # runs the eval
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Reuse everything from train.py — this imports the model, config constants,
# BG loader, and all the helpers. train.py's module-scope code runs (loads
# amplitude_samples.npy, defines classes/functions) but its main() only
# runs under __name__ == '__main__' so this is safe.
import train
from train import (
    GaussianMixtureModel,
    load_bg_tiles,
    compute_forbidden_mask,
    sample_amplitude,
)


# =============================================================================
# Eval-only helpers (from cell 26 of the notebook, stripped)
# =============================================================================
def synthesize_test_tile(bg_tile, num_gaus_low, num_gaus_high):
    """Simpler particle synthesis used for eval — no rotation, no defocus, no
    motion, no min-separation rejection. Returns (image, positions, amps)."""
    H, W = bg_tile.shape
    image = bg_tile.copy().astype(np.float32)

    forbidden = compute_forbidden_mask(bg_tile)
    if train.MARGIN > 0:
        forbidden[:train.MARGIN, :]  = True
        forbidden[-train.MARGIN:, :] = True
        forbidden[:, :train.MARGIN]  = True
        forbidden[:, -train.MARGIN:] = True
    pys, pxs = np.where(~forbidden)
    active_frac = pys.size / forbidden.size

    nominal = np.random.randint(num_gaus_low, num_gaus_high)
    scaled = nominal * active_frac
    n_floor = int(scaled)
    n_place = n_floor + (1 if np.random.random() < (scaled - n_floor) else 0)
    if pys.size == 0:
        n_place = 0

    positions, amplitudes = [], []
    x_vec = np.arange(W); y_vec = np.arange(H)

    for _ in range(n_place):
        k = np.random.randint(0, pys.size)
        x_c = pxs[k] + np.random.uniform(0, 1)
        y_c = pys[k] + np.random.uniform(0, 1)
        amp = float(sample_amplitude())
        sx = max(train.SIGMA_MIN,
                 np.random.normal(loc=train.SIGMA_X_LOC, scale=train.SIGMA_X_SCALE))
        sy = max(train.SIGMA_MIN,
                 np.random.normal(loc=train.SIGMA_Y_LOC, scale=train.SIGMA_Y_SCALE))
        g_x = np.exp(-((x_vec - x_c) ** 2) / (2 * sx ** 2))
        g_y = np.exp(-((y_vec - y_c) ** 2) / (2 * sy ** 2))
        image -= np.outer(g_y, g_x) * amp
        positions.append((x_c, y_c))
        amplitudes.append(amp)

    image = np.clip(image, 0, 65535)
    return (image,
            np.array(positions) if positions else np.zeros((0, 2)),
            np.array(amplitudes))


def compute_forbidden_mask_fast(image, buffer_px=None, threshold=None, margin=None):
    if buffer_px is None: buffer_px = train.BUFFER_PX
    if threshold is None: threshold = train.MARKER_THRESHOLD
    if margin    is None: margin    = train.MARGIN

    H, W = image.shape
    non_marker = (image > threshold).astype(np.uint8)
    if non_marker.all():
        forbidden = np.zeros((H, W), dtype=bool)
    else:
        dist = cv2.distanceTransform(non_marker, cv2.DIST_L2, 3)
        forbidden = dist <= buffer_px
    if margin > 0:
        forbidden[:margin, :]  = True
        forbidden[-margin:, :] = True
        forbidden[:, :margin]  = True
        forbidden[:, -margin:] = True
    return forbidden


def detect_via_nms(p_map_np, mu_map_np, forbidden_mask, p_threshold,
                   nms_kernel=5, refine_subpixel=True):
    """Pure NMS peak-picking on p_map, with μ-refined sub-pixel positions.
    Same algorithm as run_inference/analyze_assay.detect_via_nms_xy."""
    H, W = p_map_np.shape
    above = (p_map_np > p_threshold) & ~forbidden_mask
    if not above.any():
        return np.array([]), np.array([])
    kernel = np.ones((nms_kernel, nms_kernel), dtype=np.uint8)
    p_max = cv2.dilate(p_map_np.astype(np.float32), kernel)
    peaks = above & (p_map_np >= p_max - 1e-9)
    yi, xi = np.where(peaks)
    if refine_subpixel:
        rx = xi.astype(np.float64) + mu_map_np[0, yi, xi]
        ry = yi.astype(np.float64) + mu_map_np[1, yi, xi]
    else:
        rx = xi.astype(np.float64); ry = yi.astype(np.float64)
    return rx, ry


def match_detections_fast(true_pos, pred_x, pred_y, match_radius=10):
    """Greedy nearest-neighbour matching under match_radius. Returns list of
    (i_true, i_pred, dist) for matches; unmatched preds -> FP; unmatched
    trues -> FN."""
    if len(true_pos) == 0 or len(pred_x) == 0:
        return [], list(range(len(pred_x))), list(range(len(true_pos)))
    matched = set()
    matched_true = set()
    out = []
    # Score all pairs, sort by distance ascending.
    tx, ty = true_pos[:, 0], true_pos[:, 1]
    D = np.sqrt((tx[:, None] - pred_x[None, :]) ** 2
                + (ty[:, None] - pred_y[None, :]) ** 2)
    flat = [(D[i, j], i, j) for i in range(D.shape[0]) for j in range(D.shape[1])
            if D[i, j] <= match_radius]
    flat.sort()
    for d, i, j in flat:
        if i in matched_true or j in matched:
            continue
        matched_true.add(i); matched.add(j)
        out.append((i, j, d))
    unmatched_pred = [j for j in range(len(pred_x)) if j not in matched]
    unmatched_true = [i for i in range(len(true_pos)) if i not in matched_true]
    return out, unmatched_pred, unmatched_true


def run_validation_slim(model, device, bg_tiles, num_test_tiles=200,
                        num_gaus_low=1, num_gaus_high=10,
                        p_threshold=0.006, nms_kernel=7, match_radius=10):
    """Slim replacement for cell 26's run_validation_test. Skips all viz /
    threshold-sweep / amplitude analysis / benchmark; returns just the
    aggregate metrics."""
    per_tile = []
    with torch.no_grad():
        for ti in range(num_test_tiles):
            bg_tile = bg_tiles[np.random.randint(0, len(bg_tiles))]
            image, true_pos, _ = synthesize_test_tile(
                bg_tile, num_gaus_low, num_gaus_high)

            # Normalize with clean-bg mask (matches training's evaluate())
            valid = bg_tile > train.MARKER_THRESHOLD
            if valid.any():
                m = image[valid].mean(); s = image[valid].std()
            else:
                m = image.mean(); s = image.std()
            if s == 0: s = 1
            norm = (image - m) / s

            x_in = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                enabled=(device.type == "cuda")):
                p, pxyn_mean, _, _ = model(x_in, training=False)
            p_map  = p[0, 0].float().cpu().numpy()
            mu_map = pxyn_mean[0, :2].float().cpu().numpy()

            forbidden = compute_forbidden_mask_fast(bg_tile)
            pred_x, pred_y = detect_via_nms(
                p_map, mu_map, forbidden, p_threshold, nms_kernel=nms_kernel)

            matches, unmatched_pred, unmatched_true = match_detections_fast(
                true_pos, pred_x, pred_y, match_radius=match_radius)
            per_tile.append({
                "true_n": len(true_pos),
                "pred_n": len(pred_x),
                "tp":     len(matches),
                "fp":     len(unmatched_pred),
                "fn":     len(unmatched_true),
                "dists":  [d for _, _, d in matches],
            })

    TP = sum(r["tp"] for r in per_tile)
    FP = sum(r["fp"] for r in per_tile)
    FN = sum(r["fn"] for r in per_tile)
    P = TP / (TP + FP) if TP + FP > 0 else 0.0
    R = TP / (TP + FN) if TP + FN > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    all_dists = [d for r in per_tile for d in r["dists"]]
    all_true = np.array([r["true_n"] for r in per_tile])
    all_pred = np.array([r["pred_n"] for r in per_tile])
    count_mae = float(np.mean(np.abs(all_true - all_pred)))
    count_bias = float(np.mean(all_pred - all_true))  # +ve = over-count
    loc_rmse = float(np.sqrt(np.mean(np.square(all_dists)))) if all_dists else float("nan")
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": P, "recall": R, "f1": F1,
        "count_mae": count_mae, "count_bias": count_bias,
        "loc_rmse_px": loc_rmse,
        "n_tiles": num_test_tiles,
        "mean_true_n": float(all_true.mean()),
        "mean_pred_n": float(all_pred.mean()),
    }


# =============================================================================
# Model discovery
# =============================================================================
_TANH_RE = re.compile(r"tanh_(\d+(?:\.\d+)?)")


def discover_sweep_models(sweep_dir):
    """Returns list of {'path': ..., 'tanh_scale': ..., 'label': ...} sorted
    by tanh_scale ascending."""
    paths = sorted(glob.glob(os.path.join(sweep_dir, "tanh_*.pth")))
    out = []
    for p in paths:
        m = _TANH_RE.search(os.path.basename(p))
        if not m:
            print(f"  skipping (couldn't parse tanh_scale): {p}")
            continue
        ts = float(m.group(1))
        out.append({"path": p, "tanh_scale": ts, "label": f"tanh_{ts}"})
    out.sort(key=lambda d: d["tanh_scale"])
    return out


def load_sweep_model(pth_path, tanh_scale, device):
    model = GaussianMixtureModel(num_channels=1)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device).eval()
    (model._orig_mod if hasattr(model, "_orig_mod") else model).tanh_scale = tanh_scale
    return model


# =============================================================================
# Driver
# =============================================================================
def main(args):
    sweep_dir = os.path.abspath(args.sweep_dir)
    os.makedirs(sweep_dir, exist_ok=True)

    models = discover_sweep_models(sweep_dir)
    if not models:
        sys.exit(f"error: no tanh_*.pth files found under {sweep_dir}")
    print(f"Found {len(models)} models:")
    for m in models:
        print(f"  {m['label']:12s} <- {os.path.basename(m['path'])}")

    print(f"\nLoading BG tiles from {args.bg_folder} ...")
    # We only need the train pool for the eval (matches notebook's setup)
    bg_tiles, _ = load_bg_tiles(args.bg_folder, args.train_fov_count)
    print(f"  {len(bg_tiles)} training tiles loaded")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    density_ranges = args.density_max
    print(f"Density ranges (num_gaus_high): {density_ranges}")
    print(f"Tiles per (model × density): {args.num_test_tiles}")

    rows = []
    for m in models:
        print(f"\n=== {m['label']} ===")
        model = load_sweep_model(m["path"], m["tanh_scale"], device)
        for d_max in density_ranges:
            np.random.seed(args.seed)   # Same tiles for every (model, density)
            t0 = time.time()
            r = run_validation_slim(
                model, device, bg_tiles,
                num_test_tiles=args.num_test_tiles,
                num_gaus_low=args.density_min,
                num_gaus_high=d_max,
                p_threshold=args.p_threshold,
                nms_kernel=args.nms_kernel,
                match_radius=args.match_radius,
            )
            dt = time.time() - t0
            r.update({
                "tanh_scale":       m["tanh_scale"],
                "density_max":      d_max,
                "density_min":      args.density_min,
                "p_threshold":      args.p_threshold,
                "nms_kernel":       args.nms_kernel,
                "match_radius_px":  args.match_radius,
                "wall_time_s":      dt,
            })
            rows.append(r)
            print(f"  density [{args.density_min}, {d_max}):  "
                  f"P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
                  f"MAE={r['count_mae']:.2f}  bias={r['count_bias']:+.2f}  "
                  f"loc_rmse={r['loc_rmse_px']:.2f}px  "
                  f"({dt:.1f}s)")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(sweep_dir, "eval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    ts_vals = sorted(df["tanh_scale"].unique())
    cmap = plt.get_cmap("viridis", max(len(ts_vals), 2))
    for i, ts in enumerate(ts_vals):
        g = df[df["tanh_scale"] == ts].sort_values("density_max")
        col = cmap(i)
        axes[0].plot(g["density_max"], g["precision"], "-o", color=col, label=f"tanh_{ts}")
        axes[0].plot(g["density_max"], g["recall"], "--s", color=col, alpha=0.6)
        axes[1].plot(g["density_max"], g["f1"], "-o", color=col, label=f"tanh_{ts}")
        axes[2].plot(g["density_max"], g["count_mae"], "-o", color=col, label=f"tanh_{ts}")

    axes[0].set_title("Precision (solid) / Recall (dashed)")
    axes[0].set_xlabel("density max (num_gaus_high)")
    axes[0].set_ylabel("Precision, Recall")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.02)

    axes[1].set_title("F1")
    axes[1].set_xlabel("density max"); axes[1].set_ylabel("F1")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.02)

    axes[2].set_title("Count MAE (lower = better)")
    axes[2].set_xlabel("density max"); axes[2].set_ylabel("|pred - true| mean")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    fig.suptitle(f"Synthetic eval sweep (n_tiles per point = {args.num_test_tiles}, "
                 f"p_thr={args.p_threshold}, nms={args.nms_kernel})", fontsize=11)
    fig.tight_layout()
    png_path = os.path.join(sweep_dir, "eval_curves.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Synthetic eval across density ranges for the tanh_scale sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sweep-dir", default="runs/sweep/",
                   help="Directory containing tanh_*.pth checkpoints "
                        "(populated by organize_sweep.sh/.ps1).")
    p.add_argument("--bg-folder", default="demo_0001/background_defect/",
                   help="Same BG folder used for training — needed to build "
                        "test tiles with matching background distribution.")
    p.add_argument("--train-fov-count", type=int, default=104,
                   help="Same split as training (deterministic under "
                        "SPLIT_SEED=42).")
    p.add_argument("--density-min", type=int, default=1)
    p.add_argument("--density-max", type=int, nargs="+", default=[10, 15, 20, 25, 30],
                   help="One eval per value; each is num_gaus_high (exclusive) "
                        "with density_min as num_gaus_low.")
    p.add_argument("--num-test-tiles", type=int, default=200)
    p.add_argument("--p-threshold", type=float, default=0.006)
    p.add_argument("--nms-kernel", type=int, default=7)
    p.add_argument("--match-radius", type=float, default=10.0,
                   help="Detection <-> GT matching radius (native px in the "
                        "model-input coordinate system, i.e. downsampled px).")
    p.add_argument("--seed", type=int, default=0,
                   help="Test-tile seed — reset before each density range so "
                        "the same test set is used across models.")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
