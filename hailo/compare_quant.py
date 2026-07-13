"""Compare FP-optimized emulation vs quantized emulation on the eval set.

RUN IN: analysis_venv  (needs cv2, scipy — no torch, no hailo)
FROM:   ~/loca

Reads:  hailo/preds_fp.npy, hailo/preds_quant.npy, hailo/eval_set.npy,
        hailo/postprocess.json (for tanh_scale)
Prints: per-context precision / recall / localization RMSE and the delta.

This is the number that decides whether quantization is production-ready.
Per-layer SNR (from build.py analyze) tells you WHERE precision is lost;
this tells you whether it cost you a particle.

Hailo outputs come back NHWC — we transpose to CHW for the detection code.
"""
import argparse
import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from loca_core import (
    detect_via_nms, compute_forbidden_mask_fast, match_detections_fast,
    MARKER_THRESHOLD, window_size,
)


HERE = _HERE
EVAL_PATH     = os.path.join(HERE, 'eval_set.npy')
PREDS_FP_PATH = os.path.join(HERE, 'preds_fp.npy')
PREDS_Q_PATH  = os.path.join(HERE, 'preds_quant.npy')
POSTPROC_PATH = os.path.join(HERE, 'postprocess.json')

# Hailo -> semantic mapping (mirrors EXPECTED_OUTPUT_MAP in build.py).
NAME_P     = 'output_layer2'   # (128, 128, 1)
NAME_MU_XY = 'output_layer1'   # (128, 128, 2)


def _load_postproc():
    with open(POSTPROC_PATH) as f:
        return json.load(f)


def _load_preds(path):
    obj = np.load(path, allow_pickle=True).item()
    p_nhwc  = obj[NAME_P]        # (N, 128, 128, 1)
    mu_nhwc = obj[NAME_MU_XY]    # (N, 128, 128, 2)
    assert p_nhwc.shape[1:]  == (128, 128, 1), p_nhwc.shape
    assert mu_nhwc.shape[1:] == (128, 128, 2), mu_nhwc.shape
    return p_nhwc, mu_nhwc


def _run_detection(p_nhwc, mu_nhwc, eval_tiles_nhwc, gt_centers, gt_counts,
                    tanh_scale, p_threshold, nms_kernel, match_radius):
    """For each tile: transpose to CHW, apply tanh_scale, build forbidden mask
    from the raw tile, run NMS, match against GT. Aggregate metrics."""
    N = p_nhwc.shape[0]
    TP = FP = FN = 0
    loc_errors = []

    for i in range(N):
        # NHWC -> CHW  (drop batch dim, transpose spatial-last -> channel-first)
        p_map  = p_nhwc[i, :, :, 0].astype(np.float32)                     # (H, W)
        mu_map = np.transpose(mu_nhwc[i], (2, 0, 1)).astype(np.float32)    # (2, H, W)
        mu_map = mu_map * tanh_scale                                        # host-side scale

        # Forbidden mask from the *raw* tile (matches inference; also matches
        # how run_validation_test builds forbidden from the input image).
        # Note eval_set stores normalized tiles, not raw — for the eval fixture
        # we use the normalized tile's mask, which is the same forbidden regions
        # (marker locations don't move under standardization; only the threshold
        # boundary shifts). Set threshold to the normalized-domain equivalent:
        # since normalize is (x - m) / s, threshold = (MARKER_THRESHOLD - m) / s.
        # For simplicity we use the fact that the marker/edge geometry is fixed
        # per-tile — normalization doesn't reorder pixels — and pass a
        # conservative "no forbidden" mask; forbidden was baked into GT during
        # synthesis (compute_forbidden_mask() was used to reject placements),
        # so any FPs in that region are also legitimately trainable regions.
        # If you want strict apples-to-apples with the real-inference path,
        # rebuild the raw tile from bg_tiles_eval + un-normalize. Not done here
        # because the eval set's purpose is quant-vs-FP delta, not absolute
        # ground-truth vs prediction.
        tile_norm = eval_tiles_nhwc[i, :, :, 0]
        # Rebuild an approximate "raw" tile for forbidden-mask purposes by
        # z-scoring back — but since we don't have the original m, s, use the
        # tile's own std to detect marker regions: markers are the darkest
        # pixels of the tile.
        # Empirically markers are much darker than the normalized mean — use
        # the 5th percentile as a marker cutoff.
        marker_cutoff = np.percentile(tile_norm, 5)
        pseudo_raw = tile_norm - marker_cutoff + MARKER_THRESHOLD + 1
        forbidden = compute_forbidden_mask_fast(pseudo_raw)

        cx, cy = detect_via_nms(p_map, mu_map, forbidden,
                                 p_threshold=p_threshold,
                                 nms_kernel=nms_kernel)

        n_gt = int(gt_counts[i])
        true_pos = gt_centers[i, :n_gt] if n_gt > 0 else np.zeros((0, 2), dtype=np.float32)

        tp, fp, fn, matches = match_detections_fast(true_pos, cx, cy,
                                                     match_radius=match_radius)
        TP += tp; FP += fp; FN += fn
        loc_errors.extend(m[2] for m in matches)

    P  = TP / (TP + FP) if TP + FP else 0.0
    R  = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    rmse = float(np.sqrt(np.mean(np.square(loc_errors)))) if loc_errors else float('nan')
    return dict(TP=TP, FP=FP, FN=FN, P=P, R=R, F1=F1, loc_rmse=rmse,
                n_matches=len(loc_errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p-threshold', type=float, default=0.004,
                    help='detection threshold on p_map (matches real-data best F1)')
    ap.add_argument('--nms-kernel',  type=int,   default=5)
    ap.add_argument('--match-radius', type=int,  default=10,
                    help='match radius in model px (128 space)')
    args = ap.parse_args()

    post = _load_postproc()
    tanh_scale = float(post['tanh_scale'])
    print(f"[cfg] tanh_scale = {tanh_scale}  (from postprocess.json)")
    print(f"[cfg] p_threshold={args.p_threshold}  nms_kernel={args.nms_kernel}  "
          f"match_radius={args.match_radius}px")

    eval_obj = np.load(EVAL_PATH, allow_pickle=True).item()
    eval_tiles = eval_obj['tiles']
    gt_centers = eval_obj['centers']
    gt_counts  = eval_obj['counts']
    print(f"[data] eval tiles: {eval_tiles.shape}   total GT particles: {int(gt_counts.sum())}")

    p_fp, mu_fp = _load_preds(PREDS_FP_PATH)
    p_q,  mu_q  = _load_preds(PREDS_Q_PATH)

    fp_metrics = _run_detection(
        p_fp, mu_fp, eval_tiles, gt_centers, gt_counts,
        tanh_scale, args.p_threshold, args.nms_kernel, args.match_radius,
    )
    q_metrics = _run_detection(
        p_q, mu_q, eval_tiles, gt_centers, gt_counts,
        tanh_scale, args.p_threshold, args.nms_kernel, args.match_radius,
    )

    print(f"\n{'='*60}")
    print(f"  FP-OPTIMIZED vs QUANTIZED  (over {len(eval_tiles)} tiles)")
    print(f"{'='*60}")
    header = f"{'metric':<12}  {'fp_opt':>10}  {'quantized':>10}  {'Δ (q-fp)':>10}"
    print(header)
    for k in ['TP', 'FP', 'FN']:
        d = q_metrics[k] - fp_metrics[k]
        print(f"{k:<12}  {fp_metrics[k]:>10d}  {q_metrics[k]:>10d}  {d:>+10d}")
    for k in ['P', 'R', 'F1', 'loc_rmse']:
        d = q_metrics[k] - fp_metrics[k]
        print(f"{k:<12}  {fp_metrics[k]:>10.4f}  {q_metrics[k]:>10.4f}  {d:>+10.4f}")


if __name__ == '__main__':
    main()
