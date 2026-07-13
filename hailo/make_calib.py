"""Generate the Hailo calibration and eval sets.

RUN IN: analysis_venv  (needs cv2, scipy, PIL — no torch, no hailo).
FROM:   ~/loca  (or set --repo-root)

Outputs (written to hailo/):
  calib_set.npy   (N_CALIB, 128, 128, 1) float32  NHWC — for optimize()
  eval_set.npy    dict-in-npy with keys:
    'tiles'   (N_EVAL, 128, 128, 1) float32 NHWC
    'centers' (N_EVAL, K_max, 2) float32   x, y in tile coords, -1 padding
    'counts'  (N_EVAL,)          int32     number of real particles per tile

Both use bg_tiles_eval — the HELD-OUT FOVs — so activation ranges for
calibration and eval accuracy aren't fit to backgrounds the model memorized
at training.

Preprocessing MUST match deployment: tiles go through the same normalize_tile
as inference (loca_core.normalize_tile). Preprocessing mismatch is Hailo's #1
named cause of quantization accuracy loss.
"""
import argparse
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from loca_core import (
    load_split, generate_simulation_image, normalize_tile,
    window_size,
)


def _synth_and_normalize(bg_tiles, num_gaus_low, num_gaus_high, rng):
    """Returns (normalized_tile (H,W) float32, gt_centers (K,2) or empty)."""
    # generate_simulation_image uses np.random directly; seed via env is
    # awkward, so we drive it deterministically by reseeding here.
    np.random.set_state(rng.get_state())
    image, labels, _ = generate_simulation_image(
        img_size=window_size, bg_tiles=bg_tiles,
        num_gaus_low=num_gaus_low, num_gaus_high=num_gaus_high,
        labels_channel=3,
    )
    rng.set_state(np.random.get_state())

    presence = labels[2] > 0
    ys, xs = np.where(presence)
    if len(ys):
        # sub-pixel: labels[0] = frac_x, labels[1] = frac_y at the floored center
        cx = xs.astype(np.float32) + labels[0, ys, xs].astype(np.float32)
        cy = ys.astype(np.float32) + labels[1, ys, xs].astype(np.float32)
        centers = np.stack([cx, cy], axis=1)
    else:
        centers = np.zeros((0, 2), dtype=np.float32)

    tile_norm = normalize_tile(image).astype(np.float32)
    return tile_norm, centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-calib',       type=int, default=2048,
                    help='calibration set size (opt level 4 requires >=1024)')
    ap.add_argument('--n-eval',        type=int, default=128,
                    help='eval set size (quantized vs FP comparison)')
    ap.add_argument('--num-gaus-low',  type=int, default=0)
    ap.add_argument('--num-gaus-high', type=int, default=50)
    ap.add_argument('--seed',          type=int, default=0)
    ap.add_argument('--repo-root',     type=str, default=None,
                    help='override repo root (default: parent of hailo/)')
    ap.add_argument('--out-dir',       type=str, default=_HERE,
                    help='where to write calib_set.npy and eval_set.npy')
    args = ap.parse_args()

    # Load the FOV split (same seed/permutation as the notebook).
    if args.repo_root:
        bg_folder = os.path.join(args.repo_root, 'demo_0001/background/')
        bg_train, bg_eval, _, _ = load_split(bg_folder=bg_folder)
    else:
        bg_train, bg_eval, _, _ = load_split()

    print(f"\nsynthesizing {args.n_calib} calibration tiles from "
          f"{len(bg_eval)} eval bg tiles...")

    rng = np.random.RandomState(args.seed)

    calib = np.empty((args.n_calib, window_size[0], window_size[1], 1), dtype=np.float32)
    for i in range(args.n_calib):
        tile, _ = _synth_and_normalize(bg_eval, args.num_gaus_low, args.num_gaus_high, rng)
        calib[i, :, :, 0] = tile
        if (i + 1) % 256 == 0:
            print(f"  calib {i+1}/{args.n_calib}")

    # --- assertions BEFORE writing ---
    assert calib.shape == (args.n_calib, window_size[0], window_size[1], 1), \
        f"calib shape wrong: {calib.shape}"
    assert calib.dtype == np.float32, f"calib dtype wrong: {calib.dtype}"

    calib_path = os.path.join(args.out_dir, 'calib_set.npy')
    np.save(calib_path, calib)
    print(f"\nwrote {calib_path}")
    print(f"  shape={calib.shape}  dtype={calib.dtype}  layout=NHWC")
    print(f"  value range: [{calib.min():.3f}, {calib.max():.3f}]  "
          f"mean={calib.mean():.3f}  std={calib.std():.3f}")

    # --- eval set ---
    print(f"\nsynthesizing {args.n_eval} eval tiles + ground-truth centers...")
    eval_tiles = np.empty((args.n_eval, window_size[0], window_size[1], 1), dtype=np.float32)
    centers_list = []
    counts       = np.empty((args.n_eval,), dtype=np.int32)
    for i in range(args.n_eval):
        tile, centers = _synth_and_normalize(bg_eval, args.num_gaus_low, args.num_gaus_high, rng)
        eval_tiles[i, :, :, 0] = tile
        centers_list.append(centers)
        counts[i] = len(centers)

    # Pad centers to a rectangular (N, K_max, 2) with -1 for missing.
    k_max = max(1, max(len(c) for c in centers_list))
    centers_padded = np.full((args.n_eval, k_max, 2), -1.0, dtype=np.float32)
    for i, c in enumerate(centers_list):
        if len(c):
            centers_padded[i, :len(c), :] = c

    assert eval_tiles.shape == (args.n_eval, window_size[0], window_size[1], 1)
    assert eval_tiles.dtype == np.float32

    eval_path = os.path.join(args.out_dir, 'eval_set.npy')
    np.save(eval_path, {
        'tiles': eval_tiles,
        'centers': centers_padded,
        'counts': counts,
    })
    print(f"\nwrote {eval_path}")
    print(f"  tiles   shape={eval_tiles.shape}  dtype={eval_tiles.dtype}  layout=NHWC")
    print(f"  centers shape={centers_padded.shape}  (padded with -1)")
    print(f"  counts  mean={counts.mean():.1f}  min={counts.min()}  max={counts.max()}")


if __name__ == '__main__':
    main()
