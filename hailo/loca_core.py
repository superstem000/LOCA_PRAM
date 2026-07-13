"""Extracted from LOCA_PRAM_main_downsampled.ipynb — synthesis + preprocessing
+ detection functions the Hailo pipeline needs.

⚠ COPY, NOT REIMPLEMENTATION. Keep in sync with the notebook manually.
Whenever the following notebook cells change:
  - config cell   (DOWNSCALE_FACTOR, SIGMA_*, MARKER_THRESHOLD, AMP_*, ...)
  - bg-loader cell   (load_fov_red, tile_fovs, BG_FOLDER, TRAIN_FOV_COUNT, SPLIT_SEED)
  - generate_simulation_image cell
  - real-data normalize_tile cell
  - detect_via_nms / compute_forbidden_mask_fast cell
… mirror the edits here.

The notebook itself CANNOT be imported: it has `import napari` at top level and
napari isn't available in the analysis venv (Qt GUI, headless WSL).
"""
import glob
import os
import numpy as np
import cv2
from PIL import Image
from scipy import stats as _stats


# =====================================================================
# DOWNSCALE + PSF + MARKER CONFIG   (mirrors notebook cell 3)
# =====================================================================
DOWNSCALE_FACTOR = 2

BASE_WINDOW_SIZE = (256, 256)
window_size = (BASE_WINDOW_SIZE[0] // DOWNSCALE_FACTOR,
               BASE_WINDOW_SIZE[1] // DOWNSCALE_FACTOR)

# PSF sigmas (measured, then divided by DOWNSCALE_FACTOR).
BASE_SIGMA_X_LOC,   BASE_SIGMA_X_SCALE = 9.79, 1.32
BASE_SIGMA_Y_LOC,   BASE_SIGMA_Y_SCALE = 7.88, 1.38
BASE_SIGMA_MIN     = 2.5
BASE_MARGIN        = 12

SIGMA_X_LOC   = BASE_SIGMA_X_LOC   / DOWNSCALE_FACTOR
SIGMA_X_SCALE = BASE_SIGMA_X_SCALE / DOWNSCALE_FACTOR
SIGMA_Y_LOC   = BASE_SIGMA_Y_LOC   / DOWNSCALE_FACTOR
SIGMA_Y_SCALE = BASE_SIGMA_Y_SCALE / DOWNSCALE_FACTOR
SIGMA_MIN     = BASE_SIGMA_MIN     / DOWNSCALE_FACTOR
MARGIN        = max(1, int(round(BASE_MARGIN / DOWNSCALE_FACTOR)))

# amplitude distribution — Beta fit to real PSF amplitudes.
# `amplitude_samples.npy` lives at the repo root; scripts must run from ~/loca
# or set the path explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_amp = np.load(os.path.join(_REPO_ROOT, 'amplitude_samples.npy'))
AMP_FLOOR  = 12.0
AMP_CEIL   = float(_amp.max() + 0.03 * (_amp.max() - _amp.min()))
AMP_BETA_A, AMP_BETA_B, _, _ = _stats.beta.fit(
    _amp, floc=AMP_FLOOR, fscale=AMP_CEIL - AMP_FLOOR)


def sample_amplitude(size=None):
    return _stats.beta.rvs(AMP_BETA_A, AMP_BETA_B, loc=AMP_FLOOR,
                           scale=AMP_CEIL - AMP_FLOOR, size=size)


MARKER_THRESHOLD = 15
BUFFER_SIGMA_MULT = 5.0
BUFFER_PX = int(round(BUFFER_SIGMA_MULT * max(SIGMA_X_LOC, SIGMA_Y_LOC)))


def compute_forbidden_mask(tile, buffer_px=None, threshold=None):
    if buffer_px is None:
        buffer_px = BUFFER_PX
    if threshold is None:
        threshold = MARKER_THRESHOLD
    marker = tile <= threshold
    if buffer_px <= 0 or not marker.any():
        return marker
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~marker)
    return dist <= buffer_px


def compute_forbidden_mask_fast(image, buffer_px=None, threshold=None, margin=None):
    """cv2.distanceTransform variant + edge margin (mirrors notebook cell 26)."""
    if buffer_px is None: buffer_px = BUFFER_PX
    if threshold is None: threshold = MARKER_THRESHOLD
    if margin is None:    margin = MARGIN

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


def downsample_image(img, factor=None):
    if factor is None:
        factor = DOWNSCALE_FACTOR
    if factor == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


GAUSSIAN_BLUR_SIGMA = 0.0

def smooth_image(img, sigma=None):
    if sigma is None:
        sigma = GAUSSIAN_BLUR_SIGMA
    if sigma <= 0:
        return img
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)


# =====================================================================
# BG-loader (mirrors notebook cell 7)
# =====================================================================
BG_FOLDER          = 'demo_0001/background/'
TRAIN_FOV_COUNT    = 200
SPLIT_SEED         = 42
MIN_PLACEABLE_FRAC = 0.05


def load_fov_red(path):
    """Open a JPG FOV and return the red channel as float32 (0-255 range)."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.float32)
    if DOWNSCALE_FACTOR != 1:
        arr = downsample_image(arr)
    return arr


def _placeable_frac(tile):
    forbidden = compute_forbidden_mask(tile)
    if MARGIN > 0:
        forbidden[:MARGIN, :] = True
        forbidden[-MARGIN:, :] = True
        forbidden[:, :MARGIN] = True
        forbidden[:, -MARGIN:] = True
    return float((~forbidden).mean())


def tile_fovs(fov_paths, tile_size, label=""):
    tiles = []
    n_dropped = 0
    for p in fov_paths:
        img = load_fov_red(p)
        h, w = img.shape
        for i in range(h // tile_size[0]):
            for j in range(w // tile_size[1]):
                tile = img[i*tile_size[0]:(i+1)*tile_size[0],
                           j*tile_size[1]:(j+1)*tile_size[1]]
                if _placeable_frac(tile) < MIN_PLACEABLE_FRAC:
                    n_dropped += 1
                    continue
                tile = smooth_image(tile)
                tiles.append(tile)
    arr = np.asarray(tiles)
    print(f"  {label}: {len(fov_paths)} FOVs -> {len(arr)} tiles "
          f"(dropped {n_dropped} with placeable_frac < {MIN_PLACEABLE_FRAC:.0%})")
    return arr


def load_split(bg_folder=None, tile_size=None):
    """Reproduce the notebook's FOV split exactly. Returns bg_tiles_train,
    bg_tiles_eval, plus the FOV path lists."""
    if bg_folder is None:
        bg_folder = os.path.join(_REPO_ROOT, BG_FOLDER)
    if tile_size is None:
        tile_size = window_size
    all_fovs = sorted(glob.glob(os.path.join(bg_folder, '*.jpg')))
    assert len(all_fovs) > TRAIN_FOV_COUNT, \
        f"need > {TRAIN_FOV_COUNT} FOVs in {bg_folder}, found {len(all_fovs)}"

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(all_fovs))
    train_fovs = [all_fovs[i] for i in perm[:TRAIN_FOV_COUNT]]
    eval_fovs  = [all_fovs[i] for i in perm[TRAIN_FOV_COUNT:]]

    print(f"Loaded {len(all_fovs)} FOVs from {bg_folder}")
    print(f"  train: {len(train_fovs)} FOVs")
    print(f"  eval : {len(eval_fovs)} FOVs (held out)")

    bg_tiles_train = tile_fovs(train_fovs, tile_size, label="train")
    bg_tiles_eval  = tile_fovs(eval_fovs,  tile_size, label="eval ")
    return bg_tiles_train, bg_tiles_eval, train_fovs, eval_fovs


# =====================================================================
# Synthesis (mirrors notebook cell 9) — with augmentations + empty-tile prob
# =====================================================================
def generate_simulation_image(img_size, bg_tiles, num_gaus_low=1, num_gaus_high=50, labels_channel=3):
    bg_idx = np.random.randint(0, len(bg_tiles))
    gauss_base = bg_tiles[bg_idx].copy()
    bg_clean   = bg_tiles[bg_idx]

    forbidden = compute_forbidden_mask(bg_clean)
    if MARGIN > 0:
        forbidden[:MARGIN, :] = True
        forbidden[-MARGIN:, :] = True
        forbidden[:, :MARGIN] = True
        forbidden[:, -MARGIN:] = True

    placeable_ys, placeable_xs = np.where(~forbidden)
    active_frac = placeable_ys.size / forbidden.size

    x_vec = np.arange(img_size[1])
    y_vec = np.arange(img_size[0])

    labels = np.zeros((labels_channel, img_size[0], img_size[1]))

    EMPTY_TILE_PROB = 0.15
    if np.random.random() < EMPTY_TILE_PROB:
        n_to_place = 0
    else:
        nominal_count = np.random.randint(num_gaus_low, num_gaus_high)
        scaled = nominal_count * active_frac
        n_floor = int(scaled)
        n_to_place = n_floor + (1 if np.random.random() < (scaled - n_floor) else 0)

    if placeable_ys.size == 0:
        n_to_place = 0

    MIN_SEPARATION = 5.0
    used_locations = []
    Xg, Yg = np.meshgrid(x_vec, y_vec, indexing='xy')

    defocus       = np.random.uniform(1.0, 1.25)
    motion_theta  = np.random.uniform(0.0, 2.0 * np.pi)
    motion_length = np.random.uniform(0.0, 2.5)
    sigma_motion  = motion_length / 2.35
    _cm, _sm = np.cos(motion_theta), np.sin(motion_theta)
    Sigma_motion = sigma_motion ** 2 * np.array([
        [_cm * _cm, _cm * _sm],
        [_cm * _sm, _sm * _sm],
    ])

    for _ in range(n_to_place):
        placed = False
        for _attempt in range(50):
            k = np.random.randint(0, placeable_ys.size)
            iy_base = placeable_ys[k]
            ix_base = placeable_xs[k]
            x_c = ix_base + np.random.uniform(0, 1)
            y_c = iy_base + np.random.uniform(0, 1)
            if all((x_c - ux) ** 2 + (y_c - uy) ** 2 >= MIN_SEPARATION ** 2
                   for ux, uy in used_locations):
                used_locations.append((x_c, y_c))
                placed = True
                break
        if not placed:
            continue

        amplitude = float(sample_amplitude())

        raw_sigma_x = np.random.normal(loc=SIGMA_X_LOC, scale=SIGMA_X_SCALE)
        raw_sigma_y = np.random.normal(loc=SIGMA_Y_LOC, scale=SIGMA_Y_SCALE)
        sigma_x = max(SIGMA_MIN, raw_sigma_x) * defocus
        sigma_y = max(SIGMA_MIN, raw_sigma_y) * defocus
        amplitude_eff = amplitude / (defocus ** 2)

        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_psf = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        Sigma_psf   = R_psf @ np.diag([sigma_x ** 2, sigma_y ** 2]) @ R_psf.T
        Sigma_total = Sigma_psf + Sigma_motion
        Sigma_inv   = np.linalg.inv(Sigma_total)

        Xr = Xg - x_c
        Yr = Yg - y_c
        q = (Sigma_inv[0, 0] * Xr * Xr
             + 2.0 * Sigma_inv[0, 1] * Xr * Yr
             + Sigma_inv[1, 1] * Yr * Yr)
        particle = amplitude_eff * np.exp(-0.5 * q)

        gauss_base -= particle

        ix, iy = int(x_c), int(y_c)
        if 0 <= ix < img_size[1] and 0 <= iy < img_size[0]:
            labels[0, iy, ix] = x_c - ix
            labels[1, iy, ix] = y_c - iy
            labels[2, iy, ix] = 1.0

    gauss_base = np.clip(gauss_base, 0, 65535)
    return gauss_base, labels, bg_clean


# =====================================================================
# Normalization at inference (mirrors notebook cell 30)
# =====================================================================
def normalize_tile(image):
    """Per-tile standardization over pixels above MARKER_THRESHOLD. This is
    the EXACT function used at real-image inference. Calibration MUST use
    this same function or quantization will be fit to the wrong activation
    distribution."""
    valid = image > MARKER_THRESHOLD
    if valid.any():
        m, s = image[valid].mean(), image[valid].std()
    else:
        m, s = image.mean(), image.std()
    if s == 0:
        s = 1
    return (image - m) / s


# =====================================================================
# Detection (mirrors notebook cell 26) — pure NMS on p, μ-refined
# =====================================================================
def detect_via_nms(p_map_np, mu_map_np, forbidden_mask, p_threshold,
                   nms_kernel=5, refine_subpixel=True):
    """Pure max-pool NMS on p. mu_map_np shape (2, H, W) after any host-side
    tanh_scale multiplication."""
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
    if len(true_pos) == 0:
        return 0, len(pred_x), 0, []
    if len(pred_x) == 0:
        return 0, 0, len(true_pos), []
    matched = set(); out = []
    for pi in range(len(pred_x)):
        d = np.hypot(true_pos[:, 0] - pred_x[pi], true_pos[:, 1] - pred_y[pi])
        for ti in np.argsort(d):
            if d[ti] > match_radius:
                break
            if ti not in matched:
                matched.add(ti)
                out.append((ti, pi, d[ti]))
                break
    tp = len(out)
    return tp, len(pred_x) - tp, len(true_pos) - tp, out
