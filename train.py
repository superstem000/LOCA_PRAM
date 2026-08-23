"""LOCA-PRAM training script — standalone version of
LOCA_PRAM_main_downsampled.ipynb (cells 0-1, 3, 7, 9, 12, 16, 18, 21)
with all test / visualization blocks removed and every PSF / density /
placement / training knob exposed via argparse.

Design (Option A from the ablation planning):
    All numerical constants live as MODULE-LEVEL globals with the notebook's
    defaults baked in. `main()` overrides those globals from CLI args BEFORE
    any function that reads them (dataset workers, model, loss, train loop)
    is constructed. All downstream code reads the globals directly, matching
    the notebook's closure behaviour verbatim.

Usage:
    python train.py --tanh-scale 2.0 \
                    --output-dir runs/tanh_2.0/ \
                    --max-epoch 2000

    ./ablate_tanh.sh          # runs tanh_scale = 1, 2, 4 sequentially
"""
from __future__ import annotations

import argparse
import datetime
import gc
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
from torch.distributions import Distribution
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # headless (no viz windows during batch runs)
import matplotlib.pyplot as plt
import cv2

Image.MAX_IMAGE_PIXELS = None


# ============================================================================

# Config constants (defaults from cell 3; overridden from CLI in main())

# ============================================================================

# =====================================================================
# DOWNSCALE + PSF + MARKER CONFIG
# =====================================================================
DOWNSCALE_FACTOR = 2

BASE_WINDOW_SIZE = (256, 256)
window_size = (BASE_WINDOW_SIZE[0] // DOWNSCALE_FACTOR,
               BASE_WINDOW_SIZE[1] // DOWNSCALE_FACTOR)

# PSF sigmas: direct measurements from new PSF analysis, divided by DOWNSCALE_FACTOR.
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

# ---- PSF amplitude distribution ----
# Empirical PSF amplitudes (n=145 after outlier removal) modeled as a Beta
# bounded to [AMP_FLOOR, AMP_CEIL]: smooth, right-skewed, and (unlike the old
# Gaussian) strictly cannot emit sub-noise amplitudes. Source: amplitude_samples.npy
# from the PSF-analysis cell.
# NOTE: amplitude is an INTENSITY, not a length, so it is NOT divided by
# DOWNSCALE_FACTOR — average-pool downsampling preserves the peak of a
# well-sampled Gaussian (same as the old code, which used the native 31.22).
import numpy as np
from scipy import stats as _stats
_amp = np.load('amplitude_samples.npy')
AMP_FLOOR  = 12.0
AMP_CEIL   = float(_amp.max() + 0.03 * (_amp.max() - _amp.min()))
AMP_BETA_A, AMP_BETA_B, _, _ = _stats.beta.fit(
    _amp, floc=AMP_FLOOR, fscale=AMP_CEIL - AMP_FLOOR)

def sample_amplitude(size=None):
    """Particle amplitude(s) drawn from the fitted Beta (native intensity units)."""
    return _stats.beta.rvs(AMP_BETA_A, AMP_BETA_B, loc=AMP_FLOOR,
                           scale=AMP_CEIL - AMP_FLOOR, size=size)

# Marker / banned-region config.
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


def downsample_image(img, factor=None):
    if factor is None:
        factor = DOWNSCALE_FACTOR
    if factor == 1:
        return img
    import cv2
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


print(f"DOWNSCALE_FACTOR = {DOWNSCALE_FACTOR}")
print(f"window_size      = {window_size}")
print(f"sigma_x ~ N({SIGMA_X_LOC:.2f}, {SIGMA_X_SCALE:.2f})")
print(f"sigma_y ~ N({SIGMA_Y_LOC:.2f}, {SIGMA_Y_SCALE:.2f})")
print(f"amplitude ~ Beta(a={AMP_BETA_A:.2f}, b={AMP_BETA_B:.2f}) on [{AMP_FLOOR:.1f}, {AMP_CEIL:.1f}]")
print(f"placement margin = {MARGIN}, marker buffer = {BUFFER_PX} px ({BUFFER_SIGMA_MULT:.1f}*sigma)")
print(f"gaussian blur    = {GAUSSIAN_BLUR_SIGMA} px")


# ============================================================================
# Augmentation / density / placement DEFAULTS — CLI overridable
# (mirrors the values in cell 9 of the notebook; overridden in main())
# ============================================================================
DEFOCUS_SHARP_PROB  = 0.55     # P(no defocus per tile); rest sampled in [MIN, MAX]
DEFOCUS_BLUR_MIN    = 1.3
DEFOCUS_BLUR_MAX    = 1.8
MOTION_LENGTH_MAX   = 1.5      # native px; 0 disables motion blur
EMPTY_TILE_PROB     = 0.15
MIN_SEPARATION      = 5.0      # native model px
PARTICLES_MIN       = 0
PARTICLES_MAX       = 40



# ============================================================================
# Background loader helpers — pulled from cell 7
# ============================================================================
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



# ============================================================================
# Background loader — wraps cell 7's module-level code as a function so it
# can be called after CLI has set MARGIN / MARKER_THRESHOLD / etc.
# ============================================================================
def load_bg_tiles(bg_folder, train_fov_count, split_seed=42, min_placeable_frac=0.05):
    """Returns (bg_tiles_train, bg_tiles_eval). Same behaviour as cell 7,
    just parameterised. Uses module globals: window_size, MARGIN,
    DOWNSCALE_FACTOR, GAUSSIAN_BLUR_SIGMA."""
    all_fovs = sorted(glob.glob(os.path.join(bg_folder, '*.jpg')))
    assert len(all_fovs) > train_fov_count, (
        f"need > {train_fov_count} FOVs in {bg_folder}, found {len(all_fovs)}")
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(all_fovs))
    train_fovs = [all_fovs[i] for i in perm[:train_fov_count]]
    eval_fovs  = [all_fovs[i] for i in perm[train_fov_count:]]
    print(f"Loaded {len(all_fovs)} FOVs from {bg_folder}")
    print(f"  train: {len(train_fovs)} FOVs")
    print(f"  eval : {len(eval_fovs)} FOVs (held out)")

    def _tile(fov_paths, label):
        tiles = []
        n_dropped = 0
        for p in fov_paths:
            img = load_fov_red(p)
            h, w = img.shape
            for i in range(h // window_size[0]):
                for j in range(w // window_size[1]):
                    tile = img[i*window_size[0]:(i+1)*window_size[0],
                               j*window_size[1]:(j+1)*window_size[1]]
                    if _placeable_frac(tile) < min_placeable_frac:
                        n_dropped += 1
                        continue
                    tile = smooth_image(tile)
                    tiles.append(tile)
        arr = np.asarray(tiles)
        print(f"  {label}: {len(fov_paths)} FOVs -> {len(arr)} tiles "
              f"(dropped {n_dropped} with placeable_frac < {min_placeable_frac:.0%})")
        return arr

    bg_train = _tile(train_fovs, "train")
    bg_eval  = _tile(eval_fovs,  "eval ")
    print(f"Tile value range (train): [{bg_train.min():.1f}, {bg_train.max():.1f}]")
    return bg_train, bg_eval



# ============================================================================

# generate_simulation_image (cell 9) — CLI-driven defocus / motion / density

# ============================================================================

def generate_simulation_image(img_size, bg_tiles, num_gaus_low=1, num_gaus_high=40, labels_channel=3):
    """
    Generates a simulated image with synthetic PSF particles on a real
    background tile (sampled at random from bg_tiles).
    Amplitude now drawn from the fitted Beta distribution (see config cell).
    """
    bg_idx = np.random.randint(0, len(bg_tiles))
    gauss_base = bg_tiles[bg_idx].copy()
    bg_clean   = bg_tiles[bg_idx]

    # ---- Forbidden mask: markers dilated by BUFFER_PX, plus edge margin ----
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

    # ---- Particle count: 15% empty tiles, otherwise density-aware ----
    # EMPTY_TILE_PROB is a module global (see top of file); do NOT shadow here.
    if np.random.random() < EMPTY_TILE_PROB:
        n_to_place = 0
    else:
        nominal_count = np.random.randint(num_gaus_low, num_gaus_high)
        scaled = nominal_count * active_frac
        n_floor = int(scaled)
        n_to_place = n_floor + (1 if np.random.random() < (scaled - n_floor) else 0)

    if placeable_ys.size == 0:
        n_to_place = 0

    # Min-separation rejection sampling + random PSF rotation.
    # MIN_SEPARATION is a module global (see top of file); do NOT shadow here.
    used_locations = []
    Xg, Yg = np.meshgrid(x_vec, y_vec, indexing='xy')

    # ---- Per-tile domain-gap augmentations ----
    # Defocus: symmetric widening, integrated intensity preserved.
    # Motion blur: rank-1 covariance term along a per-tile drift axis.
    # Both drawn once per tile (FOV-wide physical effects).
    # Defocus: Bernoulli-gated. CLI-configurable via DEFOCUS_SHARP_PROB,
    # DEFOCUS_BLUR_MIN, DEFOCUS_BLUR_MAX.
    if np.random.random() < DEFOCUS_SHARP_PROB:
        defocus = 1.0
    else:
        defocus = np.random.uniform(DEFOCUS_BLUR_MIN, DEFOCUS_BLUR_MAX)
    #defocus = 1.8
    motion_theta  = np.random.uniform(0.0, 2.0 * np.pi)
    motion_length = np.random.uniform(0.0, MOTION_LENGTH_MAX)   # px
    sigma_motion  = motion_length / 2.35                  # FWHM → σ
    _cm, _sm = np.cos(motion_theta), np.sin(motion_theta)
    Sigma_motion = sigma_motion ** 2 * np.array([
        [_cm * _cm, _cm * _sm],
        [_cm * _sm, _sm * _sm],
    ])
    for _ in range(n_to_place):
        # 1. Sample center from placeable pixels (rejection-sampled for min separation)
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
            continue   # too crowded — skip this particle

        # 2. Sample amplitude from fitted Beta
        amplitude = float(sample_amplitude())

        # 3. Sample sigma (PSF-measured + downscaled values), apply defocus
        raw_sigma_x = np.random.normal(loc=SIGMA_X_LOC, scale=SIGMA_X_SCALE)
        raw_sigma_y = np.random.normal(loc=SIGMA_Y_LOC, scale=SIGMA_Y_SCALE)
        sigma_x = max(SIGMA_MIN, raw_sigma_x) * defocus
        sigma_y = max(SIGMA_MIN, raw_sigma_y) * defocus
        amplitude_eff = amplitude / (defocus ** 2)   # preserve integrated intensity

        # 4. Random PSF rotation theta ∈ [-π/2, π/2), then combine with motion blur
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_psf = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        Sigma_psf   = R_psf @ np.diag([sigma_x ** 2, sigma_y ** 2]) @ R_psf.T
        Sigma_total = Sigma_psf + Sigma_motion
        Sigma_inv   = np.linalg.inv(Sigma_total)

        # 5. Render using full covariance (handles anisotropic motion axis)
        Xr = Xg - x_c
        Yr = Yg - y_c
        q = (Sigma_inv[0, 0] * Xr * Xr
             + 2.0 * Sigma_inv[0, 1] * Xr * Yr
             + Sigma_inv[1, 1] * Yr * Yr)
        particle = amplitude_eff * np.exp(-0.5 * q)

        gauss_base -= particle

        # 6. Update labels (one-hot at the integer center)
        ix, iy = int(x_c), int(y_c)
        if 0 <= ix < img_size[1] and 0 <= iy < img_size[0]:
            labels[0, iy, ix] = x_c - ix
            labels[1, iy, ix] = y_c - iy
            labels[2, iy, ix] = 1.0

    gauss_base = np.clip(gauss_base, 0, 65535)
    return gauss_base, labels, bg_clean


# ============================================================================

# GaussianDataset (cell 12)

# ============================================================================

class GaussianDataset(Dataset):
    def __init__(self, num_batches, bg_tiles, img_size=(512, 512), transform=None):
        self.transform = transform
        self.num_batches = num_batches
        self.bg_tiles = bg_tiles
        self.img_size = img_size
        self.labels_channel = 3
        
        # --- DYNAMIC DENSITY CALCULATION ---
        # Goal: Keep particle-to-image coverage ratio consistent with the original version (20% coverage)
        
        # 1. Reference (Original Setup)
        # 50 particles (radius ~3px) in 84x84
        ref_coverage_ratio = 0.20 
        
        # 2. New Setup (Current Config)
        new_window_area = self.img_size[0] * self.img_size[1]
        
        
        # PSF area shrank ~16x (sigma 4x smaller); bumped 4x to 0-23 so we
        # actually get overlapping particles for the model to learn that case.
        # Density range is CLI-configurable via PARTICLES_MIN/MAX module globals.
        self.num_gaus_low = PARTICLES_MIN
        self.num_gaus_high = PARTICLES_MAX
        
        # print(f"Density Calculation for {img_size}:")
        # print(f"  - Particle Dimensions: ~{int(new_sigma_x*4)}x{int(new_sigma_y*4)} pixels")
        # print(f"  - Max Particles set to: {self.num_gaus_high}")

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # We call the generate function we defined in the previous cell
        image, labels, bg_image = generate_simulation_image(
            img_size=self.img_size, 
            bg_tiles=self.bg_tiles, 
            num_gaus_low=self.num_gaus_low, 
            num_gaus_high=self.num_gaus_high,
            labels_channel=self.labels_channel
        )
        
        # Standardize / Normalize -- ignore marker pixels only (match real inference).
        valid = bg_image > MARKER_THRESHOLD          # was: bg_image > MARKER_THRESHOLD
        if valid.any():
            m = image[valid].mean()
            s = image[valid].std()
        else:
            m = image.mean()
            s = image.std()
        if s == 0:
            s = 1
        image = (image - m) / s
        
        # Normalize bg_image with the SAME (m, s) as input so the bg head
        # supervises in the same normalized space.
        bg_image = (bg_image - m) / s

        # Convert to Tensor
        image = torch.from_numpy(image).float().unsqueeze(0) # (1, H, W)
        labels = torch.from_numpy(labels).float()            # (3, H, W)
        bg_image = torch.from_numpy(bg_image).float().unsqueeze(0)
        
        # Apply transforms if they exist
        if self.transform:
            image = self.transform(image)
            
        return image, labels, bg_image

# Asynchronous data loading function for batches
def async_load_batch(data_loader):
    """Load batches asynchronously using a thread pool."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_batch, batch) for batch in data_loader]
        for future in futures:
            yield future.result()

def load_batch(batch):
    """Simulate batch loading (you can customize this to fit your process)."""
    time.sleep(0.1)  # Simulate a delay (remove or adjust depending on your loading time)
    return batch

# Sanity Check: Visualizing asynchronously generated data
def sanity_check(images):
    plt.figure(figsize=(8, 8))
    # Loop over generated data for visualization and checking
    for i in range(images.shape[0]):
        if i < 4 :
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.colorbar()
            plt.title(f"Generated Image {i+1}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

    
    
        
        
        
        
        
    


# ============================================================================

# Model + loss (cell 16)

# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

# ============================================================================
# NORM SWITCH — 'batch' (Hailo-deployable) | 'group' (accuracy ceiling)
# BatchNorm2d, InstanceNorm2d, LayerNorm are supported by Hailo DFC v3.31.0;
# GroupNorm is not. 'group' is preserved so the pre-Hailo model can still be
# trained as an accuracy ceiling.
# ============================================================================
NORM = 'batch'   # 'batch' | 'group'


def _make_norm(num_channels, norm_groups=6):
    if NORM == 'batch':
        return nn.BatchNorm2d(num_channels)
    if NORM == 'group':
        return nn.GroupNorm(num_groups=norm_groups, num_channels=num_channels)
    raise ValueError(f"unknown NORM={NORM!r}")


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, norm_groups=6, dilation=1):
        super().__init__()
        # bias=False because both convs are immediately followed by a norm layer
        # whose beta absorbs the effective bias.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                       padding=dilation, dilation=dilation, bias=False),
            _make_norm(out_channels, norm_groups),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                       padding=dilation, dilation=dilation, bias=False),
            _make_norm(out_channels, norm_groups),
            nn.ELU(),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, norm_groups=6):
        super().__init__()
        # No norm here — bias stays True.
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                       padding='same', bias=True),
        )

    def forward(self, x):
        return self.up(x)


class multi_head(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, norm_groups=6):
        super().__init__()
        # First conv is followed by norm -> bias=False. Final 1x1 conv has no
        # norm after it and must keep bias=True (the p-head's bias init of -8.1
        # depends on this bias existing).
        self.multi_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                       padding='same', bias=False),
            _make_norm(in_channels, norm_groups),
            nn.ELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                       padding='same', bias=True),
        )

    def forward(self, x):
        return self.multi_head(x)


class GaussianMixtureModel(nn.Module):
    def __init__(self, num_channels) -> None:
        super().__init__()
        self.num_channels = num_channels

        # =====================================================================
        # ENCODER — 3 levels (36 -> 72 -> 144) + dilated bottleneck (288)
        # Resolution for 128^2 input: 128 -> 64 -> 32 -> 16 (bottleneck)
        # =====================================================================
        self.Conv1 = conv_block(num_channels, 36, norm_groups=6)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = conv_block(36, 72, norm_groups=6)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv3 = conv_block(72, 144, norm_groups=6)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck at 16x16 (128^2 input) — dilations (2, 4).
        # Feature-map spans of 5 and 9 px, mapping to ~40 and ~72 input pixels
        # — 4-7x the PSF width, healthy without over-integrating.
        self.Conv4 = nn.Sequential(
            conv_block(144, 288, norm_groups=6, dilation=2),
            conv_block(288, 288, norm_groups=6, dilation=4),
        )

        # =====================================================================
        # DECODER — 3 upsampling levels back to full resolution (128x128)
        # =====================================================================
        self.Up3 = up_conv(288, 144, norm_groups=6)
        self.Up_conv3 = conv_block(288, 144, norm_groups=6)   # 144 skip + 144 up = 288

        self.Up2 = up_conv(144, 72, norm_groups=6)
        self.Up_conv2 = conv_block(144, 72, norm_groups=6)    # 72 skip + 72 up = 144

        self.Up1 = up_conv(72, 36, norm_groups=6)
        self.Up_conv1 = conv_block(72, 36, norm_groups=6)     # 36 skip + 36 up = 72

        # tanh curriculum: scale on the mu-offset tanh output. Kept as a plain
        # attribute (not a buffer/parameter). Applied host-side at export.
        self.tanh_scale = 6.0

        # =====================================================================
        # FIVE NAMED HEADS — split from the old 3-channel mu head so we no
        # longer need an in-place slice assignment (which exports as ScatterND).
        # =====================================================================
        self.head_p     = multi_head(36, 1, norm_groups=6)
        self.head_mu_xy = multi_head(36, 2, norm_groups=6)
        self.head_mu_a  = multi_head(36, 1, norm_groups=6)
        self.head_sigma = multi_head(36, 3, norm_groups=6)
        self.head_bg    = multi_head(36, 1, norm_groups=6)

        self.initialize_weights()

        # =================================================================
        # p-head init: bias -8.1 -> sigmoid(-8.1) ~ 0.0003, weights zero.
        # Runs AFTER initialize_weights so it's not overwritten.
        # =================================================================
        nn.init.constant_(self.head_p.multi_head[-1].bias, -8.1)
        nn.init.zeros_(self.head_p.multi_head[-1].weight)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_heads(self, x):
        """Encoder+decoder+heads, activations applied, no scalar scaling.
        Returned tensors:  p (B,1,H,W), mu_xy (B,2,H,W), mu_a (B,1,H,W),
                           sigma (B,3,H,W), bg (B,1,H,W)
        """
        # --- Encoder ---
        x1 = self.Conv1(x)                          # (B, 36, 128, 128)
        x2 = self.Conv2(self.Maxpool1(x1))          # (B, 72,  64,  64)
        x3 = self.Conv3(self.Maxpool2(x2))          # (B, 144, 32,  32)
        x4 = self.Conv4(self.Maxpool3(x3))          # (B, 288, 16,  16)

        # --- Decoder ---
        d3 = self.Up3(x4)                            # (B, 144, 32, 32)
        d3 = torch.cat((x3, d3), dim=1)              # (B, 288, 32, 32)
        d3 = self.Up_conv3(d3)                       # (B, 144, 32, 32)

        d2 = self.Up2(d3)                            # (B, 72, 64, 64)
        d2 = torch.cat((x2, d2), dim=1)              # (B, 144, 64, 64)
        d2 = self.Up_conv2(d2)                       # (B, 72, 64, 64)

        d1 = self.Up1(d2)                            # (B, 36, 128, 128)
        d1 = torch.cat((x1, d1), dim=1)              # (B, 72, 128, 128)
        d1 = self.Up_conv1(d1)                       # (B, 36, 128, 128)

        # --- Heads (activations only; scalars applied outside) ---
        p     = torch.sigmoid(self.head_p(d1))       # (B, 1, H, W)
        mu_xy = torch.tanh(self.head_mu_xy(d1))      # (B, 2, H, W)
        mu_a  = torch.sigmoid(self.head_mu_a(d1))    # (B, 1, H, W)
        sigma = torch.sigmoid(self.head_sigma(d1))   # (B, 3, H, W)
        bg    = self.head_bg(d1)                     # (B, 1, H, W), linear
        return p, mu_xy, mu_a, sigma, bg

    def forward(self, x, training=False):
        """Rebuilds the exact 4-tuple the training/eval/detection code expects,
        preserving the scalar factors on the host side (tanh_scale, x100,
        x1.4 + 0.1) via torch.cat instead of in-place slice assignment.
        """
        p, mu_xy, mu_a, sigma, bg = self.forward_heads(x)
        pxyn_mean = torch.cat([mu_xy * self.tanh_scale, mu_a * 100.0], dim=1)  # (B, 3, H, W)
        pxy_std   = sigma * 2.4 + 0.1                                          # sigma in (0.1, 2.5)
        return p, pxyn_mean, pxy_std, bg

    def forward_export(self, x):
        """Inference-only path for ONNX export. Only p and mu_xy leave the
        chip; mu_a, sigma, bg are training-only (consumed by GMMLoss).
        Scalars (tanh_scale) applied host-side.
        """
        p, mu_xy, _, _, _ = self.forward_heads(x)
        return p, mu_xy




# =========================================================================
# HELPER: get_ground_truth_targets  (unchanged)
# =========================================================================
def get_ground_truth_targets(labels, xy_coord):
    batch_size = labels.shape[0]
    targets_list = []
    for b in range(batch_size):
        presence_map = labels[b, 2, ...] > 0
        indices = torch.nonzero(presence_map, as_tuple=False)
        if len(indices) == 0:
            targets_list.append(None)
            continue
        y_idx, x_idx = indices[:, 0], indices[:, 1]
        vals = labels[b, :, y_idx, x_idx].t()
        grid_x = xy_coord[0, 0, y_idx, x_idx]
        grid_y = xy_coord[0, 1, y_idx, x_idx]
        vals[:, 0] = vals[:, 0] + grid_x
        vals[:, 1] = vals[:, 1] + grid_y
        targets_list.append(vals)
    return targets_list


# =========================================================================
# LOSS — GMMLoss with all fixes
# =========================================================================
class GMMLoss(nn.Module):
    def __init__(self, img_size, device) -> None:
        super(GMMLoss, self).__init__()
        x_coord = torch.linspace(start=0.0, end=img_size, steps=img_size + 1)[:-1].unsqueeze(0) + 0.5
        y_coord = torch.linspace(start=0.0, end=img_size, steps=img_size + 1)[:-1].unsqueeze(1) + 0.5
        self._xy_coord = torch.cat((
            x_coord.expand(1, 1, img_size, img_size),
            y_coord.expand(1, 1, img_size, img_size)
        ), 1).to(device=device)

    def forward(self, p, pxyn_mean, pxy_std, bg_pred, bg_target, labels):
        batch_size = p.shape[0]

        # ==============================================================
        # 1. COUNT LOSS — Bernoulli-Gaussian log-prob (paper style).
        #    Treats each pixel as Bernoulli(p); total count ≈ Normal(Σp, Σp(1-p)).
        #    The variance term Σp(1-p) is maximized at p=0.5 and zero at p∈{0,1},
        #    so minimizing -log_prob also drives p toward binary values for free.
        # ==============================================================
        p_sum = p.sum(dim=(-2, -1)).squeeze(-1)                                  # (B,)
        p_var = (p - p ** 2).sum(dim=(-2, -1)).squeeze(-1).clamp(min=1e-6)        # (B,)
        labels_intensity = labels[:, 2, ...] > 0
        gt_count = labels_intensity.sum(dim=(-2, -1)).float()                     # (B,)
        count_dist = torch.distributions.Normal(p_sum, p_var.sqrt())
        count_loss = -count_dist.log_prob(gt_count).mean()

        # BCE loss removed — paper-style count_loss (Bernoulli-Gaussian) above is
        # the per-pixel pressure that drives p toward binary, plus GMM matches GT.

        # ==============================================================
        # 4. GMM LOSS — full mixture (DECODE-style). Categorical over all
        #    pixels with Σp_all normalization. Per-pixel µ/σ contributions
        #    via MixtureSameFamily; the implicit bg suppression from the
        #    denominator (1/Σp_all per pixel) is preserved naturally.
        # ==============================================================
        gt_targets_list = get_ground_truth_targets(labels, self._xy_coord)
        total_gmm_log_prob = 0
        valid_batches = 0

        p_flat = p.view(batch_size, -1)
        p_normed = p_flat / (p_flat.sum(dim=1, keepdim=True) + 1e-6)

        pxyn_mean_abs = pxyn_mean.clone()
        pxyn_mean_abs[:, 0, ...] = pxyn_mean_abs[:, 0, ...] + self._xy_coord[:, 0, ...]
        pxyn_mean_abs[:, 1, ...] = pxyn_mean_abs[:, 1, ...] + self._xy_coord[:, 1, ...]
        mu_flat  = pxyn_mean_abs.view(batch_size, 3, -1).permute(0, 2, 1)
        std_flat = pxy_std.view(batch_size, 3, -1).permute(0, 2, 1)

        for b in range(batch_size):
            targets = gt_targets_list[b]
            if targets is None or targets.shape[0] == 0:
                continue
            valid_batches += 1
            mix  = torch.distributions.Categorical(probs=p_normed[b] + 1e-8)
            comp = torch.distributions.Independent(
                       torch.distributions.Normal(mu_flat[b], std_flat[b] + 1e-6), 1)
            gmm  = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
            log_probs = gmm.log_prob(targets)
            total_gmm_log_prob += log_probs.sum()

        gmm_loss = -(total_gmm_log_prob / valid_batches) if valid_batches > 0 \
                   else torch.tensor(0.0, device=p.device)

        # ---- Gentle Bayesian prior on std: prefer tight Gaussians ----
        # At std cap=5 and typical mean(pxy_std) ~3 mid-training, this adds
        # ~3 to gmm_loss (currently ~150). Small fraction (~2%) of total loss
        # so detection isn't disrupted, but every pixel gets a steady nudge
        # toward committing to tighter Gaussians where it can. Pixels with
        # accurate means CAN tighten without penalty; ring/edge pixels with
        # tanh-saturated means CAN'T (target density at GT drops if std small),
        # so they're discouraged. Doesn't change the mixture's categorical
        # structure, so overlap handling is preserved.
        STD_PENALTY_WEIGHT = 0.0
        std_per_pixel = pxy_std[:, :2, ...].mean(dim=1, keepdim=True)   # (B, 1, H, W); exclude vestigial I_VAR
        p_weighted_std = (std_per_pixel * p).sum() / (p.sum() + 1e-6)
        self.last_p_weighted_std = p_weighted_std.item()
        self.last_std_penalty_contribution = (STD_PENALTY_WEIGHT * p_weighted_std).item()
        gmm_loss = gmm_loss + STD_PENALTY_WEIGHT * p_weighted_std

        P_ENTROPY_WEIGHT = 0.0
        p_flat = p.view(p.shape[0], -1)   # (B, N)
        p_norm = p_flat / (p_flat.sum(dim=1, keepdim=True) + 1e-6)
        p_entropy = -(p_norm * torch.log(p_norm + 1e-8)).sum(dim=1)
        p_entropy_mean = p_entropy.mean()
        self.last_p_entropy = p_entropy_mean.item()
        self.last_p_entropy_contribution = (P_ENTROPY_WEIGHT * p_entropy_mean).item()
        gmm_loss = gmm_loss + P_ENTROPY_WEIGHT * p_entropy_mean

        # ==============================================================
        # 5. BG LOSS — MSE between predicted bg and the normalized clean bg.
        #    Forces the shared backbone to disentangle bg structure from
        #    particle signal; the p-head then reads cleaner features.
        # ==============================================================
        bg_loss = F.mse_loss(bg_pred, bg_target)

        return count_loss, gmm_loss, bg_loss


# ============================================================================

# Training driver (cell 18)

# ============================================================================

def get_loss_weights(epoch):
    """Phased loss weights. bg stays at 1000 throughout.
    Phase 1 (0-300):    bg dominant, others at 1/10 normal — bg head pretraining
    Phase 2 (300-600):  others grow linearly back to normal
    Phase 3+ (600+):    bg=1000 stable, others at standard values
    """
    if epoch < 300:
        return {'count': 1.0, 'gmm': 0.1, 'bg': 1000.0}
    elif epoch < 600:
        prog = (epoch - 300) / 300
        return {
            'count': 1.0 + (10.0 - 1.0) * prog,
            'gmm':   0.1 + (1.0 - 0.1) * prog,
            'bg':    1000.0,
        }
    else:
        return {'count': 10.0, 'gmm': 1.0, 'bg': 1000.0}


def get_tanh_scale(epoch):
    """Tanh curriculum: 3.0 during cloud phase (0-1500), linear 3.0->1.0
    during cosine LR phase (1500-3000), 1.0 during refinement (3000+)."""
    if epoch < 1500:
        return 1.0
    elif epoch < 3000:
        return 1.0 - 0.0 * (epoch - 1500) / 1500
    else:
        return 1.0


def visualize_result(epoch, GT, images, p, pxy_mean, pxy_std,
                     true_counts_list, predicted_counts_p_list,
                     save_image=False, output_dir=None, eval=False):
    fig_key = ['IMG',
               'LABEL_GT',
               'P_COUNT',
               'X_MEAN',
               'Y_MEAN',
               'INTENSI',
               'X_VAR',
               'Y_VAR',
               'I_VAR',
               ]

    fig_dict = {'IMG'      : images[-1].detach().cpu(),
                'LABEL_GT' : GT[-1, ...].detach().cpu(),
                'P_COUNT'  : p[-1].detach().cpu().numpy(),
                'X_MEAN'   : pxy_mean[-1, 0, ...].detach().cpu().numpy(),
                'Y_MEAN'   : pxy_mean[-1, 1, ...].detach().cpu().numpy(),
                'INTENSI'  : pxy_mean[-1, 2, ...].detach().cpu().numpy(),
                'X_VAR'    : pxy_std[-1, 0, ...].detach().cpu().numpy(),
                'Y_VAR'    : pxy_std[-1, 1, ...].detach().cpu().numpy(),
                'I_VAR'    : pxy_std[-1, 2, ...].detach().cpu().numpy(),
                }

    if not epoch % 10:
        px = 1 / plt.rcParams['figure.dpi']
        fig, axs = plt.subplots(2, 5, figsize=(2500 * px, 1000 * px))

        for i, ax in enumerate(axs.flatten()):
            if i == 9:
                if len(true_counts_list) > 0:
                    max_count = max(max(true_counts_list), max(predicted_counts_p_list)) + 1
                    true_line = [[0, max_count]] * 2
                    ax.plot(*true_line, 'r-', alpha=0.5)
                    ax.scatter(true_counts_list, predicted_counts_p_list, s=10, c="blue", alpha=0.5)
                ax.set_title("Counting Accuracy")
                ax.set_xlabel("True Counts")
                ax.set_ylabel("Predicted Counts")
                continue

            IMG = np.squeeze(fig_dict[fig_key[i]])
            a = ax.imshow(IMG, cmap='viridis' if 'VAR' in fig_key[i] else 'gray')
            ax.set_title(fig_key[i])
            fig.colorbar(a, ax=ax)

        if save_image and output_dir:
            mode = 'eval' if eval else 'train'
            save_path = f'{output_dir}/{epoch}_epoch_{mode}.png'
            plt.savefig(save_path, dpi=300)

        if not epoch % 50:
            plt.show()
        else:
            plt.close()


def evaluate(model, loss_fun, data_loader, device, epoch, output_dir):
    model.eval()
    loss = 0
    true_counts_list, predicted_counts_p_list = [], []

    with torch.no_grad():
        for images, labels, bg_target in data_loader:
            images    = images.to(device=device)
            labels    = labels.to(device=device)
            bg_target = bg_target.to(device=device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                p, pxy_mean, pxy_std, bg_pred = model(images, training=False)

            # Cast bf16 model outputs back to fp32 for stable loss computation
            p        = p.float()
            pxy_mean = pxy_mean.float()
            pxy_std  = pxy_std.float()
            bg_pred  = bg_pred.float()

            # ---- 4 loss components (bg head reinstated) ----
            count_loss, gmm_loss, bg_loss = loss_fun(
                p, pxy_mean, pxy_std, bg_pred, bg_target, labels)
            _w = get_loss_weights(epoch)
            loss += (_w['count'] * count_loss + _w['gmm'] * gmm_loss + _w['bg'] * bg_loss).mean()

            for GT_single, p_single in zip(labels[:, -1, ...] > 0, p):
                true_counts = torch.sum(GT_single).item()
                predicted_counts_p = torch.sum(p_single).item()
                true_counts_list.append(true_counts)
                predicted_counts_p_list.append(predicted_counts_p)

        GT_batch = labels[:, -1, ...] > 0
        visualize_result(epoch, GT_batch, images, p, pxy_mean, pxy_std,
                         true_counts_list, predicted_counts_p_list,
                         save_image=True, output_dir=output_dir, eval=True)

    return loss


def _unwrapped_state_dict(model):
    """Return a state_dict whose keys match the original (uncompiled) model.
    torch.compile wraps the model and prefixes keys with "_orig_mod.".
    Stripping that prefix keeps saved checkpoints loadable into a fresh
    (non-compiled) GaussianMixtureModel at inference time."""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    return model.state_dict()


def train(model, num_epoch, start_epoch, optimizer, scheduler, loss_fun,
          device, writer, output_dir, bg_tiles, window_size,
          bg_tiles_eval=None):
    # bg_tiles_eval is the held-out FOV pool; falls back to bg_tiles if
    # not supplied (legacy single-pool behavior).
    if bg_tiles_eval is None:
        bg_tiles_eval = bg_tiles
    model.to(device=device)

    train_loss = torch.zeros(num_epoch)
    test_loss  = torch.zeros(num_epoch)

    best_loss = np.inf

    # --- BATCH SIZE ---
    # Bumped to 32 with accumulation_steps=1 (effective batch = 32, same as
    # the old 8*4=32 configuration but no accumulation = far fewer kernel
    # launches and better GPU pipeline utilization on an A6000-class card.
    # LR unchanged because effective batch is identical.
    actual_dataloader_batch_size = 16
    accumulation_steps = 4  # effective batch = 32

    num_samples_train = 600
    num_samples_eval  = 200

    print(f"Training Config: Image {window_size}, "
          f"Effective Batch {actual_dataloader_batch_size * accumulation_steps}")

    pbar = tqdm(range(start_epoch, num_epoch), total=num_epoch, initial=start_epoch)

    for epoch in pbar:
        epoch_start_time = time.time()
        model.train()

        # ---- per-epoch curriculum (tanh + loss weights) ----
        _ts = get_tanh_scale(epoch)
        _w  = get_loss_weights(epoch)
        (model._orig_mod if hasattr(model, '_orig_mod') else model).tanh_scale = _ts

        batch_loss_list = []
        count_loss_list, gmm_loss_list, bg_loss_list = [], [], []
        p_entropy_list, std_pen_list = [], []
        true_counts_list, predicted_counts_p_list = [], []

        train_dataset = GaussianDataset(num_batches=num_samples_train,
                                         bg_tiles=bg_tiles, img_size=window_size)
        train_dataloader = DataLoader(train_dataset,
                                       batch_size=actual_dataloader_batch_size,
                                       shuffle=True, num_workers=0, pin_memory=True)

        optimizer.zero_grad()

        for i, (images, labels, bg_target) in enumerate(train_dataloader):
            images    = images.to(device=device)
            labels    = labels.to(device=device)
            bg_target = bg_target.to(device=device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                p, pxy_mean, pxy_std, bg_pred = model(images, training=True)

            # Cast bf16 model outputs back to fp32 for stable loss computation
            p        = p.float()
            pxy_mean = pxy_mean.float()
            pxy_std  = pxy_std.float()
            bg_pred  = bg_pred.float()

            # ---- 4 loss components (bg head reinstated) ----
            count_loss, gmm_loss, bg_loss = loss_fun(
                p, pxy_mean, pxy_std, bg_pred, bg_target, labels)

            # ==============================================================
            #   count_loss (MSE):   ×0.5
            #   gmm_loss (NLL):     ×1.0
            #   bg_loss (MSE):      ×1.0 — auxiliary, supervises shared backbone
            #                      to separate bg structure from particle signal.
            # ==============================================================
            loss = (_w['count'] * count_loss) + (_w['gmm'] * gmm_loss) + (_w['bg'] * bg_loss)

            loss_item = loss.item()

            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                optimizer.step()
                optimizer.zero_grad()

            batch_loss_list.append(loss_item)
            count_loss_list.append(count_loss.item())
            gmm_loss_list.append(gmm_loss.item())
            bg_loss_list.append(bg_loss.item())
            p_entropy_list.append(loss_fun.last_p_entropy)
            std_pen_list.append(loss_fun.last_p_weighted_std)

            for GT_single, p_val in zip(labels[:, -1, ...] > 0, p):
                true_counts_list.append(torch.sum(GT_single).item())
                predicted_counts_p_list.append(torch.sum(p_val).item())

        if (len(train_dataloader) % accumulation_steps != 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()
            optimizer.zero_grad()

        # Visualize
        GT_batch = labels[:, -1, ...] > 0
        visualize_result(epoch, GT_batch, images, p, pxy_mean, pxy_std,
                         true_counts_list, predicted_counts_p_list,
                         save_image=True, output_dir=output_dir, eval=False)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        train_loss[epoch] = np.mean(batch_loss_list)

        # --- EVALUATION ---
        eval_dataset = GaussianDataset(num_batches=num_samples_eval,
                                        bg_tiles=bg_tiles_eval, img_size=window_size)
        eval_dataloader = DataLoader(eval_dataset,
                                      batch_size=actual_dataloader_batch_size,
                                      shuffle=True, num_workers=0, pin_memory=True)

        loss_test = evaluate(model=model, loss_fun=loss_fun,
                              data_loader=eval_dataloader, device=device,
                              epoch=epoch, output_dir=output_dir)
        test_loss[epoch] = loss_test.item()

        if writer:
            writer.add_scalar('Loss/Count',      np.mean(count_loss_list), epoch + 1)
            writer.add_scalar('Loss/GMM',        np.mean(gmm_loss_list),   epoch + 1)
            writer.add_scalar('Loss/BG',         np.mean(bg_loss_list),    epoch + 1)
            writer.add_scalar('Loss/PEntropy',   np.mean(p_entropy_list),  epoch + 1)  # spread of p (lower = more concentrated)
            writer.add_scalar('Loss/StdPenalty', np.mean(std_pen_list),    epoch + 1)  # p-weighted mean std (lower = tighter Gaussians at firing pixels)
            writer.add_scalar('Loss/Train',      train_loss[epoch],        epoch + 1)
            writer.add_scalar('Loss/Test',       test_loss[epoch],         epoch + 1)
            writer.add_scalar('LR',              lr,                       epoch + 1)
            writer.add_scalar('TanhScale',       _ts,                      epoch + 1)
            writer.add_scalar('Weights/count',   _w['count'],              epoch + 1)
            writer.add_scalar('Weights/gmm',     _w['gmm'],                epoch + 1)
            writer.add_scalar('Weights/bg',      _w['bg'],                 epoch + 1)

        pbar.set_description(
            f"Epoch {epoch+1} | Loss: {np.mean(batch_loss_list):.4f} "
            f"| Cnt: {np.mean(count_loss_list):.1f} "
            f"| GMM: {np.mean(gmm_loss_list):.1f} "
            f"| BG: {np.mean(bg_loss_list):.3f} "
            f"| Test: {loss_test.item():.4f}")

        # --- SAVING LOGIC ---
        if loss_test < best_loss:
            best_loss = loss_test
            torch.save(_unwrapped_state_dict(model), f'{output_dir}/LOCA_PRAM_BEST.pth')

        if (epoch + 1) % 20 == 0:
            chk_name = f'{output_dir}/LOCA_PRAM_checkpoint_epoch_{epoch+1}.pth'
            torch.save(_unwrapped_state_dict(model), chk_name)

    return train_loss, test_loss, model


# ============================================================================
# LR schedule — proportional to --max-epoch
# ============================================================================
def build_lr_lambda(max_epoch, warmup_epochs, flat_frac, tail_min_frac):
    """Warmup -> flat -> cosine -> min-frac.
        [0, warmup_epochs)           linear 0 -> 1.0
        [warmup_epochs, flat_end)    flat at 1.0
        [flat_end, max_epoch)        cosine 1.0 -> tail_min_frac
        [max_epoch, inf)             tail_min_frac
    flat_end = warmup_epochs + int((max_epoch - warmup_epochs) * flat_frac)
    """
    flat_end = warmup_epochs + int((max_epoch - warmup_epochs) * flat_frac)
    cosine_end = max_epoch

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if epoch < flat_end:
            return 1.0
        if epoch < cosine_end:
            tail_progress = (epoch - flat_end) / (cosine_end - flat_end)
            return tail_min_frac + (1.0 - tail_min_frac) * 0.5 * (1 + math.cos(math.pi * tail_progress))
        return tail_min_frac
    print(f"lr schedule: warmup(0-{warmup_epochs}) -> flat(-{flat_end}) -> cosine(-{cosine_end}) -> tail@{tail_min_frac}")
    return lr_lambda



# ============================================================================
# CLI
# ============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="LOCA-PRAM training (v9 batchnorm) — all knobs CLI-configurable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- MODEL OUTPUT ---
    p.add_argument("--tanh-scale", type=float, default=6.0,
                   help="Scalar multiplier on the mu_xy tanh output. Set at model "
                        "construction and NOT annealed per-epoch.")

    # --- PSF GEOMETRY (native px, divided by --downscale-factor internally) ---
    p.add_argument("--downscale-factor", type=int, default=2)
    p.add_argument("--sigma-x-mean", type=float, default=9.79)
    p.add_argument("--sigma-x-std",  type=float, default=1.32)
    p.add_argument("--sigma-y-mean", type=float, default=7.88)
    p.add_argument("--sigma-y-std",  type=float, default=1.38)
    p.add_argument("--sigma-min",    type=float, default=2.5)

    # --- AUGMENTATION ---
    p.add_argument("--defocus-sharp-prob", type=float, default=0.55,
                   help="P(no defocus per tile); remainder sampled uniform in "
                        "[--defocus-blur-min, --defocus-blur-max].")
    p.add_argument("--defocus-blur-min", type=float, default=1.3)
    p.add_argument("--defocus-blur-max", type=float, default=1.8)
    p.add_argument("--motion-length-max", type=float, default=1.5,
                   help="Max motion-blur length in native px; 0 disables.")
    p.add_argument("--empty-tile-prob", type=float, default=0.15)

    # --- AMPLITUDE / DENSITY / PLACEMENT ---
    p.add_argument("--amp-floor", type=float, default=12.0,
                   help="Lower bound of the Beta amplitude distribution.")
    p.add_argument("--particles-per-tile-min", type=int, default=0)
    p.add_argument("--particles-per-tile-max", type=int, default=40)
    p.add_argument("--min-separation", type=float, default=5.0,
                   help="Rejection-sampling min separation (native px).")

    # --- MARKERS ---
    p.add_argument("--marker-threshold", type=int, default=15)
    p.add_argument("--buffer-sigma-mult", type=float, default=5.0)

    # --- TRAINING ---
    p.add_argument("--max-epoch", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--bg-folder", default="demo_0001/background_defect/")
    p.add_argument("--train-fov-count", type=int, default=104)
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint .pth to resume from.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write checkpoints and TB logs. Auto-timestamped "
                        "under runs/ if omitted.")

    # --- LR SCHEDULE ---
    p.add_argument("--warmup-epochs", type=int, default=50)
    p.add_argument("--flat-frac", type=float, default=0.4,
                   help="Fraction of (max_epoch - warmup) spent at the flat LR.")
    p.add_argument("--tail-min-frac", type=float, default=0.05,
                   help="Min LR at end, as fraction of peak LR.")

    return p.parse_args(argv)



# ============================================================================
# main() — orchestrates: override globals from CLI, load BG, build model,
# train.
# ============================================================================
def main(args):
    # ---- 1. Override module globals from CLI (must happen BEFORE any function
    # or dataset worker reads them). ----
    global DOWNSCALE_FACTOR, BASE_WINDOW_SIZE, window_size
    global BASE_SIGMA_X_LOC, BASE_SIGMA_X_SCALE, BASE_SIGMA_Y_LOC, BASE_SIGMA_Y_SCALE
    global BASE_SIGMA_MIN, BASE_MARGIN, BUFFER_SIGMA_MULT
    global SIGMA_X_LOC, SIGMA_X_SCALE, SIGMA_Y_LOC, SIGMA_Y_SCALE, SIGMA_MIN
    global MARGIN, BUFFER_PX
    global MARKER_THRESHOLD, AMP_FLOOR
    global DEFOCUS_SHARP_PROB, DEFOCUS_BLUR_MIN, DEFOCUS_BLUR_MAX
    global MOTION_LENGTH_MAX, EMPTY_TILE_PROB, MIN_SEPARATION
    global PARTICLES_MIN, PARTICLES_MAX

    DOWNSCALE_FACTOR    = args.downscale_factor
    BASE_SIGMA_X_LOC    = args.sigma_x_mean
    BASE_SIGMA_X_SCALE  = args.sigma_x_std
    BASE_SIGMA_Y_LOC    = args.sigma_y_mean
    BASE_SIGMA_Y_SCALE  = args.sigma_y_std
    BASE_SIGMA_MIN      = args.sigma_min
    BUFFER_SIGMA_MULT   = args.buffer_sigma_mult
    MARKER_THRESHOLD    = args.marker_threshold
    AMP_FLOOR           = args.amp_floor
    DEFOCUS_SHARP_PROB  = args.defocus_sharp_prob
    DEFOCUS_BLUR_MIN    = args.defocus_blur_min
    DEFOCUS_BLUR_MAX    = args.defocus_blur_max
    MOTION_LENGTH_MAX   = args.motion_length_max
    EMPTY_TILE_PROB     = args.empty_tile_prob
    MIN_SEPARATION      = args.min_separation
    PARTICLES_MIN       = args.particles_per_tile_min
    PARTICLES_MAX       = args.particles_per_tile_max

    # Recompute derived constants
    window_size   = (BASE_WINDOW_SIZE[0] // DOWNSCALE_FACTOR,
                     BASE_WINDOW_SIZE[1] // DOWNSCALE_FACTOR)
    SIGMA_X_LOC   = BASE_SIGMA_X_LOC   / DOWNSCALE_FACTOR
    SIGMA_X_SCALE = BASE_SIGMA_X_SCALE / DOWNSCALE_FACTOR
    SIGMA_Y_LOC   = BASE_SIGMA_Y_LOC   / DOWNSCALE_FACTOR
    SIGMA_Y_SCALE = BASE_SIGMA_Y_SCALE / DOWNSCALE_FACTOR
    SIGMA_MIN     = BASE_SIGMA_MIN     / DOWNSCALE_FACTOR
    MARGIN        = max(1, int(round(BASE_MARGIN / DOWNSCALE_FACTOR)))
    BUFFER_PX     = int(round(BUFFER_SIGMA_MULT * max(SIGMA_X_LOC, SIGMA_Y_LOC)))

    print("=== CONFIG ===")
    print(f"  tanh_scale        = {args.tanh_scale}")
    print(f"  window_size       = {window_size}  (native {BASE_WINDOW_SIZE}, downscale x{DOWNSCALE_FACTOR})")
    print(f"  sigma_x model-px  = N({SIGMA_X_LOC:.2f}, {SIGMA_X_SCALE:.2f})")
    print(f"  sigma_y model-px  = N({SIGMA_Y_LOC:.2f}, {SIGMA_Y_SCALE:.2f})")
    print(f"  defocus           = sharp {DEFOCUS_SHARP_PROB:.0%} else U[{DEFOCUS_BLUR_MIN}, {DEFOCUS_BLUR_MAX}]")
    print(f"  motion_max        = {MOTION_LENGTH_MAX}")
    print(f"  empty_tile_prob   = {EMPTY_TILE_PROB}")
    print(f"  min_separation    = {MIN_SEPARATION} native px")
    print(f"  particles/tile    = [{PARTICLES_MIN}, {PARTICLES_MAX})")
    print(f"  amp_floor         = {AMP_FLOOR}")
    print(f"  marker_threshold  = {MARKER_THRESHOLD}, buffer_px = {BUFFER_PX}")

    # ---- 2. Output dir ----
    ts = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_dir = args.output_dir or f"runs/tanh_{args.tanh_scale}_{ts}/"
    if not output_dir.endswith("/"):
        output_dir += "/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  output_dir        = {output_dir}")

    # Dump the full CLI config to the output dir for reproducibility
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        import json as _json
        _json.dump(vars(args), f, indent=2, sort_keys=True)

    # ---- 3. Torch / device ----
    Distribution.set_default_validate_args(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device            = {device}")

    # ---- 4. Load BG tiles ----
    bg_tiles_train, bg_tiles_eval = load_bg_tiles(
        args.bg_folder, args.train_fov_count)

    # ---- 5. Build model with the correct tanh_scale ----
    model = GaussianMixtureModel(num_channels=1)
    # Set tanh_scale ONCE. The training loop (get_tanh_scale) has been
    # neutralised so it won't override this.
    model.tanh_scale = args.tanh_scale
    model.to(device)
    print(f"  model.tanh_scale  = {model.tanh_scale}")

    loss_fun = GMMLoss(img_size=window_size[0], device=device)
    writer = SummaryWriter(output_dir, flush_secs=10)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    lr_lambda = build_lr_lambda(
        args.max_epoch, args.warmup_epochs,
        args.flat_frac, args.tail_min_frac)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- 6. Optional resume ----
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        try:
            filename = os.path.basename(args.resume)
            raw_epoch_part = filename.split("_")[-1]
            epoch_str = raw_epoch_part.split(".")[0]
            start_epoch = int(epoch_str) + 1
            print(f"Resuming from epoch: {start_epoch}")
        except (IndexError, ValueError):
            print("Could not parse epoch number from filename. Starting from 0.")
        # Fast-forward scheduler
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        # Re-pin tanh_scale after loading (state_dict doesn't contain it)
        (model._orig_mod if hasattr(model, "_orig_mod") else model).tanh_scale = args.tanh_scale
    else:
        print("Training from scratch.")

    # Quick sanity check
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 1, window_size[0], window_size[1]).to(device)
        p_init, _, _, _ = model(dummy)
        print(f"Initial p.sum() = {p_init.sum().item():.1f}  (target range: 0-7)")
    model.train()

    print("Starting training...")
    train_loss, test_loss, model = train(
        model=model,
        num_epoch=args.max_epoch,
        start_epoch=start_epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fun=loss_fun,
        device=device,
        writer=writer,
        output_dir=output_dir,
        bg_tiles=bg_tiles_train,
        bg_tiles_eval=bg_tiles_eval,
        window_size=window_size,
    )

    # Dump loss arrays for downstream plotting
    np.save(os.path.join(output_dir, "train_loss.npy"), np.asarray(train_loss))
    np.save(os.path.join(output_dir, "test_loss.npy"),  np.asarray(test_loss))

    # Final loss plot (saved, not shown)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_loss, "s-", label="Train")
    ax.plot(test_loss,  "o-", label="Test")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title(f"Model loss (final test: {test_loss[-1]:.2f}, tanh_scale={args.tanh_scale})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=120)
    plt.close(fig)

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main(parse_args())
