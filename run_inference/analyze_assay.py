"""Standalone LOCA-PRAM assay analyzer.

Two run modes:

1) Single assay (existing):
       python analyze_assay.py --data-dir <one_assay_folder> --area 1.2
   Writes into `<data-dir>/analysis/` — per_tile.csv, summary_per_cycle.csv,
   counts_over_time.png, per_tile_over_time.png, boxplot_over_time.png,
   spatial_heatmap_last_cycle.png, per_tile_over_time.html,
   per_tile_density_over_time.html, and cycle_XXXX/<basename>.png per image.

2) Batch across a root folder of assays (new):
       python analyze_assay.py --assays-root <parent> --area 1.2 \
                                [--top-k K] [--middle-k K]
   Iterates every direct subfolder that contains cycle_*/demo_* dirs.
   Skips inference for assays that already have `analysis/per_tile.csv`
   (use --force to re-run). --top-k picks the K highest-count tiles per
   cycle; --middle-k picks the K tiles centered on the median. Either can
   be set (or both — they're independent). Per assay, each writes:
     <assay>/analysis/summary_per_cycle_{topK,middleK}.csv
     <assay>/analysis/counts_over_time_{topK,middleK}.png
     <assay>/analysis/boxplot_over_time_{topK,middleK}.png
   When any assay folder name starts with `NNN.N` (concentration),
   cross-assay concentration aggregation lands in `<assays-root>/analysis/`.
   Tiles are pooled across every folder at a concentration BEFORE any
   selection, so --top-k / --middle-k at concentration level pick K across
   the full pool, not per-folder K. For each of full / topK / middleK:
     per_concentration_per_cycle[_topK,_middleK].csv       summary stats
     counts_over_time_by_concentration[_topK,_middleK].png mean+ribbon lines
     boxplot_by_concentration[_topK,_middleK].png          grouped box per
                                                           (cycle, conc)
                                                           to show range

Detection settings (model, threshold, NMS kernel, forbidden mask, dedup)
match `LOCA_PRAM_batch_assays_eval.ipynb` exactly, so counts are numerically
identical to that notebook at the same defaults.

`plotly` is optional — install it (`pip install plotly`) to also get the
interactive HTML plots; without it, only the PNGs are written.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# Model architecture — v9 batchnorm variant with split heads
# (mirrors NORM='batch' cell in LOCA_PRAM_main_downsampled.ipynb)
#
# Key differences from the pre-v9 GroupNorm build:
#   - BatchNorm2d in every conv_block / multi_head norm slot (switchable via
#     NORM); conv layers before a norm now use bias=False.
#   - The old 3-channel mu head is split into head_mu_xy (2ch, tanh-scaled)
#     and head_mu_a (1ch, sigmoid*100). p, sigma, bg are also named heads.
#   - tanh_scale = 6.0 (final training value; was 1.0 pre-v9).
#   - pxy_std = sigmoid(sigma) * 2.4 + 0.1  (was * 1.4 + 0.1).
#
# forward() still returns the same (p, pxyn_mean, pxy_std, bg) 4-tuple, so
# downstream inference code (infer_positions, detect_via_nms_xy, ...) is
# unchanged.
# =============================================================================
NORM = "batch"   # "batch" | "group"


def _make_norm(num_channels, norm_groups=6):
    if NORM == "batch":
        return nn.BatchNorm2d(num_channels)
    if NORM == "group":
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
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                       padding="same", bias=True),
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
                       padding="same", bias=False),
            _make_norm(in_channels, norm_groups),
            nn.ELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                       padding="same", bias=True),
        )

    def forward(self, x):
        return self.multi_head(x)


class GaussianMixtureModel(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

        # --- Encoder: 3 levels + dilated bottleneck ---
        self.Conv1 = conv_block(num_channels, 36, norm_groups=6)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = conv_block(36, 72, norm_groups=6)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = conv_block(72, 144, norm_groups=6)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv4 = nn.Sequential(
            conv_block(144, 288, norm_groups=6, dilation=2),
            conv_block(288, 288, norm_groups=6, dilation=4),
        )

        # --- Decoder: 3 upsampling levels back to full resolution ---
        self.Up3 = up_conv(288, 144, norm_groups=6)
        self.Up_conv3 = conv_block(288, 144, norm_groups=6)
        self.Up2 = up_conv(144, 72, norm_groups=6)
        self.Up_conv2 = conv_block(144, 72, norm_groups=6)
        self.Up1 = up_conv(72, 36, norm_groups=6)
        self.Up_conv1 = conv_block(72, 36, norm_groups=6)

        # Applied host-side to mu_xy at inference. Set to the final training
        # value (6.0 for v9). Not a buffer — not in state_dict.
        self.tanh_scale = 6.0

        # Five named heads (split from the old 3-channel mu head).
        self.head_p     = multi_head(36, 1, norm_groups=6)
        self.head_mu_xy = multi_head(36, 2, norm_groups=6)
        self.head_mu_a  = multi_head(36, 1, norm_groups=6)
        self.head_sigma = multi_head(36, 3, norm_groups=6)
        self.head_bg    = multi_head(36, 1, norm_groups=6)

        self.initialize_weights()

        # p-head init: bias -8.1 -> sigmoid ~ 0.0003. Overwritten by the
        # checkpoint at load time anyway.
        nn.init.constant_(self.head_p.multi_head[-1].bias, -8.1)
        nn.init.zeros_(self.head_p.multi_head[-1].weight)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_heads(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool1(x1))
        x3 = self.Conv3(self.Maxpool2(x2))
        x4 = self.Conv4(self.Maxpool3(x3))
        d3 = self.Up3(x4);  d3 = torch.cat((x3, d3), dim=1); d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3);  d2 = torch.cat((x2, d2), dim=1); d2 = self.Up_conv2(d2)
        d1 = self.Up1(d2);  d1 = torch.cat((x1, d1), dim=1); d1 = self.Up_conv1(d1)
        p     = torch.sigmoid(self.head_p(d1))
        mu_xy = torch.tanh(self.head_mu_xy(d1))
        mu_a  = torch.sigmoid(self.head_mu_a(d1))
        sigma = torch.sigmoid(self.head_sigma(d1))
        bg    = self.head_bg(d1)
        return p, mu_xy, mu_a, sigma, bg

    def forward(self, x, training=False):
        p, mu_xy, mu_a, sigma, bg = self.forward_heads(x)
        pxyn_mean = torch.cat([mu_xy * self.tanh_scale, mu_a * 100.0], dim=1)
        pxy_std = sigma * 2.4 + 0.1
        return p, pxyn_mean, pxy_std, bg


# =============================================================================
# Pipeline constants (verbatim from LOCA_PRAM_batch_assays_eval.ipynb)
# =============================================================================
DOWNSCALE_FACTOR = 2
BASE_WINDOW_SIZE = (256, 256)
WINDOW_SIZE = (BASE_WINDOW_SIZE[0] // DOWNSCALE_FACTOR,
               BASE_WINDOW_SIZE[1] // DOWNSCALE_FACTOR)
NATIVE_WINDOW = (WINDOW_SIZE[0] * DOWNSCALE_FACTOR,
                 WINDOW_SIZE[1] * DOWNSCALE_FACTOR)

MARKER_THRESHOLD = 15
BASE_SIGMA_X_LOC = 9.79
BASE_SIGMA_Y_LOC = 7.88
BASE_MARGIN = 12
SIGMA_X_LOC = BASE_SIGMA_X_LOC / DOWNSCALE_FACTOR
SIGMA_Y_LOC = BASE_SIGMA_Y_LOC / DOWNSCALE_FACTOR
MARGIN = max(1, int(round(BASE_MARGIN / DOWNSCALE_FACTOR)))
BUFFER_SIGMA_MULT = 5.0
BUFFER_PX = int(round(BUFFER_SIGMA_MULT * max(SIGMA_X_LOC, SIGMA_Y_LOC)))
EDGE_MARGIN_TEST = 6

MIN_AREA = 1
DEFAULT_THRESHOLD = 0.006
DEFAULT_NMS_KERNEL = 7
OVERLAP_NATIVE = 2 * EDGE_MARGIN_TEST * DOWNSCALE_FACTOR
DEDUP_RADIUS_NATIVE = 3

DEMO_GLOBS = ("cycle_*", "demo_*")


# =============================================================================
# Inference helpers (verbatim from LOCA_PRAM_batch_assays_eval.ipynb)
# =============================================================================
def load_real_native(path):
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def downsample_image(img, factor=DOWNSCALE_FACTOR):
    if factor == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def normalize_tile(image):
    valid = image > MARKER_THRESHOLD
    if valid.any():
        m, s = image[valid].mean(), image[valid].std()
    else:
        m, s = image.mean(), image.std()
    if s == 0:
        s = 1
    return (image - m) / s


def compute_forbidden_mask_fast(image, buffer_px=BUFFER_PX,
                                threshold=MARKER_THRESHOLD,
                                margin=EDGE_MARGIN_TEST):
    H, W = image.shape
    non_marker = (image > threshold).astype(np.uint8)
    if non_marker.all():
        forbidden = np.zeros((H, W), dtype=bool)
    else:
        dist = cv2.distanceTransform(non_marker, cv2.DIST_L2, 3)
        forbidden = dist <= buffer_px
    if margin > 0:
        forbidden[:margin, :] = True
        forbidden[-margin:, :] = True
        forbidden[:, :margin] = True
        forbidden[:, -margin:] = True
    return forbidden


def detect_via_nms_xy(p_map_np, mu_map_np, forbidden_mask, p_threshold,
                      nms_kernel=5, refine_subpixel=True):
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
        rx = xi.astype(np.float64)
        ry = yi.astype(np.float64)
    return rx, ry


def tile_starts(total, tile_size, stride):
    if total <= tile_size:
        return [0]
    starts = list(range(0, total - tile_size + 1, stride))
    if starts[-1] + tile_size < total:
        starts.append(total - tile_size)
    return starts


def dedup_positions(positions, radius=DEDUP_RADIUS_NATIVE):
    if len(positions) <= 1:
        return np.asarray(positions)
    from scipy.spatial import cKDTree
    pts = np.asarray(positions)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(radius)
    drop = set()
    for i_, j_ in sorted(pairs):
        if i_ in drop or j_ in drop:
            continue
        drop.add(j_)
    keep = [i_ for i_ in range(len(pts)) if i_ not in drop]
    return pts[keep]


def infer_positions(model, device, path, threshold, nms_kernel,
                    overlap_native=OVERLAP_NATIVE,
                    dedup_radius=DEDUP_RADIUS_NATIVE):
    """One inference pass returning (native_image, cx, cy). Same tiled +
    edge-rejected + cross-tile-deduped path as the notebook's
    `detect_for_viz`; identical numeric behavior at matching (threshold,
    nms_kernel)."""
    img = load_real_native(path)
    h, w = img.shape
    stride_y = NATIVE_WINDOW[0] - overlap_native
    stride_x = NATIVE_WINDOW[1] - overlap_native
    starts_y = tile_starts(h, NATIVE_WINDOW[0], stride_y)
    starts_x = tile_starts(w, NATIVE_WINDOW[1], stride_x)

    all_x, all_y = [], []
    with torch.no_grad():
        for iy in starts_y:
            for ix in starts_x:
                native_tile = img[iy:iy + NATIVE_WINDOW[0],
                                  ix:ix + NATIVE_WINDOW[1]]
                raw = downsample_image(native_tile)
                norm = normalize_tile(raw)
                forbidden = compute_forbidden_mask_fast(raw)
                x_in = (torch.from_numpy(norm).float()
                        .unsqueeze(0).unsqueeze(0).to(device))
                p, pxy_mean, _, _ = model(x_in, training=False)
                p_map = p[0, 0].cpu().numpy()
                mu_map = pxy_mean[0, :2].cpu().numpy()
                cx128, cy128 = detect_via_nms_xy(
                    p_map, mu_map, forbidden, threshold, nms_kernel=nms_kernel)
                if len(cx128):
                    all_x.append(cx128 * DOWNSCALE_FACTOR + ix)
                    all_y.append(cy128 * DOWNSCALE_FACTOR + iy)

    if not all_x:
        return img, np.array([]), np.array([])
    pts = np.stack([np.concatenate(all_x), np.concatenate(all_y)], axis=1)
    deduped = dedup_positions(pts, radius=dedup_radius)
    return img, deduped[:, 0], deduped[:, 1]


# =============================================================================
# Filename parsers
# =============================================================================
_TILE_RE = re.compile(r"tile_(\d+)_(\d+)_", re.IGNORECASE)
_CYCLE_RE = re.compile(r"(?:cycle|demo)_(\d+)", re.IGNORECASE)
_CONC_RE = re.compile(r"^(\d+\.\d+)")


def parse_tile_pos(filename):
    m = _TILE_RE.match(os.path.basename(filename))
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_cycle_num(demo_name):
    m = _CYCLE_RE.match(os.path.basename(demo_name))
    return int(m.group(1)) if m else None


def parse_concentration(assay_name):
    """`006.0_batch_A` -> '006.0'. Returns None if the folder name doesn't
    start with a decimal number."""
    m = _CONC_RE.match(os.path.basename(assay_name))
    return m.group(1) if m else None


# =============================================================================
# Rendering
# =============================================================================
def render_annotated(img, cx, cy, count, out_path,
                     marker_px=24, marker_lw=2):
    """PNG of `img` with detections X-marked and a count box at top-center."""
    p_lo, p_hi = np.percentile(img, [1, 99])
    norm = np.clip((img - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)
    bgr = cv2.cvtColor((norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for x, y in zip(cx, cy):
        cv2.drawMarker(bgr, (int(round(x)), int(round(y))),
                       color=(0, 0, 255),
                       markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=marker_px, thickness=marker_lw)

    text = f"detections: {int(count)}"
    H, W = bgr.shape[:2]
    scale = max(0.8, min(3.0, W / 900.0))
    thickness = max(2, int(round(scale * 1.5)))
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(6, int(round(scale * 8)))
    x0 = (W - tw) // 2
    y0 = pad + th + pad
    cv2.rectangle(bgr,
                  (x0 - pad, y0 - th - pad),
                  (x0 + tw + pad, y0 + baseline + pad),
                  (0, 0, 0), -1)
    cv2.putText(bgr, text, (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)

    cv2.imwrite(str(out_path), bgr)


# =============================================================================
# Analysis — baseline
# =============================================================================
def build_per_tile_and_summary(rows_df, area_per_tile, area_units,
                               minutes_per_cycle):
    """From the one-row-per-image inference table, build:
       - per_tile: one row per (cycle, tile). Duplicate captures at the
         same (cycle, tile) are collapsed to their mean.
       - summary: one row per cycle. Aggregates the per_tile table.
    """
    df = rows_df.copy()
    df["cycle"] = df["demo"].apply(parse_cycle_num)
    df = df.dropna(subset=["cycle"])
    df["cycle"] = df["cycle"].astype(int)
    df["t_min"] = (df["cycle"] - 1) * minutes_per_cycle

    positions = df["image"].apply(parse_tile_pos)
    tile_matched = positions.notna().sum()
    if tile_matched > 0:
        df["tile_row"] = positions.apply(lambda p: p[0] if p else None)
        df["tile_col"] = positions.apply(lambda p: p[1] if p else None)
        df = df.dropna(subset=["tile_row", "tile_col"])
        df["tile_row"] = df["tile_row"].astype(int)
        df["tile_col"] = df["tile_col"].astype(int)
        df["tile"] = df.apply(lambda r: f"{r.tile_row},{r.tile_col}", axis=1)
        tile_source = f"parsed from filename ({tile_matched}/{len(rows_df)})"
    else:
        df = df.sort_values(["cycle", "image"]).reset_index(drop=True)
        df["tile"] = "ord_" + df.groupby("cycle").cumcount().astype(str)
        df["tile_row"] = np.nan
        df["tile_col"] = np.nan
        tile_source = "ordinal within cycle (no `tile_r_c_` filenames found)"

    per_tile = (df.groupby(["cycle", "t_min", "tile", "tile_row", "tile_col"],
                           dropna=False, as_index=False)
                  .agg(n_images=("n_detections", "size"),
                       n_detections=("n_detections", "mean")))
    per_tile["density"] = per_tile["n_detections"] / area_per_tile
    per_tile["area_per_tile"] = area_per_tile
    per_tile["area_units"] = area_units
    per_tile = per_tile.sort_values(["cycle", "tile_row", "tile_col", "tile"]) \
                       .reset_index(drop=True)

    summary = (per_tile.groupby(["cycle", "t_min"], as_index=False)
                       .agg(n_tiles=("tile", "nunique"),
                            n_images=("n_images", "sum"),
                            detections_sum=("n_detections", "sum"),
                            detections_mean=("n_detections", "mean"),
                            detections_std=("n_detections", "std"),
                            density_mean=("density", "mean"),
                            density_std=("density", "std")))
    summary["area_per_tile"] = area_per_tile
    summary["area_units"] = area_units
    summary = summary.sort_values("cycle").reset_index(drop=True)

    return per_tile, summary, tile_source


# =============================================================================
# Analysis — top-K
# =============================================================================
def top_k_per_cycle(per_tile, k):
    """Take the top K tiles by n_detections at each cycle. Returns a df with
    the same columns as per_tile. When a cycle has fewer than K tiles, all
    of them are kept (`n_tiles_used` in the summary records the actual K)."""
    if k is None or len(per_tile) == 0:
        return per_tile.iloc[0:0].copy()
    picks = []
    for cyc, g in per_tile.groupby("cycle"):
        picks.append(g.nlargest(min(k, len(g)), "n_detections"))
    return pd.concat(picks, ignore_index=True) if picks else per_tile.iloc[0:0].copy()


def middle_k_per_cycle(per_tile, k):
    """Take the middle K tiles by n_detections at each cycle (centered on
    the median). Filters out both the low tail (imaging failures / empty
    tiles) and the high tail (unusually hot tiles). When a cycle has
    fewer than K tiles, all of them are kept."""
    if k is None or len(per_tile) == 0:
        return per_tile.iloc[0:0].copy()
    picks = []
    for cyc, g in per_tile.groupby("cycle"):
        n = len(g)
        if n <= k:
            picks.append(g)
            continue
        g_sorted = g.sort_values("n_detections", kind="stable").reset_index(drop=True)
        start = (n - k) // 2
        picks.append(g_sorted.iloc[start:start + k])
    return pd.concat(picks, ignore_index=True) if picks else per_tile.iloc[0:0].copy()


def build_summary_from_subset(per_tile_subset, per_tile_all, k):
    """Per-cycle summary derived from a K-tile subset (top-K, middle-K, ...).
    Emits `n_tiles_used` (actual subset size at each cycle) and
    `n_tiles_available` (total tiles at that cycle) so quality shifts are
    visible."""
    if len(per_tile_subset) == 0:
        return pd.DataFrame()
    available = (per_tile_all.groupby("cycle")["tile"].nunique()
                             .rename("n_tiles_available"))
    summary = (per_tile_subset.groupby(["cycle", "t_min"], as_index=False)
                             .agg(n_tiles_used=("tile", "nunique"),
                                  detections_sum=("n_detections", "sum"),
                                  detections_mean=("n_detections", "mean"),
                                  detections_std=("n_detections", "std"),
                                  density_mean=("density", "mean"),
                                  density_std=("density", "std")))
    summary["k_requested"] = int(k)
    summary = summary.merge(available.reset_index(), on="cycle", how="left")
    cols = ["cycle", "t_min", "k_requested", "n_tiles_used",
            "n_tiles_available", "detections_sum", "detections_mean",
            "detections_std", "density_mean", "density_std"]
    return summary[cols].sort_values("cycle").reset_index(drop=True)


# =============================================================================
# Analysis — per concentration (cross-assay)
# =============================================================================
def pool_by_concentration(entries, area_per_tile):
    """entries: [{'assay': str, 'concentration': str, 'per_tile': df}, ...]
    Returns {concentration: pooled_df} where pooled_df concatenates every
    folder's per_tile rows at that concentration and adds an `assay` column
    (so n_assays / n_tiles can be tracked). `density` is recomputed from
    `n_detections` with the current --area so units are consistent across
    assays that may have been analyzed with a different area setting.

    This is the pool used for cross-concentration analysis: top-K / mid-K
    selections at concentration level operate on this pool, so the K
    highest-count tiles are picked across ALL folders at a concentration,
    not per folder.
    """
    by_conc = {}
    for e in entries:
        df = e["per_tile"].copy()
        df["assay"] = e["assay"]
        df["density"] = df["n_detections"] / area_per_tile
        by_conc.setdefault(e["concentration"], []).append(df)
    return {c: pd.concat(dfs, ignore_index=True) for c, dfs in by_conc.items()}


def summarize_per_conc_pool(pooled_by_conc, area_per_tile, area_units):
    """One row per (concentration, cycle) with count/density stats over the
    pooled tiles. `pooled_by_conc` must be a {conc: df} dict where each df
    already has `assay`, `density`, `n_detections`, `cycle`, `t_min` columns
    (i.e., the output of `pool_by_concentration` or a subset thereof)."""
    rows = []
    for conc, pooled in pooled_by_conc.items():
        if len(pooled) == 0:
            continue
        for (cycle, t_min), g in pooled.groupby(["cycle", "t_min"]):
            rows.append({
                "concentration":     conc,
                "cycle":             int(cycle),
                "t_min":             float(t_min),
                "n_assays":          int(g["assay"].nunique()),
                "n_tiles":           int(len(g)),
                "detections_sum":    int(g["n_detections"].sum()),
                "detections_mean":   float(g["n_detections"].mean()),
                "detections_std":    float(g["n_detections"].std()),
                "detections_median": float(g["n_detections"].median()),
                "density_mean":      float(g["density"].mean()),
                "density_std":       float(g["density"].std()),
                "area_per_tile":     area_per_tile,
                "area_units":        area_units,
            })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .sort_values(["concentration", "cycle"])
              .reset_index(drop=True))


def subset_pool_by_conc(pooled_by_conc, subset_fn, k):
    """Apply `subset_fn` (top_k_per_cycle / middle_k_per_cycle) INSIDE each
    concentration's pooled df — so the K tiles are picked across all folders
    at that concentration, not per folder. Returns {conc: subset_df}."""
    out = {}
    for conc, pooled in pooled_by_conc.items():
        sub = subset_fn(pooled, k)
        if len(sub):
            out[conc] = sub
    return out


# =============================================================================
# Visualizations — baseline
# =============================================================================
def plot_counts_over_time(summary, out_path, thr, nms_k, title=None):
    """Mean ± std ribbon across tiles + per-cycle sum on twin axis."""
    if len(summary) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ts = summary["t_min"].values
    m = summary["detections_mean"].values
    s = summary["detections_std"].fillna(0).values
    ax.plot(ts, m, "-o", color="steelblue", lw=2, label="mean per tile")
    ax.fill_between(ts, m - s, m + s, alpha=0.2, color="steelblue",
                    label="± 1 std")
    ax.set_xlabel("time (min, first cycle = t=0)")
    ax.set_ylabel("# detections per tile", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(ts, summary["detections_sum"].values, "-s", color="crimson",
             lw=1.8, label="sum across tiles")
    ax2.set_ylabel("total # detections (sum across tiles)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    ax.set_title(title or f"Detections over time  (thr={thr}, nms={nms_k})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_tile_over_time(per_tile, out_path, thr, nms_k, tile_source):
    if len(per_tile) == 0:
        return
    tiles = sorted(per_tile["tile"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in tiles:
        g = per_tile[per_tile["tile"] == t].sort_values("t_min")
        ax.plot(g["t_min"], g["n_detections"], "-", alpha=0.35,
                lw=0.9, color="steelblue")
    mean_line = per_tile.groupby("t_min")["n_detections"].mean().sort_index()
    ax.plot(mean_line.index, mean_line.values, "-o", color="crimson",
            lw=2.5, markersize=5, label="mean across tiles")
    ax.set_xlabel("time (min, first cycle = t=0)")
    ax.set_ylabel(f"# detections per tile  (thr={thr}, nms={nms_k})")
    ax.set_title(f"Per-tile detections over time  "
                 f"({len(tiles)} tiles; tile id: {tile_source})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_tile_over_time_html(per_tile, out_path, thr, nms_k, tile_source,
                                 y_col="n_detections",
                                 y_label="# detections per tile",
                                 title="Per-tile detections over time"):
    """Interactive Plotly HTML. Click legend to toggle a tile, double-click
    to isolate it. Skipped silently if plotly isn't installed."""
    if len(per_tile) == 0:
        return False
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"    [skip] {os.path.basename(out_path)}: install plotly "
              "(pip install plotly) to get the interactive HTML output.")
        return False

    tiles_df = (per_tile[["tile", "tile_row", "tile_col"]]
                .drop_duplicates("tile"))
    has_pos = tiles_df["tile_row"].notna().any()
    if has_pos:
        tiles_df = tiles_df.sort_values(["tile_row", "tile_col"])
    else:
        tiles_df = tiles_df.sort_values("tile")

    fig = go.Figure()
    hover_y = "%{y:.3f}" if y_col == "density" else "%{y:.0f}"

    for _, r in tiles_df.iterrows():
        tile = r["tile"]
        g = per_tile[per_tile["tile"] == tile].sort_values("t_min")
        if has_pos:
            group = f"row {int(r['tile_row'])}"
        else:
            group = "tiles"
        fig.add_trace(go.Scatter(
            x=g["t_min"].tolist(),
            y=g[y_col].tolist(),
            mode="lines+markers",
            name=str(tile),
            legendgroup=group,
            legendgrouptitle_text=group,
            line=dict(width=1.2, color="steelblue"),
            marker=dict(size=4),
            opacity=0.55,
            hovertemplate=(f"tile {tile}<br>"
                           f"t=%{{x}} min<br>"
                           f"{y_label}={hover_y}<extra></extra>"),
        ))

    mean_line = per_tile.groupby("t_min")[y_col].mean().sort_index()
    fig.add_trace(go.Scatter(
        x=mean_line.index.tolist(),
        y=mean_line.values.tolist(),
        mode="lines+markers",
        name="MEAN across tiles",
        line=dict(width=3.5, color="crimson"),
        marker=dict(size=8),
        hovertemplate=(f"MEAN<br>t=%{{x}} min<br>"
                       f"{y_label}={hover_y}<extra></extra>"),
    ))

    fig.update_layout(
        title=(f"{title}  "
               f"({len(tiles_df)} tiles; tile id: {tile_source}; "
               f"thr={thr}, nms={nms_k})"),
        xaxis_title="time (min, first cycle = t=0)",
        yaxis_title=y_label,
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            groupclick="toggleitem",
        ),
        margin=dict(r=200, b=110),
    )
    fig.add_annotation(
        text=("<b>Tip:</b> click a legend entry to hide/show that tile. "
              "<b>Double-click</b> to isolate it (hide all others); "
              "double-click again to show all."),
        xref="paper", yref="paper",
        x=0.0, y=-0.18,
        showarrow=False, align="left",
        font=dict(size=11, color="dimgray"),
    )

    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return True


def plot_boxplot_over_time(per_tile, out_path, thr, nms_k, minutes_per_cycle,
                           title=None):
    if len(per_tile) == 0:
        return
    by_t = per_tile.groupby("t_min")["n_detections"].apply(list)
    ts = sorted(by_t.index)
    data = [by_t[t] for t in ts]
    width = max(minutes_per_cycle * 0.6, 0.5)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(ts) + 4), 5))
    ax.boxplot(data, positions=ts, widths=width,
               showmeans=True, meanline=True)
    ax.set_xlabel("time (min, first cycle = t=0)")
    ax.set_ylabel(f"# detections per tile  (thr={thr}, nms={nms_k})")
    ax.set_title(title or "Detections over time — box & whisker")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_heatmap_last_cycle(per_tile, out_path, thr, nms_k):
    """Grid heatmap of counts at the final cycle. Skips when no tile
    positions were parsed (tile_row/tile_col are NaN)."""
    if len(per_tile) == 0 or per_tile["tile_row"].isna().all():
        return False
    last_cycle = per_tile["cycle"].max()
    g = per_tile[per_tile["cycle"] == last_cycle]
    if g["tile_row"].isna().all():
        return False
    rows = sorted(per_tile["tile_row"].dropna().astype(int).unique())
    cols = sorted(per_tile["tile_col"].dropna().astype(int).unique())
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: i for i, c in enumerate(cols)}
    grid = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for _, r in g.iterrows():
        if pd.notna(r.tile_row) and pd.notna(r.tile_col):
            grid[row_idx[int(r.tile_row)], col_idx[int(r.tile_col)]] = r.n_detections

    fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(cols) + 3),
                                    max(4, 0.35 * len(rows) + 2)))
    im = ax.imshow(grid, cmap="viridis", origin="upper",
                   interpolation="nearest", aspect="equal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8, rotation=45)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel("tile col")
    ax.set_ylabel("tile row")
    ax.set_title(f"Detections at final cycle ({last_cycle})  "
                 f"(thr={thr}, nms={nms_k})")

    for i in range(len(rows)):
        for j in range(len(cols)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(round(v))}",
                        ha="center", va="center",
                        color="white" if v < np.nanmax(grid) * 0.6 else "black",
                        fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="# detections")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# =============================================================================
# Visualizations — concentration
# =============================================================================
def plot_counts_over_time_by_concentration(per_conc, out_path, thr, nms_k,
                                            title_suffix=""):
    """One line per concentration, mean-vs-time, with ±std ribbon per line
    and per-cycle sum on twin axis (dashed)."""
    if len(per_conc) == 0:
        return
    concs = sorted(per_conc["concentration"].unique(),
                   key=lambda c: float(c))
    cmap = plt.get_cmap("viridis", max(len(concs), 2))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax2 = ax.twinx()
    for i, c in enumerate(concs):
        g = per_conc[per_conc["concentration"] == c].sort_values("t_min")
        col = cmap(i)
        n_assays = int(g["n_assays"].max())
        ax.plot(g["t_min"], g["detections_mean"], "-o", color=col, lw=2,
                label=f"{c}  (n_assays={n_assays})")
        s = g["detections_std"].fillna(0)
        ax.fill_between(g["t_min"],
                        g["detections_mean"] - s,
                        g["detections_mean"] + s,
                        alpha=0.12, color=col)
        ax2.plot(g["t_min"], g["detections_sum"], "--", color=col,
                 lw=1.0, alpha=0.7)
    ax.set_xlabel("time (min, first cycle = t=0)")
    ax.set_ylabel("mean # detections per tile (across pooled tiles)")
    ax2.set_ylabel("sum # detections across pooled tiles (dashed)",
                   fontsize=9, color="dimgray")
    ax2.tick_params(axis="y", labelsize=8, colors="dimgray")
    ax.set_title(f"Detections over time by concentration{title_suffix}  "
                 f"(thr={thr}, nms={nms_k})")
    ax.legend(title="concentration", fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_by_concentration(pooled_by_conc, out_path, thr, nms_k,
                                   minutes_per_cycle, title_suffix=""):
    """Grouped boxplot: at each cycle timestamp, one box per concentration
    side-by-side. Shows the full tile-count distribution (median line, IQR
    box, whiskers, mean marker) so the range across pooled tiles is visible
    for every (concentration, cycle) pair. `pooled_by_conc` is
    {concentration: df_with_n_detections_and_t_min}."""
    if not pooled_by_conc:
        return
    concs = sorted(pooled_by_conc.keys(), key=lambda c: float(c))
    n_conc = len(concs)

    all_t = sorted({float(t)
                    for df in pooled_by_conc.values()
                    for t in df["t_min"].unique()})
    if not all_t:
        return

    # Layout: box widths and per-concentration horizontal offsets scaled to
    # the tightest cycle spacing so boxes don't overlap between time points.
    step = float(np.min(np.diff(all_t))) if len(all_t) > 1 else float(minutes_per_cycle)
    slot = step / (n_conc + 1)
    box_width = slot * 0.75
    offsets = [(-(n_conc - 1) / 2 + i) * slot for i in range(n_conc)]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(all_t) * n_conc + 4), 5.5))
    cmap = plt.get_cmap("viridis", max(n_conc, 2))

    for i, conc in enumerate(concs):
        by_t = pooled_by_conc[conc].groupby("t_min")["n_detections"].apply(list)
        positions, data = [], []
        for t in all_t:
            if t in by_t.index:
                positions.append(t + offsets[i])
                data.append(list(by_t[t]))
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions, widths=box_width,
                        patch_artist=True, showmeans=True, meanline=True,
                        medianprops=dict(color="black", lw=1.2),
                        meanprops=dict(color="crimson", lw=1.2, ls="--"))
        for patch in bp["boxes"]:
            patch.set_facecolor(cmap(i))
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

    ax.set_xticks(all_t)
    ax.set_xticklabels([f"{t:.0f}" for t in all_t])
    ax.set_xlabel("time (min, first cycle = t=0)")
    ax.set_ylabel(f"# detections per tile  (thr={thr}, nms={nms_k})")
    ax.set_title(f"Detections per tile by concentration{title_suffix}  "
                 f"(median = black, mean = crimson dashed)")
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=cmap(i), alpha=0.55, ec="black",
                      lw=0.8, label=str(c))
        for i, c in enumerate(concs)
    ]
    ax.legend(handles=legend_patches, title="concentration",
              fontsize=9, loc="best")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================
def discover_cycles(data_dir, image_ext):
    """Return [(cycle_dir, [image_paths]), ...] sorted by folder name."""
    demo_dirs = sorted({p for pat in DEMO_GLOBS
                        for p in glob.glob(os.path.join(data_dir, pat))
                        if os.path.isdir(p)})
    out = []
    for d in demo_dirs:
        imgs = sorted(glob.glob(os.path.join(d, f"*{image_ext}")))
        if imgs:
            out.append((d, imgs))
    return out


def discover_assays(root, image_ext):
    """Every direct subdir of `root` that contains at least one cycle_*/demo_*
    folder with images. Skips any subfolder named literally 'analysis' (our
    own output at the root level)."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        if os.path.basename(d) == "analysis":
            continue
        if discover_cycles(d, image_ext):
            out.append(d)
    return out


def resolve_model_path(user_path):
    if os.path.isabs(user_path):
        return user_path if os.path.isfile(user_path) else None
    for c in (os.path.join(os.getcwd(), user_path),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), user_path)):
        if os.path.isfile(c):
            return c
    return None


def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GaussianMixtureModel(num_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # eval() is REQUIRED for BatchNorm2d — it switches BN to use the tracked
    # running_mean / running_var instead of the current batch's stats (batch
    # size is 1 during tile-by-tile inference, so training-mode BN would be
    # garbage).
    model.to(device).eval()
    # tanh_scale is a plain attribute (not in state_dict). GaussianMixtureModel
    # already defaults it to the v9 final training value (6.0); re-pin here so
    # the value is explicit in the inference path.
    (model._orig_mod if hasattr(model, "_orig_mod") else model).tanh_scale = 6.0
    return model, device


def infer_assay(model, device, data_dir, args):
    """Run inference on every image in every cycle folder under data_dir.
    Writes per-image annotated PNGs (unless --no-images). Returns the raw
    one-row-per-image DataFrame."""
    cycles = discover_cycles(data_dir, args.image_ext)
    if not cycles:
        return pd.DataFrame()

    analysis_dir = os.path.join(data_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    rows = []
    t0 = time.time()
    for cycle_dir, imgs in tqdm(cycles, desc=f"  cycles", leave=False):
        cycle_name = os.path.basename(cycle_dir)
        if not args.no_images:
            cycle_out = os.path.join(analysis_dir, cycle_name)
            os.makedirs(cycle_out, exist_ok=True)
        for path in tqdm(imgs, desc=f"    {cycle_name}", leave=False):
            try:
                img, cx, cy = infer_positions(
                    model, device, path,
                    threshold=args.threshold, nms_kernel=args.nms_kernel)
            except Exception as e:
                print(f"    !! {path}: {e}")
                continue
            n = int(len(cx))
            rows.append({
                "demo":         cycle_name,
                "image":        os.path.basename(path),
                "path":         path,
                "n_detections": n,
                "image_h":      img.shape[0],
                "image_w":      img.shape[1],
            })
            if not args.no_images:
                base = os.path.splitext(os.path.basename(path))[0] + ".png"
                render_annotated(img, cx, cy, n,
                                 os.path.join(cycle_out, base))
    dt = time.time() - t0
    print(f"    inference: {len(rows)} images in {dt:.1f}s")
    return pd.DataFrame(rows)


def write_baseline_outputs(rows_df, data_dir, args):
    """Compute per_tile + summary, write baseline CSVs, PNGs, and HTMLs.
    Returns (per_tile, summary, tile_source)."""
    analysis_dir = os.path.join(data_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    per_tile, summary, tile_source = build_per_tile_and_summary(
        rows_df,
        area_per_tile=args.area,
        area_units=args.area_units,
        minutes_per_cycle=args.minutes_per_cycle,
    )
    per_tile.to_csv(os.path.join(analysis_dir, "per_tile.csv"), index=False)
    summary.to_csv(os.path.join(analysis_dir, "summary_per_cycle.csv"),
                    index=False)

    plot_counts_over_time(summary,
                          os.path.join(analysis_dir, "counts_over_time.png"),
                          args.threshold, args.nms_kernel)
    plot_per_tile_over_time(per_tile,
                            os.path.join(analysis_dir, "per_tile_over_time.png"),
                            args.threshold, args.nms_kernel, tile_source)
    plot_boxplot_over_time(per_tile,
                           os.path.join(analysis_dir, "boxplot_over_time.png"),
                           args.threshold, args.nms_kernel,
                           args.minutes_per_cycle)
    plot_spatial_heatmap_last_cycle(
        per_tile,
        os.path.join(analysis_dir, "spatial_heatmap_last_cycle.png"),
        args.threshold, args.nms_kernel)
    plot_per_tile_over_time_html(
        per_tile,
        os.path.join(analysis_dir, "per_tile_over_time.html"),
        args.threshold, args.nms_kernel, tile_source,
        y_col="n_detections",
        y_label="# detections per tile",
        title="Per-tile detections over time")
    plot_per_tile_over_time_html(
        per_tile,
        os.path.join(analysis_dir, "per_tile_density_over_time.html"),
        args.threshold, args.nms_kernel, tile_source,
        y_col="density",
        y_label=f"density (count / {args.area_units})",
        title="Per-tile density over time")
    return per_tile, summary, tile_source


def write_subset_outputs(per_tile, data_dir, args, k, subset_fn,
                          suffix, label):
    """Write summary + counts/boxplot PNGs for a per-cycle K-tile subset.
    `suffix` goes in filenames (e.g. `topK`, `middleK`); `label` goes in
    plot titles (e.g. `top 10`, `middle 10`)."""
    if k is None:
        return
    analysis_dir = os.path.join(data_dir, "analysis")
    per_tile_subset = subset_fn(per_tile, k)
    if len(per_tile_subset) == 0:
        return
    summary = build_summary_from_subset(per_tile_subset, per_tile, k)
    summary.to_csv(
        os.path.join(analysis_dir, f"summary_per_cycle_{suffix}.csv"),
        index=False)
    plot_counts_over_time(summary,
        os.path.join(analysis_dir, f"counts_over_time_{suffix}.png"),
        args.threshold, args.nms_kernel,
        title=f"Detections over time — {label}  "
              f"(thr={args.threshold}, nms={args.nms_kernel})")
    plot_boxplot_over_time(per_tile_subset,
        os.path.join(analysis_dir, f"boxplot_over_time_{suffix}.png"),
        args.threshold, args.nms_kernel, args.minutes_per_cycle,
        title=f"Detections over time — {label}, box & whisker")


def process_assay(model_getter, data_dir, args):
    """Full end-to-end for one assay. Skips inference and baseline outputs
    if `analysis/per_tile.csv` already exists (unless --force). Always
    (re)runs the top-K step when --top-k is set. Returns the per_tile
    DataFrame (or None if unavailable)."""
    assay_name = os.path.basename(data_dir)
    analysis_dir = os.path.join(data_dir, "analysis")
    per_tile_path = os.path.join(analysis_dir, "per_tile.csv")

    if os.path.isfile(per_tile_path) and not args.force:
        print(f"[{assay_name}] baseline analysis exists — loading per_tile.csv")
        per_tile = pd.read_csv(per_tile_path)
    else:
        model, device = model_getter()
        print(f"[{assay_name}] running inference")
        rows_df = infer_assay(model, device, data_dir, args)
        if len(rows_df) == 0:
            print(f"[{assay_name}] no images processed")
            return None
        per_tile, _, tile_source = write_baseline_outputs(
            rows_df, data_dir, args)
        print(f"[{assay_name}] wrote baseline outputs "
              f"(tile source: {tile_source})")

    if args.top_k is not None and len(per_tile) > 0:
        write_subset_outputs(per_tile, data_dir, args, args.top_k,
                              top_k_per_cycle, "topK", f"top {args.top_k}")
        print(f"[{assay_name}] wrote top-{args.top_k} outputs")

    if args.middle_k is not None and len(per_tile) > 0:
        write_subset_outputs(per_tile, data_dir, args, args.middle_k,
                              middle_k_per_cycle, "middleK",
                              f"middle {args.middle_k}")
        print(f"[{assay_name}] wrote middle-{args.middle_k} outputs")

    return per_tile


def run(args):
    # --- source selection ---
    if bool(args.data_dir) == bool(args.assays_root):
        sys.exit("error: pass exactly one of --data-dir or --assays-root")

    model_path = resolve_model_path(args.model_path)
    if not model_path:
        sys.exit(f"error: model file not found: {args.model_path}")

    if args.assays_root:
        root = os.path.abspath(args.assays_root)
        if not os.path.isdir(root):
            sys.exit(f"error: --assays-root {root} is not a directory")
        targets = discover_assays(root, args.image_ext)
        if not targets:
            sys.exit(f"error: no assay subfolders under {root}")
    else:
        target = os.path.abspath(args.data_dir)
        if not os.path.isdir(target):
            sys.exit(f"error: --data-dir {target} is not a directory")
        targets = [target]
        root = None

    # --- header ---
    print(f"model    : {model_path}")
    print(f"area     : {args.area} {args.area_units} per tile")
    print(f"detector : threshold={args.threshold}, nms_kernel={args.nms_kernel}")
    if args.top_k is not None:
        print(f"top-K    : {args.top_k} tiles per cycle")
    if root is not None:
        print(f"root     : {root}")
    print(f"assays   : {len(targets)}")
    for t in targets:
        print(f"           - {os.path.basename(t)}")

    # --- lazy model loader (only loads if any assay needs inference) ---
    _cached = {"model": None, "device": None}

    def get_model():
        if _cached["model"] is None:
            _cached["model"], _cached["device"] = load_model(model_path)
            print(f"loaded model on {_cached['device']}")
        return _cached["model"], _cached["device"]

    # --- per-assay pass ---
    entries = []
    t_all = time.time()
    for data_dir in targets:
        per_tile = process_assay(get_model, data_dir, args)
        if per_tile is None or len(per_tile) == 0:
            continue
        if root is not None:
            conc = parse_concentration(os.path.basename(data_dir))
            if conc is not None:
                entries.append({
                    "assay":         os.path.basename(data_dir),
                    "concentration": conc,
                    "per_tile":      per_tile,
                })

    # --- per-concentration pass (only under --assays-root) ---
    if root is not None:
        matched = len(entries)
        skipped = len(targets) - matched
        print(f"\nconcentration parsing: matched {matched}/{len(targets)} assays"
              + (f"; {skipped} skipped (no NNN.N prefix)" if skipped else ""))

        if matched > 0:
            root_analysis = os.path.join(root, "analysis")
            os.makedirs(root_analysis, exist_ok=True)

            # Pool tiles once per concentration; subset flags below all work
            # off this pool so K tiles are picked ACROSS folders at each
            # concentration (not per folder).
            pooled_full = pool_by_concentration(entries, args.area)

            modes = [(None, None, "", "")]                    # full baseline
            if args.top_k is not None:
                modes.append((args.top_k, top_k_per_cycle,
                              "_topK", f"  (top {args.top_k})"))
            if args.middle_k is not None:
                modes.append((args.middle_k, middle_k_per_cycle,
                              "_middleK", f"  (middle {args.middle_k})"))

            for k_val, subset_fn, file_suffix, title_suffix in modes:
                if k_val is None:
                    pooled_mode = pooled_full
                else:
                    pooled_mode = subset_pool_by_conc(
                        pooled_full, subset_fn, k_val)
                if not pooled_mode:
                    continue

                summary_df = summarize_per_conc_pool(
                    pooled_mode, args.area, args.area_units)

                if len(summary_df):
                    csv_path = os.path.join(
                        root_analysis,
                        f"per_concentration_per_cycle{file_suffix}.csv")
                    summary_df.to_csv(csv_path, index=False)
                    plot_counts_over_time_by_concentration(
                        summary_df,
                        os.path.join(root_analysis,
                                      f"counts_over_time_by_concentration{file_suffix}.png"),
                        args.threshold, args.nms_kernel,
                        title_suffix=title_suffix)
                    print(f"wrote per_concentration_per_cycle{file_suffix}.csv")
                    print(f"wrote counts_over_time_by_concentration{file_suffix}.png")

                plot_boxplot_by_concentration(
                    pooled_mode,
                    os.path.join(root_analysis,
                                  f"boxplot_by_concentration{file_suffix}.png"),
                    args.threshold, args.nms_kernel,
                    args.minutes_per_cycle,
                    title_suffix=title_suffix)
                print(f"wrote boxplot_by_concentration{file_suffix}.png")

    print(f"\ntotal wall time: {time.time()-t_all:.1f}s")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="LOCA-PRAM standalone assay analyzer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir",
                     help="Single assay folder (containing cycle_XXXX/ or "
                          "demo_XXXX/ subfolders).")
    src.add_argument("--assays-root",
                     help="Parent folder containing many assay subfolders; "
                          "every direct subdir with cycle_*/demo_* inside "
                          "is processed. Concentration aggregation runs "
                          "automatically for subfolders whose names start "
                          "with `NNN.N` (e.g. `006.0_...`).")
    p.add_argument("--area", required=True, type=float,
                   help="Physical area of one image/tile FOV. Units are your "
                        "choice; density is reported as count / area.")
    p.add_argument("--area-units", default="mm^2",
                   help="Label for the area units (echoed into CSVs).")
    p.add_argument("--model-path", default="pram_dense_final_v11.pth",
                   help="Path to the .pth checkpoint (v9 batchnorm build "
                        "expected — the arch here has split heads and "
                        "BatchNorm; older v7 GroupNorm checkpoints won't "
                        "load without swapping NORM back to 'group' and "
                        "reverting to the pre-v9 head layout). Relative "
                        "paths resolve against cwd then this script's dir.")
    p.add_argument("--image-ext", default=".jpg",
                   help="Extension for input images inside each cycle folder.")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Detection probability threshold.")
    p.add_argument("--nms-kernel", type=int, default=DEFAULT_NMS_KERNEL,
                   help="Non-max-suppression kernel size in downsampled px.")
    p.add_argument("--minutes-per-cycle", type=float, default=5.0,
                   help="Real minutes between cycles; used for time-axis labels.")
    p.add_argument("--no-images", action="store_true",
                   help="Skip per-image annotated PNG output "
                        "(CSVs and summary plots still get written).")
    p.add_argument("--top-k", type=int, default=None,
                   help="If set, additionally produce top-K aggregation: the "
                        "K highest-count tiles at each cycle are pooled into "
                        "summary_per_cycle_topK.csv, counts_over_time_topK.png, "
                        "and boxplot_over_time_topK.png. Under --assays-root, "
                        "matching per-concentration top-K outputs are also "
                        "written.")
    p.add_argument("--middle-k", type=int, default=None,
                   help="If set, additionally produce middle-K aggregation: "
                        "the K tiles centered on the median at each cycle "
                        "(filters both the bottom tail of imaging failures "
                        "and the top tail of unusually hot tiles). Files "
                        "mirror --top-k with `_middleK` suffix. Can be used "
                        "together with --top-k.")
    p.add_argument("--force", action="store_true",
                   help="Under --assays-root, re-run inference and baseline "
                        "analysis even when analysis/per_tile.csv already "
                        "exists.")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
