"""
Synthetic Background Generator v3
=================================
Instead of inpainting, this:
1. Crops center 50% of the image
2. Divides into small patches (e.g. 64x64)
3. Classifies patches as "clean" (no particles, no holes)
4. Extracts the smooth illumination gradient
5. Extracts noise texture from clean patches (by subtracting local gradient)
6. Generates a full-size synthetic background by:
   - Recreating the gradient
   - Randomly sampling clean noise patches and stitching them

This preserves the REAL noise texture (spatial correlation, distribution, grain)
without any inpainting artifacts.

Usage:
    python generate_synthetic_bg_v3.py Picture3.png
    python generate_synthetic_bg_v3.py Picture3.png --patch-size 64 --dark-threshold 80
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import argparse
import os

Image.MAX_IMAGE_PIXELS = None


def load_image(path):
    img = Image.open(path)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def crop_center(image, fraction=0.5):
    """Crop the center portion of the image."""
    h, w = image.shape
    crop_h = int(h * fraction)
    crop_w = int(w * fraction)
    y_start = (h - crop_h) // 2
    x_start = (w - crop_w) // 2
    return image[y_start:y_start + crop_h, x_start:x_start + crop_w]


def extract_gradient(image, sigma=200):
    """Extract smooth illumination gradient via heavy Gaussian blur."""
    return gaussian_filter(image, sigma=sigma)


def classify_patches(region, patch_size=64, dark_threshold=80,
                     std_range=(1.5, 12.0), intensity_percentile=(5, 95)):
    """
    Divide region into patches and classify each as clean or dirty.

    A patch is "clean" if:
      - No pixels below dark_threshold (no holes/stripes)
      - Std within expected range (not too flat = hole edge, not too noisy = particle)
      - Min pixel value is reasonable (no dark particle centers)

    Returns:
      clean_patches: list of (y, x, patch) tuples
      dirty_patches: list of (y, x, patch) tuples
      clean_mask: boolean grid of which patches are clean
    """
    h, w = region.shape
    ny = h // patch_size
    nx = w // patch_size

    # Global stats for reference
    global_mean = np.mean(region[region > dark_threshold])
    global_std = np.std(region[region > dark_threshold])

    clean_patches = []
    dirty_patches = []
    clean_mask = np.zeros((ny, nx), dtype=bool)

    for iy in range(ny):
        for ix in range(nx):
            y0 = iy * patch_size
            x0 = ix * patch_size
            patch = region[y0:y0 + patch_size, x0:x0 + patch_size]

            # Check criteria
            has_dark = np.any(patch < dark_threshold)
            patch_std = np.std(patch)
            patch_min = np.min(patch)
            patch_mean = np.mean(patch)

            # Min value check: if any pixel is much darker than local mean,
            # likely a particle (particles are dark spots)
            min_deviation = (patch_mean - patch_min) / (patch_std + 1e-6)

            is_clean = (
                not has_dark and
                std_range[0] < patch_std < std_range[1] and
                min_deviation < 4.0  # no single pixel much darker than surroundings
            )

            if is_clean:
                clean_patches.append((y0, x0, patch.copy()))
                clean_mask[iy, ix] = True
            else:
                dirty_patches.append((y0, x0, patch.copy()))

    return clean_patches, dirty_patches, clean_mask


def extract_noise_textures(clean_patches, gradient, patch_size=64):
    """
    For each clean patch, subtract the local gradient to get pure noise texture.

    Returns list of noise-only patches (zero-mean texture).
    """
    noise_patches = []
    for (y0, x0, patch) in clean_patches:
        local_gradient = gradient[y0:y0 + patch_size, x0:x0 + patch_size]
        noise = patch - local_gradient
        noise_patches.append(noise)

    return noise_patches


def synthesize_background(target_h, target_w, gradient_full, noise_patches,
                          patch_size=64, blend_margin=8):
    """
    Generate a full-size synthetic background:
      1. Use the full gradient as the smooth base
      2. For each patch-sized region, randomly sample a clean noise patch
      3. Add noise to gradient with blending at boundaries

    blend_margin: pixels of overlap for smooth transitions between patches.
    """
    result = gradient_full.copy()

    ny = (target_h + patch_size - 1) // patch_size
    nx = (target_w + patch_size - 1) // patch_size

    n_noise = len(noise_patches)

    # Step size with overlap for blending
    step = patch_size - blend_margin

    # Create blend weights (raised cosine for smooth transitions)
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    # Taper edges
    for i in range(blend_margin):
        fade = (i + 1) / (blend_margin + 1)
        weight[i, :] *= fade          # top
        weight[-(i+1), :] *= fade     # bottom
        weight[:, i] *= fade          # left
        weight[:, -(i+1)] *= fade     # right

    # Accumulators for weighted blending
    noise_accum = np.zeros((target_h, target_w), dtype=np.float32)
    weight_accum = np.zeros((target_h, target_w), dtype=np.float32)

    y = 0
    while y < target_h:
        x = 0
        while x < target_w:
            # Random noise patch
            idx = np.random.randint(0, n_noise)
            noise = noise_patches[idx]

            # Handle edge cases (patch extends beyond image)
            ph = min(patch_size, target_h - y)
            pw = min(patch_size, target_w - x)

            noise_accum[y:y+ph, x:x+pw] += noise[:ph, :pw] * weight[:ph, :pw]
            weight_accum[y:y+ph, x:x+pw] += weight[:ph, :pw]

            x += step
        y += step

    # Normalize by weight
    weight_accum = np.maximum(weight_accum, 1e-6)
    noise_layer = noise_accum / weight_accum

    result += noise_layer
    return result


def generate_synthetic_background(image_path, crop_fraction=0.5,
                                   patch_size=64, dark_threshold=80,
                                   gradient_sigma=200, output_size=None):
    """Full pipeline."""
    # 1. Load
    image = load_image(image_path)
    orig_h, orig_w = image.shape
    if output_size is None:
        output_size = (orig_h, orig_w)
    print(f"Original image: {orig_w} x {orig_h}")

    # 2. Crop center
    region = crop_center(image, crop_fraction)
    rh, rw = region.shape
    print(f"Cropped center: {rw} x {rh}")

    # 3. Extract gradient from cropped region
    gradient = extract_gradient(region, sigma=gradient_sigma)

    # 4. Classify patches
    clean, dirty, clean_mask = classify_patches(
        region, patch_size=patch_size, dark_threshold=dark_threshold)
    total = clean_mask.size
    n_clean = len(clean)
    print(f"Patches: {n_clean} clean / {total} total ({100*n_clean/total:.1f}%)")

    if n_clean < 10:
        raise ValueError(f"Only {n_clean} clean patches found. "
                         "Try lowering dark_threshold or increasing std_range.")

    # 5. Extract noise textures
    noise_patches = extract_noise_textures(clean, gradient, patch_size)
    noise_stds = [np.std(p) for p in noise_patches]
    print(f"Noise texture stats: mean_std={np.mean(noise_stds):.2f}, "
          f"range=[{np.min(noise_stds):.2f}, {np.max(noise_stds):.2f}]")

    # 6. Create full-size gradient
    # Extrapolate the gradient to cover the full original image size
    gradient_full = extract_gradient(image, sigma=gradient_sigma)
    # Crop/resize to output size if needed
    if output_size != (orig_h, orig_w):
        # Simple: just recompute at output size by tiling
        gradient_full = tile_gradient(gradient, output_size)
    # For same size as original, use the gradient from the full image directly

    # 7. Synthesize
    print(f"Synthesizing {output_size[1]} x {output_size[0]} background...")
    synthetic = synthesize_background(
        output_size[0], output_size[1], gradient_full, noise_patches,
        patch_size=patch_size)

    synthetic = np.clip(synthetic, 0, 255)

    debug = {
        'original': image,
        'cropped': region,
        'gradient': gradient,
        'clean_mask': clean_mask,
        'clean_patches': clean,
        'noise_patches': noise_patches,
        'gradient_full': gradient_full,
        'synthetic': synthetic,
        'patch_size': patch_size,
    }

    return synthetic, debug


def tile_gradient(gradient_crop, output_size):
    """Tile a gradient to output size using mirror tiling."""
    ph, pw = gradient_crop.shape
    th, tw = output_size

    mirror_h = np.concatenate([gradient_crop, gradient_crop[::-1, :]], axis=0)
    mirror = np.concatenate([mirror_h, mirror_h[:, ::-1]], axis=1)

    mh, mw = mirror.shape
    reps_y = (th + mh - 1) // mh
    reps_x = (tw + mw - 1) // mw
    tiled = np.tile(mirror, (reps_y, reps_x))
    return tiled[:th, :tw]


def visualize_pipeline(debug, save_path=None):
    """Show all steps of the pipeline."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Row 1: extraction
    axes[0, 0].imshow(debug['original'], cmap='gray')
    axes[0, 0].set_title(f"Original ({debug['original'].shape[1]}x{debug['original'].shape[0]})")

    axes[0, 1].imshow(debug['cropped'], cmap='gray')
    axes[0, 1].set_title('Cropped Center')

    axes[0, 2].imshow(debug['gradient'], cmap='gray')
    axes[0, 2].set_title('Extracted Gradient')

    # Patch classification overlay
    region = debug['cropped']
    ps = debug['patch_size']
    clean_mask = debug['clean_mask']
    overlay = np.stack([region/255]*3, axis=-1)  # grayscale to RGB
    ny, nx = clean_mask.shape
    for iy in range(ny):
        for ix in range(nx):
            if not clean_mask[iy, ix]:
                y0, x0 = iy * ps, ix * ps
                overlay[y0:y0+ps, x0:x0+ps, 0] = np.minimum(
                    overlay[y0:y0+ps, x0:x0+ps, 0] + 0.3, 1.0)  # red tint
    axes[0, 3].imshow(overlay)
    n_clean = clean_mask.sum()
    axes[0, 3].set_title(f'Patch Classification\n'
                          f'(green=clean {n_clean}, red=dirty {clean_mask.size - n_clean})')

    # Row 2: synthesis
    # Sample noise patches
    noise_patches = debug['noise_patches']
    n_show = min(16, len(noise_patches))
    grid_size = int(np.ceil(np.sqrt(n_show)))
    noise_grid = np.zeros((grid_size * ps, grid_size * ps))
    for i in range(n_show):
        gy, gx = i // grid_size, i % grid_size
        noise_grid[gy*ps:(gy+1)*ps, gx*ps:(gx+1)*ps] = noise_patches[i]
    axes[1, 0].imshow(noise_grid, cmap='gray')
    axes[1, 0].set_title(f'Sample Noise Patches ({n_show})')

    axes[1, 1].imshow(debug['gradient_full'], cmap='gray')
    axes[1, 1].set_title('Full Gradient')

    axes[1, 2].imshow(debug['synthetic'], cmap='gray')
    axes[1, 2].set_title(f"Synthetic Background\n({debug['synthetic'].shape[1]}x{debug['synthetic'].shape[0]})")

    # Zoomed comparison
    zh, zw = 300, 300
    ch, cw = region.shape
    cy, cx = ch // 3, cw // 3
    orig_zoom = region[cy:cy+zh, cx:cx+zw]
    synth_crop = debug['synthetic']
    # Match location roughly
    sy = debug['original'].shape[0]//2 - ch//2 + cy
    sx = debug['original'].shape[1]//2 - cw//2 + cx
    synth_zoom = synth_crop[sy:sy+zh, sx:sx+zw]
    comparison = np.hstack([orig_zoom, synth_zoom])
    axes[1, 3].imshow(comparison, cmap='gray')
    axes[1, 3].axvline(x=zw, color='red', linewidth=2)
    axes[1, 3].set_title('Zoomed: Original | Synthetic')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle('Synthetic Background: Patch-Based Texture Synthesis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    plt.show()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(
        description='Generate synthetic background using clean texture extraction')
    parser.add_argument('input_image', help='Path to reference image')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--crop-fraction', type=float, default=0.5)
    parser.add_argument('--patch-size', type=int, default=64,
                        help='Size of texture patches (default: 64)')
    parser.add_argument('--dark-threshold', type=float, default=80,
                        help='Pixels below this are holes (default: 80)')
    parser.add_argument('--gradient-sigma', type=float, default=200,
                        help='Blur sigma for gradient extraction (default: 200)')
    parser.add_argument('--no-viz', action='store_true')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input_image)[0]
        args.output = f"{base}_synthetic_v3.png"

    synthetic, debug = generate_synthetic_background(
        args.input_image,
        crop_fraction=args.crop_fraction,
        patch_size=args.patch_size,
        dark_threshold=args.dark_threshold,
        gradient_sigma=args.gradient_sigma,
    )

    Image.fromarray(np.clip(synthetic, 0, 255).astype(np.uint8)).save(args.output)
    print(f"Saved: {args.output}")

    if not args.no_viz:
        viz_path = args.output.replace('.png', '_viz.png')
        visualize_pipeline(debug, save_path=viz_path)