"""
Mechanism figure 2: DWT-LL block perturbation heatmap.
Replays the real greedy_block_attack (distortion_optimization.py) on the
same watermarked image from mech_fig1 (data/raw/0000.png, embed() with the
exact params build_verified_dataset.py used), with the exact seed that
image actually got in the real build (seed=42+idx, idx=0 for 0000.png).
Captures the real per-block DC-coefficient deltas the attack applied
(now exposed via distortion_optimization.py's `deltas` return key) and
overlays them on the original image.
"""
import time
t0 = time.perf_counter()

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.watermark.embed import DWT_DCT_SVD, generate_watermark
from src.attacks.distortion_optimization import greedy_block_attack

OUT = Path("results/figures/mechanism")
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
ALPHA = 0.02
method = DWT_DCT_SVD(dwt_level=2, block_size=8)
watermark = generate_watermark((IMG_SIZE // 4 // 8,) * 2, seed=42)

raw_bgr = cv2.imread("data/raw/0000.png")
raw_bgr = cv2.resize(raw_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
wm_rgb = method.embed(raw_rgb, watermark, ALPHA)
wm_bgr = cv2.cvtColor(wm_rgb, cv2.COLOR_RGB2BGR)

result = greedy_block_attack(
    wm_rgb, raw_rgb, method, watermark, ALPHA,
    threshold=0.5, epsilon=16.0, max_iters=150, seed=42,  # seed=42+idx(0), matches the real build exactly
)
deltas = result["deltas"]
print(f"attack: converged={result['converged']}, iters={result['iters_used']}, score={result['score']:.4f}")

# Upsample the 7x7 block-delta grid to pixel resolution for overlay (each
# block covers 32x32 pixels: 8 DWT-LL px/block * 2**dwt_level).
block_px = method.block_size * (2 ** method.dwt_level)
wm_h, wm_w = deltas.shape
delta_map = np.repeat(np.repeat(deltas, block_px, axis=0), block_px, axis=1)
dmax = np.abs(deltas).max() if np.abs(deltas).max() > 0 else 1.0

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
axes[0].imshow(raw_rgb.astype(np.uint8))
axes[0].set_title("Original image", fontsize=10)
axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].imshow(raw_rgb.astype(np.uint8))
im = axes[1].imshow(delta_map, cmap="RdBu_r", vmin=-dmax, vmax=dmax, alpha=0.55)
# grid lines at block boundaries
for i in range(wm_h + 1):
    axes[1].axhline(i * block_px - 0.5, color="black", linewidth=0.3, alpha=0.4)
for j in range(wm_w + 1):
    axes[1].axvline(j * block_px - 0.5, color="black", linewidth=0.3, alpha=0.4)
axes[1].set_title(f"Per-block DC-coefficient delta applied\n({(deltas != 0).sum()}/{deltas.size} blocks perturbed)", fontsize=10)
axes[1].set_xticks([]); axes[1].set_yticks([])
plt.colorbar(im, ax=axes[1], fraction=0.046, label="DC delta (red=+, blue=-)")

fig.suptitle("distortion_optimization.py: which DWT-LL blocks were perturbed, and in which direction", fontsize=12, fontweight="bold")
fig.text(0.5, 0.01,
          f"data/raw/0000.png, replayed with the exact attack params/seed used in the real build "
          f"(alpha=0.02, epsilon=16, seed=42). Attack converged={result['converged']} in {result['iters_used']} block trials.",
          ha="center", fontsize=8, style="italic")
fig.subplots_adjust(bottom=0.16, top=0.85, left=0.03, right=0.95, wspace=0.1)
fig.savefig(OUT / "mech2_block_perturbation_heatmap.png", dpi=300)
plt.close(fig)

elapsed = time.perf_counter() - t0
print(f"mech2 done in {elapsed:.2f}s")
