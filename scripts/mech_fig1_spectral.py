"""
Mechanism figure 1: spectral residual signature.
Uses data/raw/0000.png (real source), regenerates the pre-attack watermarked
intermediate via the real DWT_DCT_SVD.embed() with the exact params
build_verified_dataset.py used for this image (alpha=0.02, block_size=8,
dwt_level=2, watermark_seed=42) -- that intermediate was never persisted to
disk by the pipeline, only the final post-attack result was. The
attacked-and-removed panel is the real saved artifact, data/removed_v2/0000.png.
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

removed_bgr = cv2.imread("data/removed_v2/0000.png")
removed_rgb = cv2.cvtColor(removed_bgr, cv2.COLOR_BGR2RGB)

def luminance(rgb):
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YUV)[:, :, 0]

def fft_logmag(y):
    F = np.fft.fftshift(np.fft.fft2(y))
    return np.log1p(np.abs(F))

y_clean = luminance(raw_rgb)
y_wm = luminance(wm_rgb)
y_removed = luminance(removed_rgb)

mag_clean = fft_logmag(y_clean)
mag_wm = fft_logmag(y_wm)
mag_removed = fft_logmag(y_removed)
diff_clean_removed = mag_removed - mag_clean

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
vmax = max(mag_clean.max(), mag_wm.max(), mag_removed.max())
for ax, mag, title in zip(
    axes[:3], [mag_clean, mag_wm, mag_removed],
    ["(a) Clean original\ndata/raw/0000.png", "(b) Watermarked (pre-attack)\nregenerated via embed()",
     "(c) Attacked & removed\ndata/removed_v2/0000.png"],
):
    im = ax.imshow(mag, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=axes[2], fraction=0.046)

dmax = np.abs(diff_clean_removed).max()
im2 = axes[3].imshow(diff_clean_removed, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
axes[3].set_title("Difference: (c) - (a) log-magnitude\n(red=excess, blue=suppressed)", fontsize=10)
axes[3].set_xticks([]); axes[3].set_yticks([])
plt.colorbar(im2, ax=axes[3], fraction=0.046)

fig.suptitle("FFT magnitude spectrum (log scale, Y channel): clean vs. watermarked vs. attacked-and-removed", fontsize=12, fontweight="bold")
fig.text(0.5, 0.01,
          "One image (data/raw/0000.png / data/removed_v2/0000.png, distortion_optimization attack, verify()-confirmed removed). "
          "Center = low frequency, edges = high frequency.",
          ha="center", fontsize=8, style="italic")
fig.subplots_adjust(bottom=0.16, top=0.82, left=0.03, right=0.98, wspace=0.15)
fig.savefig(OUT / "mech1_spectral_residual.png", dpi=300)
plt.close(fig)

elapsed = time.perf_counter() - t0
print(f"mech1 done in {elapsed:.2f}s")
