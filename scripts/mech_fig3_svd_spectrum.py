"""
Mechanism figure 3: SVD singular-value spectrum, original vs. watermarked.
Uses the real DWT_DCT_SVD.embed() code path directly (same DWT/DCT/SVD calls
embed() itself makes) on data/raw/0000.png, for one representative 8x8
DWT-LL block (block index [0,0]), showing exactly what embed() modifies:
only the top singular value S[0] is scaled by (1 + alpha*sign); S[1:] are
untouched by construction.
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
BLOCK = (0, 0)  # (i, j) block index to inspect
method = DWT_DCT_SVD(dwt_level=2, block_size=8)
watermark = generate_watermark((IMG_SIZE // 4 // 8,) * 2, seed=42)

raw_bgr = cv2.imread("data/raw/0000.png")
raw_bgr = cv2.resize(raw_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
wm_rgb = method.embed(raw_rgb, watermark, ALPHA)

# Recompute exactly what embed()/extract() compute internally for this block.
y_orig = cv2.cvtColor(raw_rgb.astype(np.float32), cv2.COLOR_RGB2YUV)[:, :, 0]
y_wm = cv2.cvtColor(wm_rgb.astype(np.float32), cv2.COLOR_RGB2YUV)[:, :, 0]
LL_orig, _ = method._dwt2(y_orig)
LL_wm, _ = method._dwt2(y_wm)

i, j = BLOCK
r, c = i * method.block_size, j * method.block_size
block_orig = LL_orig[r:r + method.block_size, c:c + method.block_size]
block_wm = LL_wm[r:r + method.block_size, c:c + method.block_size]

_, S_orig, _ = np.linalg.svd(cv2.dct(block_orig), full_matrices=False)
_, S_wm, _ = np.linalg.svd(cv2.dct(block_wm), full_matrices=False)

bit = int(watermark[i, j])
pct_change_s0 = (S_wm[0] - S_orig[0]) / S_orig[0] * 100

other_idx_pct = (S_wm[1:] - S_orig[1:]) / S_orig[1:] * 100
max_other_pct = np.abs(other_idx_pct).max()

fig, ax = plt.subplots(figsize=(7.5, 6.5))
idx = np.arange(1, len(S_orig) + 1)
ax.plot(idx, S_orig, marker="o", color="#4a5568", label="Original", linewidth=2)
ax.plot(idx, S_wm, marker="s", color="#c05621", label="Watermarked", linewidth=2, linestyle="--")
ax.set_yscale("log")
ax.set_xlabel("Singular value index")
ax.set_ylabel("Singular value (log scale)")
ax.set_xticks(idx)
ax.set_title(f"SVD spectrum, DWT-LL block {BLOCK} (watermark bit={bit})", pad=14)
ax.legend(loc="lower left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate(f"S[0]: {S_orig[0]:.1f} -> {S_wm[0]:.1f} ({pct_change_s0:+.2f}%)",
            xy=(1, S_wm[0]), xytext=(2.2, S_orig[0] * 0.5),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

fig.subplots_adjust(bottom=0.24, top=0.88, left=0.15, right=0.96)
fig.text(0.5, 0.03,
          f"data/raw/0000.png, DWT-LL block {BLOCK}, alpha=0.02. embed()'s formula analytically rescales only S[0]\n"
          f"by (1 + alpha*sign); empirically S[1:] also shift by up to {max_other_pct:.2f}% here, from uint8/colorspace\n"
          "rounding in the full embed() roundtrip, not from the SVD-modification step itself.",
          ha="center", fontsize=8, style="italic", wrap=True)
fig.savefig(OUT / "mech3_svd_spectrum.png", dpi=300)
plt.close(fig)

elapsed = time.perf_counter() - t0
print(f"mech3 done in {elapsed:.2f}s")
print(f"S_orig: {S_orig}")
print(f"S_wm:   {S_wm}")
