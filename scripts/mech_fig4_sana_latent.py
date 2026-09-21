"""
Mechanism figure 4: SANA-VAE (DC-AE) latent visualization.
Real image -> real model (mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers, the
same checkpoint src/attacks/diffusion_regen.py uses) -> real latent tensor
-> real decode. PCA-reduces the 32-channel latent to 3 components for an
RGB-like display.
"""
import time
t0 = time.perf_counter()

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.attacks.diffusion_regen import resolve_device, load_dc_ae, _round_to_multiple

OUT = Path("results/figures/mechanism")
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
device = resolve_device("auto")
model = load_dc_ae(device=device)
t_loaded = time.perf_counter()
print(f"model loaded in {t_loaded - t0:.2f}s on {device}")

raw_bgr = cv2.imread("data/raw/0000.png")
work_size = _round_to_multiple(IMG_SIZE, 32)
raw_bgr = cv2.resize(raw_bgr, (work_size, work_size), interpolation=cv2.INTER_AREA)
raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

x = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
x = x.to(device=device, dtype=model.dtype)

with torch.no_grad():
    latent = model.encode(x).latent
    recon = model.decode(latent).sample

recon_img = ((recon.clamp(-1, 1) + 1.0) / 2.0 * 255.0).squeeze(0).permute(1, 2, 0).to(torch.float32).cpu().numpy().astype(np.uint8)

# latent: (1, C, h, w) -> PCA-reduce channel dim to 3 for RGB-like display
lat = latent.squeeze(0).to(torch.float32).cpu().numpy()  # (C, h, w)
C, h, w = lat.shape
flat = lat.reshape(C, h * w).T  # (h*w, C)
flat_centered = flat - flat.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(flat_centered, full_matrices=False)
pca3 = (flat_centered @ Vt[:3].T).reshape(h, w, 3)
for k in range(3):
    ch = pca3[:, :, k]
    lo, hi = np.percentile(ch, [1, 99])
    pca3[:, :, k] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)

elapsed_compute = time.perf_counter() - t_loaded

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
axes[0].imshow(raw_rgb)
axes[0].set_title(f"Original image\n({work_size}x{work_size})", fontsize=10)
axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].imshow(pca3)
axes[1].set_title(f"Latent (PCA channels->RGB)\nshape={tuple(lat.shape)}, 32x spatial compression", fontsize=10)
axes[1].set_xticks([]); axes[1].set_yticks([])

axes[2].imshow(recon_img)
axes[2].set_title("Reconstructed (decode)", fontsize=10)
axes[2].set_xticks([]); axes[2].set_yticks([])

fig.suptitle("SANA-VAE (DC-AE) encode -> latent -> decode roundtrip", fontsize=12, fontweight="bold")
fig.text(0.5, 0.02,
          "data/raw/0000.png through mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers (real checkpoint, same as "
          "src/attacks/diffusion_regen.py). PCA fit on this image's own latent channels, top-3 components shown as RGB.",
          ha="center", fontsize=8, style="italic", wrap=True)
fig.subplots_adjust(bottom=0.18, top=0.82, left=0.03, right=0.98, wspace=0.15)
fig.savefig(OUT / "mech4_sana_vae_latent.png", dpi=300)
plt.close(fig)

total = time.perf_counter() - t0
print(f"mech4 total: {total:.2f}s (model load {t_loaded - t0:.2f}s, compute {elapsed_compute:.2f}s)")
