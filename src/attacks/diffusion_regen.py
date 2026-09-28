"""
Diffusion-regeneration attack: VAE encode/decode roundtrip through SANA's
Deep Compression Autoencoder (DC-AE), used as a fast/weak generative-manifold
reprojection attack ("SANA-VAE" in arXiv:2604.25491, "The Forensic Cost of
Watermark Removal", Evennou & Kijak 2026).

Source / provenance (see phase2_tools_evaluation.md for the full evaluation
of this and three other candidate tools):
  - Model: mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers (HuggingFace)
  - Reference impl: https://github.com/mit-han-lab/efficientvit (Apache-2.0)
  - Introduced in: Chen, Cai, et al., "Deep Compression Autoencoder for
    Efficient High-Resolution Diffusion Models" (arXiv:2410.10733), used in
    SANA (Xie et al., ICLR 2025, arXiv:2410.10629)
  - Integration: HuggingFace `diffusers` AutoencoderDC, diffusers>=0.33
    (pinned in pyproject.toml; AutoencoderDC was added in diffusers 0.32,
    but 0.32 hits an unrelated import-order bug in this environment, so the
    floor is set to the first version confirmed working here, 0.33.1)

Mechanism: passing an image through DC-AE's encoder then decoder projects it
through a heavily compressed (32x spatial) latent bottleneck and back. This
acts as a generative-manifold reprojection that can degrade a fragile
watermark signal while leaving the image visually close to the original --
paper 2 describes this as "faster but typically weaker" than full diffusion
purification (DiffPure) or ControlNet-guided regeneration (CtrlRegen).

CLI: python -m src.attacks.diffusion_regen --input data/watermarked --output data/removed_diffusion
"""

import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# transformers (a diffusers dependency) probes for TensorFlow/Flax at import
# time; in environments where those are installed but built against an
# incompatible numpy ABI, that probe crashes the import even though this
# module only ever uses the PyTorch backend. Force torch-only detection
# before diffusers (and transitively transformers) is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

DEFAULT_MODEL_ID = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dc_ae(model_id: str = DEFAULT_MODEL_ID, device: torch.device | None = None, dtype=torch.float32):
    from diffusers import AutoencoderDC

    device = device or resolve_device("auto")
    model = AutoencoderDC.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return model


def _round_to_multiple(value: int, multiple: int = 32) -> int:
    return max(multiple, round(value / multiple) * multiple)


@torch.no_grad()
def apply_diffusion_regen(
    image: np.ndarray,
    model,
    device: torch.device,
    img_size: int = 224,
) -> np.ndarray:
    """Encode/decode roundtrip attack. `image` is BGR uint8 (OpenCV convention,
    matches distortion.py), returns BGR uint8 of the same shape as input."""
    h, w = image.shape[:2]
    # DC-AE f32 needs spatial dims divisible by 32; round the working size.
    work_size = _round_to_multiple(img_size, 32)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (work_size, work_size), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    x = x.to(device=device, dtype=model.dtype)

    latent = model.encode(x).latent
    recon = model.decode(latent).sample

    recon = ((recon.clamp(-1, 1) + 1.0) / 2.0 * 255.0)
    recon = recon.squeeze(0).permute(1, 2, 0).to(torch.float32).cpu().numpy().astype(np.uint8)

    recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
    return cv2.resize(recon_bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def run_attack_pipeline(
    input_dir: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    device_name: str = "auto",
    img_size: int = 224,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    model = load_dc_ae(model_id, device)

    for img_path in sorted(input_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        attacked = apply_diffusion_regen(img, model, device, img_size)
        cv2.imwrite(str(output_dir / img_path.name), attacked)


def main():
    parser = argparse.ArgumentParser(description="Diffusion-regeneration attack (DC-AE encode/decode roundtrip)")
    parser.add_argument("--input", type=Path, required=True, help="Watermarked images dir")
    parser.add_argument("--output", type=Path, required=True, help="Output dir (removed)")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--img-size", type=int, default=224, help="Working resolution, must be a multiple of 32")
    args = parser.parse_args()

    run_attack_pipeline(args.input, args.output, args.model_id, args.device, args.img_size)
    print(f"Attacked {len(list(args.output.glob('*.png')))} images")


if __name__ == "__main__":
    main()
