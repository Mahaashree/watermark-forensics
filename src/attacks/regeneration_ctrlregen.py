"""
Regeneration attack: CtrlRegen (ControlNet-guided diffusion regeneration
from clean noise), wired against arbitrary input images -- scheme-agnostic,
does not need to know anything about how the watermark was embedded.

Source / provenance:
  - Paper: Liu, Song, Ci, Zhang, Wang, Shou, Bu, "Image Watermarks Are
    Removable Using Controllable Regeneration from Clean Noise,"
    arXiv:2410.05470.
  - Official repo: https://github.com/yepengliu/CtrlRegen, pinned commit
    6671b4b3a83f0f43e04dc1a08952b27d26197429 (fetched 2026-09-20).
  - Vendored, byte-for-byte unmodified, at third_party/ctrlregen/ (the repo
    has no pip package -- its own documented usage is clone-and-import from
    the same directory, so vendoring is not a packaging choice made here).
  - Model weights: yepengliu/ctrlregen (spatial + semantic control nets),
    SG161222/Realistic_Vision_V4.0_noVAE (base SD1.5-family checkpoint),
    facebook/dinov2-giant (image encoder), stabilityai/sd-vae-ft-mse.

License status: the CtrlRegen source repository has **no LICENSE file** as
of the pinned commit (confirmed via the GitHub API and a full file listing,
not just an unclear tag). Used here under default copyright for internal
development only, per explicit project instruction -- see
phase2_tools_evaluation.md for the full note. This blocks CtrlRegen from
any reproducibility-package or redistribution use until the authors
clarify licensing terms.

Bug fixed relative to the upstream demo notebook: the notebook hardcodes
`torch_dtype=torch.float16` unconditionally, even when
`torch.cuda.is_available()` is False. PyTorch's CPU backend does not
support fp16 for most conv/attention ops, so running the notebook's own
code as-is on CPU either errors outright or silently mixes dtypes. This
module instead selects `torch.float16` only when actually running on CUDA
and `torch.float32` otherwise, consistently for every component (base
pipeline, ControlNet, VAE, image encoder) -- not a partial/mixed-dtype
patch.

CLI: python -m src.attacks.regeneration_ctrlregen --input data/watermarked --output data/removed_ctrlregen --original data/raw
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Same workaround as src/attacks/diffusion_regen.py: transformers (a
# diffusers dependency) probes for TensorFlow/Flax at import time; in this
# environment those are installed but built against a numpy ABI
# incompatible with the numpy 2.x actually installed, which crashes the
# import even though this module only ever uses the PyTorch backend.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler, AutoencoderKL
# Import directly from the canny submodule, not `controlnet_aux` itself --
# the package's __init__.py eagerly imports MediapipeFaceDetector (unused
# here), which pulls in mediapipe -> tensorflow and hits the same numpy-2
# ABI crash as the transformers TF/JAX probe above, but as a hard top-level
# import in controlnet_aux's own __init__.py, not gated by USE_TF.
from controlnet_aux.canny import CannyDetector
from transformers import AutoModel, AutoImageProcessor

_THIRD_PARTY_DIR = str(Path(__file__).resolve().parents[2] / "third_party" / "ctrlregen")
if _THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_DIR)

from custom_i2i_pipeline import CustomStableDiffusionControlNetImg2ImgPipeline  # noqa: E402
from utils import color_match  # noqa: E402

BASE_MODEL_ID = "SG161222/Realistic_Vision_V4.0_noVAE"
CTRLREGEN_REPO = "yepengliu/ctrlregen"
SPATIAL_SUBFOLDER = "spatialnet_ckp/spatial_control_ckp_14000"
SEMANTIC_SUBFOLDER = "semanticnet_ckp/models"
SEMANTIC_WEIGHT_NAME = "semantic_control_ckp_435000.bin"
IMAGE_ENCODER_ID = "facebook/dinov2-giant"
VAE_ID = "stabilityai/sd-vae-ft-mse"


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipeline(device: torch.device):
    # The dtype fix: fp16 only on CUDA, fp32 everywhere else, applied
    # uniformly to every component so nothing ends up mismatched.
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    spatialnet = [ControlNetModel.from_pretrained(
        CTRLREGEN_REPO, subfolder=SPATIAL_SUBFOLDER, torch_dtype=dtype
    )]
    pipe = CustomStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=spatialnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.costum_load_ip_adapter(CTRLREGEN_REPO, subfolder=SEMANTIC_SUBFOLDER, weight_name=SEMANTIC_WEIGHT_NAME)
    pipe.image_encoder = AutoModel.from_pretrained(IMAGE_ENCODER_ID).to(device, dtype=dtype)
    pipe.feature_extractor = AutoImageProcessor.from_pretrained(IMAGE_ENCODER_ID)
    pipe.vae = AutoencoderKL.from_pretrained(VAE_ID).to(device, dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_ip_adapter_scale(1.0)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    return pipe


def regenerate(pipe, image_bgr: np.ndarray, step: float, seed: int, size: int = 512) -> np.ndarray:
    """One CtrlRegen regeneration pass. `image_bgr` is OpenCV BGR uint8
    (matches distortion.py/diffusion_regen.py convention); returns the same.
    `step` in (0, 1] controls removal strength vs. consistency (paper's own
    parameter -- higher pushes further toward pure regeneration)."""
    device = pipe.device
    dtype = pipe.unet.dtype

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)

    processor = CannyDetector()
    control_img = processor(pil_img, low_threshold=100, high_threshold=150)

    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        "best quality, high quality",
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        image=[pil_img],
        control_image=[control_img],
        ip_adapter_image=[pil_img],
        strength=step,
        generator=generator,
        num_inference_steps=50,
        controlnet_conditioning_scale=1.0,
        guidance_scale=2.0,
        control_guidance_start=0,
        control_guidance_end=1,
    ).images[0]
    result = color_match(pil_img, result)

    out_rgb = np.array(result.resize((image_bgr.shape[1], image_bgr.shape[0]), Image.BILINEAR))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def run_attack_pipeline(
    input_dir: Path,
    output_dir: Path,
    step: float = 0.5,
    seed: int = 0,
    device_name: str = "auto",
    img_size: int = 512,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    pipe = load_pipeline(device)

    for img_path in sorted(input_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        attacked = regenerate(pipe, img, step, seed, img_size)
        cv2.imwrite(str(output_dir / img_path.name), attacked)


def main():
    parser = argparse.ArgumentParser(description="CtrlRegen regeneration attack (ControlNet-guided diffusion regeneration)")
    parser.add_argument("--input", type=Path, required=True, help="Watermarked images dir")
    parser.add_argument("--output", type=Path, required=True, help="Output dir (removed)")
    parser.add_argument("--step", type=float, default=0.5, help="Removal strength vs. consistency, in (0, 1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--img-size", type=int, default=512, help="Working resolution (CtrlRegen's base model is SD1.5-scale, trained at 512)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    run_attack_pipeline(args.input, args.output, args.step, args.seed, args.device, args.img_size)
    print(f"Attacked {len(list(args.output.glob('*.png')))} images in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
