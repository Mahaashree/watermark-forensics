"""
Build a dataset where labels come from actual watermark verification after a
randomized-strength attack, instead of "which folder we put it in".

Every image goes through a randomly-sized attack (some barely attacked, some
heavily attacked) so both classes share overlapping surface artifacts. The
label is decided by whether DWT_DCT_SVD.verify() still detects the watermark.

Attack mix: each image gets, at random, one of three attacks:
  - CtrlRegen regeneration (--ctrlregen-frac, default 0.0 -- opt-in only,
    see the WARNING below), OR
  - the distortion-optimization attack (--optimization-frac, default 0.5;
    src.attacks.distortion_optimization, greedy zeroth-order search
    targeting DWT-LL block DC coefficients -- UnMarker-inspired, our own
    implementation, see phase2_tools_evaluation.md section 5), OR
  - the generic distortion pipeline (JPEG/noise/blur/resize), the
    remainder.
All three paths go through the exact same mandatory JPEG-80 re-encode and
verify()-based labeling below -- there is no separate pipeline or separate
label rule for any attack type.

WARNING on --ctrlregen-frac: CtrlRegen is a full Stable-Diffusion-scale
regeneration pipeline, measured at ~24 minutes per image on CPU (see
phase2_tools_evaluation.md). It defaults to 0.0 (disabled) for exactly this
reason -- setting it above a small handful of images without a GPU will
make a full dataset run take days, not minutes. Its pipeline (~9.8GB of
weights) is loaded once, lazily, only if --ctrlregen-frac > 0.

CLI: python scripts/build_verified_dataset.py --raw data/raw --out-present data/watermarked_v2 --out-removed data/removed_v2
"""

import argparse
import random
import cv2
import numpy as np
from pathlib import Path

from src.watermark.embed import DWT_DCT_SVD, generate_watermark, jpeg_reencode
from src.attacks.distortion import apply_distortion
from src.attacks.distortion_optimization import greedy_block_attack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-present", type=Path, default=Path("data/watermarked_v2"))
    parser.add_argument("--out-removed", type=Path, default=Path("data/removed_v2"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--dwt-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-quality", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--optimization-frac", type=float, default=0.5,
                         help="Fraction of images attacked with distortion_optimization instead of the generic distortion pipeline")
    parser.add_argument("--opt-epsilon", type=float, default=16.0, help="L-infinity pixel budget for distortion_optimization")
    parser.add_argument("--opt-max-iters", type=int, default=150, help="Max block trials for distortion_optimization")
    parser.add_argument("--ctrlregen-frac", type=float, default=0.0,
                         help="Fraction of images attacked with CtrlRegen (regeneration) instead of the other two attacks. "
                              "WARNING: measured ~24 min/image on CPU -- default 0.0 (opt-in only). See module docstring.")
    parser.add_argument("--ctrlregen-step", type=float, default=0.5, help="CtrlRegen removal-strength/consistency step, in (0, 1]")
    parser.add_argument("--ctrlregen-device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--ctrlregen-img-size", type=int, default=512, help="CtrlRegen's own working resolution (SD1.5-scale, trained at 512)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    method = DWT_DCT_SVD(dwt_level=args.dwt_level, block_size=args.block_size)
    wm_shape = (args.img_size // (2 ** args.dwt_level) // args.block_size,) * 2
    watermark = generate_watermark(wm_shape, args.seed)

    args.out_present.mkdir(parents=True, exist_ok=True)
    args.out_removed.mkdir(parents=True, exist_ok=True)

    ctrlregen_pipe = None
    if args.ctrlregen_frac > 0:
        # Imported lazily -- diffusers/controlnet_aux pull in heavy ML deps
        # (and the transformers TF/JAX import-time probe) that the common
        # case (ctrlregen_frac=0) shouldn't have to pay for.
        from src.attacks.regeneration_ctrlregen import resolve_device, load_pipeline, regenerate as ctrlregen_regenerate
        ctrlregen_device = resolve_device(args.ctrlregen_device)
        print(f"Loading CtrlRegen pipeline on {ctrlregen_device} (one-time; ~9.8GB download if not cached)...")
        ctrlregen_pipe = load_pipeline(ctrlregen_device)

    n_present, n_removed = 0, 0
    n_optimization_attack = 0
    n_optimization_removed_after_jpeg = 0
    n_ctrlregen_attack = 0
    n_ctrlregen_removed_after_jpeg = 0

    for idx, img_path in enumerate(sorted(args.raw.glob("*.png"))):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue
        raw_bgr = cv2.resize(raw_bgr, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        wm_rgb = method.embed(raw_rgb, watermark, args.alpha)
        wm_bgr = cv2.cvtColor(wm_rgb, cv2.COLOR_RGB2BGR)

        roll = random.random()
        use_ctrlregen_attack = ctrlregen_pipe is not None and roll < args.ctrlregen_frac
        use_optimization_attack = (
            not use_ctrlregen_attack
            and roll < args.ctrlregen_frac + args.optimization_frac
        )

        if use_ctrlregen_attack:
            n_ctrlregen_attack += 1
            attacked = ctrlregen_regenerate(
                ctrlregen_pipe, wm_bgr, args.ctrlregen_step,
                seed=args.seed + idx, size=args.ctrlregen_img_size,
            )
        elif use_optimization_attack:
            n_optimization_attack += 1
            opt_result = greedy_block_attack(
                wm_rgb, raw_rgb, method, watermark, args.alpha,
                threshold=args.threshold, epsilon=args.opt_epsilon,
                max_iters=args.opt_max_iters, seed=args.seed + idx,
            )
            attacked = cv2.cvtColor(opt_result["image"], cv2.COLOR_RGB2BGR)
        else:
            # Randomized attack strength: spans "barely touched" to "heavily attacked"
            num_passes = random.choice([0, 1, 1, 2, 2, 3, 4, 5])
            jpeg_quality = random.randint(5, 90)
            gaussian_std = random.uniform(2, 40)
            blur_kernel = random.choice([1, 1, 3, 5, 7, 9])
            resize_factor = random.uniform(0.25, 1.0)

            attacked = wm_bgr
            for _ in range(num_passes):
                attacked = apply_distortion(attacked, jpeg_quality, gaussian_std, blur_kernel, resize_factor)

        # Mandatory for every attack class, no exceptions -- this is the
        # format-leak fix; skipping it for one attack type would reopen it.
        attacked = jpeg_reencode(attacked, args.save_quality)

        attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
        extracted = method.extract(attacked_rgb, raw_rgb, args.alpha)
        present, corr = method.verify(extracted, watermark, args.threshold)

        if use_optimization_attack and not present:
            n_optimization_removed_after_jpeg += 1
        if use_ctrlregen_attack and not present:
            n_ctrlregen_removed_after_jpeg += 1

        out_dir = args.out_present if present else args.out_removed
        cv2.imwrite(str(out_dir / img_path.name), attacked)

        if present:
            n_present += 1
        else:
            n_removed += 1

    print(f"Watermark present: {n_present}, removed: {n_removed}")
    if n_optimization_attack:
        print(
            f"distortion_optimization attack: {n_optimization_attack} images attacked, "
            f"{n_optimization_removed_after_jpeg}/{n_optimization_attack} still 'removed' "
            f"after mandatory JPEG-{args.save_quality} re-encode "
            f"({n_optimization_removed_after_jpeg / n_optimization_attack:.1%} survival)"
        )
    if n_ctrlregen_attack:
        print(
            f"CtrlRegen attack: {n_ctrlregen_attack} images attacked, "
            f"{n_ctrlregen_removed_after_jpeg}/{n_ctrlregen_attack} still 'removed' "
            f"after mandatory JPEG-{args.save_quality} re-encode "
            f"({n_ctrlregen_removed_after_jpeg / n_ctrlregen_attack:.1%} survival)"
        )


if __name__ == "__main__":
    main()
