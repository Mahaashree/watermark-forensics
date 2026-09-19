"""
v3: same verification-based labeling as build_verified_dataset.py, but Layer 2
now composes the targeted coefficient-domain attack (attacks the DWT/DCT/SVD
embedding domain directly) with the existing generic pixel-domain distortion,
instead of generic distortion alone. Fixes the root cause found in Task 1
(phase0-closeout-handoff.md): generic-only attacks never touch the embedding
domain, so removal residue was indistinguishable from plain degradation.

CLI: python scripts/build_verified_dataset_v3.py --raw data/raw --out-present data/watermarked_v3 --out-removed data/removed_v3
"""

import argparse
import random
import cv2
import numpy as np
from pathlib import Path

from src.watermark.embed import DWT_DCT_SVD, generate_watermark, jpeg_reencode
from src.attacks.distortion import apply_distortion
from src.attacks.coefficient_attack import apply_coefficient_attack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-present", type=Path, default=Path("data/watermarked_v3"))
    parser.add_argument("--out-removed", type=Path, default=Path("data/removed_v3"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--dwt-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-quality", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--coeff-strength-max", type=float, default=0.3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    method = DWT_DCT_SVD(dwt_level=args.dwt_level, block_size=args.block_size)
    wm_shape = (args.img_size // (2 ** args.dwt_level) // args.block_size,) * 2
    watermark = generate_watermark(wm_shape, args.seed)

    args.out_present.mkdir(parents=True, exist_ok=True)
    args.out_removed.mkdir(parents=True, exist_ok=True)

    n_present, n_removed = 0, 0

    for img_path in sorted(args.raw.glob("*.png")):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue
        raw_bgr = cv2.resize(raw_bgr, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        wm_rgb = method.embed(raw_rgb, watermark, args.alpha)
        wm_bgr = cv2.cvtColor(wm_rgb, cv2.COLOR_RGB2BGR)

        # Randomized attack strength, same draw order for every image.
        num_passes = random.choice([0, 1, 1, 2, 2, 3, 4, 5])
        jpeg_quality = random.randint(5, 90)
        gaussian_std = random.uniform(2, 40)
        blur_kernel = random.choice([1, 1, 3, 5, 7, 9])
        resize_factor = random.uniform(0.25, 1.0)
        coeff_strength = random.uniform(0.0, args.coeff_strength_max)

        # Targeted attack first (embedding domain), then generic pixel-domain
        # distortion, matching a realistic remove-then-recompress workflow.
        attacked = apply_coefficient_attack(wm_bgr, args.dwt_level, args.block_size, coeff_strength)
        for _ in range(num_passes):
            attacked = apply_distortion(attacked, jpeg_quality, gaussian_std, blur_kernel, resize_factor)
        attacked = jpeg_reencode(attacked, args.save_quality)

        attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
        extracted = method.extract(attacked_rgb, raw_rgb, args.alpha)
        present, corr = method.verify(extracted, watermark, args.threshold)

        out_dir = args.out_present if present else args.out_removed
        cv2.imwrite(str(out_dir / img_path.name), attacked)

        if present:
            n_present += 1
        else:
            n_removed += 1

    print(f"Watermark present: {n_present}, removed: {n_removed}")


if __name__ == "__main__":
    main()
