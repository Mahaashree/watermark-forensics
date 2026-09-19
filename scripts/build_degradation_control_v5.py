"""
Task 1 control, v5: RNG-replay of build_verified_dataset_v5.py's exact
per-image draw sequence, applied to the raw (never watermarked) image.
Same coeff/generic attack severity per filename as its
watermarked_v5/removed_v5 counterpart -- watermark presence is the only
difference.

CLI: python scripts/build_degradation_control_v5.py --raw data/raw --out data/control_v5
"""

import argparse
import random
import cv2
import numpy as np
from pathlib import Path

from src.watermark.embed import generate_watermark, jpeg_reencode
from src.attacks.distortion import apply_distortion
from src.attacks.coefficient_attack import apply_coefficient_attack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/control_v5"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--dwt-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-quality", type=int, default=80)
    parser.add_argument("--coeff-strength-max", type=float, default=0.8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    wm_shape = (args.img_size // (2 ** args.dwt_level) // args.block_size,) * 2
    generate_watermark(wm_shape, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for img_path in sorted(args.raw.glob("*.png")):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue
        raw_bgr = cv2.resize(raw_bgr, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)

        num_passes = random.choice([0, 1, 1, 2, 2, 3, 4, 5])
        jpeg_quality = random.randint(5, 90)
        gaussian_std = random.uniform(2, 40)
        blur_kernel = random.choice([1, 1, 3, 5, 7, 9])
        resize_factor = random.uniform(0.25, 1.0)
        coeff_strength = random.uniform(0.0, args.coeff_strength_max)

        attacked = apply_coefficient_attack(raw_bgr, args.dwt_level, args.block_size, coeff_strength)
        for _ in range(num_passes):
            attacked = apply_distortion(attacked, jpeg_quality, gaussian_std, blur_kernel, resize_factor)
        attacked = jpeg_reencode(attacked, args.save_quality)

        cv2.imwrite(str(args.out / img_path.name), attacked)
        n_saved += 1

    print(f"Control images saved: {n_saved}")


if __name__ == "__main__":
    main()
