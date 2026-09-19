"""
Build non-watermarked degradation control set (Task 1, phase0-closeout-handoff.md).

Mirrors build_verified_dataset.py's RNG sequence exactly, but skips the
watermark embed step. Same seed, same draw order -> matched attack severity
per filename, watermark presence as the only difference.

CLI: python scripts/build_degradation_control.py --raw data/raw --out data/control_v2
"""

import argparse
import random
import cv2
import numpy as np
from pathlib import Path

from src.watermark.embed import generate_watermark, jpeg_reencode
from src.attacks.distortion import apply_distortion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/control_v2"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--dwt-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-quality", type=int, default=80)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    wm_shape = (args.img_size // (2 ** args.dwt_level) // args.block_size,) * 2
    generate_watermark(wm_shape, args.seed)  # consume numpy RNG, keep streams in sync

    args.out.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for img_path in sorted(args.raw.glob("*.png")):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue
        raw_bgr = cv2.resize(raw_bgr, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)

        # Same draw order as build_verified_dataset.py, for every image.
        num_passes = random.choice([0, 1, 1, 2, 2, 3, 4, 5])
        jpeg_quality = random.randint(5, 90)
        gaussian_std = random.uniform(2, 40)
        blur_kernel = random.choice([1, 1, 3, 5, 7, 9])
        resize_factor = random.uniform(0.25, 1.0)

        attacked = raw_bgr
        for _ in range(num_passes):
            attacked = apply_distortion(attacked, jpeg_quality, gaussian_std, blur_kernel, resize_factor)
        attacked = jpeg_reencode(attacked, args.save_quality)

        cv2.imwrite(str(args.out / img_path.name), attacked)
        n_saved += 1

    print(f"Control images saved: {n_saved}")


if __name__ == "__main__":
    main()
