"""
Build the "Original" (never-watermarked) class for 3-way classification, with
the exact same resize + JPEG re-encode formatting used for watermarked_v2/
removed_v2, so the classifier can't shortcut on file-format artifacts instead
of real watermark signal.
CLI: python -m scripts.build_original_class --raw data/raw --out data/original_v2 --save-quality 80
"""

import argparse
import cv2
from pathlib import Path

from src.watermark.embed import jpeg_reencode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--save-quality", type=int, default=80)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted(args.raw.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
        img = jpeg_reencode(img, args.save_quality)
        cv2.imwrite(str(args.out / img_path.name), img)
        n += 1

    print(f"Wrote {n} original-class images to {args.out}")


if __name__ == "__main__":
    main()
