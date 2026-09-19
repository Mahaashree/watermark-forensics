"""
Task 2: BMP shortcut-control ablation. Re-encodes an image directory to
lossless BMP. cv2.imread decodes PNG and BMP to identical pixel arrays
(both lossless), so this isolates whether the classifier depends on any
file-format-specific signal vs. pixel content alone.

CLI: python scripts/convert_to_bmp.py --input data/watermarked_v4 --output data/watermarked_v4_bmp
"""

import argparse
import cv2
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted(args.input.glob("*.png")):
        img = cv2.imread(str(img_path))
        cv2.imwrite(str(args.output / f"{img_path.stem}.bmp"), img)
        n += 1
    print(f"Converted {n} images: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
