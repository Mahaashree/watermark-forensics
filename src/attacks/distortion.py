"""
Distortion attack: JPEG recompression + Gaussian noise + Gaussian blur + resize.
Supports multiple passes for stronger watermark removal.
CLI: python -m src.attacks.distortion --input ... --output ...
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def apply_distortion(
    image: np.ndarray,
    jpeg_quality: int = 30,
    gaussian_std: float = 15.0,
    blur_kernel: int = 5,
    resize_factor: float = 0.5,
) -> np.ndarray:
    h, w = image.shape[:2]

    # 1. Resize down + up (common real-world distortion)
    if resize_factor < 1.0:
        small = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)),
                           interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. JPEG recompression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, enc = cv2.imencode('.jpg', image, encode_param)
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 3. Gaussian noise
    noise = np.random.normal(0, gaussian_std, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Gaussian blur
    if blur_kernel > 1:
        img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

    return img


def jpeg_reencode(image: np.ndarray, quality: int) -> np.ndarray:
    _, enc = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def run_attack_pipeline(
    input_dir: Path, output_dir: Path,
    jpeg_quality: int = 30, gaussian_std: float = 15.0,
    blur_kernel: int = 5, resize_factor: float = 0.5,
    num_passes: int = 3, save_quality: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(input_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        attacked = img
        for _ in range(num_passes):
            attacked = apply_distortion(attacked, jpeg_quality, gaussian_std, blur_kernel, resize_factor)
        if save_quality is not None:
            attacked = jpeg_reencode(attacked, save_quality)
        cv2.imwrite(str(output_dir / img_path.name), attacked)


def main():
    parser = argparse.ArgumentParser(description="Distortion attack (JPEG + noise + blur + resize)")
    parser.add_argument("--input", type=Path, required=True, help="Watermarked images dir")
    parser.add_argument("--output", type=Path, required=True, help="Output dir (removed)")
    parser.add_argument("--jpeg-quality", type=int, default=30)
    parser.add_argument("--gaussian-std", type=float, default=15.0)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--resize-factor", type=float, default=0.5)
    parser.add_argument("--num-passes", type=int, default=3)
    parser.add_argument("--save-quality", type=int, default=None, help="Final JPEG re-encode quality (eliminates format leak between classes)")
    args = parser.parse_args()

    run_attack_pipeline(
        args.input, args.output,
        args.jpeg_quality, args.gaussian_std, args.blur_kernel,
        args.resize_factor, args.num_passes, args.save_quality,
    )
    print(f"Attacked {len(list(args.output.glob('*.png')))} images")


if __name__ == "__main__":
    main()