"""
Targeted coefficient-domain attack.

Perturbs the dominant SVD singular value of each DCT block inside the DWT
LL sub-band -- the exact coefficient DWT_DCT_SVD.embed() modifies to encode
a watermark bit. Image-in/image-out, no watermark awareness (same contract
as distortion.py's apply_distortion): works on any image, watermarked or not.

Unlike distortion.py (pixel-domain: resize/jpeg/noise/blur), this attacks
the embedding domain directly, so it can defeat the watermark without
requiring heavy generic degradation.

CLI: python -m src.attacks.coefficient_attack --input ... --output ...
"""

import argparse
import cv2
import numpy as np
import pywt
from pathlib import Path


def apply_coefficient_attack(
    image: np.ndarray,
    dwt_level: int = 2,
    block_size: int = 8,
    strength: float = 0.15,
) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    y = yuv[:, :, 0]

    coeffs = pywt.wavedec2(y, 'haar', level=dwt_level)
    LL, details = coeffs[0], coeffs[1:]
    h, w = LL.shape
    nb_h, nb_w = h // block_size, w // block_size

    LL_mod = LL.copy()
    for i in range(nb_h):
        for j in range(nb_w):
            r, c = i * block_size, j * block_size
            block = LL_mod[r:r + block_size, c:c + block_size]
            dct_block = cv2.dct(block)
            U, S, Vt = np.linalg.svd(dct_block, full_matrices=False)
            # Random re-perturbation of the exact coefficient the watermark
            # bit lives in -- scrambles the embedded up/down signal without
            # knowing which direction it was pushed.
            S[0] = S[0] * (1 + np.random.uniform(-strength, strength))
            dct_mod = U @ np.diag(S) @ Vt
            LL_mod[r:r + block_size, c:c + block_size] = cv2.idct(dct_mod)

    attacked_y = pywt.waverec2([LL_mod] + details, 'haar')
    attacked_y = np.clip(attacked_y[:h * (2 ** dwt_level), :w * (2 ** dwt_level)], 0, 255)

    yuv_out = yuv.copy()
    yuv_out[:, :, 0] = attacked_y
    rgb_out = cv2.cvtColor(yuv_out, cv2.COLOR_YUV2RGB)
    rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


def run_attack_pipeline(
    input_dir: Path, output_dir: Path,
    dwt_level: int = 2, block_size: int = 8, strength: float = 0.15,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(input_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        attacked = apply_coefficient_attack(img, dwt_level, block_size, strength)
        cv2.imwrite(str(output_dir / img_path.name), attacked)


def main():
    parser = argparse.ArgumentParser(description="Targeted coefficient-domain attack")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dwt-level", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.15)
    args = parser.parse_args()

    run_attack_pipeline(args.input, args.output, args.dwt_level, args.block_size, args.strength)
    print(f"Attacked {len(list(args.output.glob('*.png')))} images")


if __name__ == "__main__":
    main()
