"""
DWT-DCT-SVD Watermarking with swappable interface.
CLI: python -m src.watermark.embed --input ... --output ... [--verify ...]
"""

import argparse
import cv2
import numpy as np
import pywt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple


class WatermarkMethod(ABC):
    @abstractmethod
    def embed(self, image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
        pass

    @abstractmethod
    def extract(self, watermarked: np.ndarray, original: np.ndarray, alpha: float) -> np.ndarray:
        pass

    @abstractmethod
    def verify(self, extracted: np.ndarray, original_watermark: np.ndarray, threshold: float = 0.5) -> Tuple[bool, float]:
        """Returns (is_present, correlation_score)"""
        pass


class DWT_DCT_SVD(WatermarkMethod):
    def __init__(self, dwt_level: int = 2, block_size: int = 8):
        self.dwt_level = dwt_level
        self.block_size = block_size
        self.wavelet = 'haar'

    def _dwt2(self, img: np.ndarray) -> Tuple[np.ndarray, list]:
        coeffs = pywt.wavedec2(img, self.wavelet, level=self.dwt_level)
        return coeffs[0], coeffs[1:]

    def _idwt2(self, LL: np.ndarray, details: list) -> np.ndarray:
        return pywt.waverec2([LL] + details, self.wavelet)

    def embed(self, image: np.ndarray, watermark: np.ndarray, alpha: float) -> np.ndarray:
        if image.ndim == 3:
            yuv = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2YUV)
            y = yuv[:, :, 0]
        else:
            y = image.astype(np.float32)

        h, w = y.shape
        LL, details = self._dwt2(y)
        LL_h, LL_w = LL.shape

        wm_h, wm_w = watermark.shape
        assert wm_h <= LL_h // self.block_size and wm_w <= LL_w // self.block_size

        LL_mod = LL.copy()
        for i in range(wm_h):
            for j in range(wm_w):
                r, c = i * self.block_size, j * self.block_size
                block = LL_mod[r:r+self.block_size, c:c+self.block_size]
                dct_block = cv2.dct(block)
                U, S, Vt = np.linalg.svd(dct_block, full_matrices=False)
                S[0] = S[0] * (1 + alpha * (2 * watermark[i, j] - 1))
                dct_mod = U @ np.diag(S) @ Vt
                LL_mod[r:r+self.block_size, c:c+self.block_size] = cv2.idct(dct_mod)

        watermarked_y = self._idwt2(LL_mod, details)
        watermarked_y = np.clip(watermarked_y, 0, 255)

        if image.ndim == 3:
            yuv[:, :, 0] = watermarked_y
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB).astype(np.uint8)
        return watermarked_y.astype(np.uint8)

    def extract(self, watermarked: np.ndarray, original: np.ndarray, alpha: float) -> np.ndarray:
        if watermarked.ndim == 3:
            y_w = cv2.cvtColor(watermarked.astype(np.float32), cv2.COLOR_RGB2YUV)[:, :, 0]
        else:
            y_w = watermarked.astype(np.float32)
        if original.ndim == 3:
            y_o = cv2.cvtColor(original.astype(np.float32), cv2.COLOR_RGB2YUV)[:, :, 0]
        else:
            y_o = original.astype(np.float32)

        LL_w, _ = self._dwt2(y_w)
        LL_o, _ = self._dwt2(y_o)

        wm_h = LL_w.shape[0] // self.block_size
        wm_w = LL_w.shape[1] // self.block_size
        extracted = np.zeros((wm_h, wm_w), dtype=np.float32)

        for i in range(wm_h):
            for j in range(wm_w):
                r, c = i * self.block_size, j * self.block_size
                _, S_w, _ = np.linalg.svd(cv2.dct(LL_w[r:r+self.block_size, c:c+self.block_size]), full_matrices=False)
                _, S_o, _ = np.linalg.svd(cv2.dct(LL_o[r:r+self.block_size, c:c+self.block_size]), full_matrices=False)
                ratio = S_w[0] / (S_o[0] + 1e-8)
                extracted[i, j] = 1.0 if ratio > 1.0 else 0.0
        return extracted

    def verify(self, extracted: np.ndarray, original_watermark: np.ndarray, threshold: float = 0.5) -> Tuple[bool, float]:
        corr = np.corrcoef(extracted.flatten(), original_watermark.flatten())[0, 1]
        if np.isnan(corr):
            corr = 0.0
        return corr > threshold, float(corr)


def generate_watermark(shape: Tuple[int, int], seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return (np.random.rand(*shape) > 0.5).astype(np.float32)


def jpeg_reencode(image: np.ndarray, quality: int) -> np.ndarray:
    _, enc = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def run_watermark_pipeline(
    input_dir: Path, output_dir: Path,
    method: WatermarkMethod, watermark: np.ndarray, alpha: float,
    img_size: int = 224, save_quality: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(input_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        wm = method.embed(img, watermark, alpha)
        wm_bgr = cv2.cvtColor(wm, cv2.COLOR_RGB2BGR)
        if save_quality is not None:
            wm_bgr = jpeg_reencode(wm_bgr, save_quality)
        cv2.imwrite(str(output_dir / img_path.name), wm_bgr)


def verify_watermarks(
    wm_dir: Path, orig_dir: Path,
    method: WatermarkMethod, watermark: np.ndarray,
    alpha: float, threshold: float = 0.5, img_size: int = 224
) -> dict:
    results = {"passed": 0, "failed": 0, "correlations": []}
    for wm_path in sorted(wm_dir.glob("*.png")):
        orig_path = orig_dir / wm_path.name
        if not orig_path.exists():
            continue
        wm_img = cv2.imread(str(wm_path))
        orig_img = cv2.imread(str(orig_path))
        wm_img = cv2.resize(wm_img, (img_size, img_size))
        orig_img = cv2.resize(orig_img, (img_size, img_size))
        wm_img = cv2.cvtColor(wm_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        ext = method.extract(wm_img, orig_img, alpha)
        passed, corr = method.verify(ext, watermark, threshold)
        results["correlations"].append(corr)
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
    results["mean_corr"] = float(np.mean(results["correlations"])) if results["correlations"] else 0.0
    return results


def main():
    parser = argparse.ArgumentParser(description="DWT-DCT-SVD Watermark Embed/Verify")
    parser.add_argument("--input", type=Path, help="Input directory (raw images)")
    parser.add_argument("--output", type=Path, help="Output directory (watermarked)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Watermark strength")
    parser.add_argument("--block-size", type=int, default=8, help="DCT block size")
    parser.add_argument("--dwt-level", type=int, default=2, help="DWT decomposition level")
    parser.add_argument("--seed", type=int, default=42, help="Watermark seed")

    parser.add_argument("--verify", action="store_true", help="Verify watermarks instead of embedding")
    parser.add_argument("--watermarked", type=Path, help="Watermarked images dir (for verify)")
    parser.add_argument("--original", type=Path, help="Original images dir (for verify)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Correlation threshold")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for resizing")
    parser.add_argument("--save-quality", type=int, default=None, help="JPEG re-encode quality (eliminates format leak between classes)")

    args = parser.parse_args()

    method = DWT_DCT_SVD(dwt_level=args.dwt_level, block_size=args.block_size)
    wm_shape = (224 // (2**args.dwt_level) // args.block_size,) * 2
    watermark = generate_watermark(wm_shape, args.seed)

    if args.verify:
        assert args.watermarked and args.original, "Need --watermarked and --original for verify"
        results = verify_watermarks(
            args.watermarked, args.original, method, watermark, args.alpha, args.threshold, args.img_size
        )
        print(f"Passed: {results['passed']}, Failed: {results['failed']}, Mean corr: {results['mean_corr']:.4f}")
    else:
        assert args.input and args.output, "Need --input and --output for embedding"
        run_watermark_pipeline(args.input, args.output, method, watermark, args.alpha, args.img_size, args.save_quality)
        print(f"Embedded watermark into {len(list(args.output.glob('*.png')))} images")


if __name__ == "__main__":
    main()