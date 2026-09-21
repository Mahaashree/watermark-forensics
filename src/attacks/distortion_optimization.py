"""
Distortion-optimization attack: iterative perturbation directly targeting the
spectral-domain carriers our own DWT-DCT-SVD watermark uses (the top singular
value of each 8x8 DCT block inside the DWT LL subband), driving
src.watermark.embed.DWT_DCT_SVD.verify()'s correlation score below its
detection threshold while bounding perceptual distortion with an
L-infinity pixel budget.

Algorithmic inspiration: UnMarker (Kassis & Hengartner, IEEE S&P 2025,
arXiv:2405.08363) attacks watermarks by perturbing spectral-amplitude
carriers rather than raw pixels. UnMarker itself is disqualified for direct
integration (see phase2_tools_evaluation.md, section 5: non-commercial
license, GPU>=32GB / CUDA-only wheel pins, and hard-coupling to 10
predefined published watermarking schemes with no generic single-image
path). THIS FILE IS OUR OWN FROM-SCRATCH IMPLEMENTATION, written against our
own verifier (src.watermark.embed.DWT_DCT_SVD -- there is no separate
src/watermark/verify.py in this codebase; verify()/extract() live on that
class) -- it does not reuse, port, or depend on any UnMarker code.

Why zeroth-order, not gradient-based: DWT_DCT_SVD.extract()/verify() are
pure NumPy/OpenCV/PyWavelets (pywt.wavedec2, cv2.dct, np.linalg.svd, a hard
`1.0 if ratio > 1.0 else 0.0` per-block threshold, np.corrcoef) -- there are
no PyTorch tensors anywhere in that pipeline, so there is no computational
graph to backprop through, and the hard per-block threshold would kill
gradient signal even if the upstream ops were reimplemented in a
differentiable framework.

What gets perturbed, why, and how the optimizer was chosen (two iterations
of this attack failed before this one worked -- kept here because the
failures are informative about this verifier's score landscape): the
watermark's detectable signal is the top singular value S[0] of the DCT of
each 8x8 LL-subband block; DWT_DCT_SVD.embed() moves S[0] by a
multiplicative +/-alpha shift via the block's own SVD. (1) Perturbing raw
spatial LL pixels with independent random noise per pixel measurably
changed the image but never flipped a single block's bit -- noise spread
across all 64 coefficients of a block mostly lands off the dominant
singular direction. (2) Restricting the search to one scalar per block
(the block's DCT DC coefficient, which empirically moves S[0] almost 1:1)
and driving it with SPSA (a random +/-1 direction shared across all 49
blocks, finite-difference gradient estimate) still failed: each block's
extracted bit is a hard threshold, so the score is a step function with a
wide flat region around any single random direction's probe magnitude --
score_plus/score_minus came back identical for every direction tried,
giving SPSA a zero gradient estimate to work from every time. (3) What
works: because the wm_h x wm_w blocks tile the LL subband disjointly (each
block's DC term maps to its own non-overlapping pixel patch -- confirmed
empirically), one block's perturbation can never push another block's
pixels past the L-infinity budget. That makes a **greedy per-block
coordinate search** both correct and cheap: for each block, in random
order, trial the two directions at the block's own max-available DC
magnitude (given the remaining L-infinity budget), keep whichever direction
(if either) lowers verify()'s observed correlation score, else leave that
block unperturbed. This never inspects S[0]/S_o[0] directly -- every
decision is made purely from verify()'s returned scalar, so it stays a
black-box, zeroth-order search over a 49-dimensional discrete space, not
an analytical inversion of embed(). Restricting that search to the DC
coefficient of the exact blocks the watermark lives in -- rather than
raw pixel space (~150k dims) or the full LL subband (3136 coefficients) --
is the concrete form "spectral-amplitude targeting" takes here.

Distortion budget: L-infinity, not SSIM. Chosen because (a) it can be
enforced exactly with a clip at every step, with no differentiable or
approximate perceptual-metric machinery needed in a zeroth-order loop;
(b) it matches the perturbation-budget convention of the PGD/adversarial-
example lineage this attack is modeled on; (c) the disjoint-block tiling
property above only holds cleanly for a per-pixel budget (L-infinity) --
SSIM is a joint, non-separable metric over the whole image, so it can't be
allocated per-block independently the way this attack needs. SSIM is
still measured and reported in the demo as a descriptive, non-enforced
sanity check on the result, not as the constraint itself.

CLI: python -m src.attacks.distortion_optimization --input data/watermarked --output data/removed_optim --original data/raw
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path

from src.watermark.embed import DWT_DCT_SVD, generate_watermark


def linf_project(candidate: np.ndarray, reference: np.ndarray, epsilon: float) -> np.ndarray:
    delta = np.clip(candidate - reference, -epsilon, epsilon)
    return np.clip(reference + delta, 0, 255)


def _score(method: DWT_DCT_SVD, image: np.ndarray, original: np.ndarray, watermark: np.ndarray, alpha: float, threshold: float) -> float:
    extracted = method.extract(image, original, alpha)
    _, corr = method.verify(extracted, watermark, threshold)
    return corr


def _apply_dc_deltas(LL: np.ndarray, deltas: np.ndarray, block_size: int) -> np.ndarray:
    """Add deltas[i, j] to the DC coefficient (DCT[0,0]) of LL block (i, j)."""
    LL_new = LL.copy()
    wm_h, wm_w = deltas.shape
    for i in range(wm_h):
        for j in range(wm_w):
            r, c = i * block_size, j * block_size
            block = LL_new[r:r + block_size, c:c + block_size]
            dct_block = cv2.dct(block)
            dct_block[0, 0] += deltas[i, j]
            LL_new[r:r + block_size, c:c + block_size] = cv2.idct(dct_block)
    return LL_new


def greedy_block_attack(
    watermarked: np.ndarray,
    original: np.ndarray,
    method: DWT_DCT_SVD,
    watermark: np.ndarray,
    alpha: float,
    threshold: float = 0.5,
    epsilon: float = 16.0,
    max_iters: int = 150,
    seed: int = 0,
) -> dict:
    """Greedy zeroth-order coordinate search over one scalar per LL block
    (each block's DCT DC coefficient), L-infinity-bounded in pixel space
    around `watermarked`. See the module docstring for why this replaced an
    SPSA-based first attempt.

    One iteration = one block trial (both +/- directions tried, better one
    kept if it improves verify()'s score, else left alone). Blocks are
    visited in a random (seeded) order, at most once each per full pass;
    `max_iters` also upper-bounds the number of block trials, so it can be
    set below the total block count to see how quickly the score falls.

    Requires `original` (the pre-watermark source image) because
    DWT_DCT_SVD.extract() is defined relative to it -- consistent with how
    the rest of this codebase already uses verify() (see
    scripts/build_verified_dataset.py), not an extra assumption introduced
    here.

    Returns: dict(image, score, passed, iters_used, converged, elapsed_s).
    """
    rng = np.random.default_rng(seed)
    watermarked_f = watermarked.astype(np.float32)
    original_f = original.astype(np.float32)

    y_o = cv2.cvtColor(original_f, cv2.COLOR_RGB2YUV)[:, :, 0]
    LL_o, _ = method._dwt2(y_o)
    wm_h = LL_o.shape[0] // method.block_size
    wm_w = LL_o.shape[1] // method.block_size

    # A delta of size d applied to one block's DC coefficient reconstructs,
    # empirically, to a pixel-space delta of roughly d / (2**dwt_level *
    # block_size) -- one factor of 2**dwt_level from the DWT reconstruction
    # and one factor of block_size from the DC term's energy spreading
    # evenly across the block on idct (both confirmed by direct
    # measurement). epsilon is specified in pixel units for an intuitive
    # CLI; this converts it to the DC-coefficient magnitude available.
    dc_to_pixel_scale = (2 ** method.dwt_level) * method.block_size
    dc_budget = epsilon * dc_to_pixel_scale

    yuv = cv2.cvtColor(watermarked_f, cv2.COLOR_RGB2YUV)
    LL, details = method._dwt2(yuv[:, :, 0])

    def image_from_deltas(deltas: np.ndarray) -> np.ndarray:
        LL_new = _apply_dc_deltas(LL, deltas, method.block_size)
        y_new = method._idwt2(LL_new, details)
        yuv_new = yuv.copy()
        yuv_new[:, :, 0] = y_new
        img = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)
        # Blocks tile the LL subband disjointly, so per-block max-magnitude
        # deltas can only ever hit exactly epsilon per pixel, never exceed
        # it -- this clip is a safety net, not load-bearing.
        return linf_project(img, watermarked_f, epsilon)

    t0 = time.perf_counter()
    deltas = np.zeros((wm_h, wm_w), dtype=np.float32)
    best_score = _score(method, watermarked_f, original_f, watermark, alpha, threshold)
    converged = best_score <= threshold
    order = rng.permutation(wm_h * wm_w)
    iters_used = 0

    for it in range(min(max_iters, len(order))):
        if converged:
            break
        iters_used = it + 1
        i, j = divmod(int(order[it]), wm_w)

        best_local_score, best_local_sign = best_score, None
        for sign in (1.0, -1.0):
            trial = deltas.copy()
            trial[i, j] = sign * dc_budget
            score = _score(method, image_from_deltas(trial), original_f, watermark, alpha, threshold)
            if score < best_local_score:
                best_local_score, best_local_sign = score, sign

        if best_local_sign is not None:
            deltas[i, j] = best_local_sign * dc_budget
            best_score = best_local_score
        converged = best_score <= threshold

    best_image = image_from_deltas(deltas)
    elapsed = time.perf_counter() - t0
    return {
        "image": np.clip(np.round(best_image), 0, 255).astype(np.uint8),
        "score": best_score,
        "passed": best_score <= threshold,  # "passed" = attack succeeded (watermark no longer detected)
        "iters_used": iters_used,
        "converged": converged,
        "elapsed_s": elapsed,
        "deltas": deltas,  # per-block DC-coefficient delta actually applied to `best_image`, for inspection/visualization
    }


def run_attack_pipeline(
    input_dir: Path,
    original_dir: Path,
    output_dir: Path,
    method: DWT_DCT_SVD,
    watermark: np.ndarray,
    alpha: float,
    threshold: float = 0.5,
    epsilon: float = 16.0,
    max_iters: int = 150,
    seed: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"attacked": 0, "defeated_verify": 0, "iters": []}

    for wm_path in sorted(input_dir.glob("*.png")):
        orig_path = original_dir / wm_path.name
        if not orig_path.exists():
            continue
        watermarked = cv2.cvtColor(cv2.imread(str(wm_path)), cv2.COLOR_BGR2RGB)
        original = cv2.cvtColor(cv2.imread(str(orig_path)), cv2.COLOR_BGR2RGB)
        if watermarked.shape != original.shape:
            original = cv2.resize(original, (watermarked.shape[1], watermarked.shape[0]))

        result = greedy_block_attack(
            watermarked, original, method, watermark, alpha,
            threshold, epsilon, max_iters, seed,
        )
        out_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / wm_path.name), out_bgr)

        summary["attacked"] += 1
        summary["iters"].append(result["iters_used"])
        if result["passed"]:
            summary["defeated_verify"] += 1

    return summary


def main():
    parser = argparse.ArgumentParser(description="Distortion-optimization attack (greedy zeroth-order search over DWT-LL block DC coefficients, L-inf bounded)")
    parser.add_argument("--input", type=Path, required=True, help="Watermarked images dir")
    parser.add_argument("--output", type=Path, required=True, help="Output dir (removed)")
    parser.add_argument("--original", type=Path, required=True, help="Pre-watermark source images dir (needed by extract())")
    parser.add_argument("--alpha", type=float, default=0.1, help="Watermark strength used at embed time")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--dwt-level", type=int, default=2)
    parser.add_argument("--watermark-seed", type=int, default=42, help="Seed used to generate the ground-truth watermark bits")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=16.0, help="L-infinity pixel budget (0-255 scale)")
    parser.add_argument("--max-iters", type=int, default=150, help="Max block trials (one per iteration)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for block visitation order")
    args = parser.parse_args()

    method = DWT_DCT_SVD(dwt_level=args.dwt_level, block_size=args.block_size)
    wm_shape = (args.img_size // (2 ** args.dwt_level) // args.block_size,) * 2
    watermark = generate_watermark(wm_shape, args.watermark_seed)

    summary = run_attack_pipeline(
        args.input, args.original, args.output, method, watermark, args.alpha,
        args.threshold, args.epsilon, args.max_iters, args.seed,
    )
    mean_iters = np.mean(summary["iters"]) if summary["iters"] else 0.0
    print(
        f"Attacked {summary['attacked']} images, "
        f"defeated verify() on {summary['defeated_verify']}/{summary['attacked']}, "
        f"mean iters {mean_iters:.1f}"
    )


if __name__ == "__main__":
    main()
