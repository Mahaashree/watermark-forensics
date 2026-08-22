# Experiment 1 — Baseline Reproduction

**Date:** 2026-07-14
**Status:** Complete

## Setup

- **Dataset:** 100 DIV2K images, split into 140 train / 30 val / 30 test (watermarked + removed = 200 total)
- **Watermark:** DWT-DCT-SVD, alpha=0.1, block_size=8, dwt_level=2, seed=42
- **Attack:** 6-pass distortion (JPEG q=2, Gaussian std=45, blur kernel=13, resize 0.1x) — aggressive enough to break watermark verification
- **Model:** ConvNeXt-Tiny (28M params, pretrained on ImageNet)
- **Training:** AdamW, lr=1e-4, batch_size=16, 12 epochs with early stopping (patience=5)

## Watermark Verification Results

| Set | Passed | Failed | Mean Correlation |
|-----|--------|--------|-----------------|
| Watermarked | 97/100 | 3/100 | 0.8721 |
| Removed (attacked) | 14/100 | 86/100 | 0.2989 |

The distortion attack drops verification rate from 97% to 14%, confirming the watermark is effectively removed.

## Classifier Results

| Split | Accuracy | AUROC | Precision | Recall | F1 |
|-------|----------|-------|-----------|--------|----|
| Train | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Test | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Observations

1. **The classifier achieves perfect separation.** This is expected at this scale — the distortion attack (6-pass extreme JPEG + noise + blur + resize) creates very obvious visual degradation that a pretrained ConvNeXt-Tiny can easily detect. The task at this stage is distinguishing "clean watermarked" from "heavily distorted" images.

2. **Small dataset caveat.** With only 100 base images (200 total after attack), overfitting is likely. The 100% accuracy reflects the simplicity of this binary task, not robustness. Phase 2 should use a larger dataset and more realistic attacks.

3. **Watermark robustness.** The DWT-DCT-SVD watermark is surprisingly robust — even with aggressive distortion (JPEG quality=2, Gaussian std=45, blur kernel=13, 6 passes), 14% of images still pass verification. This confirms the watermark has genuine resilience, and removing it requires significant image degradation.

4. **Attack-classifier gap.** The attack that removes the watermark also degrades image quality enough that a classifier trivially distinguishes the two classes. The interesting question for Phase 2 is whether a more subtle attack (diffusion regeneration) can remove the watermark while preserving image quality enough to fool the classifier.

## Files Produced

- `checkpoints/best_model.pt` — trained model weights
- `results/test_metrics.json` — test set evaluation metrics
- `results/logs/train.log` — training log

## Next Steps (Phase 2)

- Replace distortion attack with diffusion-based regeneration
- Test on larger dataset
- Measure whether the classifier still detects removal when the attack is subtle
- Integrate SynthID-Bypass / ComfyUI
