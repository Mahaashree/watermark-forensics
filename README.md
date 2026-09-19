# Watermark-Removal Forensics — Phase 1 Baseline Reproduction

End-to-end pipeline: **Watermark → Attack → Dataset → Classifier → Evaluation**

## Quick Start (Colab T4)

```bash
# 1. Install
pip install -e .


# 2. Download DIV2K validation (~800 images)
python scripts/download_div2k.py --max-images 800

# 3. Embed DWT-DCT-SVD watermark (class 0)
python -m src.watermark.embed --input data/raw --output data/watermarked \
  --alpha 0.1 --block-size 8 --dwt-level 2

# 4. Verify watermark present
python -m src.watermark.embed --verify --watermarked data/watermarked --original data/raw

# 4. Apply distortion attack (class 1)
python -m src.attacks.distortion --input data/watermarked --output data/removed \
  --jpeg-quality 50 --gaussian-std 5 --blur-kernel 3

# 5. Verify attack removes watermark
python -m src.watermark.embed --verify --watermarked data/removed --original data/raw

# 6. Build stratified splits (70/15/15)
python -m src.data.build_splits --watermarked data/watermarked --removed data/removed --splits data/splits

# 7. Train ConvNeXt-Tiny classifier
python -m src/train.py --config configs/default.yaml

# 8. Evaluate on test set
python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test
```

## Project Structure
```
watermark-forensics-implementation/
├── configs/default.yaml           # All hyperparameters
├── scripts/download_div2k.py      # DIV2K downloader (mirror-first)
├── src/
│   ├── watermark/embed.py         # DWT-DCT-SVD + CLI (embed/verify)
│   ├── attacks/distortion.py      # JPEG + noise + blur + CLI
│   ├── attacks/diffusion_regen.py # Phase 2 stub
│   ├── edits/generic_edit.py      # Phase 3 stub
│   ├── data/build_splits.py       # Stratified splits + CLI
│   ├── data/dataset.py            # PyTorch Dataset + DataLoaders
│   ├── models/classifier.py       # ConvNeXt-Tiny via timm
│   ├── train.py                   # Training loop (OOM fallback, AMP)
│   ├── evaluate.py                # Metrics (acc, AUROC, P/R/F1)
│   └── utils/{seed.py,logging.py} # Reproducibility + logging
├── experiments/exp1_baseline_repro/
├── notebooks/exploration.ipynb
├── checkpoints/                   # gitignored
└── results/{logs,figures}/        # gitignored
```

## Configuration
All hyperparameters in `configs/default.yaml`:
- `train.batch_size: 32` (auto-fallback to 16 on OOM)
- `model.name: "convnext_tiny"` (pretrained ImageNet)
- `train.mixed_precision: true` (uses `torch.amp`, falls back to `torch.cuda.amp`)
- `watermark.alpha: 0.1`, `block_size: 8`, `dwt_level: 2` → ~49-bit watermark
- `attack.jpeg_quality: 50`, `gaussian_std: 5.0`, `blur_kernel: 3`

## Augmentation Policy (Phase 0, Task 4)

Train-time augmentation in `src/data/dataset.py` is deliberately minimal:
`Resize` + `HorizontalFlip(p=0.5)` only, matching the eval-time transform
plus one safe augmentation. No `RandomResizedCrop`, no `ColorJitter`.

**Why:** the classifier's target signal is a watermark-removal residue
living in the Y-channel DCT/SVD domain, embedded on a fixed 8x8 block
grid (`src/watermark/embed.py`). Two of the previously-applied
augmentations directly threaten that signal:
- `RandomResizedCrop` resamples pixels at arbitrary offsets/scales,
  breaking alignment with the 8x8 embedding grid and blurring the exact
  high-frequency pattern the residue lives in.
- `ColorJitter`'s brightness/contrast terms rescale pixel values, which
  directly perturbs the Y-channel SVD singular values the watermark (and
  its removal-residue) live on — the same coefficient
  `DWT_DCT_SVD.embed()` modifies to encode a bit.

`HorizontalFlip` is kept: image size (224) is divisible by the DWT/block
grid (8), so a full-image horizontal flip maps block boundaries onto
block boundaries — it mirrors spatial layout without resampling pixels
or rescaling values, so it doesn't disturb the residue signal.

This applies going forward from Phase 0 Task 4; results reported before
this point (`results/v2`–`results/v5`) were trained under the old
`RandomResizedCrop` + `ColorJitter` policy and are not re-run against
this change unless noted.

## Alpha History (Phase 0, Task 5)

`configs/default.yaml` states `watermark.alpha: 0.1`, but no reported
result was ever produced with it — this was a documentation gap, not a
config anyone should run expecting the numbers below. Actual alpha per
config, in order used:

| Config | alpha | Why |
|---|---|---|
| `default.yaml` | 0.1 | Phase 1 original; construction-labeled, superseded, never produced a reported number |
| `v2.yaml` / `v3.yaml` | 0.02 | Tuned down until watermark removal succeeded on ~50% of images post-attack (a workable train/test class balance), not for realistic deployment strength. **Known limitation**, not a deployment-representative setting. |
| `v4.yaml` | 0.08 | Task 1 found alpha=0.02 leaves no detectable post-removal residue (control AUROC ≈ 0.51, chance) — raised to test whether a stronger embedding survives removal at all. |
| `v5.yaml` | 0.15 | Confirmed the residue signal plateaus (~0.64 control AUROC) rather than continuing to improve with alpha — see `results/task1_writeup.md`. Current value; still tuned for signal detectability, not deployment realism. |

None of these alpha values should be read as "what a real deployed
watermark would use" — all were chosen to make this research pipeline
answer methodology questions (class balance, residue detectability), not
to model production watermark strength.

## Phase 1 Definition of Done
- [ ] Pipeline runs end-to-end with 8 commands above
- [ ] Test AUROC > 0.6 (beats random)
- [ ] Reproducible on Colab T4

## Next Phases
- **Phase 2**: Swap distortion for diffusion regeneration + SynthID-Bypass
- **Phase 3**: Generic diffusion edits for transfer test