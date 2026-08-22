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
python src/train.py --config configs/default.yaml

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

## Phase 1 Definition of Done
- [ ] Pipeline runs end-to-end with 8 commands above
- [ ] Test AUROC > 0.6 (beats random)
- [ ] Reproducible on Colab T4

## Next Phases
- **Phase 2**: Swap distortion for diffusion regeneration + SynthID-Bypass
- **Phase 3**: Generic diffusion edits for transfer test