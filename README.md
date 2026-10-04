# Watermark-Removal Forensics

End-to-end pipeline: **Watermark → Attack → Verify → Dataset → Classifier → Evaluation**

Current honest headline result (diversified attack mix, 3-seed mean, see
`research_paper_draft.txt` Section 5.3.1): **accuracy 0.7289 ± 0.034,
AUROC 0.7632 ± 0.012** on a 150-image held-out test set. This supersedes
both the earlier single-run 0.6733/0.6615 figure (Section 5.3 — trained on
different hardware, not reproducible on this machine, see Section 5.3.1)
and any "accuracy 1.0" figure that may appear in old notes/commits (a
diagnosed format/label leak, Section 4.1, discarded, not a real result).

## Quick Start (current pipeline, matches the paper draft)

```bash
# 1. Install
pip install -e .

# 2. Download DIV2K (900 images used in the paper's headline result)
python scripts/download_div2k.py --max-images 900

# 3. Build the verified dataset: embeds a watermark into every image, attacks
#    it with a randomly-chosen attack family, then re-verifies against the
#    true original to assign the label (present/removed) — NOT by which code
#    branch produced the file. See research_paper_draft.txt Section 3.2 for
#    why label-by-construction was tried first and discarded (it leaks).
python -m scripts.build_verified_dataset --raw data/raw \
  --out-present data/watermarked_v2 --out-removed data/removed_v2 \
  --save-quality 80 --alpha 0.02

# 4. Build group-aware stratified splits (70/15/15) — no source image appears
#    in more than one split, even though both classes can derive from the
#    same source photo (src/data/build_splits.py uses StratifiedGroupKFold).
python -m src.data.build_splits --watermarked data/watermarked_v2 \
  --removed data/removed_v2 --splits data/splits_v2

# 5. Train ConvNeXt-Tiny classifier (gradient-accumulation config — see
#    configs/v2_accum_long.yaml comments for why epochs=36/patience=15)
python -m src.train --config configs/v2_accum_long.yaml

# 6. Evaluate on test set
python -m src.evaluate --config configs/v2_accum_long.yaml \
  --checkpoint checkpoints_v2_accum_long/best_model.pt --split test
```

The original Phase 1 flow (`src.watermark.embed` + `src.attacks.distortion`
+ label-by-construction `build_splits`) still exists for reference but is
**known to leak** (format leak + attack-intensity leak, both documented in
`research_paper_draft.txt` Section 4.1 and `current_status.txt`) — don't
use it to reproduce a real result.

## Project Structure
```
watermark-forensics-implementation/
├── configs/v2_accum_long.yaml         # Headline-result hyperparameters
├── configs/default.yaml               # Legacy Phase 1 config (leaky, reference only)
├── scripts/download_div2k.py          # DIV2K downloader (mirror-first)
├── scripts/build_verified_dataset.py  # Current pipeline: embed+attack+verify-label
├── src/
│   ├── watermark/embed.py             # DWT-DCT-SVD + CLI (embed/verify)
│   ├── attacks/distortion.py          # Legacy single-family distortion attack
│   ├── attacks/distortion_optimization.py # UnMarker-inspired targeted attack
│   ├── attacks/diffusion_regen.py     # SANA-VAE/DC-AE regeneration attack
│   ├── attacks/regeneration_ctrlregen.py  # CtrlRegen regeneration attack
│   ├── data/build_splits.py           # Group-aware stratified splits + CLI
│   ├── data/dataset.py                # PyTorch Dataset + DataLoaders
│   ├── models/classifier.py           # ConvNeXt-Tiny via timm
│   ├── models/simple_cnn.py           # Non-finetuned baseline model
│   ├── train.py                       # Training loop (grad accum, AMP, seed override)
│   ├── evaluate.py                    # Metrics (acc, AUROC, P/R/F1, TPR@low-FPR)
│   └── utils/{seed.py,logging.py}     # Reproducibility + logging
├── experiments/exp1_baseline_repro/   # Baseline comparator (SimpleCNN + non-learned thresholds)
├── checkpoints_v2_accum_long/         # gitignored — headline model
└── results/                           # gitignored — metrics, figures, logs
```

See `ARCHITECTURE.md` for the full per-stage pipeline description and
`research_paper_draft.txt` for the complete methodology, ablations, and
results writeup, and `workplan.md` for current publication-readiness status.

## Configuration
Headline config is `configs/v2_accum_long.yaml`:
- `train.batch_size: 8`, `grad_accum_steps: 2` (effective batch 16)
- `model.name: "convnext_tiny"` (pretrained ImageNet)
- `watermark.alpha: 0.02`, `block_size: 8`, `dwt_level: 2`
- Attack mix: distortion-optimization + SANA-VAE regeneration, drawn ~50/50
  per image, both re-encoded to JPEG-80 after attack (see `scripts/build_verified_dataset.py`)

`configs/default.yaml` and `configs/mild*.yaml` are earlier, superseded
configs kept for reproducibility of the discarded Phase 1 result and the
single-attack-family reference baseline (paper draft Section 4.2) — not
the current headline number.

## Status
Phase 1-3 (pipeline, diversified attacks, 3-class extension) are done —
see `current_status.txt` and `research_paper_draft.txt` Section 5 for
full results. Gap 1 (real-world removal-tool evaluation) is open/unresolved
— see paper draft Section 6 and `workplan.md` for what's left.