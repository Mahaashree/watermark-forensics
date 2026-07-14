watermark-forensics-implementation/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                  # original clean images
│   ├── watermarked/          # after watermarking
│   ├── removed/              # after removal attack (label=1)
│   ├── edited/               # generic diffusion edits, no watermark (for transfer test)
│   └── splits/                # train/val/test csv/json index files
├── src/
│   ├── watermark/
│   │   └── embed.py          # apply watermark to clean images
│   ├── attacks/
│   │   ├── distortion.py     # simple distortion-based removal (start here)
│   │   └── diffusion_regen.py # diffusion-based regeneration removal (later)
│   ├── edits/
│   │   └── generic_edit.py   # inpainting / regional regen, no watermark (transfer test)
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset/DataLoader
│   │   └── build_splits.py   # generate train/val/test splits
│   ├── models/
│   │   └── classifier.py     # ConvNeXt-Tiny / ResNet-50 binary classifier
│   ├── train.py
│   ├── evaluate.py           # accuracy, AUROC, per-attack breakdown
│   └── utils/
│       ├── logging.py
│       └── seed.py
├── experiments/
│   ├── exp1_baseline_repro/  # reproduce base paper's setup on simple attack
│   ├── exp2_real_world_tools/ # SynthID-Bypass generalization test
│   └── exp3_edit_transfer/   # removal-residue vs edit-residue transfer test
├── notebooks/
│   └── exploration.ipynb     # Colab-friendly exploration, sanity checks
├── checkpoints/               # saved model weights (gitignored, large files)
└── results/
    ├── logs/
    └── figures/
