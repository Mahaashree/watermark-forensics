# Phase 1 — Baseline Reproduction

**Project:** Watermark-Removal Forensics in the Wild
**Phase goal:** Build a small, working, end-to-end pipeline that reproduces the core idea of the base paper (arXiv:2604.25491) on a tiny dataset, before scaling up or 
introducing real-world tools.

**Do NOT touch in this phase:** SynthID-Bypass/ComfyUI, diffusion regeneration attacks, the generic-edit transfer dataset, hyperparameter sweeps. Those are Phase 2 and Phase 
3.

---

## 1. Why this phase exists

Before testing against real-world tools or running expensive experiments, you need proof that your basic pipeline — watermark → attack → labeled dataset → classifier → 
evaluation — actually works, end to end, on a small scale. This phase is deliberately small and ugly. Speed and correctness matter more than realism right now. Everything 
built here gets reused and scaled up in later phases, so nothing here is throwaway work.

---

## 2. Repository structure

```
watermark-forensics-implementation/
├── README.md
├- pyproject.toml
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                   # original clean images
│   ├── watermarked/           # after watermarking
│   ├── removed/               # after removal attack (label = 1)
│   ├── edited/                # (Phase 3 — leave empty for now)
│   └── splits/                # train/val/test index files
├── src/
│   ├── watermark/
│   │   └── embed.py
│   ├── attacks/
│   │   ├── distortion.py      # simple distortion-based removal (Phase 1)
│   │   └── diffusion_regen.py # (Phase 2 — leave stub for now)
│   ├── edits/
│   │   └── generic_edit.py    # (Phase 3 — leave stub for now)
│   ├── data/
│   │   ├── dataset.py
│   │   └── build_splits.py
│   ├── models/
│   │   └── classifier.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       ├── logging.py
│       └── seed.py
├── experiments/
│   └── exp1_baseline_repro/
├── notebooks/
│   └── exploration.ipynb
├── checkpoints/                # gitignored
└── results/
    ├── logs/
    └── figures/
```

---

## 3. Step-by-step plan

### Step 1 — Environment setup
- [ ] Create repo, initialize git
- [ ] Set up Python 3.10+ environment (Colab default is fine)
- [ ] Install: `torch`, `torchvision`, `timm`, `scikit-learn`, `opencv-python`, `Pillow`, `pandas`, `pyyaml`
- [ ] Confirm GPU is visible (`torch.cuda.is_available()`)

**Checkpoint:** environment installs cleanly, GPU detected.

### Step 2 — Gather a small clean image set
- [ ] Collect 500-1000 clean images (any public dataset is fine for this phase — e.g. a subset of COCO, or your own photos — realism doesn't matter yet, pipeline correctness 
does)
- [ ] Place in `data/raw/`

**Checkpoint:** `data/raw/` has ~500-1000 images, all loadable without errors.

### Step 3 — Apply a watermark
- [ ] Implement a simple, working open-source post-hoc watermarking method in `src/watermark/embed.py`
- [ ] Run on all images in `data/raw/`, save to `data/watermarked/`
- [ ] Sanity check: watermark is verifiable (write a tiny verification script, confirm it detects the watermark on a few sample images)

**Checkpoint:** every image in `data/watermarked/` passes watermark verification.

### Step 4 — Apply a simple removal attack
- [ ] Implement a basic distortion-based attack in `src/attacks/distortion.py` — combine JPEG recompression + Gaussian noise + slight blur
- [ ] Run on all images in `data/watermarked/`, save to `data/removed/`
- [ ] Sanity check: confirm the watermark verifier now fails (or mostly fails) on these images — this proves the "attack" actually removes the watermark, not just degrades 
the image randomly

**Checkpoint:** watermark verification rate drops sharply on `data/removed/` vs. `data/watermarked/`.

### Step 5 — Build the labeled dataset
- [ ] Implement `src/data/build_splits.py`: label `data/watermarked/` as class 0, `data/removed/` as class 1
- [ ] Split into train/val/test (suggest 70/15/15), save index files to `data/splits/`
- [ ] Implement `src/data/dataset.py`: PyTorch `Dataset`/`DataLoader` reading from the split files

**Checkpoint:** `DataLoader` returns correctly shaped, correctly labeled batches.

### Step 6 — Build and train the classifier
- [ ] Implement `src/models/classifier.py`: ConvNeXt-Tiny via `timm` (fallback: ResNet-50 if ConvNeXt-Tiny gives issues)
- [ ] Implement `src/train.py`: standard binary classification training loop, config-driven (batch size, lr, epochs from `configs/default.yaml`)
- [ ] Train on the small dataset, log loss/accuracy per epoch to `results/logs/`
- [ ] Save best checkpoint to `checkpoints/`

**Checkpoint:** training runs to completion in under ~15-20 minutes on Colab free-tier T4, loss decreases sensibly.

### Step 7 — Evaluate
- [ ] Implement `src/evaluate.py`: compute accuracy, AUROC, precision, recall on the held-out test split
- [ ] Save results table to `results/`

**Checkpoint:** you have a single results table with real numbers — this is your first concrete result, and your reproduction baseline for comparison in later phases.

### Step 8 — Document
- [ ] Fill in `README.md`: what the repo does, how to run each stage in order, expected output at each step
- [ ] Write a short internal note (in `experiments/exp1_baseline_repro/`) summarizing what you found — even at this small scale, note whether the classifier clearly 
separates class 0 vs 1, or struggles

**Checkpoint:** someone unfamiliar with the project (a teammate, or future-you in month 3) can read the README and rerun the whole pipeline from scratch.

---

## 4. Definition of done for Phase 1

You're done with Phase 1 when:
1. The full pipeline (watermark → attack → dataset → train → evaluate) runs end-to-end without manual intervention
2. Your classifier achieves noticeably-better-than-random accuracy distinguishing watermarked vs. removal-processed images (exact number doesn't need to match the base paper 
yet — just needs to show the signal exists)
3. Repo is clean, documented, and reproducible by a teammate
4. You understand every step well enough to explain it without looking at the code

---

## 5. What comes after Phase 1

- **Phase 2:** swap the simple distortion attack for diffusion-based regeneration attacks, and begin real-world tool integration (SynthID-Bypass) — this is your actual 
"generalization to real-world tools" experiment
- **Phase 3:** build the generic diffusion-edit dataset (no watermark involved) for the removal-residue vs. edit-residue transfer test

Do not start either of these until Phase 1's definition of done is fully met. A shaky foundation here will cost you far more time later than the discipline of finishing this 
properly now.

