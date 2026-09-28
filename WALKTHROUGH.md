# Watermark-Removal Forensics — Complete Walkthrough

**Project:** Can a classifier tell whether an image's watermark was successfully
removed, based only on the image itself? Base paper: arXiv:2604.25491.

**Pipeline:** `raw image → embed watermark → attack (try to remove it) → label by
actual verification → train classifier → evaluate`

This doc summarizes everything done so far, in order, including the two dead ends
that got caught and fixed. It's written to be readable without touching the code.

---

## 1. Repo layout

```
watermark-forensics/
├── src/
│   ├── watermark/embed.py       # DWT-DCT-SVD watermark: embed + verify
│   ├── attacks/distortion.py    # JPEG + noise + blur removal attack
│   ├── attacks/diffusion_regen.py  # stub — Phase 3, not built yet
│   ├── edits/generic_edit.py    # stub — Phase 4, not built yet
│   ├── data/build_splits.py     # train/val/test split builder
│   ├── data/dataset.py          # PyTorch Dataset/DataLoader
│   ├── models/classifier.py     # ConvNeXt-Tiny (via timm)
│   ├── train.py / evaluate.py
│   └── utils/{seed,logging}.py
├── scripts/
│   ├── download_div2k.py        # pulls DIV2K images into data/raw
│   └── build_verified_dataset.py # v2 pipeline: verification-based labeling
├── configs/                     # default.yaml, mild.yaml, v2.yaml, etc.
├── data/, checkpoints/, results/  # gitignored, generated locally
├── phase1.md, phase2.md, phase3.md  # phase plans/reports (source of this doc)
├── current_status.txt, notes.txt, run.md
```

Git history (`watermark-forensics`, branch `monish`):
1. `init commit`
2. `phase1 pipeline`
3. `Phase 2 (interim): fix label leak, add verification-based dataset pipeline`
4. `Add phase3.md: plan for real-world removal attacks phase`

---

## 2. Phase 1 — Baseline pipeline

Built the full 8-step pipeline end to end:
1. Download ~100 clean images (DIV2K).
2. Embed a **DWT-DCT-SVD watermark** (`alpha=0.1`) → `data/watermarked/`.
3. Verify the watermark is detectable.
4. Apply a **distortion attack** (JPEG recompress + Gaussian noise + blur) →
   `data/removed/`.
5. Verify the attack drops the detection rate.
6. Build 70/15/15 train/val/test splits, labeling by folder (`watermarked` = 0,
   `removed` = 1).
7. Train a ConvNeXt-Tiny binary classifier.
8. Evaluate.

**Result:** 100% test accuracy, 1.0 AUROC. Looked perfect — and that was the problem.

**Caveat found immediately:** the attack used to make `removed/` was extreme
(JPEG q=2, Gaussian std=45, blur kernel=13, 6 passes), and watermarked vs. removed
images were saved in different formats (PNG vs JPEG-derived). The classifier was
almost certainly detecting **format/compression artifacts**, not the watermark's
actual survival.

---

## 3. Phase 2 (interim) — Finding and fixing two label leaks

This wasn't the diffusion-attack phase originally planned as "Phase 2" — it became
necessary hardening work once the 1.0 AUROC turned out to be fake.

### 3.1 Leak #1 — format leak
Fixed by re-encoding both classes to matching JPEG quality 80 before training.

### 3.2 Leak #2 — attack-intensity leak (the real problem)
Even with formats matched, labels were still assigned **by construction**: every
image dropped in `watermarked/` was class 0, every image put through the attack
pipeline was class 1 — regardless of whether the watermark actually survived.
Any attack strong enough to plausibly remove a DWT-DCT-SVD watermark leaves generic
degradation (noise/blur/compression) a CNN can key on directly, without reasoning
about the watermark at all. Softening the attack, expanding the dataset (100 → 900
DIV2K images), and even a "shared baseline distortion" trick (same light distortion
on both classes, extra attack only on removed) all failed to close this — the model
still hit 99.6%+ by keying on attack intensity.

**Root cause confirmed:** labels need to come from whether the watermark is actually
still detectable, not from which folder the file was written to.

### 3.3 Fix — verification-based labeling (`scripts/build_verified_dataset.py`)
New approach, one pass per raw image:
1. Embed the watermark.
2. Apply a **randomized-strength** attack (num_passes 0–5, JPEG q 5–90, Gaussian
   std 2–40, blur kernel ∈ {1,3,5,7,9}, resize 0.25–1.0) — so attack strength is no
   longer correlated with the label.
3. Re-encode to JPEG-80 (kills the format leak too).
4. Run the watermark's own `extract()`/`verify()` against the true original. The
   label is **whatever verification actually says** — "present" or "removed" — not
   which pipeline branch produced the file.

**Problem hit immediately:** class imbalance. The SVD watermark barely responds to
JPEG/blur/noise until near-destructive levels, so most images stayed "present" even
after a randomized attack:

| alpha (watermark strength) | present / removed |
|---|---|
| 0.20 (Phase 1 default) | 835 / 65 |
| 0.12 | 821 / 79 |
| 0.05 | 748 / 152 |
| 0.03 | 589 / 311 |
| **0.02** | **441 / 459** ✅ balanced |

Widening the attack range alone barely helped (812/88) — alpha (watermark strength)
was the lever that mattered, not attack range. Verified `alpha=0.02` is still a real
watermark (59/60 survival on unattacked, JPEG-80-only images) before committing to it.

**Final v2 dataset:** 441 present / 459 removed, split into train 630 (309/321),
val 135 (66/69), test 135 (66/69) — balanced and stratified. Checked for source-image
leakage across splits (not possible by construction here, but confirmed).

### 3.4 Retrained on the honest dataset
ConvNeXt-Tiny, `configs/v2.yaml`, early-stopped at epoch 7 (best val acc 0.874 at
epoch 2).

**Result (`results/v2/test_metrics.json`):**

| Metric | Value |
|---|---|
| Accuracy | 0.800 |
| AUROC | 0.888 |
| Precision | 0.850 |
| Recall | 0.739 |
| F1 | 0.791 |

Confusion matrix (rows=true, cols=pred, order=[watermarked, removed]):
```
[[57,  9],
 [18, 51]]
```

**Honest read:** this is real signal (majority-class baseline is ~51% on this
balanced test set), not a ceiling artifact. 0.888 AUROC is solid-but-not-perfect —
the classifier ranks a random removed image above a random watermarked one ~89% of
the time. Weak point: **recall (0.74)** — it misses ~1 in 4 real removals, biased
toward predicting "still watermarked," which matters most for a forensic tool.
Caveats: small dataset (630 train images), one watermark method, one image source
(DIV2K), single seed/run, and the attack-strength distribution was hand-tuned via
the alpha sweep specifically to balance classes — not a sample of "real-world"
attack strength.

---

## 4. Phase 3 (planned, not started) — Real-world removal attacks

Written as a plan (`phase3.md`), not yet executed. Everything up to here used only
`src/attacks/distortion.py` (synthetic JPEG/blur/noise/resize). Phase 3's question:
does the 0.888 AUROC signal survive attacks nobody hand-tuned for balance?

Planned steps:
1. **Close cheap gaps first:** class-weighted loss / threshold tuning for the recall
   gap; re-run v2 training with 2–3 seeds for a stability range; explicit
   leakage check once diffusion attacks add multiple outputs per raw image.
2. **Implement `src/attacks/diffusion_regen.py`:** img2img regeneration (e.g. Stable
   Diffusion, strength 0.1–0.6), randomized per image the same way the v2 dataset
   was, verification-based labeling again.
3. **Integrate SynthID-Bypass / ComfyUI** as a real-world attack on the same
   watermarked inputs — noting that SynthID-Bypass targets Google's SynthID, not
   this project's DWT-DCT-SVD watermark, so a weak/no effect is itself a finding.
4. **Rebuild dataset (v3)** mixing distortion + diffusion-regen + SynthID-Bypass,
   retrain, and evaluate **per attack type**, not just pooled.
5. **Document** with the same honesty standard as the Phase 2 writeup.

Phase 4 (generic diffusion edits, no watermark, for a removal-vs-edit-residue
transfer test) is explicitly blocked until Phase 3's per-attack-type results land.

---

## 5. Practical run notes (`run.md`)

- Local machine is **CPU-only** — real training runs belong on Colab T4.
- `src/train.py` / `src/evaluate.py` must be invoked as modules
  (`python -m src.train ...`), not as scripts — they use absolute imports.
- `configs/default.yaml` currently points at `data/splits_jpeg80_extreme` (the only
  split with all images present locally); `mild.yaml` / `mild_fixed.yaml` are stale
  until `data/watermarked/` is regenerated.
- The v2 (honest) pipeline:
  ```bash
  python -m scripts.build_verified_dataset --raw data/raw \
    --out-present data/watermarked_v2 --out-removed data/removed_v2 \
    --save-quality 80 --alpha 0.02
  python -m src.data.build_splits --watermarked data/watermarked_v2 \
    --removed data/removed_v2 --splits data/splits_v2
  python -m src.train --config configs/v2.yaml
  python -m src.evaluate --config configs/v2.yaml \
    --checkpoint checkpoints_v2/best_model.pt --split test
  ```

---

## 6. Where things stand right now

- **Done:** end-to-end pipeline works; two label leaks (format, attack-intensity)
  found and fixed; an honest, non-trivial baseline established (80% accuracy /
  0.888 AUROC on verification-labeled, distortion-only data).
- **Not started:** diffusion regeneration attacks, SynthID-Bypass integration,
  per-attack-type evaluation, generic-edit transfer test (Phase 4).
- **Open small items:** recall gap on "removed" class, multi-seed stability check,
  explicit leakage check for future multi-attack datasets.
