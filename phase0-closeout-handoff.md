# Watermark-Forensics — Phase 0 Close-Out Handoff

Repo: https://github.com/Mahaashree/watermark-forensics/
Base paper: Evennou & Kijak, arXiv:2604.25491, "The Forensic Cost of Watermark Removal"

## Context (read this first)

We're building a classifier that detects whether an image had its AI watermark
removed, even when removal succeeded and the watermark can't be recovered. The
core claim we're testing: removal leaves a distinct statistical residue.

**Current baseline (do not touch, this is the reference to beat/validate against):**
- Dataset: DIV2K, n≈900, watermark embedded via DWT/DCT/SVD, S' = S + α·W, α=0.02
- Attacks (Layer 2): recompression (JPEG q5-90), additive noise (σ2-40), blur
  (k∈{1,3,5,7,9}), rescale (0.25-1.00), 0-5 repetition passes, JPEG-80 re-encode
  on both classes (this re-encode step is the fix for a prior format-leak bug —
  do not remove it)
- Labels: from `verify()` output, not folder membership (this is the fix for a
  prior construction-leak bug — do not revert to folder-based labels)
- Class 0 (present) n=441, Class 1 (removed) n=459. Train 630 (309/321), Val 135
  (66/69), Test 135 (66/69)
- Model: ConvNeXt-Tiny (timm, ImageNet-pretrained) + custom head (dropout 0.1,
  linear 768→2), AdamW lr=1e-4 wd=1e-4, CosineAnnealingLR, early stop patience 5
- Result: AUROC=0.888, acc=0.800, precision=0.850, recall=0.739, F1=0.791

**Two known bugs already fixed** (context for why controls below matter):
1. Format leak — attack params in `configs/default.yaml` were too extreme
   (jpeg_quality=2, gaussian_std=45.0, num_passes=6), trivially separable.
   Fixed by randomizing attack strength to decorrelate degradation from label.
2. Label leak — labels came from folder membership, not actual verifier output.
   Fixed by running `verify()` against all attacked images.

**Do not treat 0.888 AUROC as final.** Two more possible leaks are still
unvalidated (below). History says: assume leakage until proven otherwise.

---

## Task list — strict priority order. Do not skip ahead.

### Task 1 — Non-watermarked degradation control (NEW, highest priority)

**Why:** Current "attacks" (recompression, noise, blur, rescale, repetition) are
generic signal degradations, not watermark-targeted removal attacks. They never
touch the watermark's embedding domain (DWT/DCT/SVD) with intent to defeat the
verifier. Risk: the classifier may just be learning "this image was degraded"
rather than "this image had its watermark removed."

**What to build:**
1. Take a held-out subset of *non-watermarked* source images (skip Layer 1
   entirely — no embed step)
2. Apply the exact same Layer 2 degradation pipeline to them
   (recompression/noise/blur/rescale/repetition, same param ranges, same JPEG-80
   re-encode)
3. Run the existing trained classifier on: {watermarked+attacked} (class 1)
   vs {non-watermarked+degraded} (new control class)
4. Report AUROC for this comparison

**Interpretation:**
- High AUROC here (classifier easily tells the two apart) → good sign, residue
  is watermark-removal-specific, not generic degradation
- Low/chance AUROC here (classifier can't tell them apart, or worse, confuses
  degraded-only images with "removed" class) → classifier is confounded by
  generic degradation, not detecting removal-specific residue. Phase 0 result
  is unreliable and pipeline needs rework before any Gap 1/2 work.

**Deliverable:** one AUROC number + short writeup of which case we're in.

---

### Task 2 — BMP shortcut-control ablation (non-negotiable per project notes 5.6)

**Why:** Rule out a third leak channel through compression artifacts, even
after JPEG-80 re-encoding fix.

**What to build:**
1. Take the full train/val/test set (clean + attacked, both classes)
2. Re-encode every image to lossless BMP
3. Retrain (or at minimum re-evaluate the existing model) on this BMP version
4. Report AUROC

**Interpretation:**
- AUROC collapses to ~0.5 → expected/correct, means the model isn't relying on
  compression-format signal
- AUROC stays high → leak still present somewhere, needs investigation before
  trusting any downstream numbers

**Deliverable:** AUROC on BMP set, pass/f
ail against the ~0.5 expectation.

---

### Task 3 — Add TPR@low-FPR to evaluate.py

**Why:** Field-standard metric (see Cozzolino & Verdoliva, Synthbuster
literature), currently missing from our eval script. Needed for every result
from here forward, including the Gap 1 generalization-gap table.

**What to add:**
- TPR @ 1% FPR
- TPR @ 0.1% FPR
- Compute from the existing ROC curve data (sklearn `roc_curve`, then find the
  TPR at the FPR threshold closest to 0.01 / 0.001)
- Add to `evaluate.py` output alongside existing AUROC/acc/precision/recall/F1

**Deliverable:** updated `evaluate.py`, re-run on current baseline to report
these two numbers for the existing model.

---

### Task 4 — Resolve augmentation strategy

**Why:** Currently undecided/undocumented. Standard ImageNet-style
augmentation (random crop, flip, color jitter) risks destroying the very
residue signal we're trying to detect. Need an explicit, justified choice.

**What to do:**
1. Check `configs/default.yaml` / training script for whatever augmentation is
   currently silently applied
2. Decide: keep minimal/no augmentation that could disturb high-frequency
   residue (e.g. avoid aggressive JPEG-like augmentation, avoid strong color
   jitter), OR justify why current augmentation is safe
3. Document the decision and rationale in the README or a methodology note —
   this needs to be stated explicitly, not left implicit

**Deliverable:** documented augmentation decision + rationale, config updated
to match.

---

### Task 5 — Fix alpha documentation contradiction

**Why:** `configs/default.yaml` states alpha=0.1, but actual training used
alpha=0.02 (tuned to hit ~50/50 removal-success balance across classes). This
is currently undocumented and will look like an inconsistency to anyone
reading the repo.

**What to do:**
1. Update config default or add a comment clarifying 0.02 is what was actually
   used for the reported results
2. Add a line to README/notes: "alpha=0.02 was tuned for dataset class
   balance (~50/50 removal success), not for realistic deployment strength —
   this is a known limitation, not a deployment-representative setting"

**Deliverable:** config/doc fix, one paragraph of honest caveat text.

---

## Definition of done for Phase 0

All 5 tasks above complete, with:
- Task 1 and 2 results explicitly stated as pass/fail against expected
  direction (not just "here's a number")
- Task 3 integrated into standard eval output going forward
- Task 4 and 5 documented in repo, not just fixed silently

**Do not start on Gap 1 (real-world tool testing) or Gap 2 (transfer matrix)
until all 5 are done.** If Task 1 or Task 2 fails (leak detected), stop and
flag it — don't proceed to Gap 1/2 on a confounded baseline, that's how we got
the 1.00 AUROC bug the first time.

## What's explicitly NOT in this handoff (comes after Phase 0 closes)

- Gap 1: naming and testing real, downloadable removal tools (not
  reimplementations)
- Gap 2: train-on-removal/test-on-edits transfer matrix experiments
- Confirming whether Evennou & Kijak v2 does bidirectional transfer testing
  (this is a paper-reading task, not engineering — Mahaa is handling this
  separately)
