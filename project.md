# Watermark-Removal Forensics

## What this project is

This project builds and honestly evaluates a forensic image classifier
that tries to answer one question: **if an invisible watermark is removed
from an image, can a machine-learning model detect that the removal
happened — without ever seeing the original image to compare against?**

It's not a watermarking scheme paper, and it's not a watermark-removal
attack paper. It's a *forensics* project: given only a finished image, can
a trained classifier tell "this still has its watermark" apart from "this
had a watermark and it's gone now," the same way image forensics more
broadly tries to tell "real photo" apart from "AI-generated/edited photo."

The motivating real-world scenario: as invisible watermarks (SynthID,
C2PA-style provenance marks, etc.) get proposed as a way to label
AI-generated content, people will build tools to strip them. A forensic
detector that can flag "this watermark was removed" — even without
knowing which removal tool was used — would be a useful signal. This
project asks whether that's actually achievable, and reports what it
found, including the parts that didn't work.

## The pipeline, end to end

```
Raw photo (DIV2K dataset)
   -> embed watermark (DWT-DCT-SVD)              [class: Present]
   -> attack it (randomly chosen attack family)
   -> re-verify against the true original          -> label: Present or Removed
   -> group-aware train/val/test split (no source photo crosses a split)
   -> train ConvNeXt-Tiny classifier
   -> evaluate: accuracy, AUROC, precision/recall/F1, confusion matrix
```

The critical design decision, arrived at the hard way (see "Bugs found and
fixed" below): **the label is never "which folder did this file come
from."** It's always "did the watermark's own verification check still
detect it, after whatever happened to this image." That distinction is
the difference between a classifier that measures something real and one
that's silently cheating.

### 1. Watermark embedding
A DWT-DCT-SVD scheme (Lai & Tsai, 2010): the image is wavelet-decomposed,
partitioned into 8×8 blocks, each block's dominant singular value is
nudged up or down according to a watermark bit, and the image is
reconstructed. Deliberately tuned weak (`alpha=0.02`) so it's a real but
fragile watermark — survivable under light processing, breakable under a
genuine removal attempt, not an indestructible toy.

### 2. Attack families
Every watermarked image is attacked by one of two structurally different
methods, drawn at random:
- **Distortion-optimization** — a project-original, greedy/gradient-free
  search over the watermark's own frequency-domain coefficients, inspired
  by (but independent of) the UnMarker attack (Kassis & Hengartner, IEEE
  S&P 2025).
- **SANA-VAE / DC-AE regeneration** — an encode/decode roundtrip through a
  pretrained deep-compression autoencoder (Xie et al., ICLR 2025), a
  lighter-weight relative of full diffusion-purification attacks like
  DiffPure.

A third method (CtrlRegen, diffusion-based) was integrated and works, but
wasn't used in the final dataset build because its source repository ships
without a LICENSE file — flagged as a reproducibility/redistribution
concern rather than quietly ignored.

### 3. Verification-based labeling
After attack, every image is re-checked against its true original using
the watermark's own `verify()` function. The label is whatever that check
says — "Present" or "Removed" — not which code path produced the file.
This is the fix for the project's most important early bug (below).

### 4. Group-aware splitting
Train/val/test splits are built with `StratifiedGroupKFold`, keyed on the
source photo's filename, so no single photo's derivatives can end up
split across train and test — a subtler leakage mode than the label leaks,
caught and fixed separately.

### 5. Classifier
ConvNeXt-Tiny (ImageNet-pretrained, ~27.8M params) with a small
Dropout→Linear head, fine-tuned end-to-end. AdamW, cosine LR schedule,
mixed precision, gradient accumulation, early stopping.

## Key results (current, as of 2026-10-04)

| Result | Value |
|---|---|
| Headline (2-class, present vs. removed, diversified attacks, 3-seed mean) | **Accuracy 0.7289 ± 0.034, AUROC 0.7632 ± 0.012** |
| Reference baseline (single distortion-only attack, easier task) | Accuracy 0.800, AUROC 0.888 |
| 2-layer CNN baseline (trained, same recipe) | Accuracy 0.6667, AUROC 0.4420 — collapses to majority-class, below-chance ranking |
| Non-learned image-quality thresholds (Laplacian var. / JPEG blockiness) | AUROC 0.52–0.59 |
| 3-class extension (adds "never watermarked" as a third class), present-vs-removed subset | Accuracy 0.7467, AUROC 0.7929 (on a different dataset variant — not a direct comparison) |

The headline result clears both baselines by a wide margin, which is the
basic sanity check that the classifier is learning something real rather
than a trivial shortcut.

## Bugs found and fixed (why the methodology is the actual strength here)

1. **Format leak.** Early version saved "Present" images as PNG and
   "Removed" images as JPEG-derived. The classifier hit 100% accuracy —
   because it was detecting file format, not watermark survival. Fixed by
   re-encoding both classes identically.
2. **Attack-intensity leak.** Even after fixing the format leak, labels
   were still assigned by "which folder was this attacked into," not by
   actual watermark survival. Any attack strong enough to plausibly remove
   the watermark also leaves generic degradation (noise/blur/compression)
   a CNN can key on directly — making the task trivially easy regardless
   of dataset size, without the model ever reasoning about the watermark.
   Fixed by switching to verification-based labeling (step 3 above).
3. **Source-image leakage.** A separate audit found the original
   `build_splits.py` had no grouping logic, and would leak a source
   photo's derivatives across train/test under some dataset layouts. Fixed
   with `StratifiedGroupKFold`, verified zero leakage.
4. **Cross-hardware reproducibility gap** (found 2026-10-04). The
   project's headline checkpoint was trained on Apple Silicon (MPS).
   Retraining the identical, identically-seeded pipeline on a CUDA GPU
   produced a measurably different dataset realization and a different
   result (AUROC 0.763 vs. the original 0.662) — not because of a code
   bug, but because the watermark's deliberate fragility means small
   cross-backend floating-point differences are enough to flip some
   images' verification outcome. Documented transparently as a finding,
   not silently patched over; the current, CUDA-trained dataset is now
   the dataset of record, backed by a 3-seed mean rather than a single run.

Each of these was caught by deliberately interrogating a suspiciously good
result rather than accepting it — the project's stated methodological
stance throughout.

## The two open research questions

The project frames its actual research contribution as two sub-questions:

- **Gap 1 — real-world generalization.** Does the classifier's skill
  transfer to a watermark-removal tool it never trained on, built
  independently by someone else — as opposed to one of this project's own
  attack implementations? **Status: unresolved, zero evidence.** Four
  real tools were investigated (DiffPure, the released UnMarker tool,
  `remove-ai-watermarks`, Synthid-Bypass); all four were disqualified or
  impractical, originally due to CPU/MPS-only compute, now (CUDA
  available) mostly due to multi-GB download size and setup complexity. A
  resume attempt on `remove-ai-watermarks` is the most promising
  unfinished thread — it got past model loading before stalling mid
  download.
- **Gap 2 — removal-residue specificity.** Is the classifier detecting
  something specific to watermark removal, or just generic
  editing/regeneration artifacts that would appear in any heavily-edited
  image? **Status: partially addressed, inconclusive.** A control
  comparison (verified-removed images vs. a matched-severity,
  never-watermarked control corpus) shows a declining trend across
  training runs that doesn't fit a clean story — the most-trusted,
  properly-converged model scores *lowest* on this control, the opposite
  of what "better training = better at the real task" would predict. Left
  as an open, honestly-reported finding rather than explained away.

## What's genuinely done vs. what isn't

**Done and solid:** the full pipeline, two label-leakage bugs caught and
fixed, group-aware splitting, a real baseline comparison (classifier beats
both a trained 2-layer CNN and non-learned heuristics by a wide margin), a
3-seed headline result with confidence intervals, a related-work section
situating the project against prior watermarking/forensics literature, and
a transparently documented cross-hardware reproducibility finding.

**Not done:** Gap 1 has no evidence at all. Gap 2 has one inconclusive
data point. Three analysis sections (batch-size-confound isolation,
per-attack-family breakdown, 3-class comparison) still reference the
original (now-unreproducible) checkpoint and haven't been re-verified
against the current dataset. The test set is small (n=150, 50 negatives),
limiting how precise any low-false-positive-rate claim can be.

## Publication status

Currently best described as a rigorous progress report / negative-results
case study, not a completed journal paper. Its genuine strength is
methodological: the leakage bugs, the baseline comparison, and the
cross-hardware reproducibility finding are real, useful, citable
contributions on their own, independent of whether Gap 1 ever closes. The
realistic near-term path is an arXiv preprint plus a workshop/short paper
at a forensics-specific venue (e.g. ACM IH&MMSec, IEEE WIFS), with the
abstract reframed to lead with the methodology and reproducibility
findings rather than presenting the still-open Gap 1 as an unmet promise.
A full journal submission would additionally want Gap 1 closed (or
explicitly descoped) and the three stale analysis sections re-verified.

## Where things live

| What | Where |
|---|---|
| Full academic writeup (methodology, all results, limitations) | `research_paper_draft.txt` |
| Per-stage technical architecture | `ARCHITECTURE.md` |
| Quickstart / reproduction commands | `README.md` |
| Publication-readiness tracking | `workplan.md` |
| Dated session/history log | `current_status.txt` |
| Source-image leakage audit | `leak_check_report.md` |
| Real-tool evaluation attempts (Gap 1) | `phase2_tools_evaluation.md` |
| How the classifier actually works, plain-language | `guide.md` |
| Embedding/attack/dataset code | `src/watermark/`, `src/attacks/`, `scripts/build_verified_dataset.py` |
| Classifier/training/eval code | `src/models/`, `src/train.py`, `src/evaluate.py` |
| Headline config | `configs/v2_accum_long.yaml` |
