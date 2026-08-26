# Phase 2 (interim) — Fixing the Label Leak, Getting an Honest Baseline

**Status:** Not the "Phase 2" originally scoped in `phase1.md` (diffusion regeneration attacks +
SynthID-Bypass). This is hardening work that had to happen first: Phase 1's headline result
(100% accuracy, 1.0 AUROC) turned out to be fake, and this phase fixes that before any
diffusion-attack work is worth doing. Real Phase 2 (diffusion regen, SynthID-Bypass) still hasn't
started — see "What's actually next" at the bottom.

---

## 1. Why this phase exists

Phase 1 reported 100% accuracy / 1.0 AUROC. That number was investigated across two prior
sessions (see `current_status.txt`, 2026-08-25 and 2026-08-26 updates) and found to be an
artifact, not a real result — twice:

1. **Format leak:** watermarked images were saved as PNG, removed images as JPEG-derived. Fixed
   by re-encoding both classes to matching JPEG quality.
2. **Attack-intensity leak:** even after fixing the format leak, labels were still assigned "by
   construction" — every image in the `watermarked/` folder is class 0, every image put through
   the attack pipeline is class 1. Any attack strong enough to plausibly remove a DWT-DCT-SVD
   watermark leaves generic degradation (noise/blur/compression artifacts) that a CNN can key on
   directly, without ever reasoning about the watermark. The classifier was learning "how hard
   was this image attacked," not "is the watermark still there."

This phase replaces construction-based labels with **verification-based labels**: every image
gets a randomly-sized attack, and the label is decided by actually running the watermark
method's own `extract()`/`verify()` against the true original — not by which folder it came
from. If the signal survives this, it's real.

---

## 2. What was done

### 2.1 Built the verification-labeled dataset (`scripts/build_verified_dataset.py`)

For each of the 900 raw images (DIV2K):
- Embed the watermark.
- Apply a **randomized** attack: `num_passes` 0–5, JPEG quality 5–90, Gaussian noise std 2–40,
  blur kernel ∈ {1,3,5,7,9}, resize factor 0.25–1.0 — so both "barely touched" and "heavily
  attacked" images exist, and neither class is defined by attack strength.
- Re-encode to JPEG-80 (kills the format leak).
- Run `DWT_DCT_SVD.extract()` + `verify()` against the true original. Label = watermark still
  detected (`present`) vs. not (`removed`).

### 2.2 Found and fixed a class-imbalance problem

First run (`alpha=0.2`, the Phase-1 default): **835 present / 65 removed**. The watermark
perturbs the dominant singular value of 8×8 DCT blocks in the wavelet LL band — a component
that JPEG/blur/noise barely touch until the attack is near-destructive. Widening the attack
range alone (wider jpeg/noise/blur/resize bounds) barely moved this: **812/88**.

Swept `--alpha` (watermark strength) instead, since that directly controls how much attack is
needed to flip a bit:

| alpha | present / removed |
|---|---|
| 0.20 | 835 / 65 |
| 0.12 | 821 / 79 |
| 0.05 | 748 / 152 |
| 0.03 | 589 / 311 |
| 0.02 | 441 / 459 |

Verified `alpha=0.02` is still a legitimate watermark, not a degenerate no-op: on unattacked
images (only the mandatory JPEG-80 save re-encode applied), it survives verification **59/60**
times. Used `alpha=0.02` for the final dataset: **441 present / 459 removed**.

### 2.3 Rebuilt splits and retrained

- `data/splits_v2`: train 630 (309/321), val 135 (66/69), test 135 (66/69) — balanced, stratified.
- Checked for source-image leakage across splits (same raw image appearing in two splits, or
  as both classes): **zero** — each raw image maps to exactly one output file by construction,
  so this wasn't actually possible, but it's confirmed.
- Retrained ConvNeXt-Tiny from scratch on `configs/v2.yaml`. Early-stopped at epoch 7 (best val
  acc 0.874 at epoch 2); train accuracy climbed to 0.94 while val loss rose after epoch 2–3 —
  ordinary, expected overfitting on a genuinely hard task, not the old leak.

---

## 3. Result

```
accuracy:  0.800
auroc:     0.888
precision: 0.850   (of images predicted "removed", 85% actually were)
recall:    0.739   (of truly "removed" images, 74% were caught)
f1:        0.791
confusion matrix (rows=true, cols=pred, order=[watermarked, removed]):
  [[57,  9],
   [18, 51]]
```

## 4. How good is this, honestly

- **It's real.** Test set is balanced (66 present / 69 removed), so the majority-class baseline
  is ~51%. 80% accuracy and 0.888 AUROC are well above that, and there's no leakage explaining
  it away this time — this is the model doing genuine forensic work.
- **0.888 AUROC is a solid, not spectacular, signal.** For reference: 0.5 = no signal, 1.0 =
  perfect separation (which is what the fake Phase-1 result claimed). 0.888 means the model
  ranks a random removed image above a random watermarked one ~89% of the time — good separation
  with real overlap in the hard cases, which is exactly what you'd expect from a subtle
  forensic task on a robust-ish watermark.
- **Recall (0.74) is the weak point.** The model misses about 1 in 4 actual removals (18/69
  false negatives) — it's biased toward predicting "still watermarked." For a forensic/detection
  tool this matters more than precision usually would, since a missed removal is the failure
  mode you care most about.
- **Caveats that limit how much to trust this number:**
  - Small dataset — 630 train images total, only one watermark method (DWT-DCT-SVD), one raw
    image source (DIV2K).
  - Single train/eval run, single seed — no confidence interval. `notes.txt` already flagged
    this concern for the old (fake) 1.0 AUROC; it applies here too, just less severely since
    80%/0.888 isn't sitting at a ceiling.
  - The randomized-attack distribution was hand-tuned (via the alpha sweep) specifically to
    produce a balanced dataset — that's a legitimate way to get labels, but it means the
    train/test distribution of attack strengths is an artifact of this tuning, not a sample of
    "real-world" attacks.

**Bottom line:** this is a credible, honest baseline for "can a classifier detect DWT-DCT-SVD
watermark survival after generic distortion attacks" — moderate-to-good, not solved. It's the
number Phase 1 should have reported instead of the fake 1.0.

---

## 5. What's actually next

The real Phase 2 from `phase1.md` hasn't started yet:
- Swap/add diffusion regeneration attacks (`src/attacks/diffusion_regen.py` is currently a stub)
  instead of relying only on the distortion pipeline.
- Integrate SynthID-Bypass / ComfyUI as a real-world removal attack, not a synthetic one.
- Re-measure whether the classifier still detects removal when the attack is a plausible
  real-world tool rather than parameterized JPEG/blur/noise.

Smaller items worth doing first, cheaply:
- Class-weighted loss or threshold tuning to close the recall gap on the "removed" class.
- Re-run with 2–3 different seeds to see how stable 80%/0.888 actually is on this dataset size.
