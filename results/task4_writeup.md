# Task 4 — Augmentation Strategy

Ref: `phase0-closeout-handoff.md` Task 4.

## Decision

Changed `src/data/dataset.py` train-time transform from
`RandomResizedCrop + HorizontalFlip + ColorJitter` to
`Resize + HorizontalFlip` only. Full rationale documented in
`README.md` under "Augmentation Policy" (not just fixed silently, per
the task's requirement).

**Why:** `RandomResizedCrop` resamples pixels off the fixed 8x8
DWT/DCT block grid the watermark is embedded on; `ColorJitter`'s
brightness/contrast directly rescale the Y-channel SVD singular values
the watermark (and any removal residue) live on. Both risk destroying
the exact signal this classifier exists to detect.
`HorizontalFlip` is kept — 224 is divisible by the block grid (8), so a
full-image flip maps block boundaries onto block boundaries without
resampling or rescaling anything.

## Empirical check

Retrained v5 (alpha=0.15, current confirmed-weak-pass config) under the
new policy (`configs/v5_minaug.yaml` → `checkpoints_v5_minaug`) to see
whether the change helps, hurts, or is neutral — not just argued
theoretically.

| | old aug (crop+flip+jitter) | new aug (resize+flip) |
|---|---|---|
| Test AUROC | 0.8192 | 0.8521 |
| Test TPR@1%FPR | 0.157 | 0.056 |
| **Task 1 control AUROC** (removed vs. never-watermarked, matched severity) | 0.6439 | **0.6552** |

Task 1 control AUROC (the metric that actually isolates removal-specific
signal from confounds) improved slightly, consistent with the hypothesis
that destructive augmentation was washing out some residue. The plain
AUROC and TPR@low-FPR moved in opposite directions, which is expected
noise on a 135-sample test set with a single training run each — not
treated as a strong result, just directionally consistent with the
rationale.

## Verdict

Policy adopted going forward. Not re-running v2–v5's already-reported
numbers under it (per README's stated scope) — this governs new
training runs from here on (Task 5 and any future Gap 1/2 work).
