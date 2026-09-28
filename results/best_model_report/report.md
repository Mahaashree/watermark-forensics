# Best Model Report -- checkpoints_v2_accum_long/best_model.pt

Test split, n=150 (52 present / 98 removed). All numbers below are copied
verbatim from already-committed JSON (`results/v2_accum_long/test_metrics.json`,
`results/attack_breakdown/accum_long_{optimization,sana}.json`) or derived from the same
cached real inference pass whose AUROC/confusion-matrix were verified to match those files
exactly (`/tmp/accum_long_roc_data.npz`). Nothing here was recomputed independently or re-rounded.

## Pooled test metrics

| Metric | Value |
|---|---|
| Accuracy | 0.6733 |
| AUROC | 0.6615 |
| Precision | 0.7025 |
| Recall | 0.8673 |
| F1 | 0.7763 |
| TPR@1.92%FPR (nearest achievable to 1%) | 0.1735 |
| TPR@1.92%FPR (nearest achievable to 0.1%) | 0.1735 |
| Loss | 0.6411 |

**FPR resolution caveat:** n_negative (present class) = 52, so the empirical ROC curve's FPR only
takes discrete steps of 1/52 ~= 1.92% -- there is no real operating point at exactly 1% or 0.1% FPR.
Both rows above are reported at the nearest achievable FPR **at or above** the requested target
(1.92% in both cases here), not interpolated between the neighboring ROC points, since interpolating
on only 52 negatives would imply a precision this sample size doesn't support. Read these as "TPR at
the tightest FPR this test set can actually demonstrate," not as the TPR at exactly 1%/0.1% FPR.

**Bias callout:** the model is biased toward predicting "Removed." Confusion matrix (rows=true, cols=pred,
labels=[Present, Removed]):

```
              Pred Present   Pred Removed
True Present        16             36
True Removed        13             85
```

Of 52 true "Present" images, only 16 (31%) are correctly identified --
the other 36 (69%) are misclassified as "Removed." This is a real finding about
the model's operating behavior, not just a number to read off the matrix: at this threshold, the model
is much better at confirming a removal than at confirming a watermark is still intact.

**PR curve truncation check:** the precision-recall curve in `pr_curve.png` can look like it cuts off
around recall~=0.85 in some previews. Checked directly against the saved PNG's pixel data and the
underlying `recall` array (not just the screenshot): recall genuinely spans the full [0, 1] range, and
plotted content extends to the figure's right edge (verified: non-white pixels reach column 1757 of
1800, including the "1.0" tick label). The plotting code's axis limits were already `ax.set_xlim(0, 1)`.
**Conclusion: this was a rendering/preview artifact, not a bug** -- no code change was needed, and the
figure was not regenerated with different axis limits (there was nothing to fix).

## Per-attack-type breakdown

| Metric | distortion_optimization (n=83, n_present=9, n_removed=74) | SANA-VAE (n=67, n_present=43, n_removed=24) |
|---|---|---|
| Precision | 0.9437 (n_present=9) | 0.3600 (n_present=43) |
| Recall | 0.9054 (n_removed=74) | 0.7500 (n_removed=24) |
| F1 | 0.9241 (n=83) | 0.4865 (n=67) |
| AUROC | 0.7688 (n=83) | 0.6483 (n=67) |
| Accuracy | 0.8675 (n=83) | 0.4328 (n=67) |

The model is substantially stronger on distortion_optimization-attacked images (AUROC 0.7688)
than on SANA-VAE-attacked images (AUROC 0.6483) -- consistent with the earlier attack-mix
finding that attack family, not batch size, drives the AUROC gap.

**Small-sample caveat:** distortion_optimization's present class has only n=9 examples
in this subset. Its precision, recall, F1, and AUROC above are computed over that small a class and are
high-variance as a result -- they should not be read as a confident, stable measurement of how this
model handles distortion_optimization specifically, only as a rough, noisy signal from this one test split.

## Inference demo notes

(See `results/inference_demo/` for the full 6-example, 4-panel visual demo.) One example
(`data/raw/0300.png`, flowers) has a pre-attack watermarked `verify()` correlation of only **0.525** --
barely above the 0.5 presence threshold before any attack was applied. This indicates some source
images embed the watermark weakly at alpha=0.02 even with no attack in play, which is itself a source
of "removed" labels unrelated to attack strength and worth keeping in mind when reading per-image results.

## Figures

- `confusion_matrix.png` -- raw counts + row-normalized, labeled Present/Removed
- `roc_curve.png` -- ROC with AUC annotated
- `pr_curve.png` -- Precision-Recall curve (AP=0.7967), base rate=0.653; verified not truncated, see note above
- `confidence_histogram.png` -- softmax P(removed), split by true class
- `per_attack_type_table.png` -- rendered version of the table above, with per-cell sample sizes
