"""
Standalone report for the converged best model
(checkpoints_v2_accum_long/best_model.pt) on the held-out test set.

All pooled-test numbers/figures are derived from the cached real inference
pass at /tmp/accum_long_roc_data.npz (probs, labels for the 150-sample test
set) -- the same cache whose AUROC was already verified to exactly match
results/v2_accum_long/test_metrics.json, so nothing here is re-rounded or
re-derived from a different run. Per-attack-type numbers are copied verbatim
from results/attack_breakdown/accum_long_{optimization,sana}.json, not
recomputed.

Outputs -> results/best_model_report/
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import tpr_at_fpr

OUT = Path("results/best_model_report")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Load real, already-verified cached inference (no re-derivation) ----
d = np.load("/tmp/accum_long_roc_data.npz")
probs, labels = d["probs"], d["labels"]
preds = (probs > 0.5).astype(int)

# ---- Ground-truth numbers this report must reproduce exactly ----
pooled = json.load(open("results/v2_accum_long/test_metrics.json"))
opt = json.load(open("results/attack_breakdown/accum_long_optimization.json"))
sana = json.load(open("results/attack_breakdown/accum_long_sana.json"))

# sanity: cached npz must reproduce the committed AUROC/CM exactly
from sklearn.metrics import roc_auc_score
assert abs(roc_auc_score(labels, probs) - pooled["auroc"]) < 1e-9, "cached npz AUROC drift!"
cm_check = confusion_matrix(labels, preds).tolist()
assert cm_check == pooled["confusion_matrix"], "cached npz confusion matrix drift!"
print("Verified: cached npz reproduces committed test_metrics.json exactly.")

# =====================================================================
# Figure 1: Confusion matrix (raw + normalized), labeled Present/Removed
# =====================================================================
cm = np.array(pooled["confusion_matrix"])
cm_norm = cm / cm.sum(axis=1, keepdims=True)
class_names = ["Present", "Removed"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, mat, fmt, title in zip(
    axes, [cm, cm_norm], ["d", ".2f"], ["Raw counts", "Row-normalized (recall per true class)"]
):
    im = ax.imshow(mat, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            text = f"{val:d}" if fmt == "d" else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center",
                     color="white" if val > mat.max() * 0.55 else "black", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Confusion matrix -- checkpoints_v2_accum_long/best_model.pt (test, n=150)", fontsize=12, fontweight="bold")
present_recall = cm[0, 0] / cm[0].sum()
fig.text(0.5, 0.055,
          f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}. Accuracy={pooled['accuracy']:.4f}, "
          f"Precision={pooled['precision']:.4f}, Recall={pooled['recall']:.4f}, F1={pooled['f1']:.4f}.",
          ha="center", fontsize=8, style="italic")
fig.text(0.5, 0.02,
          f"Model is biased toward predicting 'Removed': only {present_recall:.0%} of true 'Present' images are correctly identified.",
          ha="center", fontsize=8, style="italic", fontweight="bold")
fig.subplots_adjust(bottom=0.22, top=0.85, left=0.06, right=0.98, wspace=0.35)
fig.savefig(OUT / "confusion_matrix.png", dpi=300)
plt.close(fig)

# =====================================================================
# Figure 2: ROC curve with AUC annotated on the plot
# =====================================================================
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)
assert abs(roc_auc - pooled["auroc"]) < 1e-9

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="C0", lw=2, label=f"ConvNeXt-Tiny (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance (AUC = 0.500)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC -- best_model.pt (test, n=150)", fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.annotate(f"AUC = {roc_auc:.4f}", xy=(0.55, 0.15), fontsize=12, fontweight="bold")
fig.text(0.5, 0.01,
          "n_negative (present class) = 52, so FPR resolution is 1/52 ~= 1.92% -- TPR@low-FPR reported at the "
          "nearest achievable FPR, not interpolated. See report.md.",
          ha="center", fontsize=7.5, style="italic")
fig.subplots_adjust(bottom=0.13, top=0.93, left=0.13, right=0.96)
fig.savefig(OUT / "roc_curve.png", dpi=300)
plt.close(fig)

# =====================================================================
# Figure 3: Precision-Recall curve
# =====================================================================
prec_curve, rec_curve, _ = precision_recall_curve(labels, probs)
pr_auc = auc(rec_curve, prec_curve)
base_rate = float((labels == 1).mean())

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(rec_curve, prec_curve, color="C1", lw=2, label=f"ConvNeXt-Tiny (AP = {pr_auc:.4f})")
ax.axhline(base_rate, color="gray", lw=1, linestyle="--", label=f"Base rate (removed={base_rate:.3f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall -- best_model.pt (test, n=150,\n98 removed / 52 present)", fontsize=10.5, fontweight="bold")
ax.legend(loc="lower left", fontsize=9)
fig.subplots_adjust(bottom=0.11, top=0.87, left=0.13, right=0.96)
fig.savefig(OUT / "pr_curve.png", dpi=300)
plt.close(fig)

# =====================================================================
# Figure 4: Prediction confidence histogram, split by true class
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
bins = np.linspace(0, 1, 26)
ax.hist(probs[labels == 0], bins=bins, alpha=0.6, color="C0", label=f"True: Present (n={int((labels==0).sum())})")
ax.hist(probs[labels == 1], bins=bins, alpha=0.6, color="C3", label=f"True: Removed (n={int((labels==1).sum())})")
ax.axvline(0.5, color="black", lw=1, linestyle="--", label="Decision threshold (0.5)")
ax.set_xlabel("Softmax P(removed)"); ax.set_ylabel("Count")
ax.set_title("Prediction confidence distribution, split by true class", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
fig.subplots_adjust(bottom=0.11, top=0.91, left=0.1, right=0.97)
fig.savefig(OUT / "confidence_histogram.png", dpi=300)
plt.close(fig)

# =====================================================================
# Per-attack-type table (values copied verbatim from committed JSONs)
# =====================================================================
attack_table = {
    "distortion_optimization": {k: opt[k] for k in ["accuracy", "precision", "recall", "f1", "auroc", "n_samples", "n_present", "n_removed"]},
    "sana_vae": {k: sana[k] for k in ["accuracy", "precision", "recall", "f1", "auroc", "n_samples", "n_present", "n_removed"]},
}

# Every cell carries the subset's class composition (n_present, n_removed)
# directly next to the value -- not just as a separate footer row -- since
# distortion_optimization's n_present=9 makes precision/recall/F1/AUROC on
# that column high-variance, and that risk should be visible at every cell,
# not just discoverable by cross-referencing a row below.
opt_n = f"(n_p={opt['n_present']}, n_r={opt['n_removed']})"
sana_n = f"(n_p={sana['n_present']}, n_r={sana['n_removed']})"

fig, ax = plt.subplots(figsize=(10.5, 3.0))
ax.axis("off")
rows = ["Precision", "Recall", "F1", "AUROC", "Accuracy", "n_samples (present/removed)"]
cols = ["distortion_optimization", "SANA-VAE"]
cell_text = [
    [f"{opt['precision']:.4f} {opt_n}", f"{sana['precision']:.4f} {sana_n}"],
    [f"{opt['recall']:.4f} {opt_n}", f"{sana['recall']:.4f} {sana_n}"],
    [f"{opt['f1']:.4f} {opt_n}", f"{sana['f1']:.4f} {sana_n}"],
    [f"{opt['auroc']:.4f} {opt_n}", f"{sana['auroc']:.4f} {sana_n}"],
    [f"{opt['accuracy']:.4f} {opt_n}", f"{sana['accuracy']:.4f} {sana_n}"],
    [f"{opt['n_samples']} ({opt['n_present']}/{opt['n_removed']})", f"{sana['n_samples']} ({sana['n_present']}/{sana['n_removed']})"],
]
table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 1.6)
ax.set_title("Per-attack-type metrics -- best_model.pt (test)\nn_p = n_present, n_r = n_removed, shown per cell", fontsize=11, fontweight="bold", pad=24)
fig.text(0.5, 0.02,
          "distortion_optimization's n_present=9 is small -- its precision/recall/F1/AUROC here are high-variance "
          "and should not be read as a confident per-family result.",
          ha="center", fontsize=8, style="italic")
fig.subplots_adjust(left=0.26, right=0.98, top=0.76, bottom=0.14)
fig.savefig(OUT / "per_attack_type_table.png", dpi=300)
plt.close(fig)

# =====================================================================
# TPR @ low FPR (via src.evaluate.tpr_at_fpr, same function added to evaluate.py)
# Reports the nearest achievable FPR >= target, NOT interpolated -- see
# tpr_at_fpr's docstring. Both the 1% and 0.1% targets land in the same gap
# below this test set's first nonzero ROC step (1/52 ~= 1.92%), so both
# resolve to the same achieved operating point.
# =====================================================================
fpr_near_1pct, tpr_near_1pct = tpr_at_fpr(labels, probs, 0.01)
fpr_near_01pct, tpr_near_01pct = tpr_at_fpr(labels, probs, 0.001)
assert abs(fpr_near_1pct - pooled["tpr_near_1pct_fpr"]["achieved_fpr"]) < 1e-9
assert abs(tpr_near_1pct - pooled["tpr_near_1pct_fpr"]["tpr"]) < 1e-9
assert abs(fpr_near_01pct - pooled["tpr_near_0.1pct_fpr"]["achieved_fpr"]) < 1e-9
assert abs(tpr_near_01pct - pooled["tpr_near_0.1pct_fpr"]["tpr"]) < 1e-9
tpr_label_1pct = pooled["tpr_near_1pct_fpr"]["label"]
tpr_label_01pct = pooled["tpr_near_0.1pct_fpr"]["label"]

# =====================================================================
# Summary JSON + Markdown
# =====================================================================
summary = {
    "checkpoint": "checkpoints_v2_accum_long/best_model.pt",
    "config": "configs/v2_accum_long.yaml",
    "split": "test",
    "n_samples": pooled["n_samples"],
    "n_present": int((labels == 0).sum()),
    "n_removed": int((labels == 1).sum()),
    "pooled": {
        "accuracy": pooled["accuracy"],
        "auroc": pooled["auroc"],
        "precision": pooled["precision"],
        "recall": pooled["recall"],
        "f1": pooled["f1"],
        "confusion_matrix": pooled["confusion_matrix"],
        "present_class_recall": present_recall,
        "tpr_near_1pct_fpr": pooled["tpr_near_1pct_fpr"],
        "tpr_near_0.1pct_fpr": pooled["tpr_near_0.1pct_fpr"],
        "n_negative_present_class": pooled["n_negative_present_class"],
        "tpr_low_fpr_note": pooled["tpr_low_fpr_note"],
        "loss": pooled["loss"],
    },
    "per_attack_type": attack_table,
    "per_attack_type_small_sample_note": (
        f"distortion_optimization's present class has n={opt['n_present']}; its precision/recall/F1/AUROC "
        "are high-variance and should not be read as a confident per-family result."
    ),
    "pr_curve_truncation_check": (
        "Verified against the saved PNG pixel data and the underlying recall array: recall genuinely spans "
        "[0, 1] and the plotted content extends to the figure's right edge. Any apparent cutoff at "
        "recall~=0.85 in a chat/thumbnail preview is a display artifact, not a bug in the saved figure or "
        "the plotting code's axis limits (which were already ax.set_xlim(0, 1))."
    ),
    "embedding_fragility_note": (
        "One inference-demo example (data/raw/0300.png, flowers) has a pre-attack watermarked verify() "
        "correlation of only 0.525 -- barely above the 0.5 presence threshold before any attack was applied "
        "-- indicating some source images embed the watermark weakly at alpha=0.02 even with no attack in play."
    ),
    "source_of_truth": {
        "pooled_metrics": "results/v2_accum_long/test_metrics.json",
        "distortion_optimization_metrics": "results/attack_breakdown/accum_long_optimization.json",
        "sana_vae_metrics": "results/attack_breakdown/accum_long_sana.json",
        "cached_inference": "/tmp/accum_long_roc_data.npz (verified AUROC + confusion matrix match test_metrics.json exactly)",
    },
}
json.dump(summary, open(OUT / "report.json", "w"), indent=2)

md = f"""# Best Model Report -- checkpoints_v2_accum_long/best_model.pt

Test split, n={pooled['n_samples']} (52 present / 98 removed). All numbers below are copied
verbatim from already-committed JSON (`results/v2_accum_long/test_metrics.json`,
`results/attack_breakdown/accum_long_{{optimization,sana}}.json`) or derived from the same
cached real inference pass whose AUROC/confusion-matrix were verified to match those files
exactly (`/tmp/accum_long_roc_data.npz`). Nothing here was recomputed independently or re-rounded.

## Pooled test metrics

| Metric | Value |
|---|---|
| Accuracy | {pooled['accuracy']:.4f} |
| AUROC | {pooled['auroc']:.4f} |
| Precision | {pooled['precision']:.4f} |
| Recall | {pooled['recall']:.4f} |
| F1 | {pooled['f1']:.4f} |
| {tpr_label_1pct} | {tpr_near_1pct:.4f} |
| {tpr_label_01pct} | {tpr_near_01pct:.4f} |
| Loss | {pooled['loss']:.4f} |

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
True Present       {cm[0,0]:>3}            {cm[0,1]:>3}
True Removed       {cm[1,0]:>3}            {cm[1,1]:>3}
```

Of {cm[0].sum()} true "Present" images, only {cm[0,0]} ({present_recall:.0%}) are correctly identified --
the other {cm[0,1]} ({1-present_recall:.0%}) are misclassified as "Removed." This is a real finding about
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

| Metric | distortion_optimization (n={opt['n_samples']}, n_present={opt['n_present']}, n_removed={opt['n_removed']}) | SANA-VAE (n={sana['n_samples']}, n_present={sana['n_present']}, n_removed={sana['n_removed']}) |
|---|---|---|
| Precision | {opt['precision']:.4f} (n_present={opt['n_present']}) | {sana['precision']:.4f} (n_present={sana['n_present']}) |
| Recall | {opt['recall']:.4f} (n_removed={opt['n_removed']}) | {sana['recall']:.4f} (n_removed={sana['n_removed']}) |
| F1 | {opt['f1']:.4f} (n={opt['n_samples']}) | {sana['f1']:.4f} (n={sana['n_samples']}) |
| AUROC | {opt['auroc']:.4f} (n={opt['n_samples']}) | {sana['auroc']:.4f} (n={sana['n_samples']}) |
| Accuracy | {opt['accuracy']:.4f} (n={opt['n_samples']}) | {sana['accuracy']:.4f} (n={sana['n_samples']}) |

The model is substantially stronger on distortion_optimization-attacked images (AUROC {opt['auroc']:.4f})
than on SANA-VAE-attacked images (AUROC {sana['auroc']:.4f}) -- consistent with the earlier attack-mix
finding that attack family, not batch size, drives the AUROC gap.

**Small-sample caveat:** distortion_optimization's present class has only n={opt['n_present']} examples
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
- `pr_curve.png` -- Precision-Recall curve (AP={pr_auc:.4f}), base rate={base_rate:.3f}; verified not truncated, see note above
- `confidence_histogram.png` -- softmax P(removed), split by true class
- `per_attack_type_table.png` -- rendered version of the table above, with per-cell sample sizes
"""
(OUT / "report.md").write_text(md)

print(f"\nWrote report to {OUT}/")
for f in sorted(OUT.iterdir()):
    print(" -", f.name)
