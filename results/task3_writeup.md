# Task 3 — TPR@low-FPR in evaluate.py

Ref: `phase0-closeout-handoff.md` Task 3.

## Change

`src/evaluate.py`: added `tpr_at_fpr(labels, probs, target_fpr)` using
`sklearn.metrics.roc_curve` — reports the best TPR achievable at
FPR ≤ target (falls back to nearest FPR point if the target isn't
reachable, e.g. small test sets). Added `tpr_at_1pct_fpr` and
`tpr_at_0.1pct_fpr` to the results dict, alongside existing
accuracy/AUROC/precision/recall/F1. No existing fields changed —
purely additive, per the task's ask.

## Re-run results

**v2** (`configs/v2.yaml`, alpha=0.02 — the handoff's literal "current
baseline," now known confounded per Task 1):

| Metric | Value |
|---|---|
| AUROC | 0.8876 |
| TPR @ 1% FPR | 0.2609 |
| TPR @ 0.1% FPR | 0.2609 |

**v5** (`configs/v5.yaml`, alpha=0.15 — current confirmed-weak-pass
model per Task 1):

| Metric | Value |
|---|---|
| AUROC | 0.8192 |
| TPR @ 1% FPR | 0.1573 |
| TPR @ 0.1% FPR | 0.1573 |

Both models show 1% and 0.1% FPR giving identical TPR — expected: test
split has only 66 (v2) / 46 (v5) negatives, so the smallest achievable
FPR step is ~1.5–2.2%, coarser than 0.1%. Both operating points collapse
onto the same nearest-achievable threshold. This is a sample-size
artifact of the eval split, not a bug — flagging it since it will look
odd in a results table.

## Reading

At field-standard low-FPR operating points, both models look
considerably weaker than their headline AUROC suggests (v2: 0.26 TPR
at ~1.5% FPR; v5: 0.16 TPR at ~2.2% FPR). This is consistent with the
Task 1 finding — AUROC is an aggregate ranking measure and doesn't by
itself reveal how much of that ranking ability survives at the strict
low-false-positive thresholds a real forensic tool would need to operate
at. Worth keeping in every future results table from here on, per the
handoff's stated reason.

## Note for future runs

TPR@low-FPR is noisy with small negative-class counts (test set: 46–66
negatives depending on version). If this metric matters for later
Gap 1/2 work, worth using a larger held-out negative pool than the
current 15% test split provides.
