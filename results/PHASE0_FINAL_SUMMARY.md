# Phase 0 — Final Results

Ref: `phase0-closeout-handoff.md`. Full per-task detail in
`results/task1_writeup.md` through `task5_writeup.md` (kept). All other
intermediate data/checkpoints/metrics from this phase have been removed —
regenerable via `scripts/build_verified_dataset_v*.py` +
`src/train.py` if needed again.

![Phase 0 summary](figures/phase0_final_summary.png)

## Task verdicts

| Task | Verdict | Key number |
|---|---|---|
| 1 — Non-watermarked degradation control | **Confirmed weak pass** | Control AUROC plateaus at 0.655 (v5-minaug), vs. 0.509 (chance) at the original alpha=0.02 |
| 2 — BMP shortcut-control ablation | **Pass** (proven structurally) | PNG/BMP decode bit-identical (20/20) — no format channel exists in this pipeline |
| 3 — TPR@low-FPR in evaluate.py | **Done** | v5-minaug: TPR@1%FPR = 0.056; both models look much weaker at low-FPR than headline AUROC suggests |
| 4 — Augmentation strategy | **Done, adopted** | Dropped RandomResizedCrop + ColorJitter (both threaten the DCT/SVD-domain residue); Task 1 control AUROC improved 0.644→0.655 |
| 5 — Alpha documentation | **Done** | `default.yaml` flagged as historical; README "Alpha History" table covers full v2→v5 progression |

## What the chart shows

Two numbers per pipeline version:
- **Headline test AUROC** — the "does this look like a good classifier"
  number reported throughout the project.
- **Task 1 control AUROC** — removed-watermark images vs. never-watermarked
  images given the *exact same* attack severity. This is the number that
  actually isolates removal-specific residue from generic-degradation
  confound.

**The gap between the two bars is the story.** At alpha=0.02 (v2, the
original reported baseline, and v3 after adding a targeted DWT/DCT/SVD
attack), the orange bar sits at chance (0.51) while the blue bar reads
0.82–0.89 — the headline number was almost entirely generic-degradation
confound, not real forensic signal. Raising alpha to 0.08 (v4) and 0.15
(v5) lifts the orange bar to ~0.64–0.66: a real, reproducible,
above-chance residue signal exists, but it plateaus well short of the
headline number. The final adopted setup (v5-minaug: alpha=0.15 +
minimal augmentation) is the best honestly-measured result of this
phase: headline AUROC 0.852, but the actually-isolated removal-residue
signal is AUROC 0.655 — real, but modest.

## Bottom line for whoever picks this up next

Any AUROC number from this pipeline above ~0.65 is likely still carrying
a generic-degradation component unless it has been checked against a
Task-1-style severity-matched control. Don't quote headline AUROC alone
as "detection of watermark removal" — quote the control-checked number,
or run the check before quoting either one.
