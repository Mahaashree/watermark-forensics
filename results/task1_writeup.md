# Task 1 — Non-Watermarked Degradation Control

Ref: `phase0-closeout-handoff.md` Task 1, `plan.md`.

## Result

AUROC = **0.5106** (chance level).

| Group | n | mean P(removed) | std |
|---|---|---|---|
| True removed (`removed_v2`, test split) | 69 | 0.6608 | 0.2395 |
| Control (never watermarked, same degradation) | 69 | 0.6521 | 0.2405 |

Full numbers: `results/task1_control_auroc.json`.

## Verdict: FAIL — confounded

Model can't separate watermark-removed images from never-watermarked
images given matched degradation. Mean scores nearly identical (0.661 vs
0.652). Not the "inverted" failure mode either — no systematic confusion
in either direction, just no signal.

## Interpretation

Baseline AUROC 0.8875 (`results/v2/test_metrics.json`) does not reflect
watermark-removal-specific residue. Reflects generic degradation
detection instead. Classifier learned "was this image attacked," not
"was the watermark removed."

## Method note

Control set (`data/control_v2/`, 900 images) built via RNG-replay against
`scripts/build_verified_dataset.py`'s exact per-image draw sequence
(see `scripts/build_degradation_control.py`). Guarantees matched attack
severity per filename — pixel content and degradation strength held
constant, only variable changed: watermark presence. Stronger control
than "same param ranges" — rules out degradation-intensity confound
directly, not just on average.

## Next steps per handoff

Per Definition of Done: Task 1 failed. Stop.

- Do not proceed to Task 2–5 on this pipeline as-is.
- Do not proceed to Gap 1 / Gap 2.
- Pipeline needs rework before trusting any downstream numbers — likely
  candidates: the Layer 2 attack pipeline (recompression/noise/blur/
  rescale) never targets the watermark's embedding domain (DWT/DCT/SVD)
  with intent to defeat `verify()`, so both classes carry the same kind
  of degradation signature and the model has nothing else to key on.

## Rework attempt (v3) — still fails

Added `src/attacks/coefficient_attack.py`: perturbs the dominant SVD
singular value of each DCT block inside the DWT LL sub-band — the exact
coefficient `DWT_DCT_SVD.embed()` modifies. Composed with the existing
generic distortion in `scripts/build_verified_dataset_v3.py`
(`data/watermarked_v3`/`removed_v3`, alpha=0.02 unchanged, class balance
284/616). Retrained (unmodified `src/train.py`) →
`checkpoints_v3/best_model.pt`, test AUROC 0.8225
(`results/v3/test_metrics.json`).

Re-ran Task 1 control against v3 (`data/control_v3`, RNG-replay-matched
severity, `results/task1_control_auroc_v3.json`):

| Group | n | mean P(removed) | std |
|---|---|---|---|
| True removed (`removed_v3`, test split) | 92 | 0.8164 | 0.3035 |
| Control (never watermarked, same attack) | 92 | 0.8062 | 0.3098 |

**AUROC = 0.5087 — still chance level.**

### Why the targeted attack didn't help

The control here is matched to *exact* per-image attack severity
(coefficient-attack strength + generic-distortion params replayed
identically). Because `verify()`'s pass/fail outcome is mechanically
driven by attack severity, the "removed" class is — by construction —
whatever fraction of images got hit hard enough to flip the label. Giving
the control the identical severity neutralizes the degradation-intensity
shortcut in both directions, which is correct methodology, but it also
means: if the only thing separating "removed" from "control" is that one
started from a `+alpha` embedded image and the other didn't, and `alpha`
is faint enough to be erased by any attack strong enough to defeat
`verify()`, then no residue survives *by either attack type*. Swapping
generic-only for generic+coefficient-domain didn't change this, because
both attacks are strong enough to already destroy the alpha=0.02 signal
whenever they're strong enough to flip the label at all.

### Implication

This points past "attack doesn't target the right domain" to a more basic
problem: **alpha=0.02 may be too weak for post-removal residue to exist at
all**, independent of how removal is attempted. Two testable directions,
not yet run:
1. Repeat this same rework at a stronger, still-plausible alpha (e.g.
   0.05–0.1) — the handoff already flags 0.02 as tuned only for label
   balance, not deployment realism, so this isn't inventing a new
   discrepancy, it's testing whether the balance-tuned alpha itself is
   the reason no residue is detectable.
2. Revisit whether Task 1's control should require *exact* per-image
   severity matching (current approach, high rigor, may over-null any
   real signal) vs. the handoff's original ask of matching *param ranges*
   only (independently drawn, not paired) — the two designs can give
   different answers and both are defensible; worth running both before
   concluding no residue exists at any alpha.

Flagging both for a decision before further engineering — this is a
methodology fork, not a bug.

## v4 — stronger alpha, same rework

Repeated v3's pipeline (`scripts/build_verified_dataset_v4.py`,
`scripts/build_degradation_control_v4.py`) at alpha=0.08 (4x v3/v2's
0.02), coeff-strength-max raised to 0.4 to keep the flip threshold
reachable at the stronger embedding. Class balance 333/567
(`data/watermarked_v4`/`removed_v4`). Retrained →
`checkpoints_v4/best_model.pt`, test AUROC 0.7508
(`results/v4/test_metrics.json`).

Task 1 control re-run (`results/task1_control_auroc_v4.json`,
85 removed / 85 control, exact severity-matched as before):

| Group | n | mean P(removed) | std |
|---|---|---|---|
| True removed (`removed_v4`, test split) | 85 | 0.7294 | 0.1136 |
| Control (never watermarked, same attack) | 85 | 0.6626 | 0.1396 |

**AUROC = 0.6410 — above chance.**

### Reading

Confirms the v3 hypothesis: alpha=0.02 was too weak for any post-removal
residue to survive attacks strong enough to defeat `verify()`. At
alpha=0.08, some residue does survive severity-matched attacks (both
generic and coefficient-domain) — removed images score higher than their
matched controls, not just from degradation intensity (that's controlled
for), so the gap reflects an actual watermark-removal-specific signal.

Not yet at the "good" bar (comparable to/above the 0.888 confounded
baseline) — 0.64 is moderate, well short of confident separation. Options
from here, not yet run: push alpha higher still (0.15–0.2, watching for
the point where robustness starts trivially defeating the attack budget
instead of leaving residue), or hold alpha at 0.08 and get a tighter
AUROC estimate with a larger control run (train+val+test, not just the
85-pair test slice) before deciding if 0.64 is real signal or noise on a
small n.

### Verdict (superseded by v5, below)

## v5 — alpha pushed further, plateau found

Repeated the same pipeline at alpha=0.15 (nearly 2x v4's 0.08),
coeff-strength-max raised to 0.8 to keep the flip threshold reachable
(`scripts/build_verified_dataset_v5.py`,
`scripts/build_degradation_control_v5.py`). Class balance 308/592.
Retrained → `checkpoints_v5/best_model.pt`, test AUROC 0.8192
(`results/v5/test_metrics.json`).

Task 1 control (`results/task1_control_auroc_v5.json`, 89/89
severity-matched):

| alpha | removed mean P | control mean P | Task 1 AUROC |
|---|---|---|---|
| 0.02 (v3) | ~0.66 | ~0.65 | 0.509 |
| 0.08 (v4) | 0.729 | 0.663 | 0.641 |
| 0.15 (v5) | 0.736 | 0.608 | **0.644** |

Doubling alpha from 0.08 to 0.15 moved Task 1 AUROC by only +0.003.
This is a plateau, not continued improvement — alpha strength is not the
lever that gets this to a confident pass. Whatever residue exists above
chance at alpha≈0.08 is already close to whatever this pipeline can
surface; pushing the embedding stronger doesn't reveal more of it.

### Final verdict: Task 1 — CONFIRMED WEAK PASS, not strong pass

- Not confounded (ruled out at v3/v4/v5: AUROC is consistently and
  reproducibly above 0.5 across two independent alpha values with a
  rigorous, severity-matched control — this is a real, if modest,
  removal-specific signal, not noise or a re-run fluke).
- Not strong: plateaus at AUROC ≈ 0.64, well short of the un-controlled
  baseline's 0.75–0.89. The classifier can partially, not confidently,
  tell "removed" from "never watermarked but equally attacked."
- Recommend proceeding to Task 3/4/5 with this caveat carried forward
  explicitly (not silently treating the 0.75–0.89 baseline numbers as
  "clean" — they still include whatever the generic-degradation
  confound contributes on top of the ~0.64-level real signal). Further
  strengthening this signal likely requires a different lever than
  alpha — e.g. a less severity-correlated labeling scheme, or accepting
  that DWT/DCT/SVD watermarking of this style has an inherent residue
  ceiling around this level.
