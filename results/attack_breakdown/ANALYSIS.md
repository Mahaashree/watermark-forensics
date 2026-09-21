# Per-attack-type breakdown + batch-size isolation attempt

Follow-up to the full dataset rebuild + retrain (commit `27bab6f`). Two
questions: (1) does the pooled test AUROC drop (0.888 -> 0.674) come from
SANA-VAE specifically being a much harder attack than distortion_optimization,
or is it uniform across both; (2) is the batch_size 16->8 workaround (used to
dodge a host memory-pressure stall) a confound in that number, and can it be
isolated with gradient accumulation.

## How the per-attack split was built

`build_verified_dataset.py` doesn't record which attack a given image got.
The RNG draw is fully deterministic and reproducible without re-running any
attacks: `random.seed(42)`, then one `random.random()` per raw image in
sorted-filename order decides `sana` vs `optimization` (verified: nothing else
in the loop touches Python's `random` module state -- `greedy_block_attack`
uses its own isolated `np.random.default_rng`, `apply_diffusion_regen` has no
randomness, and the generic-distortion branch never triggers since
`ctrlregen_frac + sana_frac + optimization_frac == 1.0` this run). The
replayed counts (462 optimization / 438 sana) match the actual build log
exactly. `attack_type_map.json` here is that replay; `test_optimization.csv`
/ `test_sana.csv` are the test-split rows filtered by it.

## Result 1: the committed batch_size=8 model (checkpoints_v2/best_model.pt)

| Subset | n | present/removed | AUROC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| distortion_optimization | 83 | 9/74 | **0.733** | 0.542 | 0.95 | 0.514 |
| SANA-VAE | 67 | 43/24 | **0.672** | 0.687 | 0.556 | 0.625 |
| pooled (reference) | 150 | 52/98 | 0.674 | 0.607 | 0.791 | 0.541 |

SANA somewhat weaker than optimization, but the gap (0.06) is much smaller
than either subset's gap from the 0.888 baseline (~0.16-0.22) -- read at the
time as "a shared factor drags both down, attack-family difficulty is a
smaller secondary effect."

## Isolating batch size: gradient accumulation

Root cause of the stall was confirmed as host memory pressure (8GB RAM,
`memory_pressure`/`vm_stat` showed ~36MB free system-wide), not sleep.
Added `grad_accum_steps` to `src/train.py` (smoke-tested for correctness and
backward-compatibility with the no-accumulation default), and retrained with
`configs/v2_accum.yaml`: `batch_size=8, grad_accum_steps=2` ->
**effective batch size 16, matching the original baseline exactly** --
including a detail beyond just batch size: ~38 optimizer steps/epoch, which
also matches the *true* batch=16 baseline's step count (tonight's batch=8
run, with no accumulation, took ~75 steps/epoch -- twice as many weight
updates per epoch as baseline, a second axis of difference that wasn't
flagged explicitly before).

## Result 2: the grad-accum model (checkpoints_v2_accum/best_model.pt) -- degenerate

Training early-stopped at epoch 6 (best val acc 0.5467, reached at epoch 1
and never beaten). The resulting checkpoint **collapsed to a trivial
always-predict-"removed" classifier**:

| Subset | n | present/removed | AUROC | Recall | Confusion matrix |
|---|---|---|---|---|---|
| distortion_optimization | 83 | 9/74 | 0.596 | 1.0 | [[0,9],[0,74]] |
| SANA-VAE | 67 | 43/24 | 0.691 | 1.0 | [[0,43],[0,24]] |
| pooled | 150 | 52/98 | 0.653 | 1.0 | [[0,52],[0,98]] |

Every present-class prediction is wrong (0 in the left column of every
matrix) -- this is not a fairly-converged model to draw a clean "attack-mix
only" delta from.

## Honest conclusion

Matching effective batch size (and optimizer-step count) to baseline did
**not** reproduce a clean, comparable trained model -- it produced a
different, worse failure mode (degenerate convergence within the fixed
12-epoch / patience-5 budget), rather than isolating attack-mix as the sole
remaining variable as hoped. Two things follow from that:

1. **Batch size / step-count remains a genuinely uncontrolled variable in
   tonight's committed 0.674 result.** The attempt to isolate it surfaced a
   real sensitivity (this harder, more diverse dataset combined with fewer
   optimizer steps/epoch is more fragile to converge within a fixed epoch
   budget) rather than resolving it. This is the explicit "leave it
   documented" fallback the task asked for, arrived at after a real attempt,
   not skipped.
2. **The per-attack-family AUROC ordering is not stable evidence.** It
   flips between the two runs (optimization > SANA at 0.733/0.672 in the
   batch=8 model; SANA > optimization at 0.691/0.596 in the degenerate
   accum model). Given one of the two runs producing this ordering is a
   collapsed classifier, neither ordering should be treated as a reliable
   signal about which attack family is intrinsically harder. The one thing
   that *is* consistent across both runs: both subsets are well below the
   0.888 baseline in both runs, which is compatible with "new attack mix
   makes the task harder" but doesn't cleanly separate that from "this
   dataset+budget combination is generally harder to optimize."

**What would actually resolve this**, if wanted: rerun the grad-accum config
with a larger epoch/patience budget (fewer optimizer steps/epoch means the
model may need more epochs to reach a comparable total step count -- 12
epochs at ~38 steps/epoch is ~456 total steps here vs. ~900 at batch=8's ~75
steps/epoch across the same 12-epoch ceiling), or profile whether a
different optimizer/LR schedule is needed for the accumulated case. Not done
tonight -- flagging rather than guessing further.

---

## Follow-up: larger epoch/patience budget (`configs/v2_accum_long.yaml`)

Same `batch_size=8, grad_accum_steps=2` (effective batch 16, matching
baseline). Budget increased 3x (`epochs=12->36`, `early_stopping_patience=
5->15`): batch=8's converged run took ~900 total optimizer steps (75/epoch
x 12 epochs); at accum's ~38 steps/epoch, 24 epochs would match that step
count, so 36 epochs gives that plus margin for this being a harder dataset,
with patience scaled by the same 3x factor so early stopping doesn't cut the
larger budget short for the same reason it cut the original one short.

**Watched closely through epoch 5-6 for the same collapse signature** (val
acc stuck at 0.5467, the val split's majority-class rate, 82/150) per
instruction, ready to stop early if it reappeared. It did not: epoch 4 broke
out to val acc **0.6333**, clearly above baseline and above that epoch's own
train accuracy -- a real learning signal, not noise. Training ran the full
19 epochs before early-stopping on patience (best val acc stayed 0.6333 from
epoch 4; never re-collapsed afterward, oscillating 0.45-0.61 for the
remaining 15 epochs without beating it).

### Result 3: the converged, budget-matched model (checkpoints_v2_accum_long/best_model.pt)

Pooled test:

| Metric | batch=8 (committed) | accum=2, long budget (this run) |
|---|---|---|
| AUROC | 0.674 | **0.662** |
| Accuracy | 0.607 | 0.673 |
| Precision | 0.791 | 0.702 |
| Recall | 0.541 | 0.867 |
| F1 | 0.642 | 0.776 |

Per-attack breakdown:

| Subset | n | present/removed | AUROC (batch=8) | AUROC (accum, long) |
|---|---|---|---|---|
| distortion_optimization | 83 | 9/74 | 0.733 | **0.769** |
| SANA-VAE | 67 | 43/24 | 0.672 | **0.648** |
| pooled | 150 | 52/98 | 0.674 | 0.662 |

### This is the resolved comparison

Two independently-trained, **both genuinely converged** (neither collapsed)
models, at two different batch-size/effective-batch-size configurations,
land within **0.012 AUROC of each other** (0.674 vs 0.662) -- both far below
the 0.888 baseline, both in the same ~0.65-0.67 range. That small a gap
between two batch-size settings, next to a ~0.22 gap from baseline, is
strong evidence that **batch size was not the dominant driver of the drop
from 0.888** -- the new, more diverse attack mix is.

The per-attack ordering also now **agrees** between the two converged runs
(distortion_optimization consistently easier than SANA-VAE: 0.733/0.672 and
0.769/0.648), reversing last section's caution about the ordering being
unstable -- that instability was specific to the degenerate run, not a real
property of the comparison. With two independently-converged models pointing
the same direction, **SANA-VAE being the harder attack family for this
classifier is now a reasonably trustworthy finding**, not just a shared
confound's artifact.

### Task 1 control -- did not confirm the same story, flagged rather than smoothed over

| Reference | AUROC |
|---|---|
| origin/v2 (original v5 run) | 0.6718 |
| batch=8 (committed, healthy) | 0.6107 |
| accum=2, long budget (this run, healthy) | **0.5712** |

This is the one result that does **not** fit a clean "batch size resolved,
attack mix is the story" narrative: the properly-converged, budget-matched
model scores *lower* on Task 1 than the batch=8 model, continuing a
downward trend across all three data points rather than converging back
toward the 0.6718 reference. `removed_mean_prob` (0.563) and
`control_mean_prob` (0.545) are close together with small standard
deviations (0.056/0.048) -- this model's probability outputs for the
removed-vs-control comparison specifically are more compressed than the
other two, even though it is not degenerate on the main present-vs-removed
task and its per-attack breakdown looks sensible. Plausible reading: the
longer training budget let this model fit the main task's decision boundary
well without necessarily learning the more subtle removal-residue signal
Task 1 isolates -- but that is a hypothesis, not confirmed here. Not
investigated further tonight; flagged rather than absorbed into the main
narrative.

### Bottom line

- **Resolved**: the pooled-AUROC batch-size confound. Two converged models,
  two batch-size settings, 0.012 AUROC apart -- the ~0.21 drop from the
  0.888 baseline is attributable to the new attack mix, not batch size.
- **Resolved, reversing earlier caution**: the per-attack-family ordering
  (SANA-VAE harder than distortion_optimization) is now consistent across
  both converged runs.
- **Not resolved**: why Task 1 control AUROC continues declining across all
  three runs (0.672 -> 0.611 -> 0.571) rather than stabilizing. This is a
  separate, real, open question -- not the same confound as the pooled
  result, and not chased further tonight per scope.
