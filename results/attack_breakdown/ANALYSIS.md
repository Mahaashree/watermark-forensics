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
