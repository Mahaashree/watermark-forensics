# Session Notes — 2026-10-04: Journal-Readiness Pass

Scope: half-day pass to move the paper closer to journal-ready. Did not
touch git (no commits/staging) per instruction — everything below is in
the working tree, uncommitted.

## What was done

### 1. Related Work section (paper draft §1.1)
Added a real related-work section — previously the paper only had a flat
tools/papers list, no literature review. Covers three bodies of prior
work: watermark embedding (Lai & Tsai 2010 — the DWT-DCT-SVD scheme this
project's embedding is directly based on; Fernandez et al. ICCV 2023
Stable Signature; Wen et al. NeurIPS 2023 Tree-Ring Watermarks), removal
attacks (Zhao et al. NeurIPS 2024 regeneration attacks; An et al. ICML
2024 WAVES benchmark), and forensic generalization (Wang et al. CVPR 2020;
Corvi et al. ICASSP 2023; Ojha et al. CVPR 2023 — the last of these is the
closest prior-work analogue to this project's own Gap 1). Every citation
was checked against a primary source via web search before being written
down, not recalled from memory. REFERENCES section updated to match.

### 2. Real baseline comparison (paper draft §5.8)
Found that a baseline comparator already existed
(`experiments/exp1_baseline_repro/baseline_check.py`: a 2-layer CNN +
non-learned Laplacian-variance / JPEG-blockiness thresholds) but had only
ever been run against the old, discarded 100-image leaky split, and its
summary line still printed the fake Phase-1 "1.0 accuracy" number. Fixed
that stale line and reran it against the real `data/splits_v2` test set.
Result: the 2-layer CNN collapses to constant majority-class prediction
(AUROC 0.44, *below* chance), the non-learned thresholds reach AUROC
0.52–0.59. Both are well short of the ConvNeXt-Tiny headline — this
directly answers the "no baseline comparison" gap flagged in `workplan.md`
with a real, just-computed number, not a fabricated one.

### 3. Multi-seed stability run (paper draft §5.3.1) — and what it uncovered
Went to reproduce the headline checkpoint (`checkpoints_v2_accum_long`)
for 2 additional seeds, per `scripts/aggregate_seed_metrics.py` (existing
but unused tooling). The checkpoint didn't exist on disk. Retraining the
identical config on this machine's CUDA GPU (RTX 3050) did **not**
reproduce the original 0.6733 acc / 0.6615 AUROC.

Investigated rather than shrugged it off:
- `results/v2_accum_long/CHECKPOINT_REFERENCE.md` already recorded that
  the original run was trained on **MPS**, not CUDA.
- Read `src/attacks/diffusion_regen.py` to rule out an unseeded RNG in the
  SANA-VAE attack as the cause — it's a deterministic `@torch.no_grad()`
  encode/decode, no generator, no sampling.
- Conclusion: the watermark is deliberately weak (`alpha=0.02`, already
  flagged in the paper's own §5.6 as borderline), so small cross-backend
  floating-point differences (MPS vs. CUDA) are enough to flip some
  images' `verify()` outcome, changing the realized present/removed class
  balance when the dataset-build step reruns on different hardware. This
  is why `data/watermarked_v2`/`data/removed_v2` on disk now (298/602)
  don't match the paper's reported 399/501 build, even though
  `scripts/build_verified_dataset.py --seed 42` was used both times.

Flagged this to you directly rather than silently picking a number or
silently "fixing" it. You chose: adopt the current on-disk dataset as
authoritative going forward.

Then ran 3 seeds (42, 1, 2) of `configs/v2_accum_long.yaml` on CUDA:

| Seed | Accuracy | AUROC  |
|------|----------|--------|
| 42   | 0.7333   | 0.7704 |
| 1    | 0.7600   | 0.7694 |
| 2    | 0.6933   | 0.7498 |
| **Mean** | **0.7289** | **0.7632** |
| Std  | 0.0336   | 0.0116 |

This is real, stable (AUROC std 0.012), and *higher* than the old
single-run number — a correction, not a regression. Documented as new
§5.3.1 in the paper, with the original run kept as a flagged historical
record (§5.3) rather than deleted, and explicit notes on which downstream
sections (§5.2 batch-size-confound isolation, §5.4 per-attack-family
breakdown, §5.7 3-class comparison) still reflect the old run and have
*not* been re-verified against the current dataset — listed in §8 Future
Work rather than silently left inconsistent.

### 4. Fixed a real bug found along the way
`data/splits_v2/test_task1_control.csv` — a file tracked in git and
required for the paper's §5.5 control comparison — had been deleted in
the working tree. Restored it from the last commit (`git checkout --
<path>`; read-only history access, no commit/staging, consistent with
your "don't touch git" instruction).

### 5. Recovered data that would otherwise have been lost
Evaluating the new seed-42 checkpoint overwrote
`results/v2_accum_long/test_metrics.json` in place (same default filename
as the original run's output). Recovered the original MPS-run numbers
from git history (`git show HEAD:... > test_metrics_original_mps.json`)
before renaming the new file — nothing from the original run was
permanently lost, but it would have been if I hadn't checked.

### 6. Documentation consistency fixes
- `README.md`: quickstart described the discarded Phase-1 leaky pipeline
  (`src.watermark.embed` + `src.attacks.distortion` + construction-based
  labels) instead of the current verification-labeled one
  (`scripts.build_verified_dataset` + group-aware splits). Rewrote it to
  match what the paper actually describes, and updated the headline
  number to the new 3-seed figure.
- `ARCHITECTURE.md`: updated the headline-number note (was already stale
  about the lost checkpoint; now reflects the 3-seed retrain and the
  hardware-reproducibility finding).
- `workplan.md`: reconciled several stale checkboxes —
  `build_splits.py`'s group-awareness was already done (just never
  checked off), the baseline comparison and related-work items are now
  genuinely done, the multi-seed item is done and its write-up explains
  why it took longer than expected.
- Paper draft: corrected several "CPU/MPS-only" claims (abstract, §6
  status, Limitations) that were stale now that this machine has a CUDA
  GPU — Gap 1 is still unresolved, but the reason changed from "hard
  compute block" to "stalled mid-download," which is a meaningfully
  different (and more fixable) problem.
- `current_status.txt`: appended a dated session log in the file's
  existing style, summarizing all of the above for continuity.

## What I deliberately did NOT do (and why)

- **Did not attempt Gap 1** (resuming the stalled `remove-ai-watermarks`
  download). Multi-GB download, already stalled once on this exact
  machine per `workplan.md` — high risk of not finishing in a half-day
  budget and tying up the GPU/bandwidth for everything else. Left as the
  top item in Future Work.
- **Did not re-run §5.2 (batch-size confound), §5.4 (per-attack
  breakdown), or §5.7 (3-class comparison)** on the current dataset. Each
  needs its own training run(s) and, for §5.7, a separate 3-class dataset
  rebuild. Flagged explicitly in the paper rather than left silently
  stale.
- **Did not commit anything to git.** Per your instruction — all changes
  above are sitting uncommitted in the working tree.
- **Did not try to fix the underlying MPS/CUDA non-determinism.** Would
  need either a stronger (less borderline) watermark alpha or
  deterministic-mode flags across every numeric library in the pipeline —
  flagged as future work, not attempted.

## Files touched this session

Modified: `research_paper_draft.txt`, `README.md`, `ARCHITECTURE.md`,
`workplan.md`, `current_status.txt`,
`experiments/exp1_baseline_repro/baseline_check.py`,
`results/v2_accum_long/CHECKPOINT_REFERENCE.md`,
`data/splits_v2/test_task1_control.csv` (restored).

New: `checkpoints_v2_accum_long/best_model.pt` (retrained),
`checkpoints_v2_accum_long_seed1/`, `checkpoints_v2_accum_long_seed2/`,
`results/v2_accum_long/test_metrics_seed{42,1,2}.json`,
`results/v2_accum_long/test_metrics_original_mps.json`,
`experiments/exp1_baseline_repro/baseline_results_v2.json`,
`results/train_seed{42_rerun,1,2}.log`.

## Recommended next half-day (priority order)

1. Resume Gap 1 (`remove-ai-watermarks invisible <img> -o <out> --force
   --cpu-offload` with `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` set, per
   `workplan.md`) — highest leverage, most at-risk of not finishing.
2. Re-run §5.2 and §5.4 on the current CUDA dataset for internal
   consistency (both are retrain-and-eval, same pattern as this session's
   §5.3.1 work, each ~10-15 min on this GPU).
3. Write a real Discussion/Conclusion section closing the "progress
   report" framing into a submittable paper narrative.
