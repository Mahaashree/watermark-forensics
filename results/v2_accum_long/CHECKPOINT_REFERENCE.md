# Checkpoint reference: checkpoints_v2_accum_long/best_model.pt

Not committed (106MB, gitignored via the project-wide `*.pt` rule, same as
every other checkpoint in this repo). This file documents provenance so the
result is traceable without the binary.

## Current (2026-10-04 retrain, CUDA)

The original MPS-trained checkpoint documented below was lost from disk.
Retraining the identical config on this machine's CUDA GPU (RTX 3050) did
not reproduce the original numbers — see `research_paper_draft.txt`
Section 5.3.1 for the full account (a hardware-dependent reproducibility
gap in the dataset-build step, not a code regression). Current status:

- **Config:** `configs/v2_accum_long.yaml` (unchanged from below)
- **Trained on:** `data/splits_v2` as it exists on disk 2026-10-04 (298
  present / 602 removed — differs from the 399/501 figure in paper draft
  Section 3.2; see Section 5.3.1 for why)
- **Device:** CUDA (RTX 3050, 6GB) via `.venv` (torch 2.6.0+cu124) — NOT
  MPS, unlike the original run below
- **3 seeds trained:** 42 (`checkpoints_v2_accum_long/`), 1
  (`checkpoints_v2_accum_long_seed1/`), 2
  (`checkpoints_v2_accum_long_seed2/`)
- **Seed 42 training log:** `results/train_seed42_rerun.log`; best epoch
  26 (val acc 0.7933), early-stopped at epoch 35
- **SHA-256 (seed 42):** `66616bc690b9c7114e6f92efdadb1fcf8a397db1ef8fdc592011b570dbb03f07`
- **Test metrics:** `results/v2_accum_long/test_metrics_seed{42,1,2}.json`.
  Note: evaluating this retrain overwrote the original run's
  `test_metrics.json` in-place (same default filename, no `--tag` on the
  first eval call) before it was renamed to `test_metrics_seed42.json` —
  the original MPS numbers were recovered separately from git history
  (commit `5c45190`) into `test_metrics_original_mps.json` so nothing was
  permanently lost, but be aware `test_metrics_seed42.json` is the new
  CUDA run, not a copy of the original despite the parallel naming.
- **3-seed aggregate:** accuracy 0.7289 ± 0.034, AUROC 0.7632 ± 0.012
  (`scripts/aggregate_seed_metrics.py results/v2_accum_long/test_metrics_seed*.json`)
- **Per-attack breakdown, Task 1 control eval:** not yet re-run on this
  retrain (the files below are from the original MPS run) — flagged as
  future work in paper draft Section 8.

To reproduce: `python -m src.train --config configs/v2_accum_long.yaml
--seed {42,1,2}` against the current `data/splits_v2`, then
`python -m src.evaluate --config configs/v2_accum_long.yaml --checkpoint
checkpoints_v2_accum_long_seed{N}/best_model.pt --split test --tag seed{N}`.

---

## Original (lost, MPS — historical record)

- **Config:** `configs/v2_accum_long.yaml` (batch_size=8, grad_accum_steps=2
  -> effective batch 16; epochs=36, early_stopping_patience=15; otherwise
  identical to the original baseline: ConvNeXt-Tiny, ImageNet-pretrained,
  lr=1e-4, weight_decay=1e-4, dropout=0.1, CosineAnnealingLR)
- **Trained on:** `data/splits_v2` (900-image rebuild, commit `27bab6f`)
- **Training log:** `results/logs_v2_accum_long/train.log`
- **Best epoch:** 4 (val acc 0.6333), early-stopped at epoch 19 (patience 15
  exhausted, never re-beaten after epoch 4)
- **SHA-256:** `7d152d9440412eb90c46e1570f8af15ea2a79b19850ae2fe3a9fe15b651e89a1`
- **Test metrics:** recovered from git history (commit `5c45190`, the
  last commit before this file was overwritten by the 2026-10-04 retrain's
  eval run) into `results/v2_accum_long/test_metrics_original_mps.json`
- **Per-attack breakdown:** `results/attack_breakdown/accum_long_optimization.json`,
  `results/attack_breakdown/accum_long_sana.json`
- **Task 1 control eval:** `results/task1_control_auroc_accum_long.json`

To reproduce: retrain with the config above against `data/splits_v2` as
currently committed (seed=42 throughout), or regenerate this checkpoint by
rerunning `python -m src.train --config configs/v2_accum_long.yaml`. Exact
byte-for-byte reproduction is not guaranteed -- MPS (used for this run) does
not give the same determinism guarantees as CUDA-deterministic mode. **In
practice, as of 2026-10-04, it was not reproducible at all on CUDA — see
above.**
