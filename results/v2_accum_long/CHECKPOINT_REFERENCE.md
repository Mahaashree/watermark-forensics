# Checkpoint reference: checkpoints_v2_accum_long/best_model.pt

Not committed (106MB, gitignored via the project-wide `*.pt` rule, same as
every other checkpoint in this repo). This file documents provenance so the
result is traceable without the binary.

- **Config:** `configs/v2_accum_long.yaml` (batch_size=8, grad_accum_steps=2
  -> effective batch 16; epochs=36, early_stopping_patience=15; otherwise
  identical to the original baseline: ConvNeXt-Tiny, ImageNet-pretrained,
  lr=1e-4, weight_decay=1e-4, dropout=0.1, CosineAnnealingLR)
- **Trained on:** `data/splits_v2` (900-image rebuild, commit `27bab6f`)
- **Training log:** `results/logs_v2_accum_long/train.log`
- **Best epoch:** 4 (val acc 0.6333), early-stopped at epoch 19 (patience 15
  exhausted, never re-beaten after epoch 4)
- **SHA-256:** `7d152d9440412eb90c46e1570f8af15ea2a79b19850ae2fe3a9fe15b651e89a1`
- **Test metrics:** `results/v2_accum_long/test_metrics.json`
- **Per-attack breakdown:** `results/attack_breakdown/accum_long_optimization.json`,
  `results/attack_breakdown/accum_long_sana.json`
- **Task 1 control eval:** `results/task1_control_auroc_accum_long.json`

To reproduce: retrain with the config above against `data/splits_v2` as
currently committed (seed=42 throughout), or regenerate this checkpoint by
rerunning `python -m src.train --config configs/v2_accum_long.yaml`. Exact
byte-for-byte reproduction is not guaranteed -- MPS (used for this run) does
not give the same determinism guarantees as CUDA-deterministic mode.
