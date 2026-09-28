# DEPRECATED — known source-image leakage

This split directory (`data/splits_jpeg80`) was built with the pre-fix
`src/data/build_splits.py`, which did a plain file-level `train_test_split`
with no concept of "source image". Under the embed-all -> attack-all layout
these splits were built from (every raw source contributes both a
watermarked and a removed derivative), that leaked source images across
train/val/test — the same underlying image's clean and removed variants could
land in different splits, letting a classifier partially memorize per-image
artifacts instead of learning a general forensic signal.

See `leak_check_report.md` (repo root) for the audit and exact overlap counts
for this directory.

**Do not use these splits for training or reported results.** Regenerate from
source with the current (group-aware) `src/data/build_splits.py`, which
splits by source-ID via `StratifiedGroupKFold` and asserts zero group
leakage before writing the CSVs.

`data/splits_v2` is unaffected by this issue (different dataset layout,
confirmed leak-free in `leak_check_report.md`) and remains valid.
