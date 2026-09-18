"""
Build stratified, group-aware train/val/test splits from watermarked (class 0)
and removed (class 1).

Group-aware: every file is assigned a source-ID (the filename stem before any
extension, e.g. "0235.png" -> "0235"), matching the derivation used in
leak_check_report.md. This is the raw source image a file was derived from.
Splitting is done with StratifiedGroupKFold so that no source-ID ever appears
in more than one split, even when a source contributes both a watermarked and
a removed derivative (the embed-all -> attack-all layout used by
data/watermarked + data/removed, and planned for Phase 2 diffusion-regen
attacks). See leak_check_report.md for the audit that found the old
plain train_test_split leaked source images across splits in that layout.

CLI: python -m src.data.build_splits --watermarked ... --removed ... --splits ...
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


def _source_id(path: Path) -> str:
    """Source-ID = filename stem before the extension, e.g. 0235.png -> 0235.

    Same derivation as leak_check_report.md's audit: every pipeline stage
    (embed.py, distortion.py, build_verified_dataset.py) preserves
    img_path.name unchanged, so the stem is a reliable source identifier.
    """
    return path.stem


def _group_stratified_split(df: pd.DataFrame, held_out_ratio: float, seed: int):
    """Split df into (kept, held_out) with no group (source-ID) crossing the
    boundary, approximately stratified by label.

    GroupShuffleSplit does not support combined group+stratify. We use
    StratifiedGroupKFold instead: pick n_splits ~= round(1 / held_out_ratio)
    and take one fold as the held-out portion. This guarantees the hard
    invariant (zero group leakage) exactly, while only approximating the
    requested ratio and class balance (StratifiedGroupKFold balances classes
    across folds as well as group sizes allow, which is not exact when group
    sizes are uneven). The actual resulting sizes are printed so any drift
    from the requested ratio is visible, not silent.
    """
    n_splits = max(2, round(1 / held_out_ratio))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kept_idx, held_out_idx = next(sgkf.split(df, df["label"], groups=df["source_id"]))
    return df.iloc[kept_idx].copy(), df.iloc[held_out_idx].copy()


def _assert_no_group_leakage(train_df, val_df, test_df) -> None:
    train_ids = set(train_df["source_id"])
    val_ids = set(val_df["source_id"])
    test_ids = set(test_df["source_id"])

    overlaps = {
        "train/val": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "val/test": val_ids & test_ids,
    }
    leaked = {pair: ids for pair, ids in overlaps.items() if ids}
    if leaked:
        details = "; ".join(
            f"{pair}: {len(ids)} shared source-IDs (e.g. {sorted(ids)[:5]})"
            for pair, ids in leaked.items()
        )
        raise AssertionError(
            f"Source-image leakage detected across splits: {details}. "
            "This should be impossible with StratifiedGroupKFold grouping by "
            "source_id — check that source_id was derived correctly."
        )


def build_splits(
    wm_dir: Path, removed_dir: Path, splits_dir: Path,
    train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    wm_files = sorted(wm_dir.glob("*.png"))
    rm_files = sorted(removed_dir.glob("*.png"))
    all_files = wm_files + rm_files

    df = pd.DataFrame({
        "path": [str(p) for p in all_files],
        "label": [0] * len(wm_files) + [1] * len(rm_files),
        "source_id": [_source_id(p) for p in all_files],
    })

    # Split off (val + test) as a group, then split that group into val/test.
    train_df, temp_df = _group_stratified_split(df, val_ratio + test_ratio, seed)
    val_size_of_temp = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = _group_stratified_split(temp_df, 1 - val_size_of_temp, seed)

    _assert_no_group_leakage(train_df, val_df, test_df)

    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df[["path", "label"]].to_csv(splits_dir / "train.csv", index=False)
    val_df[["path", "label"]].to_csv(splits_dir / "val.csv", index=False)
    test_df[["path", "label"]].to_csv(splits_dir / "test.csv", index=False)

    print(f"Train: {len(train_df)} (0:{sum(train_df.label==0)}, 1:{sum(train_df.label==1)})")
    print(f"Val:   {len(val_df)} (0:{sum(val_df.label==0)}, 1:{sum(val_df.label==1)})")
    print(f"Test:  {len(test_df)} (0:{sum(test_df.label==0)}, 1:{sum(test_df.label==1)})")
    print("No source-ID leakage across train/val/test (verified).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermarked", type=Path, required=True)
    parser.add_argument("--removed", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_splits(args.watermarked, args.removed, args.splits, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()