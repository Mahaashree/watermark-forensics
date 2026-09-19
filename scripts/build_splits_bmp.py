"""
Task 2: same stratified-split logic as src/data/build_splits.py, but globs
*.bmp instead of *.png. Kept separate rather than editing build_splits.py.

CLI: python scripts/build_splits_bmp.py --watermarked data/watermarked_v4_bmp --removed data/removed_v4_bmp --splits data/splits_v4_bmp
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def build_splits_bmp(
    wm_dir: Path, removed_dir: Path, splits_dir: Path,
    train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    wm_files = sorted(wm_dir.glob("*.bmp"))
    rm_files = sorted(removed_dir.glob("*.bmp"))

    df = pd.DataFrame({
        "path": [str(p) for p in wm_files] + [str(p) for p in rm_files],
        "label": [0] * len(wm_files) + [1] * len(rm_files)
    })

    train_df, temp_df = train_test_split(df, train_size=train_ratio, stratify=df["label"], random_state=seed)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, train_size=val_size, stratify=temp_df["label"], random_state=seed)

    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"Train: {len(train_df)} (0:{sum(train_df.label==0)}, 1:{sum(train_df.label==1)})")
    print(f"Val:   {len(val_df)} (0:{sum(val_df.label==0)}, 1:{sum(val_df.label==1)})")
    print(f"Test:  {len(test_df)} (0:{sum(test_df.label==0)}, 1:{sum(test_df.label==1)})")


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

    build_splits_bmp(args.watermarked, args.removed, args.splits, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
