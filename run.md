# How to Run

## Setup done

- Created venv at `.venv` (Python 3.13, via `python -m venv .venv`)
- Installed project + deps: `pip install -e .`
- Installed `requests` (used by `scripts/download_div2k.py`, missing from `pyproject.toml`)
- Torch installed is **CPU-only** (no local GPU detected) — training/eval will be slow. Use Colab T4 for real runs, per README.
- `src/data/dataset.py` hardened: `__getitem__` now raises a clear `FileNotFoundError` if `cv2.imread` returns `None`, and `create_dataloaders` checks `train.csv`/`val.csv`/`test.csv` exist before building datasets (instead of a raw pandas traceback).
- `configs/default.yaml` repointed at the one split that's actually usable out of the box (see below) — verified with a live dataloader smoke test (`train batch: torch.Size([4, 3, 224, 224])`, 140/30/30 train/val/test images).

## Activate venv

```bash
cd "C:\final year project\watermark-forensics"
source .venv/Scripts/activate
```

Always run commands from the repo root — configs and data paths are relative to cwd.

## Important: module invocation

`README.md` shows `python src/train.py ...` and `python src/evaluate.py ...`. These **fail** with
`ModuleNotFoundError: No module named 'src'` because both files use absolute imports
(`from src.utils.seed import ...`). Run them as modules instead:

```bash
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test
```

All other CLIs (`src.watermark.embed`, `src.attacks.distortion`, `src.data.build_splits`) already work
with `python -m ...` as written in the README.

## Current data state (checked)

- `data/raw/`, `data/watermarked/`, `data/removed/`, `data/splits/` — **empty** (gitignored, never populated on this machine). The DIV2K download was attempted but did not complete/persist — `data/raw/` does not exist. Not needed for the path below.
- `data/watermarked_jpeg80/` (100 images) + `data/removed_jpeg80/` (100 images) — **self-contained, already on disk**, referenced directly by `data/splits_jpeg80_extreme/*.csv`. This is the dataset `configs/default.yaml` now uses.
- `data/removed_mild/`, `data/removed_mild_jpeg80/` and `data/splits_mild/`, `data/splits_jpeg80/` — also present, but their `train/val/test.csv` reference `data/watermarked/...` (not `watermarked_jpeg80`), which is empty — **stale, won't load** until `data/watermarked/` is regenerated from raw images.
- `checkpoints/` — does not exist yet; created automatically on first `python -m src.train` run.

## `configs/default.yaml` — current state

```yaml
data:
  raw_dir: "data/raw"
  watermarked_dir: "data/watermarked_jpeg80"   # updated — matches what splits_jpeg80_extreme actually points to
  removed_dir: "data/removed_jpeg80"           # updated — matches what splits_jpeg80_extreme actually points to
  edited_dir: "data/edited"
  splits_dir: "data/splits_jpeg80_extreme"     # updated — only split with all images present locally
  ...
train:
  save_dir: "checkpoints"
  log_dir: "results/logs"
eval:
  save_dir: "results"
```

Note: `data.watermarked_dir` / `data.removed_dir` are only consumed by `src/data/build_splits.py` when
*building* new splits — `src/train.py` / `src/evaluate.py` read image paths straight out of the split
CSVs, so they don't need to match for training to work. They're kept in sync here for consistency /
documentation only.

Attack params under `attack:` (jpeg_quality=2, gaussian_std=45, blur_kernel=13, 6 passes) describe how
`removed_jpeg80` *would* be regenerated from scratch — they don't affect training against the existing
split, since those images are already on disk.

## Run now (no download needed)

```bash
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test
```

This trains directly against the 140/30/30 train/val/test split already present in
`data/splits_jpeg80_extreme/`.

## Full pipeline from scratch (only if you want to regenerate data)

```bash
# 1. Download DIV2K validation images
python scripts/download_div2k.py --max-images 800

# 2. Embed watermark (class 0)
python -m src.watermark.embed --input data/raw --output data/watermarked \
  --alpha 0.1 --block-size 8 --dwt-level 2

# 3. Verify watermark present
python -m src.watermark.embed --verify --watermarked data/watermarked --original data/raw

# 4. Apply distortion attack (class 1)
python -m src.attacks.distortion --input data/watermarked --output data/removed \
  --jpeg-quality 50 --gaussian-std 5 --blur-kernel 3

# 5. Verify attack removes watermark
python -m src.watermark.embed --verify --watermarked data/removed --original data/raw

# 6. Build stratified splits (70/15/15)
python -m src.data.build_splits --watermarked data/watermarked --removed data/removed --splits data/splits

# 7. Train (use python -m, not python src/train.py)
python -m src.train --config configs/default.yaml   # first repoint splits_dir back to "data/splits"

# 8. Evaluate
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test
```

## v2 pipeline — verification-labeled dataset (fixes the overfitting/label-leakage)

Labels here come from actually running `verify()` after a randomized-strength attack, not
from "which folder we put the file in" — see `current_status.txt` (2026-08-26 update) for why.
`--alpha 0.02` is required to get a balanced present/removed split; the default 0.2 makes the
watermark too robust and yields ~835/65.

```bash
# 1. Build verification-labeled dataset from data/raw (900 images already on disk)
python -m scripts.build_verified_dataset --raw data/raw \
  --out-present data/watermarked_v2 --out-removed data/removed_v2 \
  --save-quality 80 --alpha 0.02

# 2. Build stratified splits
python -m src.data.build_splits --watermarked data/watermarked_v2 --removed data/removed_v2 \
  --splits data/splits_v2

# 3. Train
python -m src.train --config configs/v2.yaml

# 4. Evaluate
python -m src.evaluate --config configs/v2.yaml --checkpoint checkpoints_v2/best_model.pt --split test
```

Last run: test accuracy 0.800, AUROC 0.888 (`results/v2/test_metrics.json`) — real signal, not the
old 100%/1.0 ceiling.

## Quickest showcase (no training — reuses existing checkpoint)

`checkpoints_v2/best_model.pt` and `data/splits_v2/` are already on disk, so evaluation alone
reproduces the headline result in seconds:

```bash
cd "C:/final year project/watermark-forensics"
source .venv/Scripts/activate
python -m src.evaluate --config configs/v2.yaml --checkpoint checkpoints_v2/best_model.pt --split test
```

Expect ~80.0% accuracy, 0.888 AUROC (matches `results/v2/test_metrics.json`).

Even faster — print the already-saved metrics with zero compute:

```bash
cat "C:/final year project/watermark-forensics/results/test_metrics.json"
```

## Existing configs

| Config | Attack strength | Splits used | Status |
|---|---|---|---|
| `configs/default.yaml` | extreme (jpeg=2, noise=45, blur=13, 6 passes) | `data/splits_jpeg80_extreme` | **ready to run**, images present |
| `configs/mild.yaml` | mild (jpeg=80, noise=2, blur=1, 1 pass) | `data/splits_mild` | stale — references empty `data/watermarked/` |
| `configs/mild_fixed.yaml` | mild + fixed format leak | `data/splits_jpeg80` | stale — references empty `data/watermarked/` |

To make `mild.yaml` / `mild_fixed.yaml` runnable, either regenerate `data/watermarked/` (steps 1–3 in the
full pipeline above), or repoint their `splits_dir` at a self-contained split the way `default.yaml` was
repointed.

## Quick sanity check (no data needed)

```bash
python -m src.watermark.embed --help
python -m src.attacks.distortion --help
python -m src.data.build_splits --help
python -m src.train --help
python -m src.evaluate --help
```

## Dataloader smoke test (verifies a config's split loads without training)

```bash
python -c "
from pathlib import Path
from src.data.dataset import create_dataloaders
train_dl, val_dl, test_dl = create_dataloaders(Path('data/splits_jpeg80_extreme'), img_size=224, batch_size=4, num_workers=0)
x, y = next(iter(train_dl))
print(x.shape, y.shape, y.tolist())
print(len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))
"
```


 1. Live metrics (real inference, ~1-2 min):
  cd "C:/final year project/watermark-forensics"
  source .venv/Scripts/activate
  python -m src.evaluate --config configs/v2.yaml --checkpoint checkpoints_v2/best_model.pt --split test

  2. Regenerate + open the graphs (instant, read-only):
  python -m scripts.make_figures
  This writes 5 PNGs to results/figures/:

  ┌─────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │            File             │                                              What it shows                                              │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ metrics_comparison.png      │ Leaky (100%/1.0 fake) vs. honest (80%/0.888 real) — your best "here's the bug we found and fixed" slide │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ confusion_matrices.png      │ Side-by-side confusion matrices, same leaky-vs-honest story                                             │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ training_curves_v2.png      │ Loss/accuracy curves for the honest run — shows real learning, not memorization                         │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ training_curves_default.png │ Same for the leaky run — for contrast                                                                   │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ alpha_sweep.png             │ How you tuned watermark strength to get a balanced dataset — nice "rigor" slide                         │
  └─────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  The leaky-vs-honest comparison (metrics + confusion matrix) is your strongest narrative beat — it shows you caught label leakage, diagnosed the root
  cause, and fixed it, which is more impressive to a committee than just "80% accuracy."