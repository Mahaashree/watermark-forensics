# How to Run (updated)

This supersedes the previous version of this file, which was written on a
different machine/checkout and pointed at data/configs that don't exist in
this working tree. Everything below reflects what's actually on disk now.

## Setup done

- `.venv` created (Python 3.13, `python -m venv .venv`), project installed via
  `pip install -e .`.
- `torch`/`torchvision` upgraded to **CUDA builds** (`torch==2.6.0+cu124`,
  `torchvision==0.21.0+cu124`) — plain `pip install torch` on Windows gives a
  CPU-only build by default, which was the initial state. GPU (RTX 3050) is
  now used automatically since all configs have `train.device: "auto"`.
- `timm` installed (was missing from a from-scratch `pip install -e .` in this
  environment).

## Activate venv (PowerShell)

```powershell
cd "C:\final year project\watermark-forensics"
.venv\Scripts\Activate.ps1
```

If blocked by execution policy:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Always run commands from the repo root.

## Module invocation

Run everything as a module (`python -m ...`), not `python src/train.py ...` —
the latter fails with `ModuleNotFoundError: No module named 'src'` because
these files use absolute imports.

## Data on disk (built from scratch this session — none of this was in the git repo; `data/` is gitignored)

- `data/raw/` — **900** images = DIV2K validation (100) + DIV2K train (800),
  via `python scripts/download_div2k.py --split valid` then `--split train`.
- `data/watermarked_v2/` — **298** images. Watermark embedded
  (`alpha=0.02`, `block_size=8`, `dwt_level=2`) and verified *present* after a
  mild JPEG-80 re-encode. Built via
  `python -m scripts.build_verified_dataset --raw data/raw --out-present data/watermarked_v2 --out-removed data/removed_v2 --save-quality 80 --alpha 0.02`.
- `data/removed_v2/` — **602** images from the same command: watermark
  embedded then attacked, verified *absent* afterward.
- `data/original_v2/` — **900** images, one per raw source, **never
  watermarked**, but resized to 224x224 + JPEG-80 re-encoded to match the
  file format of the other two classes (prevents the classifier from
  cheating on file-format artifacts instead of real watermark signal). Built
  via `python -m scripts.build_original_class --raw data/raw --out data/original_v2 --save-quality 80`.
- `data/splits_v2/` — **2-class** (Present=0 / Removed=1) leak-free,
  group-aware splits: 600 train / 150 val / 150 test. Built via
  `python -m src.data.build_splits --watermarked data/watermarked_v2 --removed data/removed_v2 --splits data/splits_v2`.
- `data/splits_3class/` — **3-class** (Present=0 / Removed=1 / Original=2)
  leak-free splits: 1200 train / 300 val / 300 test. Built via
  `python -m src.data.build_splits --watermarked data/watermarked_v2 --removed data/removed_v2 --original data/original_v2 --splits data/splits_3class`.

**Known-bad split dirs — do not use:** `data/splits`, `data/splits_full`,
`data/splits_mild`, `data/splits_jpeg80`, `data/splits_jpeg80_extreme` all
carry a `DEPRECATED_LEAKED.md` marker (pre-fix `build_splits.py` leaked
source images across train/val/test — see `leak_check_report.md`). They're
also empty in this checkout regardless, since their underlying image dirs
were never regenerated here.

## Configs

| Config | Classes | Splits dir | Checkpoint dir | Status |
|---|---|---|---|---|
| `configs/default.yaml` | 2 (Present/Removed) | `data/splits_v2` | `checkpoints/` | **trained** — test acc 0.76, AUROC 0.796 |
| `configs/3class.yaml` | 3 (Present/Removed/Original) | `data/splits_3class` | `checkpoints_3class/` | **trained** — best val acc 0.7467, test metrics not yet pulled |
| `configs/v2.yaml` | 2 | `data/splits_v2` | `checkpoints_v2/` | not used this session (duplicate of default.yaml's setup) |

## Train / evaluate

```powershell
# 2-class model
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints\best_model.pt --split test

# 3-class model
python -m src.train --config configs/3class.yaml
python -m src.evaluate --config configs/3class.yaml --checkpoint checkpoints_3class\best_model.pt --split test
```

`evaluate.py` auto-detects class count from the checkpoint's config: for 2
classes it reports the original binary metrics (accuracy/AUROC/precision/
recall/F1/TPR@low-FPR); for 3+ classes it reports macro-averaged
precision/recall/F1 and one-vs-rest macro AUROC instead.

## New: single-image inference

`src/infer.py` — feed it any image, get a Present/Removed/Original verdict
(2-class configs only print Present/Removed) with confidence and full
per-class probabilities. See `guide.md` for exactly what preprocessing
happens and why this is a "best-effort guess," not a certainty check.

```powershell
python -m src.infer --config configs/3class.yaml --checkpoint checkpoints_3class\best_model.pt --image path\to\image.png
```

## New: layer-by-layer inspection

`src/inspect_layers.py` — runs one image through the network and prints the
output shape + activation stats (mean/std/min/max) at every stage (stem,
each of the 4 ConvNeXt blocks, pooling head, classifier head), to verify the
forward pass is behaving correctly layer by layer, not just check the final
verdict.

```powershell
python -m src.inspect_layers --config configs/3class.yaml --checkpoint checkpoints_3class\best_model.pt --image path\to\image.png
```

## Quick sanity check (no data needed)

```powershell
python -m src.watermark.embed --help
python -m src.attacks.distortion --help
python -m src.data.build_splits --help
python -m src.train --help
python -m src.evaluate --help
python -m src.infer --help
python -m src.inspect_layers --help
```

## Still outstanding

- Test-set metrics for `configs/3class.yaml` haven't been pulled yet — run
  the `src.evaluate` command above with `checkpoints_3class/best_model.pt`.
- `configs/v2.yaml`, `configs/mild.yaml`, `configs/mild_fixed.yaml`,
  `configs/full.yaml`, `configs/v2_accum*.yaml`, `configs/v2_lowmem.yaml`,
  `configs/v2_weighted.yaml` exist in the repo but weren't touched this
  session — treat their `splits_dir`/data references as unverified until
  checked against what's actually on disk.
