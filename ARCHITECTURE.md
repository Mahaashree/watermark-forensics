# Architecture — Watermark-Removal Forensics

This document describes the system architecture: components, responsibilities, data
flow, and the design decisions behind them. For project history/results, see
`WALKTHROUGH.md`. For run commands, see `run.md`.

---

## 1. Architectural style

This is a **linear, file-mediated ML pipeline**, not a service. There is no
long-running process, no API, no database (see prior discussion — everything is
flat files: PNG/JPEG images, CSV splits, YAML configs, JSON metrics, `.pt`
checkpoints). Each stage is an independent CLI script that reads inputs from disk
and writes outputs to disk; stages compose by directory convention, not by
in-process calls. This is deliberate for a research pipeline: every intermediate
artifact is inspectable, re-runnable in isolation, and diffable between experiment
versions (`v1` vs `v2` directories) without re-running earlier stages.

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────┐   ┌────────────┐
│  Data        │→ │  Watermark    │→ │  Attack        │→ │  Labeling +     │→ │  Model    │→ │  Evaluation │
│  Acquisition │  │  Embedding    │  │  (removal      │  │  Splitting      │  │  Training │  │  + Reporting│
│              │  │               │  │  attempt)      │  │                 │  │           │  │             │
└─────────────┘   └──────────────┘   └───────────────┘   └────────────────┘   └──────────┘   └────────────┘
   scripts/          src/watermark/     src/attacks/        scripts/build_       src/train.py   src/evaluate.py
   download_div2k     embed.py          distortion.py       verified_dataset.py                 scripts/
                                         diffusion_regen.py  src/data/                            make_figures.py
                                         (stub)              build_splits.py
                                                              src/data/dataset.py
```

---

## 2. Components

### 2.1 Data acquisition — `scripts/download_div2k.py`
Pulls the DIV2K image set from a public mirror into `data/raw/`. This is the only
external data dependency; everything downstream is derived from these images.
Boundary: this is the system's only network I/O.

### 2.2 Watermark layer — `src/watermark/embed.py`
Defines `WatermarkMethod`, an abstract base class with `embed()`, `extract()`,
`verify()` — the seam that lets the watermarking algorithm be swapped without
touching any other layer. The only concrete implementation is `DWT_DCT_SVD`:

- **embed**: converts to YUV, takes the luma channel, 2-level Haar DWT, splits the
  LL sub-band into 8×8 blocks, DCT’s each block, perturbs the block’s dominant SVD
  singular value up or down by `alpha` depending on the corresponding watermark
  bit, inverse-DCT, inverse-DWT, reassembles YUV → RGB.
- **extract**: same transform on both the (possibly attacked) image and the true
  original; compares dominant singular values block-by-block to recover a bit per
  block.
- **verify**: Pearson correlation between the extracted bit-grid and the known
  embedded watermark, thresholded (default 0.5) → boolean "watermark present."

This module is also invoked as a library function (not just a CLI) by
`scripts/build_verified_dataset.py` — `extract()`/`verify()` are the ground-truth
oracle the whole labeling scheme depends on.

### 2.3 Attack layer — `src/attacks/`
- `distortion.py` (implemented): a composable distortion — resize-down/up → JPEG
  recompress → Gaussian noise → Gaussian blur — applicable in `num_passes`
  repetitions. Pure image-in/image-out function (`apply_distortion`), independent
  of the watermark layer, so it can be pointed at any image regardless of whether
  it's watermarked. This independence is what makes verification-based labeling
  possible: the attack doesn't know or care what the "correct" label is.
- `diffusion_regen.py` (stub, Phase 3): reserved seam for an img2img
  regeneration attack. Same expected contract as `distortion.py` — takes an
  image, returns an attacked image — so it's a drop-in alternative attack source
  once implemented.

### 2.4 Edit layer — `src/edits/generic_edit.py` (stub, Phase 4)
Reserved for the future transfer-test dataset: diffusion edits applied to
images that were **never watermarked**, to test whether the classifier's
"removal residue" detector is actually watermark-specific or just "this image
was diffusion-touched." Deliberately outside the watermark/attack/label loop —
it's a separate dataset branch, not a pipeline stage.

### 2.5 Labeling + dataset layer
- `scripts/build_verified_dataset.py` — the architecturally important piece. For
  each raw image it: embeds → applies a **randomized-strength** attack → re-encodes
  to a fixed JPEG quality → runs `extract()`/`verify()` against the true original →
  writes the result to `data/watermarked_v2/` (label "present") or
  `data/removed_v2/` (label "removed") **based on the verification outcome, not on
  which code path produced the file**. This is what closes the two label leaks
  described in `WALKTHROUGH.md` (format leak, attack-intensity leak) — labels are
  now a *measurement*, not a *construction*.
- `src/data/build_splits.py` — stratified 70/15/15 train/val/test split over
  whatever present/removed directories it's pointed at, seeded for
  reproducibility, written as CSV (`path,label`).
- `src/data/dataset.py` — `WatermarkForensicsDataset` (PyTorch `Dataset`) reads a
  split CSV, loads images via OpenCV, applies Albumentations transforms
  (augmentation on train, resize/normalize-only on val/test), returns
  `(tensor, label)`. `create_dataloaders()` wraps all three splits.

Splits reference file paths, not raw pixel data — this is why old split CSVs
"go stale" when their source image directories are deleted or regenerated (see
`run.md`); the split is a pointer layer, not a data copy.

### 2.6 Model layer — `src/models/classifier.py`
`WatermarkClassifier`: a `timm`-provided `convnext_tiny` backbone (pretrained on
ImageNet, pooled to a feature vector) + a small custom head (`Dropout` →
`Linear(feat_dim, 2)`). `ModelConfig` (a dataclass) and `from_config()` decouple
the model definition from YAML — any backbone name `timm` recognizes can be
swapped in via config alone, no code change.

### 2.7 Training / evaluation layer — `src/train.py`, `src/evaluate.py`
- `train.py`: standard supervised loop — AdamW + cosine LR schedule, optional
  AMP mixed precision on CUDA, gradient clipping, early stopping on val accuracy,
  checkpoints the best model to `<save_dir>/best_model.pt`, appends structured
  log lines to `<log_dir>/train.log`.
- `evaluate.py`: loads a checkpoint, runs inference over one split, computes
  accuracy/AUROC/precision/recall/F1/confusion matrix via scikit-learn, dumps to
  `<eval.save_dir>/<split>_metrics.json`.

Both are driven entirely by a single YAML config (`configs/*.yaml`) — one file
fully specifies data paths, watermark params (for documentation/reproducibility,
not consumed at train time), model architecture, and hyperparameters. This is why
multiple parallel configs (`default.yaml`, `v2.yaml`, `mild.yaml`, ...) exist
side by side instead of one file being edited in place: each config is a
self-contained record of one experiment.

### 2.8 Reporting layer — `scripts/make_figures.py` (added for the showcase)
Reads only already-produced artifacts (`results/*.json`, `results/logs*/train.log`)
and renders matplotlib figures (training curves, confusion matrices, metrics
comparison, alpha-sweep). Strictly downstream and read-only — it cannot affect
any earlier stage, and re-running it is always safe/idempotent.

### 2.9 Cross-cutting utilities — `src/utils/`
- `seed.py`: single `set_seed()` used by every stage that has randomness
  (splitting, training, watermark generation) — the reproducibility backbone.
- `logging.py`: `setup_logging()` — timestamps + appends to a shared log file per
  run (note: append-only, so `results/logs*/train.log` accumulates multiple
  historical runs in one file — a known quirk `make_figures.py` has to parse
  around, see §4).

---

## 3. Data flow by phase

**Phase 1 (baseline, construction-labeled):**
`data/raw` → `embed.py` → `data/watermarked*` (label 0 by folder)
`data/watermarked*` → `distortion.py` → `data/removed*` (label 1 by folder)
→ `build_splits.py` → `data/splits*` → `train.py`/`evaluate.py` → `results/*.json`

**Phase 2/interim (verification-labeled, current honest baseline):**
`data/raw` → `build_verified_dataset.py` (embed + randomized attack + verify
internally) → `data/watermarked_v2` / `data/removed_v2` (label = verify() result)
→ `build_splits.py` → `data/splits_v2` → `train.py --config configs/v2.yaml` →
`evaluate.py` → `results/v2/test_metrics.json`

**Phase 3 (planned):** same shape as Phase 2, but `diffusion_regen.py` and a
SynthID-Bypass/ComfyUI integration become additional attack sources feeding the
same verification-based labeling step, producing `data/*_v3` and per-attack-type
evaluation.

**Phase 4 (planned):** a parallel, unlabeled-by-watermark branch off `data/raw`
through `generic_edit.py` into `data/edited/`, used only for the transfer test —
does not feed the main train/eval loop.

---

## 4. Key design decisions

| Decision | Why |
|---|---|
| **Verification-based labels, not folder-based** | The core architectural fix of the project. Any label scheme where "which pipeline branch produced this file" *is* the label lets the classifier key on incidental artifacts of that branch (format, attack intensity) instead of the actual phenomenon. Routing every image through the same watermark oracle (`verify()`) after a randomized attack removes that shortcut. |
| **Attack functions decoupled from the watermark/label logic** | `apply_distortion()` takes an image, returns an image — no awareness of watermarks or labels. This is what makes it reusable both for the old construction-based pipeline and the new verification-based one, and what will let `diffusion_regen.py` slot in identically. |
| **One YAML config per experiment, never edited in place for a new experiment** | Each config is a durable, diffable record of exactly what produced a given `results/*.json`. `run.md`'s "which split does this config point to" table exists precisely because configs *were* edited in place early on and it caused confusion — new experiments now get new config files instead. |
| **Versioned data directories (`_v2`, `_jpeg80`, ...) instead of overwriting** | Lets old results stay reproducible/comparable after the labeling method changes, at the cost of disk space (mitigated by gitignoring the large ones). |
| **Abstract `WatermarkMethod` base class with one concrete implementation** | Only `DWT_DCT_SVD` exists today, but the seam is real: `embed()`/`extract()`/`verify()` is the exact contract `build_verified_dataset.py` and `embed.py`'s CLI both code against, so a second watermarking method could be added without touching either caller. |
| **No experiment tracking system (MLflow/W&B) — logs are append-only text files** | Proportionate to project scale (single researcher, single machine), but it's the reason `results/logs/train.log` mixes multiple historical runs together and why `make_figures.py` has to split the log into run-blocks by `"Device:"` lines rather than reading one clean run per file. This is the most likely place to accrue tech debt if the number of experiments grows further. |
| **CPU-first code with CUDA auto-detection (`train.device: "auto"`)** | The project runs on both a CPU-only local machine and a CUDA-equipped machine (Colab T4 or a local GPU) without config changes — `torch.device("cuda" if torch.cuda.is_available() ...)` — at the cost of local runs being slow enough that `run.md` explicitly tells the reader to expect it. |

---

## 5. Runtime / deployment view

- **No servers, no containers, no orchestration** — every stage is
  `python -m src.<module> --config configs/x.yaml`, run manually or scripted, on
  either a local machine (`.venv`, CPU or local CUDA) or Google Colab (T4 GPU).
- **State lives entirely in the filesystem** of whichever machine ran the stage;
  moving between machines (e.g. Colab → local) means moving `data/`,
  `checkpoints/`, and `results/` by hand — there's no shared storage layer.
  `run.md`'s troubleshooting notes (stale splits pointing at empty directories)
  are a direct symptom of this.
- **Version control boundary**: git tracks code, configs, small metrics JSON, and
  a couple of already-committed log files; it deliberately excludes bulk image
  data, checkpoints, and generated figures (`.gitignore`) — those are
  regeneratable from the tracked code + a config, which is the intended recovery
  path if a machine is lost.

---

## 6. Known limitations of the current architecture

- **Single point of truth for "ground truth" is the watermark method's own
  `verify()`.** If `DWT_DCT_SVD.extract()`/`verify()` has a bug, every downstream
  label is silently wrong in a way nothing else in the pipeline would catch —
  there's no independent check.
- **No per-attack-type provenance carried into the split CSVs.** `path,label` is
  the entire schema; once Phase 3 mixes distortion + diffusion-regen +
  SynthID-Bypass into one dataset, per-attack-type evaluation (a stated Phase 3
  goal) will require either extending this schema or reconstructing attack type
  from the file path — worth deciding explicitly before Phase 3 starts.
- **Log files as the only training-run record** means historical runs are only
  recoverable by parsing timestamps/config dumps out of text, as `make_figures.py`
  already has to do — this will get harder, not easier, as more configs/runs
  accumulate.
