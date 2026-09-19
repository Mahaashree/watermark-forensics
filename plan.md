# Task 1 — Non-Watermarked Degradation Control: Implementation Plan

Ref: `phase0-closeout-handoff.md`, Task 1 (highest priority, blocks everything else).

## What we're actually testing

Current baseline: AUROC=0.8875 on `configs/v2.yaml` (alpha=0.02,
`data/watermarked_v2` vs `data/removed_v2`, `checkpoints_v2/best_model.pt`,
`data/splits_v2/test.csv`, 135 test images: 66 label-0 / 69 label-1).

Open question: is the classifier keying on "watermark-removal residue" (what
we want) or on "this image went through recompression/noise/blur/rescale"
(a confound — every `removed_v2` image was attacked, `watermarked_v2` images
were attacked too but many surrendered the watermark less)? We can't tell
from the existing two classes alone because both were built from a
watermarked source. We need a third group that was **never watermarked**
but got the **same degradation**.

## Design decision: RNG-replay for exact per-image matching

The handoff only requires "same param ranges." I'm proposing something
stronger and roughly free: reproduce the *exact same per-image attack
parameters* used for each `removed_v2`/`watermarked_v2` image, applied to
the un-watermarked raw source instead. That turns this into a true paired
control (only variable that changes is "was a watermark ever embedded"),
instead of a merely range-matched one.

This works because `scripts/build_verified_dataset.py` is fully
deterministic given `--seed 42`:

1. `random.seed(42)` and `np.random.seed(42)` are set once, up front.
2. `generate_watermark(wm_shape, seed=42)` immediately calls
   `np.random.seed(42)` again and draws one `np.random.rand(...)` — this
   fixes numpy's RNG state right before the main loop starts.
3. The loop walks `sorted(raw.glob("*.png"))` — all 900 images, in order.
   For every image (regardless of what happens to it later) it draws, via
   Python's `random` module: `num_passes`, `jpeg_quality`, `gaussian_std`,
   `blur_kernel`, `resize_factor` — always in that order.
4. `apply_distortion` then draws `np.random.normal(...)` for Gaussian noise,
   once per pass — via numpy's RNG.
5. `method.embed()` (DWT/DCT/SVD) consumes **no** randomness — it's a pure
   function of the image array and alpha.

Since steps 2–4 are the only sources of randomness and neither depends on
whether embedding happened, a second script that performs the identical
setup (same seed, same `generate_watermark` call just to consume the same
numpy draw, same sorted iteration, same per-image draw order) but skips
`method.embed()` and attacks the raw image directly will reproduce
byte-identical `(num_passes, jpeg_quality, gaussian_std, blur_kernel,
resize_factor)` per filename — the Gaussian noise itself will differ (numpy
state has advanced differently since the two runs are separate processes
each starting fresh from seed 42, and no embed-time draws happen either
way, so this holds), but the *severity knobs* match exactly. That's the
part that matters for ruling out "degradation intensity" as the shortcut.

**Correctness trap to avoid:** you must iterate over all 900 raw filenames
in the full sorted order and perform every draw for every image, even for
the ~831 images you won't bother saving. Filtering the glob down to just
the 135 test-set filenames *before* the loop desyncs the RNG stream from
`build_verified_dataset.py`'s and breaks the pairing. Filter only at the
"do I write this file to disk" step, not the iteration step.

## Step 1 — `scripts/build_degradation_control.py` (new file)

Copy the structure of `scripts/build_verified_dataset.py` almost verbatim:

- Args: `--raw` (default `data/raw`), `--out` (default `data/control_v2`),
  `--img-size 224`, `--alpha 0.02` / `--block-size 8` / `--dwt-level 2`
  (only needed to compute `wm_shape` so the `generate_watermark` call
  consumes numpy's RNG identically — the watermark array itself is
  discarded), `--seed 42`, `--save-quality 80`.
- Same `random.seed` / `np.random.seed` / `generate_watermark(...)` calls
  up front, in the same order, with the same seed.
- Loop over `sorted(args.raw.glob("*.png"))` (all 900):
  - Resize raw BGR to `img_size` with `INTER_AREA` (matches
    `build_verified_dataset.py` exactly — needed so pixel-level starting
    point matches what watermarking started from).
  - Draw `num_passes`, `jpeg_quality`, `gaussian_std`, `blur_kernel`,
    `resize_factor` in that exact order (same `random.choice/randint/uniform`
    calls) — **do this unconditionally for every image**, whether or not
    you end up saving output for it.
  - `attacked = raw_bgr` (skip `method.embed()` entirely — no watermark).
  - Apply `apply_distortion` from `src/attacks/distortion.py` `num_passes`
    times with those params (this consumes the matching numpy noise draws).
  - `jpeg_reencode(attacked, save_quality=80)`.
  - Save to `args.out / img_path.name` — always, for all 900 (simplest;
    building the full set costs one script run, and gives you slack to
    inspect train/val counterparts too if you want to sanity-check later).
- No `verify()` call, no label split — one flat output folder,
  `data/control_v2/`, 900 files.

## Step 2 — build the eval subset

- Read `data/splits_v2/test.csv`.
- Keep rows where `label == 1` (the true `removed_v2` test images — 69 rows,
  confirmed by inspecting the split).
- For each, take `Path(path).name` and look up the same filename in
  `data/control_v2/`.
- Build a new CSV, e.g. `data/splits_v2/test_task1_control.csv`, with:
  - 69 rows: `path=data/removed_v2/<name>.png, label=1`
  - 69 rows: `path=data/control_v2/<name>.png, label=0`
  - (138 rows total, balanced.)

## Step 3 — score with the frozen model (no retraining)

New script `scripts/eval_task1_control.py`:

- Load `configs/v2.yaml` for `model` config only.
- Build a `WatermarkForensicsDataset` directly from
  `test_task1_control.csv` with `train=False` (same `Resize` +
  `Normalize` eval-time transform used for the reported 0.8875 result —
  no augmentation, so this is apples-to-apples with how the baseline was
  scored).
- Load `WatermarkClassifier.from_config(cfg["model"])`, load weights from
  `checkpoints_v2/best_model.pt`, `eval()`.
- For every image: `probs = softmax(logits, dim=1)[:, 1]` — same "P(removed)"
  score used everywhere else in this repo.
- `auroc = roc_auc_score(labels, probs)` where label 1 = true `removed_v2`,
  label 0 = `control_v2`.
- Also dump the two score distributions (mean/std, or a quick histogram via
  `results/figures/task1_control_hist.png` using the existing
  `scripts/make_figures.py` pattern) — a scalar AUROC can hide the "model
  actively confuses control-degraded images for removed" failure mode if
  you don't look at direction, so check whether control-image scores are
  systematically *higher* than true-removed scores (that would show up as
  AUROC < 0.5 rather than ≈ 0.5, but the histogram makes the finding legible
  in the writeup either way).
- Save `results/task1_control_auroc.json` with the same field shape as
  `results/v2/test_metrics.json` (split name, auroc, n_samples, plus which
  checkpoint/config was used) so it's directly comparable.

## Step 4 — interpretation (per handoff's stated thresholds)

- **AUROC high** (comparable to or above 0.8875) → classifier separates true
  watermark-removal residue from generic degradation → Phase 0 baseline is
  plausible, proceed to Task 2.
- **AUROC ≈ 0.5** (chance) → classifier cannot tell watermark-removed images
  from never-watermarked-but-equally-degraded images → the 0.8875 number is
  confounded by generic degradation, not removal-specific residue → **stop,
  flag it, do not proceed to Task 2–5 or Gap 1/2** until the pipeline is
  reworked (per handoff's explicit instruction).
- **AUROC < 0.5** (inverted) → worse than confounded: the model is scoring
  degraded-but-never-watermarked images as *more* "removed" than genuinely
  watermark-removed ones. Report this distinctly from the ≈0.5 case since it
  points at a different bug (e.g. attack-strength distribution mismatch
  between the two classes) rather than plain non-informativeness.

## Step 5 — deliverable

`results/task1_writeup.md`: the AUROC number, the pass/fail verdict against
the three cases above, the score-distribution figure, and 2–3 sentences of
conclusion — matching the "AUROC + short writeup of which case we're in"
deliverable the handoff asks for.

## Files touched / created

- New: `scripts/build_degradation_control.py`
- New: `data/control_v2/` (900 images, generated artifact — consider
  gitignoring like the other `data/*` dirs)
- New: `data/splits_v2/test_task1_control.csv`
- New: `scripts/eval_task1_control.py`
- New: `results/task1_control_auroc.json`, `results/figures/task1_control_hist.png`
- New: `results/task1_writeup.md`
- Untouched: `checkpoints_v2/best_model.pt`, `configs/v2.yaml`, the original
  `data/watermarked_v2` / `data/removed_v2` / `data/splits_v2` splits — this
  task only *scores* the existing frozen model, it does not retrain it.
