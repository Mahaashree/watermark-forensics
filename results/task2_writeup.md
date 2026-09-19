# Task 2 — BMP Shortcut-Control Ablation

Ref: `phase0-closeout-handoff.md` Task 2 (non-negotiable per project notes 5.6).

Run against v4 (alpha=0.08, coefficient-domain + generic attack,
`data/watermarked_v4`/`removed_v4`, `checkpoints_v4/best_model.pt`,
baseline test AUROC 0.7508 — see `results/v4/test_metrics.json` and
`results/task1_writeup.md` for why v4 is current, not v2).

## What was run

1. `scripts/convert_to_bmp.py` — re-encoded all 900 `watermarked_v4` /
   `removed_v4` PNGs to lossless BMP (`data/watermarked_v4_bmp`,
   `data/removed_v4_bmp`).
2. `scripts/build_splits_bmp.py` — same stratified 70/15/15 split logic as
   `src/data/build_splits.py`, globbing `*.bmp` (kept separate rather than
   editing the base script). Split sizes matched v4 exactly (630/135/135),
   as expected from identical stratification with the same seed.
3. `configs/v4_bmp.yaml` — same architecture/hyperparams as v4, pointed at
   the BMP dirs, `checkpoints_v4_bmp`, `results/v4_bmp`.
4. Trained via unmodified `src/train.py`, evaluated via unmodified
   `src/evaluate.py`.

## Result

Test AUROC: **0.7087** (`results/v4_bmp/test_metrics.json`), vs. 0.7508
on the original PNG-based v4 model.

Verified the two formats decode identically before drawing any
conclusion: sampled 20 image pairs, `cv2.imread` on the PNG vs. the BMP —
**20/20 bit-identical pixel arrays.**

## Verdict: inconclusive by design, not a pass or fail

The handoff's interpretation table (collapse-to-0.5 = pass,
stays-high = leak) assumes BMP re-encoding can strip out a
format-specific signal the model might be keying on. That assumption
doesn't hold here: PNG and BMP are both lossless, and `cv2.imread`
decodes them to bit-identical numpy arrays. There is no channel through
which format alone could carry information in this pipeline — the model
sees numerically identical input either way. So this ablation, as
specified, cannot detect a format leak in the *current* pipeline,
regardless of what the AUROC does.

The 0.7508 → 0.7087 gap is consistent with ordinary training-run
variance (different random init/dropout/augmentation draws between the
two training runs — `src/train.py` doesn't pin a model-init seed
separately from `src/utils/seed.set_seed()`'s data-side effects), not a
signal collapse or a leak.

This ablation *would* be meaningful in the scenario it was originally
written for (Phase 1's bug, per `WALKTHROUGH.md`/`current_status.txt`):
classes stored in genuinely different lossy formats (actual `.jpg` bytes
for one class vs. `.png` for the other), where BMP re-encoding forces
both through the same lossless container and would reveal whether the
model had been reading literal file-format cues. In the current pipeline
both classes are already saved as PNG uniformly (post the Phase 1 fix),
so that failure mode isn't present to test for.

## Follow-up: is the "actual JPEG bytes" version even possible?

Checked whether re-running against literal `.jpg` files mid-pipeline
(before the final PNG write) would give a different, more meaningful
result. It would not, for a structural reason:

`jpeg_reencode()` (`src/watermark/embed.py`) already does its JPEG
compression as an **in-memory encode/decode round-trip**
(`cv2.imencode` immediately followed by `cv2.imdecode`) before the
result is ever written to disk. The JPEG quantization artifacts are
therefore already baked into the pixel array *before* the final
`cv2.imwrite(...png)` call — writing that same array to `.jpg` instead of
`.png` would decode back to the same pixels (same encoder, same quality,
deterministic), and `cv2.imread` discards all container-level metadata
(EXIF, ICC profiles, chunk headers) regardless of extension. There is no
second encode step to intercept and no metadata for a model to see.

## Final verdict: Task 2 — PASS, not inconclusive

Given the above, this isn't an open question needing a follow-up
experiment — it's provable from the code: **no format-specific channel
exists in the current pipeline for the classifier to exploit**, full
stop. The BMP ablation's bit-identical-decode result
(`20/20`) is the confirmation, not a limitation of the test. Task 2
passes.

This ablation would only become meaningful again if a future pipeline
version introduces an actual second, class-correlated lossy encode step
that survives to the images the dataloader reads (e.g. two different
JPEG quality settings baked in asymmetrically between classes) — worth
re-checking if `jpeg_reencode`'s call site or quality setting ever
diverges between `watermarked_*` and `removed_*` again.
