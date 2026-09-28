# Source-Image Leakage Audit — data/splits_v2 (and other split variants)

**Scope:** diagnostic only. No files modified, no retraining performed.

## Method

- `build_splits.py` writes no explicit source-ID column — only `path` and `label`.
  The source identifier has to be recovered from the filename stem (`{stem}.png`,
  e.g. `0235.png`), which is preserved unchanged from `data/raw/` through every
  downstream stage (`embed.py`, `distortion.py`, `build_verified_dataset.py` all
  do `cv2.imwrite(out_dir / img_path.name, ...)`). Actual image bytes for
  `watermarked_v2` / `removed_v2` are gitignored and not present in this working
  tree (only 101 leftover files in `data/raw`, not the full ~900), so the
  perceptual-hash fallback (step 4) wasn't needed — the filename stem is a
  reliable, code-verified source ID, not a guess.
- Extracted stems from `data/splits_v2/{train,val,test}.csv`, computed pairwise
  set intersections.

## Result for `data/splits_v2`: **PASS — no leakage**

| Pair | Overlap |
|---|---|
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |

630 train / 135 val / 135 test stems, all globally unique (630+135+135 = 900,
matching the full `watermarked_v2` + `removed_v2` count from `current_status.txt`).

**Why it's clean, structurally, not by luck:** `build_verified_dataset.py`
(the v2 dataset builder) routes each raw source image to **exactly one** of
`watermarked_v2` or `removed_v2` based on the post-attack `verify()` outcome
(lines ~65-80: `out_dir = args.out_present if present else args.out_removed`).
Every source contributes exactly one derivative total — never a clean copy in
one class and a removed copy in the other. Since `build_splits.py` then does a
plain file-level stratified split over this set, and each source appears in
the input exactly once, group leakage of the kind described in the task
(source X's clean copy in train, X's removed copy in test) is **not possible**
for this dataset layout, regardless of `build_splits.py`'s lack of grouping
logic. So the current headline result (v2: 0.80 acc / 0.89 AUROC) is **not
affected** by this failure mode and does not need to be invalidated.

## Important caveat: `build_splits.py` itself has no group-awareness — and other split sets ARE leaked

I also checked the legacy split sets, which use the *original* pipeline layout
(`embed.py` watermarks **every** raw image into `data/watermarked/`, then
`distortion.py` attacks **every** one of those into `data/removed/` — i.e. every
source contributes to *both* classes, unlike v2's mutually-exclusive routing).
`build_splits.py` globs both dirs and does a plain `train_test_split` with no
source grouping, so this layout leaks by construction:

| Split set | train∩val | train∩test | val∩test |
|---|---|---|---|
| `data/splits` (v1) | 23 | 27 | 1 |
| `data/splits_mild` | 23 | 27 | 1 |
| `data/splits_full` | 183 | 189 | 47 |
| `data/splits_jpeg80` | 23 | 27 | 1 |
| `data/splits_jpeg80_extreme` | 23 | 27 | 1 |

Sample overlapping stems (splits_full): `0400, 0039, 0626, 0615, 0828, 0098, 0771, 0215`.

These numbers don't change any current conclusion — the v1/mild/full/jpeg80
results were already known-invalid due to the separately-diagnosed format/
intensity label leak documented in `current_status.txt`, and are not currently
cited as real numbers. But this is a **second, independent leakage mechanism**
in the same legacy split sets, worth knowing if anyone resurrects those runs.

## Forward-looking risk

`build_splits.py` has no source-ID column and no grouping logic at all. It
happened to produce a clean split for v2 only because `build_verified_dataset.py`
made source-to-class assignment mutually exclusive. **Phase 2 (diffusion
regeneration attacks) is planned to follow the original embed-then-attack
pattern** per `phase2.md`/README (watermark all sources, then attack all of
them) — the same layout that leaks in the table above. If `build_splits.py` is
reused as-is for Phase 2 data, it will reproduce this leak.

**Recommended fix (not applied here):** add an explicit source-ID field
(filename stem, written explicitly rather than inferred) to the manifest, and
replace `train_test_split` in `build_splits.py` with
`sklearn.model_selection.GroupShuffleSplit` (or `StratifiedGroupKFold`) keyed
on that source ID, so no source can appear in more than one split regardless
of how many derivative classes it contributes to. This should be done before
Phase 2 dataset building starts, not patched after the fact by filtering an
existing manifest.

## Verdict summary

- **`data/splits_v2` (current v2 results, 0.80 acc / 0.89 AUROC): PASS, no source
  leakage.** No action needed on the current headline number.
- **`build_splits.py` is not source-leakage-safe in general** and will leak on
  any future dataset where a source contributes to both classes (the layout
  Phase 2 is planned to use). Fix before Phase 2, not after.
