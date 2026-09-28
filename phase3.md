# Phase 3 — Real-World Removal Attacks (Diffusion Regen + SynthID-Bypass)

**Status:** This is the phase originally scoped as "Phase 2" in `phase1.md` (diffusion
regeneration attacks + SynthID-Bypass/ComfyUI integration). It got renumbered because
`phase2.md` turned into unplanned hardening work — fixing a format leak and an
attack-intensity label leak that made Phase 1's 1.0 AUROC fake. That fix produced an
honest baseline (0.800 accuracy / 0.888 AUROC, `results/v2/test_metrics.json`) using
only synthetic distortions (JPEG/blur/noise/resize). Phase 3 asks whether that signal
survives contact with attacks nobody hand-tuned for balance.

**Do NOT touch in this phase:** the generic-edit transfer dataset (`src/edits/`,
`experiments/exp3_edit_transfer/`) — that's Phase 4.

---

## 1. Why this phase exists

Every result so far — the fake 1.0 and the honest 0.888 — comes from
`src/attacks/distortion.py`: parameterized JPEG/blur/noise/resize, swept and tuned
specifically to produce a balanced present/removed split (`alpha=0.02`, see
`current_status.txt`, 2026-08-26). That's a legitimate way to get labels, but it's not
evidence the classifier generalizes to how watermarks actually get removed in the
wild — diffusion-based regeneration attacks (img2img at low noise, or full
regeneration) are the real threat model the base paper (arXiv:2604.25491) and this
whole project are ultimately about. `src/attacks/diffusion_regen.py` is currently a
stub. This phase fills it in, integrates a real-world tool (SynthID-Bypass/ComfyUI),
and re-measures.

---

## 2. Step 0 — Close out known gaps from Phase 2 first (cheap, do before scaling up)

- [ ] Class-weighted loss or threshold tuning to close the recall gap on "removed"
      (0.739 recall vs. 0.850 precision at alpha=0.02 — model under-calls removal)
- [ ] Re-run `configs/v2.yaml` training with 2–3 different seeds to check how stable
      0.800/0.888 actually is on 900 images before treating it as the number to beat
- [ ] Explicitly confirm no source-image leakage across `data/splits_v2` (noted as
      "not actually possible by construction" in `phase2.md` but worth a real check
      once diffusion attacks add new output files per raw image)

**Checkpoint:** you have a stable baseline number (with a rough confidence range, not
a single seed) to compare Phase 3 results against.

---

## 3. Step 1 — Implement diffusion regeneration attack

- [ ] Implement `src/attacks/diffusion_regen.py`: img2img regeneration at a range of
      noise/strength levels (e.g. Stable Diffusion img2img, strength 0.1–0.6) applied
      to `data/watermarked_v2/`
- [ ] Randomize strength per-image the same way `build_verified_dataset.py` randomizes
      distortion params — don't hand-pick a single strength, or you'll reintroduce the
      Phase-1-style leak (attack strength correlating with label)
- [ ] Sanity check: confirm `DWT_DCT_SVD.verify()` against the true original actually
      flips to "removed" for at least some fraction of images — if regeneration at
      every tested strength leaves the watermark intact, the attack isn't strong
      enough to be interesting, and if it destroys it 100% of the time it's not
      revealing anything either

**Checkpoint:** a diffusion-regenerated image set with a genuine present/removed split
(not 100/0), produced the same verification-based way as `data/watermarked_v2`.

## 4. Step 2 — Integrate SynthID-Bypass / ComfyUI

- [ ] Stand up SynthID-Bypass (or the ComfyUI workflow it depends on) as an attack
      option alongside the local diffusion_regen implementation
- [ ] Run it on the same `data/watermarked_v2` inputs so results are comparable
      apples-to-apples with Step 1
- [ ] Note any differences in what SynthID-Bypass targets vs. your DWT-DCT-SVD
      watermark — SynthID-Bypass is built against Google's SynthID, not the
      classical watermark used so far, so partial/no effect is itself a real finding,
      not a bug to fix

**Checkpoint:** you can produce an attacked image set via SynthID-Bypass/ComfyUI and
verify present/removed labels the same way as every prior dataset.

## 5. Step 3 — Rebuild dataset, retrain, evaluate

- [ ] Build `data/watermarked_v3`/`data/removed_v3` (or similarly named) mixing
      distortion attacks (v2) with diffusion-regen and SynthID-Bypass attacks, so the
      classifier sees the range of removal methods, not just one
- [ ] Rebuild `data/splits_v3`, checking for source-image leakage explicitly this time
- [ ] Retrain ConvNeXt-Tiny (new `configs/v3.yaml`) from scratch
- [ ] Evaluate overall, **and** broken out per attack type (distortion vs.
      diffusion-regen vs. SynthID-Bypass) — the aggregate number matters less here
      than whether the classifier holds up specifically on the real-world attacks

**Checkpoint:** a results table with accuracy/AUROC/precision/recall per attack type,
not just pooled.

## 6. Step 4 — Compare and document

- [ ] Compare per-attack-type AUROC against the Step 0 baseline (0.888 on distortion
      alone) — does it hold, drop, or is distortion now the easy case and
      diffusion-regen the hard one?
- [ ] Update `README.md` and write an honest summary (same tone as `phase2.md` section
      4, "how good is this, honestly") — call out ceiling effects, small-sample
      caveats, and whether SynthID-Bypass meaningfully attacked your specific
      watermark or not
- [ ] Log everything in `current_status.txt` the way Phase 1/2 sessions were tracked

**Checkpoint:** someone unfamiliar with the project can read the results table and
know, per attack type, whether removal is detectable — with honest caveats, not
another ceiling-effect number.

---

## 7. Definition of done for Phase 3

1. `src/attacks/diffusion_regen.py` is implemented and produces a genuine, randomized
   present/removed split under verification-based labeling (no construction-based
   leak, no fixed-strength leak)
2. SynthID-Bypass/ComfyUI is integrated and run against the same watermarked inputs
3. A classifier is retrained on a mixed-attack-type dataset and evaluated per attack
   type, not just in aggregate
4. Results are documented with the same honesty standard as `phase2.md` — real
   numbers, explicit caveats, no unexplained 1.0s

---

## 8. What comes after Phase 3

- **Phase 4** (originally scoped as "Phase 3" in `phase1.md`): build the generic
  diffusion-edit dataset (`src/edits/generic_edit.py`, no watermark involved) for the
  removal-residue vs. edit-residue transfer test — can the classifier tell "watermark
  was removed" apart from "image was diffusion-edited for an unrelated reason"?

Do not start Phase 4 until Phase 3's per-attack-type results are in — if diffusion
regeneration attacks turn out to leave a very different residue than the simple
distortion pipeline, that changes what "removal-residue" even means for the Phase 4
transfer test.
