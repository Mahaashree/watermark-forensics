# Workplan: Journal Publication Readiness

_Assessment date: 2026-10-03_

## Current Status

**Not yet journal-ready, but closer than most final-year projects.**

### Strengths
- Genuine methodological discipline: two separate label-leakage bugs were caught and fixed (a format leak, then an attack-intensity leak) rather than papered over.
- Honest, mediocre numbers were reported instead of inflated ones — the kind of rigor reviewers respect.
- Paper draft is already structured like a real paper: abstract, explicit research questions, ablations, confusion matrices, limitations, future work.

### Core Problems
1. **Central research question unanswered.** The abstract promises real-world generalization testing (does the model detect watermark removal from actual tools people use?). All four candidates were disqualified or stalled — originally on compute grounds (CPU/MPS-only), and as of 2026-10-04 this project has CUDA (RTX 3050) but a renewed attempt stalled mid-download, not on a hard compute block anymore (paper draft Section 6.4). Gap 1 still delivers zero evidence either way.
2. ~~**Inconsistent headline numbers across documents.**~~ **Updated 2026-10-04.** The headline itself changed during this pass: the original checkpoint behind 0.67 acc/0.66 AUROC no longer exists and could not be reproduced on this (CUDA) machine — it was trained on MPS, and the dataset-build step turns out not to be reproducible across hardware backends (paper draft Section 5.3.1). A 3-seed CUDA retrain gives accuracy 0.7289 ± 0.034, AUROC 0.7632 ± 0.012 — real, stable, and *better* than the old figure. README/ARCHITECTURE.md/paper draft are now consistent on this new number.
3. ~~**No baseline comparison.**~~ **Fixed 2026-10-04** — see Section 5.8 of the paper draft: a trained 2-layer CNN collapses to majority-class prediction (AUROC 0.44, below chance), and non-learned Laplacian-variance / JPEG-blockiness thresholds only reach AUROC 0.52-0.59, vs. the ConvNeXt-Tiny headline's 0.6615. Run via `experiments/exp1_baseline_repro/baseline_check.py --splits-dir data/splits_v2`.
4. ~~**No related-work section.**~~ **Fixed 2026-10-04** — see Section 1.1 of the paper draft.

---

## Must-Fix Before Submitting

- [~] Get at least one real-world watermark-removal tool running (e.g. a few hours on free-tier Colab GPU) to produce partial Gap 1 evidence — a partial result beats "we tried and got nothing." **In progress (2026-10-03):** this machine actually has CUDA (RTX 3050, 6GB — earlier disqualifications in `phase2_tools_evaluation.md` were written on a CPU/MPS-only machine and don't apply here). `remove-ai-watermarks[qwen-zimage]` installed in `.venv-gap1` and got past model loading before being stopped mid-download (Qwen-Image-2512 transformer shards are ~5GB each, 9 shards). Not yet completed — resume by rerunning `remove-ai-watermarks invisible <img> -o <out> --force --cpu-offload` with `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` set (fixes a Windows console charmap crash hit on the first attempt).
- [x] Resolve the 0.67/0.66 vs 0.80/0.89 discrepancy across README/ARCHITECTURE.md/paper draft; pick and justify one honest headline number. **Resolved (2026-10-03):** not a true discrepancy — they measure different tasks (0.6733/0.6615 = 2-class diversified-attack headline; 0.800/0.888 = 2-class single-attack reference baseline, both correctly scoped in the paper draft already). Fixed `ARCHITECTURE.md`, which was stale and only showed the 0.80/0.89 number under a "Phase 3 Planned (Not Started)" heading despite Phase 3 being done. Also found and documented a third, previously-undocumented checkpoint (`checkpoints_3class`) — its pooled macro accuracy/AUROC (0.817/0.914) is inflated by an easy "Original" class; disaggregated present-vs-removed-only metrics (0.7467/0.7929, computed via `scripts/eval_3class_present_removed_only.py`) are now in paper draft Section 5.7.
- [x] Fix `build_splits.py`'s lack of group-awareness before generating any more data. **Done (prior to this assessment):** `src/data/build_splits.py` already uses `StratifiedGroupKFold` keyed on filename-stem source-ID (see module docstring + `_assert_no_group_leakage`), and `leak_check_report.md` confirms zero group leakage on `data/splits_v2`. This checkbox was stale — no action needed, verified by reading the current source.
- [x] Add a related-work section with properly formatted citations. **Done (2026-10-04):** added as Section 1.1 in `research_paper_draft.txt`, covering watermark embedding (Lai & Tsai 2010; Fernandez et al. ICCV 2023; Wen et al. NeurIPS 2023), removal attacks (Zhao et al. NeurIPS 2024; An et al. WAVES, ICML 2024), and forensic generalization (Wang et al. CVPR 2020; Corvi et al. ICASSP 2023; Ojha et al. CVPR 2023) — all citations verified against primary sources, not recalled from memory. REFERENCES section updated to match.
- [x] Run multiple seeds and report confidence intervals. **Done 2026-10-04** — 3 seeds (42/1/2) of `configs/v2_accum_long.yaml`: accuracy 0.7289 ± 0.034, AUROC 0.7632 ± 0.012 (paper draft Section 5.3.1). Doing this surfaced a bigger issue, resolved in the same pass: the original headline checkpoint no longer existed on disk and couldn't be reproduced because it was trained on MPS, not CUDA, and the dataset-build step (`scripts/build_verified_dataset.py`) turns out not to be reproducible across hardware backends despite its `--seed` flag — see Section 5.3.1 for the full account. The new 3-seed number is real, stable, and *higher* than the old single-run figure, so this is a correction, not a regression. n is still only 50 negatives (down from 52) on the current data/splits_v2 — expanding the test set (below) remains relevant.

## Would Strengthen the Paper (Not Blocking)

- [x] Add a simple baseline model for comparison. **Done 2026-10-04** (see Must-Fix list above).
- [ ] Expand the test set size.
- [ ] Investigate the unexplained declining-AUROC trend found in the ablations (Section 5.5) instead of leaving it unexplained.
- [ ] Quantify weak-embedding label noise across the full dataset, not just the one anecdotal case currently cited.

---

## Bottom Line (updated 2026-10-04)

Substantially stronger than the 2026-10-03 assessment: related work, a
real baseline comparison, and multi-seed confidence intervals are now in
the paper, and the headline number is consistent across documents and
supported by 3 seeds rather than 1. What's left is almost entirely Gap 1
(real-tool evidence) and the two re-runs flagged in Section 8 (5.2/5.4 on
the current CUDA dataset). **Closing Gap 1 — resuming the stalled
remove-ai-watermarks download — remains the single highest-leverage fix,
and is now a bandwidth/time problem, not a hard compute blocker.**
