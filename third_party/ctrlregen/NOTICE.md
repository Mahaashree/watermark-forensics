# Vendored from CtrlRegen (unmodified)

Source: https://github.com/yepengliu/CtrlRegen
Pinned commit: 6671b4b3a83f0f43e04dc1a08952b27d26197429 (main, fetched 2026-09-20)
Paper: Liu et al., "Image Watermarks Are Removable Using Controllable
Regeneration from Clean Noise," arXiv:2410.05470.

Files here (`custom_i2i_pipeline.py`, `custom_ip_adapter.py`, `utils.py`) are
copied byte-for-byte from that commit, unmodified. They are not
pip-installable, so vendoring is the repo's own intended usage pattern
(clone + import from the same directory), not a packaging choice made here.

**License status: no LICENSE file exists in the source repository as of the
pinned commit** (confirmed via the GitHub API and a full top-level file
listing, not just an unclear/missing SPDX tag) -- code is used here under
default copyright for internal development purposes only, per explicit
project instruction. This blocks CtrlRegen from any reproducibility-package
or redistribution use until the authors clarify licensing terms; see
phase2_tools_evaluation.md for the full note. Do not redistribute this
directory outside internal development use.

Our own integration wrapper (`src/attacks/regeneration_ctrlregen.py`) does
not modify these files; it imports them and fixes the upstream demo
notebook's unconditional-float16 dtype choice (which breaks/mismatches on
CPU) at the call site instead.
