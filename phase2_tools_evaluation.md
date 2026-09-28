# Phase 2 Tools Evaluation — Real Watermark-Removal Attacks

**Scope note:** everything below (DiffPure, SANA-VAE, CtrlRegen, WMForger,
UnMarker) is **Module 2** — attack-family building blocks evaluated for use
in *constructing our own training dataset* (`build_verified_dataset.py`).
None of it is Gap 1 evidence. Gap 1 ("do detectors survive real,
independently-downloaded removal tools?") is tracked separately at the
bottom of this file, under **"Gap 1: Real-Tool Evaluation"** — do not
re-conflate the two sections.

Evaluates the four attacks used in ["The Forensic Cost of Watermark Removal:
From Dedicated Attacks to Image Editing"](https://arxiv.org/abs/2604.25491)
(Evennou & Kijak, submitted Apr 2026) for integration into
`src/attacks/diffusion_regen.py`, replacing the current stub. Per the task,
no weights have been downloaded yet — this is the pre-download review.

The paper itself gives no GitHub links or implementation details for any of
the four attacks (confirmed by reading the full text) — all four had to be
independently traced to their real public sources below.

## 1. DiffPure — Nie et al., ICML 2022

- **Source:** [github.com/NVlabs/DiffPure](https://github.com/NVlabs/DiffPure) (official)
- **License:** NVIDIA Source Code License — **non-commercial / research-and-evaluation only.** Any commercial use requires contacting NVIDIA.
- **Install:** Docker-only as documented (`diffpure.Dockerfile`), CUDA 11.0, Python 3.8. No pip package.
- **Model size:** 3 separate pretrained diffusion checkpoints required (CIFAR-10 Score SDE, ImageNet 256 Guided Diffusion, CelebA-HQ DDPM) plus classifier weights — sizes not published, but Guided Diffusion 256×256 checkpoints alone are typically ~2GB+, so this is a multi-GB, multi-file download.
- **CPU/GPU:** **GPU-required**, README states "1–4 high-end NVIDIA GPUs with 32GB memory." No CPU path.
- **Single-image use:** Not directly supported. The repo is built entirely around benchmark evaluation scripts (`eval_sde_adv.py` running AutoAttack/BPDA+EOT over CIFAR-10/ImageNet/CelebA-HQ test sets) — there's no documented "run on one arbitrary image" entrypoint; one would have to extract the purification step from the eval script by hand.
- **Verdict: disqualified for this integration.** Non-commercial license is a real constraint for research reuse, GPU memory requirement (32GB) exceeds what's assumed elsewhere in this repo (configs here target a single consumer/Colab GPU), Docker-only setup doesn't fit the project's pip-based tooling, and there's no clean single-image API — it would need real engineering to extract one.

## 2. SANA-VAE — Xie et al., ICLR 2025 (Oral)

Paper 2 uses this as a fast/weak VAE-based reconstruction attack. SANA's VAE
component is **DC-AE** (Deep Compression Autoencoder), from a companion paper
(Chen, Cai, et al., "Deep Compression Autoencoder for Efficient
High-Resolution Diffusion Models") and is released standalone.

- **Source:** [huggingface.co/docs/diffusers/api/models/autoencoder_dc](https://huggingface.co/docs/diffusers/en/api/models/autoencoder_dc) — integrated directly into HuggingFace `diffusers` as `AutoencoderDC`. Checkpoint: `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`. Reference implementation: [github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit).
- **License:** **Apache-2.0** (efficientvit repo license; `diffusers` itself is also Apache-2.0). No usage restriction.
- **Install:** `pip install diffusers torch` — no git clone, no Docker, no manual dependency wrangling. Weights auto-download via `from_pretrained` on first use (standard HF cache).
- **Model size:** ~0.3B params — roughly 0.6GB (fp16/bf16) to 1.2GB (fp32) as a single safetensors file.
- **CPU/GPU:** Standard `diffusers.ModelMixin` (plain `nn.Module`) — `.to("cpu")` or `.to("cuda")` both work. Confirmed runnable on CPU (slow, but functional — no CUDA-only ops).
- **Single-image use:** Yes, directly supported and this is exactly the standalone use case the model is designed for:
  ```python
  from diffusers import AutoencoderDC
  ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=torch.float32).to(device)
  latent = ae.encode(x).latent
  y = ae.decode(latent).sample
  ```
  An encode→decode roundtrip through this heavily-compressed (32×) latent space is exactly the "VAE reconstruction" attack mechanism the paper describes — passing an image through it acts as a generative-manifold reprojection that should degrade a fragile watermark signal while leaving the image visually intact.
- **Verdict: cleanest available install path.** Pure pip install, permissive license, small download, CPU-capable, and a documented, minimal single-image API. **Recommended for this integration.**

## 3. CtrlRegen — Liu et al., arXiv:2410.05470 (family: regeneration) — INTEGRATED

Re-evaluated in depth for integration as the second regeneration-family
attack (SANA-VAE is the first). Same disqualification checks run against
DiffPure and UnMarker were applied here first.

- **Source:** [github.com/yepengliu/CtrlRegen](https://github.com/yepengliu/CtrlRegen), pinned commit `6671b4b3a83f0f43e04dc1a08952b27d26197429`.
- **License: absent, confirmed, not merely unspecified.** Checked the GitHub API's license field (`NOASSERTION`/`other`) and the full top-level file listing directly — there is no `LICENSE` file at all in this repository. Used here under default copyright, for internal development only, per explicit instruction. **This is a hard blocker for including CtrlRegen in the paper's reproducibility package or any redistribution** until the authors add explicit terms — it is not resolved by anything done in this integration and should not be treated as such.
- **Hardware/dependencies — no hard disqualifier found:** `requirements.txt` is `diffusers==0.30.2`, `controlnet-aux==0.0.9`, `transformers==4.44.2`, `pillow==10.3.0`, `color-matcher` — no `tensorflow`, no `numpy` pin, no CUDA-only wheel index (unlike UnMarker's `cu121`-locked torch). The code itself does `device = 'cuda:0' if torch.cuda.is_available() else 'cpu'` — a real, working fallback path exists, not just a flag that's ignored.
- **Real bug found in the upstream demo, fixed in our wrapper:** the notebook hardcodes `torch_dtype=torch.float16` unconditionally, regardless of `device`. PyTorch's CPU backend doesn't support fp16 for most conv/attention ops, so running their own demo as-published on CPU would error or silently mismatch dtypes. Our wrapper (`src/attacks/regeneration_ctrlregen.py`) selects `torch.float16` only under CUDA and `torch.float32` otherwise, applied uniformly to every component.
- **Single-image path: genuinely exists, confirmed by tracing the actual code** (not assumed from the README). `ctrlregen_plus_demo.ipynb`'s `ctrl_regen_plus(input_img: PIL.Image, step, seed) -> PIL.Image` is a real function operating on one arbitrary image passed in directly — no dataset registry, no scheme config file, unlike UnMarker's `attack.py`.
- **Scheme-agnostic, confirmed:** it's an image-to-image ControlNet-guided diffusion regeneration (Canny-edge spatial control + DINOv2 semantic control, denoise from a partially-noised latent back through Stable Diffusion). It operates purely on pixels — nothing in the pipeline references or needs a watermark detector object, unlike UnMarker's `init_watermarker(evaluator)` coupling.
- **Cost that IS real, flagged rather than hidden:** required downloads (base SD1.5-family checkpoint + `facebook/dinov2-giant` image encoder + CtrlRegen's own control-net/IP-adapter weights + a VAE) total roughly 9-10GB via `from_pretrained` (repo API reports higher totals — 30GB/9GB/1.6GB/0.7GB — but that includes redundant format variants `from_pretrained` doesn't all fetch). This is much larger than SANA-VAE's ~1.2GB. It is a genuine multi-step diffusion pipeline (default 50 UNet forward passes plus a giant image encoder), so CPU runtime for one image is expected to be far slower than SANA-VAE's single encode/decode pass -- see the demo result below for the actual measured number.
- **Environment-fragility check:** same `transformers` TF/JAX import-time probe hit during SANA-VAE work recurred here (triggered merely by importing `diffusers.ControlNetModel`). No new hard-required TF/numpy dependency was introduced by CtrlRegen itself — this is the same optional, lazily-probed backend issue as before. Fixed with the same `USE_TF=0`/`USE_FLAX=0` workaround already in `diffusion_regen.py`, now also in `regeneration_ctrlregen.py`. No global version changes made.
- **Verdict: integrated**, given no hard technical disqualifier (real CPU path, no forced non-commercial/CUDA-only terms, scheme-agnostic, genuine single-image API) — but **flagged as blocked from the paper's reproducibility section pending license clarification from the authors**, and flagged as meaningfully more expensive (download size, runtime) than SANA-VAE.

**Demo result** (`data/raw/0000.png`, 1740x2040, `step=0.5`, `seed=0`, CPU-only):
- Weight download: ~9.8GB confirmed (`yepengliu/ctrlregen` 1.5GB + `Realistic_Vision_V4.0_noVAE` 4.0GB + `dinov2-giant` 4.2GB + `sd-vae-ft-mse` ~8KB delta once cached), one-time.
- Pipeline load: 26.6s (from local HF cache). **Regeneration: 1419.4s (~23.7 min)** for one image, 50 UNet steps at 512x512 on CPU — squarely in the 10-40 min range flagged before starting, not hours.
- Shape preserved: 1740x2040x3 -> 1740x2040x3.
- **L-infinity diff: 240, mean abs diff: 22.9, SSIM: 0.526, PSNR: 18.2dB** -- a large, real change (full diffusion regeneration, not a bounded perturbation like `distortion_optimization.py`), confirming genuine regeneration rather than a passthrough. SSIM/PSNR are markedly lower than SANA-VAE's single encode/decode pass (0.988 SSIM / 29.8dB), consistent with CtrlRegen being a much more aggressive, full-reconstruction attack rather than a lightweight reprojection.
- Two real bugs hit and fixed along the way (both my own integration code, not environment/version issues): (1) `controlnet_aux`'s package `__init__.py` eagerly imports an unrelated `MediapipeFaceDetector`, which pulled in a `mediapipe`->`tensorflow` chain hitting the same numpy-2 ABI issue as before -- turned out to be already guarded by the library's own `try/except ImportError` and degrades gracefully (confirmed by the "mediapipe not installed" warning appearing before continuing), so no code change was actually needed here once understood -- an initial "fix" (importing `controlnet_aux.canny` directly) was based on a wrong assumption (that this skips the package `__init__.py`; it does not) and didn't change anything. (2) A real bug: the pipeline was constructed with `controlnet` as a single-element list (matching the upstream notebook), which puts the pipeline in "multiple controlnets" mode requiring `image`/`control_image`/`ip_adapter_image` to also be passed as lists -- fixed by matching the notebook's exact call convention (including `guidance_scale=2.0`, `controlnet_conditioning_scale=1.0`, `control_guidance_start/end`, and the `color_match` post-step, which the first draft had omitted).

**Known limitation: consistency quality is poor at both tested `step` values, not just the default.** The repo/notebook state no recommended default for `step` (only `strength between 0 and 1`, illustrated with `step=0.5`) -- one follow-up run at `step=0.3` (lower removal strength, expected to trade some removal power for better consistency) gave **SSIM 0.556 / PSNR 19.5dB**, barely different from `step=0.5`'s 0.526 / 18.2dB. Per instruction, not tuning further tonight -- flagging as a known limitation rather than a resolved trade-off: on this one test image, CtrlRegen's output diverges substantially from the input at both settings tried, well under the ~0.85 SSIM bar that would indicate a genuinely tunable quality/removal trade-off. This may be specific to this image, to the `Realistic_Vision_V4.0_noVAE` base model pairing, or to running at CPU-forced fp32 rather than the authors' fp16/GPU setup -- unconfirmed, would need a GPU run or more images to disentangle.

## 4. WMForger — Souček et al., NeurIPS 2025 (Spotlight)

Full title: *"Transferable Black-Box One-Shot Forging of Watermarks via Image
Preference Models."* This one needs an explicit flag per the task: **it is
not fundamentally a removal technique.** I confirmed this by reading paper 2's
own description of it (Souček et al. 2025, "training a preference model via
reinforcement learning to distinguish watermarked from clean images, using
Fourier-domain transforms") and cross-checked it against the real paper's
title/abstract — this is a **watermark forging/transfer attack** (its
headline use case is making unwatermarked content falsely appear watermarked,
or transferring a stolen watermark onto new content), which paper 2 itself
also classifies as "not strictly removal" despite including it in its four
"dedicated attacks." The same preference-model machinery can plausibly be run
in the opposite direction (optimize an image to score as *unwatermarked*
rather than *watermarked*), which is presumably how paper 2 used it as a
removal-adjacent baseline — but this is not the tool's documented primary
purpose, and I have not confirmed a documented "removal mode" flag in the
released code.

- **Source:** [github.com/facebookresearch/videoseal](https://github.com/facebookresearch/videoseal), code in the `wmforger/` subdirectory (official, real, actively maintained by Meta FAIR — good sign vs. the "likely not public" expectation in the task brief).
- **License:** **MIT** (whole `videoseal` repo). No usage restriction.
- **Install:** `git clone` + repo-root `requirements.txt` (PyTorch 2.4.0, torchvision 0.19.0, Python 3.10), then `wget https://dl.fbaipublicfiles.com/wmforger/convnext_pref_model.pth` for the pretrained preference model.
- **Model size:** Checkpoint size not documented in what I could retrieve; a ConvNeXt-based preference model is typically in the tens-to-low-hundreds of MB range, not multi-GB — but this is an estimate, not a confirmed number.
- **CPU/GPU:** Not stated for inference/optimization specifically (only that training-from-scratch uses 8 GPUs); the `optimize_image.py` single-image path is likely far lighter than training and plausibly CPU-feasible, but unconfirmed.
- **Single-image use:** Yes, directly documented:
  ```bash
  python optimize_image.py --ckpt_path convnext_pref_model.pth --image assets/tahiti_watermarked.png
  ```
- **Verdict: available and cleanly documented, but semantically off-target for a "removal attack" integration until its removal-mode behavior is confirmed by reading `optimize_image.py`'s actual objective function.** Not disqualified — genuinely public, permissively licensed, real one-shot CLI — but flagged as needing a closer look before being wired in as a removal attack, rather than treated as equivalent to DiffPure/SANA-VAE/CtrlRegen.

## 5. UnMarker — Kassis & Hengartner, IEEE S&P 2025 (family: distortion-optimization)

Cited as the primary case study in base paper 1 (arXiv:2605.09203, "Removing
the Watermark Is Not Enough: Forensic Stealth in Generative-AI Watermark
Removal"), not paper 2. Evaluated on request as a fifth candidate attack for
`src/attacks/distortion_unmarker.py`.

- **Source:** [github.com/andrekassis/ai-watermark](https://github.com/andrekassis/ai-watermark) (official, author-maintained). Paper: arXiv:2405.08363, DOI 10.1109/SP61157.2025.00005.
- **License:** Custom "Source Code License for UnMarker" — same shape as DiffPure's: perpetual/worldwide copyright grant, but Section 3.3 restricts the Work (and any derivative works) to **non-commercial use only** ("for research or evaluation purposes only"), with a carve-out that only the licensor/its affiliates may use it commercially. Fine for this research project, but another real licensing constraint.
- **Install:** `conda create`, `git clone`, then `./install.sh` + `./download_data_and_models.sh` — the latter pulls **~30GB** of pretrained models and datasets (this is not one small checkpoint; it's the full set of detector models for every watermarking scheme the repo benchmarks against: Yu1/Yu2, PTW, HiDDeN, TreeRing, StegaStamp, StableSignature, PRC, Gs, Vine, SynthID).
- **Hardware:** README states a "high-end NVIDIA GPU with >=32GB memory" and a CUDA 12 driver. `attack.py` does expose a `--device` flag (defaulting to `"cuda"`), but `requirements.txt` pins `torch==2.2.2`/`torchvision==0.17.2` against the **`cu121` (CUDA-only) wheel index** — not a build that installs or runs meaningfully on CPU or MPS.
- **Dependency footprint:** `requirements.txt` pins ~30 exact package versions, including `numpy==1.24.1` and `tensorflow==2.9.0` as a hard (non-optional) dependency — not an optional backend probed lazily like `transformers` in the SANA-VAE case. This numpy pin is directly incompatible with the numpy 2.2.5 already installed in this environment (the same ABI conflict hit during SANA-VAE integration), and here there's no `USE_TF=0`-style workaround available, because `tensorflow` is a required top-level import for this codebase, not an optional lazy-probed backend. Installing this requirements.txt as specified would require downgrading numpy globally, which risks breaking every other package in this shared environment (as seen with the diffusers/huggingface-hub conflict already hit once this session) — exactly the kind of global-version change the task asked me not to make.
- **Single-image use: not supported.** `attack.py`'s CLI is `--output_dir`, `--attack {UnMarker,DiffusionAttack,VAEAttack,Crop,JPEG,...}`, `--evaluator {Yu1,Yu2,PTW,HiDDeN,TreeRing,StegaStamp,StableSignature,Prc,Gs,Vine,SynthID}`, `--total_imgs` (default 100) — there is no `--image`/arbitrary-input-path argument. `input_dir` is read from a per-scheme YAML config (`attack_configs/<evaluator>.yaml`), not passed by the caller.
- **Root architectural blocker, not just a CLI inconvenience:** I traced `attack.py` → `modules/attack/base_attack.py` → `BaseAttack.__init__`, which does `self.evaluator = init_watermarker(evaluator, ...)`. The `UnMark` attack (`modules/attack/unmark/unmark.py`) is built directly on top of this — it optimizes an adversarial perturbation **against a specific, already-instantiated watermark detector object** for one of the ten bundled schemes above. Despite the paper's "Universal Attack" title (referring to generalizing across those ten published schemes), the released code is not a scheme-agnostic/black-box perturbation tool that runs on an arbitrary image with no detector in the loop. Using it against *our* project's custom DWT-DCT-SVD watermark would require writing a new `init_watermarker`-compatible adapter class exposing our embed/verify logic through their interface — genuine integration engineering, not CLI wiring, and arguably drifts toward "reimplementing on top of their internals" rather than "running their released tool," which cuts against the point of this integration.
- **Verdict: disqualified, same category as DiffPure.** GPU-only (`cu121`-pinned torch wheel, no working CPU path), ~30GB mandatory download, non-commercial-only license, no single-arbitrary-image CLI, and a hard dependency (`tensorflow==2.9.0`) that would force a global numpy downgrade in this shared environment to even install — with the added blocker that the released attack is architecturally coupled to one of ten specific pre-registered watermarking schemes' detector models, not runnable against an arbitrary/unknown watermark without writing a real adapter.

**Outcome:** since the released tool itself is unusable here, the
distortion-optimization *attack family* was implemented from scratch instead
of integrated — see `src/attacks/distortion_optimization.py`. This is our
own code, not a port of UnMarker's, run directly against our own verifier
(`src.watermark.embed.DWT_DCT_SVD`); UnMarker is cited there only as the
algorithmic inspiration for targeting spectral-amplitude carriers rather
than raw pixels. Full mechanism, why a first SPSA-based attempt failed, and
demo results are in that file's docstring and the commit/PR notes for this
change.

## Summary table

| Tool / family | Source | License | Install | Size | CPU? | Single-image CLI? | Verdict |
|---|---|---|---|---|---|---|---|
| DiffPure | NVlabs/DiffPure | NVIDIA non-commercial | Docker, CUDA 11.0 | multi-GB (3 checkpoints) | No (32GB GPU) | No (benchmark-only) | Disqualified |
| SANA-VAE (DC-AE) | HF `diffusers` / mit-han-lab/efficientvit | Apache-2.0 | `pip install diffusers` | ~0.6–1.2GB | Yes | Yes, documented | **Recommended / integrated** |
| CtrlRegen (regeneration) | yepengliu/CtrlRegen, pinned `6671b4b` | **Absent (confirmed, no LICENSE file)** -- blocks reproducibility-package use | vendored (no pip pkg) + `controlnet-aux`/`color-matcher` pinned | ~9.8GB confirmed | Yes (~24 min/image on CPU) | Yes, `ctrl_regen_plus(img, step, seed)` | **Integrated** |
| WMForger | facebookresearch/videoseal/wmforger | MIT | git clone + pip + wget checkpoint | Undisclosed, likely 10s-100s MB | Unconfirmed | Yes, documented | Available but semantically forging-not-removal — needs closer look |
| UnMarker (released tool) | [andrekassis/ai-watermark](https://github.com/andrekassis/ai-watermark) | Custom, non-commercial | conda + git clone + install.sh + 30GB download | ~30GB (10 bundled scheme detectors) | No (`cu121`-only torch pin, no CPU path) | No (scheme-config only, no `--image` arg) | **Disqualified** |
| Distortion-optimization (`distortion_optimization.py`) | n/a — **our own implementation, UnMarker-inspired**, not the released tool | Project's own (see repo LICENSE) | Already in this repo, no new deps | n/a | Yes (pure NumPy/OpenCV/PyWavelets, no GPU used) | Yes, `--input`/`--output`/`--original` | **Implemented** — greedy zeroth-order search over DWT-LL block DC coefficients, L-infinity bounded |

## Recommendation

Integrate **SANA-VAE (DC-AE via `diffusers.AutoencoderDC`)** first: permissive
license, single `pip install`, sub-GB to ~1GB download, confirmed CPU-capable
(matches this project's no-GPU-guaranteed local environment), and a minimal,
well-documented single-image API that maps directly onto the "VAE
reconstruction" attack paper 2 describes.

---

# Gap 1: Real-Tool Evaluation

**This section, and only this section, is Gap 1 evidence** ("do detectors
trained on our own attack-family building blocks still work against real
tools that real users download to strip watermarks?"). Everything above
this line is Module 2 (attack families used to build our training data) --
distinct question, not to be conflated with what follows.

## remove-ai-watermarks (github.com/wiltodelta/remove-ai-watermarks)

Independently verified to exist before cloning (not assumed from prior
context): 5,624 stars at check time, Apache-2.0, real PyPI package
(`remove-ai-watermarks`).

**Setup attempted, ~2 minutes into a 60-minute budget -- stopped early on a
clean, code-verified double-disqualification, not a timeout.**

The tool has two distinct paths for invisible-watermark removal, per its own
README:

1. **`invisible` command (diffusion regeneration via a Qwen z-image
   pipeline)** -- this is the path that would generically apply to our
   custom DWT-DCT-SVD watermark regardless of scheme, the one the task
   asked to test. Traced to source (`invisible_engine.py`,
   `_internal/qwen_zimage_pipeline.py`): the engine calls
   `self._require_cuda()` directly, and the docstring states outright:
   > "ALL ARE CUDA-ONLY -- there is no CPU or MPS path for
   > invisible-watermark removal."

   This machine has no CUDA (confirmed repeatedly throughout this project --
   MPS and CPU only). **Same disqualifying pattern as DiffPure and
   UnMarker**: a real, hard, code-level GPU requirement with zero fallback,
   not a soft "slower on CPU" case.

2. **"Direct local-format disruption" (`pixels` extra,
   `microsoft_invismark.py`)** -- the README's other invisible-removal path.
   Traced to source: this is scheme-specific to Microsoft's proprietary
   **InvisMark** declaration (Paint/Photos), not a generic pixel-domain
   attack. It would not recognize or meaningfully attack our custom
   watermark at all -- **the same scheme-coupling problem that disqualified
   UnMarker** (built to strip one specific known signature, not to
   generically disrupt an arbitrary invisible watermark).

**Verdict: disqualified, both available invisible-removal paths, for two
different reasons that happen to mirror the project's two prior
disqualification patterns exactly (DiffPure/UnMarker's GPU-only
requirement, and UnMarker's scheme-coupling).** No dataset was built, no
images were run through it, no classifier score exists for this tool.

**Per instruction: not silently substituting another Module-2 attack and
calling this done.** Reporting back to pivot to Synthid-Bypass
(github.com/cebeuq/Synthid-Bypass, ComfyUI-based) as the next real-tool
candidate -- not yet attempted.

**Gap 1 status: still zero real-tool evidence.** Nothing in this section
should be read as a completed Gap 1 result; it documents a disqualification,
not a measurement. TPR@1%FPR / TPR@0.1%FPR support was not added to
`src/evaluate.py` this pass since there is no real-tool-attacked subset yet
to score with it -- deferred until a viable tool is found.
