# Phase 2 Tools Evaluation — Real Watermark-Removal Attacks

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

## 3. CtrlRegen — Liu et al., arXiv:2410.05470

- **Source:** [github.com/yepengliu/CtrlRegen](https://github.com/yepengliu/CtrlRegen) (official, exists and appears maintained). Search results reported ICLR 2025 acceptance for this work — noted here as unverified against the arXiv listing itself, treat as secondary confirmation only.
- **License:** **Not specified** in the repo — no LICENSE file content surfaced. This alone is a usage risk (default copyright, no grant to reuse) and would need direct confirmation before relying on it.
- **Install:** `git clone` + `pip install -r requirements.txt`, plus two separate model components (Semantic Control Adapter, Spatial Control Network) pulled from `huggingface.co/yepengliu/ctrlregen`. This is a ControlNet-based pipeline stacked on top of a base diffusion model, so the real dependency footprint (and download size) is larger than the two adapter files alone — a full SD-based backbone is implied but not itemized in what I could retrieve.
- **Model size:** Not disclosed for the adapters; base diffusion model dependency size not stated but implied to be a standard SD-scale checkpoint (multi-GB) given the ControlNet architecture.
- **CPU/GPU:** Not explicitly stated; a ControlNet + diffusion backbone stack effectively requires GPU in practice even if not documented as a hard requirement.
- **Single-image use:** Yes — a demo notebook (`CtrlRegen_Plus_Demo.ipynb`) processes individual images with an adjustable strength parameter.
- **Verdict: viable but not cleanest.** Real code exists and single-image use is supported, but unspecified license, undocumented total download size, and a heavier multi-component pipeline (ControlNet + backbone + 2 adapters) make this a worse first integration than SANA-VAE. Good second candidate once the license is confirmed directly with the authors or repo maintainer.

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
| CtrlRegen | yepengliu/CtrlRegen | Unspecified | git clone + pip + 2 HF adapters + backbone | Undisclosed, likely multi-GB | Unclear, likely GPU-bound | Yes (demo notebook) | Viable 2nd choice, license needs confirming |
| WMForger | facebookresearch/videoseal/wmforger | MIT | git clone + pip + wget checkpoint | Undisclosed, likely 10s-100s MB | Unconfirmed | Yes, documented | Available but semantically forging-not-removal — needs closer look |
| UnMarker (released tool) | [andrekassis/ai-watermark](https://github.com/andrekassis/ai-watermark) | Custom, non-commercial | conda + git clone + install.sh + 30GB download | ~30GB (10 bundled scheme detectors) | No (`cu121`-only torch pin, no CPU path) | No (scheme-config only, no `--image` arg) | **Disqualified** |
| Distortion-optimization (`distortion_optimization.py`) | n/a — **our own implementation, UnMarker-inspired**, not the released tool | Project's own (see repo LICENSE) | Already in this repo, no new deps | n/a | Yes (pure NumPy/OpenCV/PyWavelets, no GPU used) | Yes, `--input`/`--output`/`--original` | **Implemented** — greedy zeroth-order search over DWT-LL block DC coefficients, L-infinity bounded |

## Recommendation

Integrate **SANA-VAE (DC-AE via `diffusers.AutoencoderDC`)** first: permissive
license, single `pip install`, sub-GB to ~1GB download, confirmed CPU-capable
(matches this project's no-GPU-guaranteed local environment), and a minimal,
well-documented single-image API that maps directly onto the "VAE
reconstruction" attack paper 2 describes.
