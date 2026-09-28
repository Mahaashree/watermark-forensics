# Watermark-Forensics Implementation — Detailed Architecture

## Pipeline Overview

The end-to-end pipeline processes raw images through watermark embedding, attack, dataset generation, classifier training, and evaluation. The complete flow is:

```mermaid
flowchart TD
    RawImages -->|embed| Watermarked[Watermarked Images (class 0)]
    Watermarked -->|attack| Removed[Removed Images (class 1)]
    Watermarked + Removed -->|build_splits| Splits[Train/Val/Test CSVs]
    Splits -->|create_dataloaders| Dataset[PyTorch Dataset + DataLoaders]
    Dataset -->|train/eval| Classifier[ConvNeXt-Tiny]
    Classifier -->|evaluate| Metrics[Acc, AUROC, P/R/F1, CM]
```

---

## Stage 1: Watermark Embedding (`src/watermark/embed.py`)

### Inputs
- Raw images from `data/raw/` (RGB, resized to 224×224)
- Watermark generated via `generate_watermark(shape, seed)` — binary matrix seeded with `seed=42`
- Hyperparameters: `alpha=0.1` (watermark strength), `block_size=8`, `dwt_level=2`, wavelet=`'haar'`

### Process
1. Convert RGB → YUV, extract Y channel
2. Apply 2-level DWT → `LL` subband + 3 detail bands (`details`)
3. Partition `LL` into 8×8 blocks
4. For each block belonging to the watermark grid:
   - Apply DCT: `dct_block = cv2.dct(block)`
   - SVD: `U, S, Vt = np.linalg.svd(dct_block, full_matrices=False)`
   - Perturb singular values: `S[0] *= (1 + alpha * (2 * watermark[i, j] - 1))`
   - Inverse DCT: `dct_mod = U @ np.diag(S) @ Vt`
   - `LL_mod[r:r+8, c:c+8] = cv2.idct(dct_mod)`
5. Reconstruct via inverse DWT: `watermarked_y = _idwt2(LL_mod, details)`
6. Clip to [0, 255], convert back to RGB from YUV

### Outputs
- Watermarked images saved to `data/watermarked/`
- Verification workflow: `embed --verify` runs `extract()` → `verify()` → `(is_present, correlation_score)`

### Key Metrics
- **Watermark survival rate:** fraction of images where `verify()` returns `is_present=True`
- **Correlation score:** Pearson r between extracted and original watermark (range [-1, 1]), threshold 0.5 for present/absent
- Embedding capacity: approximately 49 bits (for 224×224 image, DWT level 2, block size 8)

---

## Stage 2: Distortion Attack (`src/attacks/distortion.py`)

### Inputs
- Watermarked images
- `jpeg_quality`: JPEG compression quality (0-100, lower = stronger compression)
- `gaussian_std`: Standard deviation of Gaussian noise added to pixel values
- `blur_kernel`: Kernel size for Gaussian blur (odd integer; 1 = no blur)
- `resize_factor`: Factor to resize down then up (0.5 = 50% → 100% of original)
- `num_passes`: Number of times to apply the full distortion pipeline

### Process (4 steps, applied `num_passes` times sequentially)
1. **Resize down + upsample:** `small = resize(image, factor, INTER_AREA)`, then `resize(small, original_size, INTER_LINEAR)`
2. **JPEG recompression:** `cv2.imencode('.jpg', image, [IMWRITE_JPEG_QUALITY, quality])` → `imdecode`
3. **Gaussian noise:** `noise = np.random.normal(0, std, image.shape)`; `clip(img.astype(float) + noise, 0, 255).astype(uint8)`
4. **Gaussian blur:** If `blur_kernel > 1`: `cv2.GaussianBlur(img, (kernel, kernel), 0)`

### Outputs
- Attacked images saved to `data/removed/` (label = class 1)
- Both watermarked and removed classes re-encoded to matching JPEG quality (80) via `--save-quality` to eliminate format leak

### Key Metrics
- **Attack intensity:** governed by all 4 params simultaneously; stronger combos more effectively remove watermark
- **Format consistency:** both classes saved as JPEG-80 after attack (fixes the Phase 1 format leak)

---

## Stage 3: Dataset Split Building (`src/data/build_splits.py`)

### Inputs
- `data/watermarked/` (class 0) and `data/removed/` (class 1) directories
- Split ratios: `train=0.70`, `val=0.15`, `test=0.15`
- `seed=42` for reproducible stratified splits

### Process
1. Collect all `.png` files from both directories, sort for deterministic ordering
2. Create DataFrame with columns `path` (string) and `label` (0 for watermarked, 1 for removed)
3. First split: `train_test_split(df, train_size=0.70, stratify=df["label"], random_state=42)` → `train_df`, `temp_df` (30%)
4. Second split: `val_size = val_ratio / (val_ratio + test_ratio) = 0.15 / 0.30 = 0.5`
   - `train_test_split(temp_df, train_size=0.5, stratify=temp_df["label"], random_state=42)` → `val_df`, `test_df`
5. Save CSVs: `train.csv`, `val.csv`, `test.csv` to `data/splits/`

### Outputs
- Three CSV files with columns `path` and `label`
- Printed class breakdowns: e.g., "Train: 900 (450/450), Val: 195 (97/98), Test: 195 (97/98)"

### Key Metrics
- **Stratification:** each split maintains approximately the same class ratio
- **Source-image leakage prevention:** each raw image maps to exactly one output file by construction

---

## Stage 4: PyTorch Dataset & DataLoaders (`src/data/dataset.py`)

### Inputs
- Split CSV files (`train.csv`, `val.csv`, `test.csv`)
- `img_size=224`, `batch_size=32` (falls back to 16 on OOM), `num_workers=4` (0 on CPU-only)
- `train=True` flag enables data augmentation

### Transforms (via `albumentations`)
- **Train:** `RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))` + `HorizontalFlip(p=0.5)` + `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)` + `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` + `ToTensorV2()`
- **Val/Test:** `Resize(224, 224)` + `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` + `ToTensorV2()`

### Outputs
- `DataLoader` yielding `(img, label)` tuples
  - `img`: `torch.Size([3, 224, 224])` float tensor, normalized
  - `label`: `torch.long` scalar (0 = watermarked, 1 = removed)
- Pin memory enabled for faster GPU transfer
- 3 return values: `train_dl, val_dl, test_dl`

### Key Metrics
- Batch size, input shape, dataset sizes per split
- Number of batches per epoch: `ceil(n_samples / batch_size)`

---

## Stage 5: ConvNeXt-Tiny Classifier (`src/models/classifier.py`)

### Architecture
- **Backbone:** `timm.create_model("convnext_tiny", pretrained=True, num_classes=0, global_pool="avg")`
  - ImageNet-pretrained, ~28.6M parameters
  - Outputs feature vector of dim 764 (after global average pooling)
- **Head:** `nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 2))`
  - Dropout for regularization
  - Final linear layer maps 768 → 2 logits (binary: watermarked/removed)

### Forward Pass
1. Input `x` → backbone (feature extraction + avg pooling) → feature vector
2. Feature → dropout → linear head → logits (2 values)
3. Softmax applied for probabilities: `probs = softmax(logits, dim=1)[:, 1]` (probability of class 1 = "removed")

### Key Metrics
- Total parameters: ~30.1M (28.6M backbone + 1.5M head)
- Feature dimension: 768
- Trainable parameters: entire backbone (frozen? no — all fine-tuned) + head

---

## Stage 6: Training Loop (`src/train.py`)

### Configuration (from `configs/default.yaml`)
- Device: `cuda` if available and `cfg["train"]["device"] != "cpu"`, else `cpu`
- Mixed precision: `torch.amp.autocast("cuda")` + `GradScaler("cuda")`, falls back to `torch.cuda.amp` if needed
- optimizer: `AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)`
- scheduler: `CosineAnnealingLR(optimizer, T_max=12)` (T_max = total epochs)
- criterion: `nn.CrossEntropyLoss()`
- Gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)`

### Training Loop (12 epochs max)
1. `train_one_epoch()`: iterate train dataloader, forward/backward with optional AMP
   - Returns `(avg_loss, accuracy)` for the epoch
2. `evaluate()`: no-grad forward on val dataloader
   - Returns `(avg_loss, accuracy, all_probs, all_labels)`
3. `scheduler.step()` after each epoch
4. **Checkpointing:** save `model.state_dict()` to `checkpoints/best_model.pt` when val accuracy improves
5. **Early stopping:** patience=5; if val acc doesn't improve for 5 consecutive epochs, halt training

### OOM Fallback
- If CUDA out-of-memory occurs, training falls back to `batch_size=16` (from `batch_size_fallback` in config)

### Key Metrics per Epoch
- Train loss, train accuracy
- Val loss, val accuracy
- Learning rate (stepped cosine decay)
- Best val accuracy seen, patience counter

---

## Stage 7: Evaluation (`src/evaluate.py`)

### Process
1. Load model from checkpoint: `model.load_state_dict(torch.load(args.checkpoint, map_location=device))`
2. Create dataloaders for specified split (`train`/`val`/`test`)
3. No-grad forward pass, accumulating:
   - `total_loss` (CrossEntropyLoss × n_samples)
   - `correct` predictions: `(logits.argmax(1) == labels).sum().item()`
   - `all_probs`: `torch.softmax(logits, dim=1)[:, 1]` (probability of class 1)
   - `all_labels`: ground truth labels
4. Compute metrics from accumulated arrays:
   - **Accuracy:** `accuracy_score(labels, preds)` where `preds = (probs > 0.5).astype(int)`
   - **AUROC:** `roc_auc_score(labels, probs)` (measures ranking quality)
   - **Precision, Recall, F1:** `precision_recall_fscore_support(labels, preds, average="binary")`
   - **Confusion Matrix:** `confusion_matrix(labels, preds).tolist()` (rows=true, cols=pred, order=[watermarked, removed])

### Outputs
- Printed JSON results to stdout
- Saved to `{save_dir}/{split}_metrics.json` (e.g., `results/test_metrics.json`)

### Key Metrics (output format)
| Metric | Formula / Description |
|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| AUROC | Probability that a random "removed" image has higher prediction score than a random "watermarked" image |
| Precision | Of predicted "removed", % that are actually removed = TP / (TP + FP) |
| Recall | Of truly "removed", % correctly caught = TP / (TP + FN) |
| F1 | Harmonic mean of precision and recall |
| Confusion Matrix | `[[TN, FP], [FN, TP]]` for binary classification |

---

## Configuration Parameters (`configs/default.yaml`)

| Section | Parameter | Value | Description |
|---|---|---|---|
| `data.raw_dir` | `"data/raw"` | Source images directory |
| `data.watermarked_dir` | `"data/watermarked_jpeg80"` | After watermark embedding |
| `data.removed_dir` | `"data/removed_jpeg80"` | After distortion attack |
| `data.splits_dir` | `"data/splits_jpeg80_extreme"` | Split CSV files |
| `data.img_size` | `224` | Input image resize dimension |
| `data.seed` | `42` | Random seed for reproducibility |
| `watermark.alpha` | `0.1` | Watermark embedding strength (critical for survival vs. attack) |
| `watermark.block_size` | `8` | DCT block size for watermark insertion |
| `watermark.dwt_level` | `2` | Number of wavelet decomposition levels |
| `attack.jpeg_quality` | `2` | Extreme compression (Phase 1); changed to 80 in v2 fix |
| `attack.gaussian_std` | `45.0` | Extreme Gaussian noise (Phase 1) |
| `attack.blur_kernel` | `13` | Extreme blur kernel (Phase 1) |
| `attack.num_passes` | `6` | Number of attack iterations (Phase 1) |
| `model.name` | `"convnext_tiny"` | Backbone architecture via timm |
| `model.pretrained` | `true` | Use ImageNet-pretrained weights |
| `model.num_classes` | `2` | Binary classification head |
| `model.dropout` | `0.1` | Dropout regularization rate |
| `train.batch_size` | `16` | Effective batch size (fallback 8 on OOM) |
| `train.lr` | `0.0001` | AdamW learning rate |
| `train.epochs` | `12` | Maximum training epochs |
| `train.mixed_precision` | `true` | Use `torch.amp` for mixed precision |
| `train.grad_clip` | `1.0` | Gradient norm clipping threshold |
| `train.early_stopping_patience` | `5` | epochs with no improvement before stopping |
| `eval.metrics` | `[accuracy, auroc, precision, recall, f1]` | Metrics to compute during evaluation |

---

## Known Issues, Fixes, and Metrics Summary

### Phase 1 (Original) — Bugs Found and Fixed
- **Result:** 100% test accuracy, 1.0 AUROC (fake — ceiling artifact)
- **Bug #1: Format leak** — Watermarked images saved as PNG, removed images as JPEG-derived. The classifier was detecting format/compression artifacts, not watermark survival.
  - **Fix:** Both classes re-encoded to matching JPEG-80 quality via `--save-quality` flag
- **Bug #2: Attack-intensity leak** — Labels assigned by construction (which folder the file was in), not by whether watermark actually survived. Classifier learned "how hard was this attacked" rather than "is the watermark still there?"
  - **Fix:** Verification-based labeling via `scripts/build_verified_dataset.py` — randomized attack strength per image, label = whatever `verify()` says after attack against true original

### Phase 2 (Interim Fixes) — Honest Baseline
- **Dataset:** `alpha=0.02` watermark strength, randomized attack range (jpeg 5-90, std 2-40, blur {1,3,5,7,9}, resize 0.25-1.0, passes 0-5)
- **Class balance:** 441 present / 459 removed (balanced)
- **Splits:** train 630 (309/321), val 135 (66/69), test 135 (66/69) — balanced and stratified
- **Retrained:** ConvNeXt-Tiny, early-stopped epoch 7 (best val acc 0.874 at epoch 2)
- **Test results:** accuracy 0.800, AUROC 0.888, precision 0.850, recall 0.739, F1 0.791
- **Confusion matrix:** `[[57, 9], [18, 51]]` (rows=true watermarked/removed, cols=pred)
  - 57 true watermarked correctly predicted
  - 18 false negatives = missed removals (recall gap)
  - 9 false alarms = watermarked incorrectly predicted as removed

### Current State (run.md)
- `configs/default.yaml` points at `data/splits_jpeg80_extreme` (140/30/30 train/val/test, self-contained on disk)
- Extreme attack params (jpeg=2, noise=45, blur=13, 6 passes) but both classes re-encoded to JPEG-80
- Ready to run: `python -m src.train --config configs/default.yaml` then `python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test`

### Open Issues
- **Recall gap:** model under-calls "removed" class (recall 0.74 vs precision 0.85) — biased toward predicting "still watermarked"
- **Consider:** class-weighted loss or threshold tuning if removed class needs prioritization
- **Multi-seed stability:** ~80%/0.89 AUROC may vary on 900-image dataset; recommend re-running with 2-3 seeds
- **Phase 3 pending:** diffusion regeneration attacks and SynthID-Bypass integration not yet started

### Phase 3 Planned (Not Started)
- Implement `src/attacks/diffusion_regen.py`: img2img regeneration (e.g., Stable Diffusion, strength 0.1-0.6), randomized per image
- Integrate SynthID-Bypass / ComfyUI as real-world attack alongside local diffusion
- Rebuild dataset mixing distortion + diffusion-regen + SynthID-Bypass attacks
- Retrain and evaluate **per attack type**, not just pooled results
- Document with honesty standard: real numbers, explicit caveats, no unexplained 1.0s