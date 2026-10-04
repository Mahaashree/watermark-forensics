# How Image Classification Works in This Project

This explains, end to end, how an image goes from a file on disk to a
Present / Removed / Original verdict.

## 1. What the classifier actually is

`src/models/classifier.py` wraps a **ConvNeXt-Tiny** (pretrained on ImageNet,
~27.8M params) with a small custom head:

```
Image (224x224x3)
   -> ConvNeXt-Tiny backbone (stem + 4 stages, 18 blocks total)
   -> Global average pool -> 768-dim feature vector
   -> Dropout(0.1)
   -> Linear(768 -> num_classes)
   -> logits -> softmax -> probabilities
```

It is a **blind classifier**: at inference time it only ever sees the target
image. It never sees "the original" to compare against. It has learned, from
training examples, statistical artifacts that tend to correlate with each
class — it is making a probabilistic guess, not a certainty check.

This is different from `src/watermark/embed.py`'s `DWT_DCT_SVD.verify()`,
which is a deterministic check but **requires the original un-watermarked
image** to compare against. You can only use `verify()` if you already have
that original; the classifier exists for the case where you don't.

## 2. The three classes it's trained to tell apart

| Label | Name | How it's built |
|---|---|---|
| 0 | **Present** | Raw image with a watermark embedded via `DWT_DCT_SVD.embed()` (DWT level 2, 8x8 blocks, DCT+SVD singular-value perturbation), verified present after a mild JPEG-80 re-encode |
| 1 | **Removed** | Raw image, watermark embedded, then attacked (JPEG re-encode + Gaussian noise + blur, randomized strength), verified **absent** afterward |
| 2 | **Original** | Raw image, **never watermarked at all** — but still resized to 224x224 and JPEG-80 re-encoded, to match the file format of the other two classes exactly |

Class 2 is deliberately put through the same resize + JPEG-80 formatting as
classes 0/1 (`scripts/build_original_class.py`). If it weren't, the
classifier could cheat by learning "is this a resized/recompressed image"
instead of "does this have a watermark" — the file-format difference alone
would separate class 2 from the others without the model ever looking at
watermark structure. This mirrors a leakage bug already found and fixed
earlier in this repo for classes 0/1 (see `leak_check_report.md`).

Splits are built with `src/data/build_splits.py`, which groups by source
image ID (`StratifiedGroupKFold`) so the same raw photo's Present/Removed/
Original derivatives never end up split across train/val/test — that would
leak near-duplicate content across the boundary.

## 3. What happens to your image at inference time

`src/infer.py` -> `load_image()`:

1. **Read**: `cv2.imread()` (loads as BGR)
2. **Color convert**: BGR -> RGB
3. **Resize**: to 224x224 (whatever the input resolution/aspect ratio was)
4. **Normalize**: ImageNet mean/std (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`)
5. **To tensor**: HWC numpy -> CHW torch tensor, batch dim added -> `(1,3,224,224)`

That tensor is the *only* thing that reaches the model. The original file on
disk is never modified — nothing is embedded, attacked, or saved anywhere
unless you explicitly run the embed/attack scripts yourself.

## 4. Forward pass -> verdict

```
logits = model(image_tensor)          # shape (1, num_classes)
probs  = softmax(logits, dim=1)[0]    # e.g. [P(Present), P(Removed), P(Original)]
verdict = argmax(probs)
```

`src/infer.py` prints the verdict, the winning probability as "confidence",
and the full probability breakdown across all classes.

## 5. Running it

```powershell
# Single-image verdict
python -m src.infer --config configs/3class.yaml --checkpoint checkpoints_3class/best_model.pt --image path\to\image.png

# See what every layer of the network outputs for that same image
python -m src.inspect_layers --config configs/3class.yaml --checkpoint checkpoints_3class/best_model.pt --image path\to\image.png

# Full metrics on the held-out test set (accuracy, macro-AUROC, confusion matrix, etc.)
python -m src.evaluate --config configs/3class.yaml --checkpoint checkpoints_3class/best_model.pt --split test
```

(For the older 2-class Present/Removed-only model, swap in
`configs/default.yaml` and `checkpoints/best_model.pt`.)

## 6. What this can and can't tell you

- **Can**: give you a calibrated-ish probability that a natural photograph
  resembles training examples of watermarked / attacked / untouched images,
  measured by held-out test accuracy and AUROC (see `results/` after running
  `evaluate.py`).
- **Can't**: give you a certain answer. It's a learned approximation, wrong
  some fraction of the time (whatever the measured test error is).
- **Can't**: say anything meaningful about images completely unlike its
  training distribution (e.g. screenshots, UI diagrams, synthetic graphics)
  — those are out-of-distribution inputs and the output is close to a
  coin flip dressed up as a percentage.
- **Can't** replace `verify()` when you actually have the original image to
  compare against — that route gives a much stronger, non-probabilistic
  answer and should be preferred whenever the original is available.
