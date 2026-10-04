"""
Single-image inference: classify one arbitrary image as watermark
present/removed using a trained checkpoint.
CLI: python -m src.infer --config configs/default.yaml --checkpoint checkpoints/best_model.pt --image path/to/image.png
"""

import argparse
import cv2
import numpy as np
import torch
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from src.models.classifier import WatermarkClassifier
from src.train import resolve_device

# label 0 = watermark present, 1 = watermark removed, 2 = never watermarked
CLASS_NAMES = ["Present", "Removed", "Original"]


def extract_layer_feature(model: WatermarkClassifier, img: torch.Tensor, layer_name: str) -> np.ndarray:
    """Run a forward pass and return a fixed-size feature vector from the
    named layer. Conv feature maps (B,C,H,W) are global-average-pooled over
    space to a (C,) vector; the already-pooled backbone head is used as-is."""
    layers = {
        "stem": model.backbone.stem,
        "stage_0": model.backbone.stages[0],
        "stage_1": model.backbone.stages[1],
        "stage_2": model.backbone.stages[2],
        "stage_3": model.backbone.stages[3],
        "norm_pre": model.backbone.norm_pre,
        "pool_head": model.backbone.head,
    }
    module = layers[layer_name]
    captured = {}

    def hook(_module, _inp, out):
        captured["out"] = out.detach()

    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        model(img)
    handle.remove()

    out = captured["out"]
    if out.dim() == 4:
        out = out.mean(dim=(2, 3))
    return out[0].cpu().numpy()


def load_ood_stats(checkpoint_path: Path):
    stats_path = checkpoint_path.parent / "ood_stats.npz"
    if not stats_path.exists():
        return None
    data = np.load(stats_path)
    return {
        "centroid_labels": data["centroid_labels"],
        "centroids": data["centroids"],
        "threshold": float(data["threshold"]),
        "percentile": float(data["percentile"]),
        "layer": str(data["layer"]),
    }


def cosine_dist_to_centroids(feat: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    feat_n = feat / (np.linalg.norm(feat) + 1e-8)
    cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    return 1 - cent_n @ feat_n


def load_image(path: Path, img_size: int) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform(image=img)["image"].unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["train"]["device"])

    model = WatermarkClassifier.from_config(cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    img = load_image(args.image, cfg["data"]["img_size"]).to(device)
    n_classes = cfg["model"]["num_classes"]
    class_names = CLASS_NAMES[:n_classes]

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    verdict = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    print(f"Image: {args.image}")

    is_ood = False
    ood_stats = load_ood_stats(args.checkpoint)
    if ood_stats is not None:
        feat = extract_layer_feature(model, img, ood_stats["layer"])
        dists = cosine_dist_to_centroids(feat, ood_stats["centroids"])
        min_dist = float(dists.min())
        is_ood = min_dist > ood_stats["threshold"]
        if is_ood:
            print(f"WARNING: out-of-distribution input (distance={min_dist:.4f} > "
                  f"threshold={ood_stats['threshold']:.4f}, the {ood_stats['percentile']:.0f}th "
                  f"percentile of real training examples). This image doesn't resemble "
                  f"anything the model trained on - the verdict below is unreliable.")

    print(f"Verdict: {verdict}{'  [UNRELIABLE - see warning above]' if is_ood else ''}")
    print(f"Confidence: {confidence:.4f}")
    for name, p in zip(class_names, probs.tolist()):
        print(f"  P({name}) = {p:.4f}")


if __name__ == "__main__":
    main()
