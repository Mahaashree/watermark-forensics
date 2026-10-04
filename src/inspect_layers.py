"""
Layer-by-layer forward pass inspection for a single input image: shows the
output shape and activation stats (mean/std/min/max) at each stage of the
network, so you can verify the classifier is actually processing the image
correctly through every layer, not just check the final verdict.
CLI: python -m src.inspect_layers --config configs/default.yaml --checkpoint checkpoints/best_model.pt --image path/to/image.png
"""

import argparse
import torch
import yaml
from pathlib import Path

from src.models.classifier import WatermarkClassifier
from src.train import resolve_device
from src.infer import load_image, CLASS_NAMES


def stats(t: torch.Tensor) -> str:
    return (f"shape={tuple(t.shape)}  mean={t.mean().item():.4f}  "
            f"std={t.std().item():.4f}  min={t.min().item():.4f}  max={t.max().item():.4f}")


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

    layers = {
        "stem": model.backbone.stem,
        "stage_0": model.backbone.stages[0],
        "stage_1": model.backbone.stages[1],
        "stage_2": model.backbone.stages[2],
        "stage_3": model.backbone.stages[3],
        "norm_pre": model.backbone.norm_pre,
        "backbone_pool_head": model.backbone.head,
        "classifier_dropout": model.head[0],
        "classifier_linear": model.head[1],
    }

    outputs = {}

    def make_hook(name):
        def hook(module, inp, out):
            outputs[name] = out.detach()
        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in layers.items()]

    img = load_image(args.image, cfg["data"]["img_size"]).to(device)

    with torch.no_grad():
        logits = model(img)

    for h in handles:
        h.remove()

    print(f"Image: {args.image}")
    print(f"Input tensor: {stats(img)}\n")

    for name in layers:
        print(f"{name:22s} {stats(outputs[name])}")

    probs = torch.softmax(logits, dim=1)[0]
    class_names = CLASS_NAMES[:cfg["model"]["num_classes"]]
    print(f"\nFinal logits: {logits[0].tolist()}")
    for name, p in zip(class_names, probs.tolist()):
        print(f"P({name})={p:.4f}")
    print(f"Verdict: {class_names[int(probs.argmax().item())]}")


if __name__ == "__main__":
    main()
