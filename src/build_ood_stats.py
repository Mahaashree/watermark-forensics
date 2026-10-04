"""
Precompute per-class feature centroids + an in-distribution distance
threshold from the training set, for out-of-distribution (OOD) flagging at
inference time. Softmax confidence alone doesn't work for this: a network can
be highly confident and still completely wrong on an input unlike anything it
trained on (as observed empirically here). Distance in the backbone's own
768-dim feature space is a better signal of "have I actually seen anything
like this before."
CLI: python -m src.build_ood_stats --config configs/3class.yaml --checkpoint checkpoints_3class/best_model.pt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from src.models.classifier import WatermarkClassifier
from src.train import resolve_device
from src.infer import load_image, extract_layer_feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=99.0,
                         help="Percentile of in-distribution distances used as the OOD threshold")
    parser.add_argument("--layer", type=str, default="pool_head",
                         choices=["stem", "stage_0", "stage_1", "stage_2", "stage_3", "norm_pre", "pool_head"],
                         help="Which layer's (globally-pooled) activations to use for OOD distance")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["train"]["device"])
    model = WatermarkClassifier.from_config(cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    train_csv = Path(cfg["data"]["splits_dir"]) / "train.csv"
    df = pd.read_csv(train_csv)

    feats_by_label = {}
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting train features"):
            img = load_image(Path(row["path"]), cfg["data"]["img_size"]).to(device)
            feat = extract_layer_feature(model, img, args.layer)
            feats_by_label.setdefault(int(row["label"]), []).append(feat)
            all_feats.append(feat)
            all_labels.append(int(row["label"]))

    all_feats = np.stack(all_feats)
    all_labels = np.array(all_labels)

    def cosine_dist(a, b):
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return 1 - a @ b

    centroids = {}
    for label, feats in feats_by_label.items():
        centroids[label] = np.mean(np.stack(feats), axis=0)

    # For each train sample, distance to its OWN class centroid -> in-distribution
    # distance profile. Threshold = the requested percentile of these distances.
    own_class_dists = np.array([
        cosine_dist(all_feats[i:i+1], centroids[all_labels[i]])[0]
        for i in range(len(all_feats))
    ])
    threshold = float(np.percentile(own_class_dists, args.percentile))

    out_path = args.checkpoint.parent / "ood_stats.npz"
    np.savez(
        out_path,
        centroid_labels=np.array(sorted(centroids.keys())),
        centroids=np.stack([centroids[l] for l in sorted(centroids.keys())]),
        threshold=threshold,
        percentile=args.percentile,
        layer=args.layer,
    )
    print(f"Saved OOD stats to {out_path}")
    print(f"Threshold (cosine distance, {args.percentile}th percentile of in-distribution): {threshold:.4f}")
    print(f"In-distribution distance range: min={own_class_dists.min():.4f} max={own_class_dists.max():.4f} mean={own_class_dists.mean():.4f}")


if __name__ == "__main__":
    main()
