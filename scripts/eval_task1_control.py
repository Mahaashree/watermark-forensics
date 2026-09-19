"""
Task 1 eval: score frozen classifier on {removed_v2 test} vs {control_v2}.
No retraining. Same P(class 1) decision function as src/evaluate.py.

CLI: python scripts/eval_task1_control.py --config configs/v2.yaml \
    --checkpoint checkpoints_v2/best_model.pt \
    --csv data/splits_v2/test_task1_control.csv
"""

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.data.dataset import WatermarkForensicsDataset
from src.models.classifier import WatermarkClassifier
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v2.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_v2/best_model.pt"))
    parser.add_argument("--csv", type=Path, default=Path("data/splits_v2/test_task1_control.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/task1_control_auroc.json"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = WatermarkForensicsDataset(args.csv, cfg["data"]["img_size"], train=False)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = WatermarkClassifier.from_config(cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dl:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    auroc = roc_auc_score(labels, probs)
    removed_scores = probs[labels == 1]
    control_scores = probs[labels == 0]

    results = {
        "task": "task1_nonwatermarked_degradation_control",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "csv": str(args.csv),
        "auroc": float(auroc),
        "n_samples": int(len(labels)),
        "n_removed": int((labels == 1).sum()),
        "n_control": int((labels == 0).sum()),
        "removed_mean_prob": float(removed_scores.mean()),
        "removed_std_prob": float(removed_scores.std()),
        "control_mean_prob": float(control_scores.mean()),
        "control_std_prob": float(control_scores.std()),
    }

    print(json.dumps(results, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
