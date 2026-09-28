"""
Score a frozen classifier on an arbitrary split CSV (path,label), reporting
the same metric set as src/evaluate.py (accuracy, AUROC, precision, recall,
F1, confusion matrix). Used to break the pooled test AUROC down by which
attack produced each image -- attack type isn't stored per-image by
build_verified_dataset.py, so the CSV passed in here is expected to already
be filtered to one attack type (see the RNG-replay note in the commit this
script ships with).

CLI: python scripts/eval_by_attack_type.py --config configs/v2_lowmem.yaml \
    --checkpoint checkpoints_v2/best_model.pt --csv /tmp/test_sana.csv
"""

import argparse
import json
import yaml
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path

from src.train import resolve_device
from src.data.dataset import WatermarkForensicsDataset
from src.models.classifier import WatermarkClassifier
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--label", type=str, default=None, help="Name for this subset in the printed report")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["train"]["device"])
    ds = WatermarkForensicsDataset(args.csv, cfg["data"]["img_size"], train=False)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = WatermarkClassifier.from_config(cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_probs, all_labels = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")  # only one class present in this subset
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()

    results = {
        "subset": args.label or str(args.csv),
        "accuracy": acc,
        "auroc": auroc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "loss": total_loss / total,
        "n_samples": int(total),
        "n_present": int((labels == 0).sum()),
        "n_removed": int((labels == 1).sum()),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
