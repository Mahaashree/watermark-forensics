"""
Evaluation: accuracy, AUROC, precision, recall, F1, confusion matrix.
CLI: python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split test
"""

import argparse
import yaml
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path
from tqdm import tqdm

from src.utils.seed import set_seed
from src.data.dataset import create_dataloaders
from src.models.classifier import WatermarkClassifier, ModelConfig
from src.train import resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=None,
                         help="Override cfg['data']['seed']. Only affects "
                              "dataloader shuffling determinism, not which "
                              "checkpoint is loaded — pass --checkpoint "
                              "explicitly for the matching seed run.")
    parser.add_argument("--tag", type=str, default=None,
                         help="Suffix for the saved metrics filename, e.g. "
                              "'seed1', so multi-seed eval runs don't "
                              "overwrite each other's {split}_metrics.json.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["data"]["seed"] = args.seed

    set_seed(cfg["data"]["seed"])
    device = resolve_device(cfg["train"]["device"])

    train_dl, val_dl, test_dl = create_dataloaders(
        Path(cfg["data"]["splits_dir"]),
        cfg["data"]["img_size"],
        cfg["train"]["batch_size"],
        cfg["data"]["num_workers"],
        cfg["data"]["seed"]
    )
    dl_map = {"train": train_dl, "val": val_dl, "test": test_dl}
    eval_dl = dl_map[args.split]

    model = WatermarkClassifier.from_config(cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(eval_dl, desc=f"Eval {args.split}"):
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
    auroc = roc_auc_score(labels, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds).tolist()

    results = {
        "split": args.split,
        "accuracy": acc,
        "auroc": auroc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "loss": total_loss / total,
        "n_samples": int(total)
    }

    print(json.dumps(results, indent=2))

    save_dir = Path(cfg["eval"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.split}_metrics_{args.tag}.json" if args.tag else f"{args.split}_metrics.json"
    with open(save_dir / filename, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
