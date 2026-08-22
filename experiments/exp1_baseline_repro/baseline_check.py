"""
Step 1: Trivial baseline check.
  Option A: 2-layer CNN trained on same data.
  Option B: Non-learned threshold on Laplacian variance + JPEG blockiness.

Usage: python experiments/exp1_baseline_repro/baseline_check.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve
)
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import WatermarkForensicsDataset
from src.models.simple_cnn import SimpleCNN


def load_split(split_csv: Path, img_size: int = 224) -> tuple:
    ds = WatermarkForensicsDataset(split_csv, img_size, train=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    paths = pd.read_csv(split_csv)["path"].tolist()
    return ds, dl, paths


def compute_laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_jpeg_blockiness(img_bgr: np.ndarray, block_size: int = 8) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    h, w = h - h % block_size, w - w % block_size
    gray = gray[:h, :w]
    diff_h = np.abs(np.diff(gray, axis=1))
    diff_v = np.abs(np.diff(gray, axis=0))
    block_boundary_mask_h = np.zeros_like(diff_h, dtype=bool)
    block_boundary_mask_v = np.zeros_like(diff_v, dtype=bool)
    for i in range(block_size - 1, h, block_size):
        if i < diff_v.shape[0]:
            block_boundary_mask_v[i, :] = True
    for j in range(block_size - 1, w, block_size):
        if j < diff_h.shape[1]:
            block_boundary_mask_h[:, j] = True
    boundary_strength = (diff_h[block_boundary_mask_h].mean() + diff_v[block_boundary_mask_v].mean()) / 2
    interior_mask_h = ~block_boundary_mask_h
    interior_mask_v = ~block_boundary_mask_v
    interior_strength = (diff_h[interior_mask_h].mean() + diff_v[interior_mask_v].mean()) / 2
    return boundary_strength / (interior_strength + 1e-8)


def eval_option_b(paths: list, labels: np.ndarray) -> dict:
    n = len(paths)
    lap_vars = np.zeros(n)
    blockiness = np.zeros(n)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        lap_vars[i] = compute_laplacian_variance(img)
        blockiness[i] = compute_jpeg_blockiness(img)

    best_lap = {}
    best_block = {}

    for name, feat in [("laplacian_var", lap_vars), ("jpeg_blockiness", blockiness)]:
        for higher_is_removed in [True, False]:
            sorted_idx = np.argsort(feat)
            best_acc = 0.0
            best_thresh = None
            for pivot in range(1, n):
                thresh = (feat[sorted_idx[pivot - 1]] + feat[sorted_idx[pivot]]) / 2
                if higher_is_removed:
                    preds = (feat > thresh).astype(int)
                else:
                    preds = (feat < thresh).astype(int)
                acc = accuracy_score(labels, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
            preds = (feat > best_thresh).astype(int) if higher_is_removed else (feat < best_thresh).astype(int)
            acc = accuracy_score(labels, preds)
            if best_acc >= 0.5:
                auc = max(roc_auc_score(labels, feat), 1 - roc_auc_score(labels, feat))
            else:
                auc = 1 - roc_auc_score(labels, feat) if roc_auc_score(labels, feat) > 0.5 else roc_auc_score(labels, feat)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
            cm = confusion_matrix(labels, preds).tolist()
            result = {"accuracy": acc, "auroc": auc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm, "threshold": float(best_thresh), "higher_is_removed": higher_is_removed}
            if name == "laplacian_var":
                best_lap = result
            else:
                best_block = result

    return {"laplacian_var": best_lap, "jpeg_blockiness": best_block}


def train_option_a(train_dl, val_dl, device, epochs=20):
    model = SimpleCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"  Epoch {epoch:02d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print(f"  Early stopping at epoch {epoch}")
                break

    return model, best_val_acc


@torch.no_grad()
def eval_option_a(model, test_dl, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    auroc = roc_auc_score(labels, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds).tolist()
    return {"accuracy": acc, "auroc": auroc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--output", type=str, default="experiments/exp1_baseline_repro/baseline_results.json")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    print("Loading splits...")
    train_ds, train_dl, train_paths = load_split(splits_dir / "train.csv")
    val_ds, val_dl, val_paths = load_split(splits_dir / "val.csv")
    test_ds, test_dl, test_paths = load_split(splits_dir / "test.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    print("\n=== Option A: 2-layer CNN ===")
    print("Training SimpleCNN...")
    model, best_val_acc = train_option_a(train_dl, val_dl, device)
    print(f"Best val acc: {best_val_acc:.4f}")
    results["option_a"] = {"val_accuracy": best_val_acc}
    for split_name, split_dl in [("train", train_dl), ("val", val_dl), ("test", test_dl)]:
        metrics = eval_option_a(model, split_dl, device)
        results["option_a"][split_name] = metrics
        print(f"  {split_name}: acc={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f} f1={metrics['f1']:.4f}")

    print("\n=== Option B: Non-learned (Laplacian variance + JPEG blockiness threshold) ===")
    for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        labels = pd.read_csv(splits_dir / f"{split_name}.csv")["label"].values
        split_results = eval_option_b(paths, labels)
        results.setdefault("option_b", {})[split_name] = split_results
        lap = split_results["laplacian_var"]
        jpeg = split_results["jpeg_blockiness"]
        print(f"  {split_name}:")
        print(f"    Laplacian var:   acc={lap['accuracy']:.4f} auroc={lap['auroc']:.4f} f1={lap['f1']:.4f} (thresh={lap['threshold']:.2f}, higher_is_removed={lap['higher_is_removed']})")
        print(f"    JPEG blockiness: acc={jpeg['accuracy']:.4f} auroc={jpeg['auroc']:.4f} f1={jpeg['f1']:.4f} (thresh={jpeg['threshold']:.4f}, higher_is_removed={jpeg['higher_is_removed']})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n=== Summary ===")
    print(f"ConvNeXt-Tiny (original):  test acc=1.0000 auroc=1.0000")
    print(f"2-layer CNN (Option A):     test acc={results['option_a']['test']['accuracy']:.4f} auroc={results['option_a']['test']['auroc']:.4f}")
    print(f"Laplacian var (Option B):   test acc={results['option_b']['test']['laplacian_var']['accuracy']:.4f} auroc={results['option_b']['test']['laplacian_var']['auroc']:.4f}")
    print(f"JPEG blockiness (Option B): test acc={results['option_b']['test']['jpeg_blockiness']['accuracy']:.4f} auroc={results['option_b']['test']['jpeg_blockiness']['auroc']:.4f}")


if __name__ == "__main__":
    main()
