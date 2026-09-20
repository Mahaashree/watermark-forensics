"""
Training loop with OOM fallback, mixed precision (torch.amp), early stopping.
CLI: python src/train.py --config configs/default.yaml [--seed N]
"""

import argparse
import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.dataset import create_dataloaders
from src.models.classifier import WatermarkClassifier, ModelConfig


def resolve_device(requested: str) -> torch.device:
    """"auto" picks cuda > mps > cpu. "cpu" always forces cpu. Anything else
    (e.g. explicit "cuda"/"mps") is used as-is if available, else falls back
    to cpu."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_class_weights(cfg, device: torch.device):
    """Config-driven, opt-in class weighting for the loss.

    cfg["train"]["class_weights"] may be:
      - absent / null (default): no weighting, identical behavior to before
        this option existed.
      - "auto": inverse-frequency weights computed from the train split's
        label counts (sklearn "balanced" formula: n_samples / (n_classes *
        count_c)).
      - a 2-element list [w0, w1]: used directly, e.g. to upweight class 1
        ("removed") beyond what frequency-balancing alone would give.
    """
    spec = cfg["train"].get("class_weights")
    if spec is None:
        return None

    if spec == "auto":
        train_csv = Path(cfg["data"]["splits_dir"]) / "train.csv"
        counts = pd.read_csv(train_csv)["label"].value_counts().sort_index()
        n = counts.sum()
        weights = [n / (len(counts) * c) for c in counts]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return torch.tensor(list(spec), dtype=torch.float32, device=device)

    raise ValueError(f"Unrecognized train.class_weights value: {spec!r}")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip, use_amp):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        all_labels.append(labels.cpu())

    return total_loss / total, correct / total, torch.cat(all_probs), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None,
                         help="Override cfg['data']['seed']. When set, save_dir "
                              "and log_dir get a _seed{N} suffix so multi-seed "
                              "runs don't clobber each other's checkpoints/logs.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["data"]["seed"] = args.seed
        cfg["train"]["save_dir"] = f"{cfg['train']['save_dir']}_seed{args.seed}"
        cfg["train"]["log_dir"] = f"{cfg['train']['log_dir']}_seed{args.seed}"

    set_seed(cfg["data"]["seed"])

    device = resolve_device(cfg["train"]["device"])

    log_dir = Path(cfg["train"]["log_dir"])
    logger = setup_logging(log_dir, "train")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    train_dl, val_dl, _ = create_dataloaders(
        Path(cfg["data"]["splits_dir"]),
        cfg["data"]["img_size"],
        cfg["train"]["batch_size"],
        cfg["data"]["num_workers"],
        cfg["data"]["seed"]
    )

    model = WatermarkClassifier.from_config(cfg["model"]).to(device)
    logger.info(f"Model: {cfg['model']['name']}, params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    class_weights = resolve_class_weights(cfg, device)
    if class_weights is not None:
        logger.info(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_acc = 0.0
    patience = 0
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_dl, optimizer, criterion, device, scaler, cfg["train"]["grad_clip"], use_amp
        )
        val_loss, val_acc, _, _ = evaluate(model, val_dl, criterion, device, use_amp)
        scheduler.step()

        logger.info(f"Epoch {epoch:03d} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            logger.info(f"  -> New best: {best_val_acc:.4f}")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()