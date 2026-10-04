"""
One-off: evaluate checkpoints_3class/best_model.pt on only the
present-vs-removed subset of the 3-class test split (excludes the
"original" class), to check whether the pooled 3-class macro metrics
(accuracy 0.817, AUROC 0.914) are inflated by the easy original-vs-rest
distinction. Score = P(removed) / (P(present) + P(removed)), renormalized
to exclude the "original" probability mass.
"""

import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.classifier import WatermarkClassifier
from src.train import resolve_device

CONFIG = "configs/3class.yaml"
CHECKPOINT = "checkpoints_3class/best_model.pt"

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

device = resolve_device(cfg["train"]["device"])
model = WatermarkClassifier.from_config(cfg["model"]).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

df = pd.read_csv(Path(cfg["data"]["splits_dir"]) / "test.csv")
df = df[df["label"].isin([0, 1])].reset_index(drop=True)  # present=0, removed=1 only

transform = A.Compose([
    A.Resize(height=cfg["data"]["img_size"], width=cfg["data"]["img_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

all_probs = []
all_labels = []

with torch.no_grad():
    for _, row in df.iterrows():
        img = cv2.imread(row["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(image=img)["image"].unsqueeze(0).to(device)
        logits = model(img)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)
        all_labels.append(int(row["label"]))

probs_full = np.stack(all_probs)
labels = np.array(all_labels)

# Renormalize to exclude P(original): score = P(removed) / (P(present) + P(removed))
denom = probs_full[:, 0] + probs_full[:, 1]
score_removed = probs_full[:, 1] / denom
preds = (score_removed > 0.5).astype(int)

acc = accuracy_score(labels, preds)
auroc = roc_auc_score(labels, score_removed)
prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
cm = confusion_matrix(labels, preds).tolist()

print(f"n_samples (present+removed only) = {len(labels)}")
print(f"n_present = {(labels == 0).sum()}, n_removed = {(labels == 1).sum()}")
print(f"Accuracy = {acc:.4f}")
print(f"AUROC = {auroc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall = {rec:.4f}")
print(f"F1 = {f1:.4f}")
print(f"Confusion matrix (rows=true present/removed, cols=pred present/removed) = {cm}")
