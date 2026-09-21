"""
Mechanism figure 5: Grad-CAM on the ConvNeXt-Tiny classifier.
Hooks the output of the last ConvNeXt stage (model.backbone.stages[-1],
before norm_pre/pooling) on checkpoints_v2_accum_long/best_model.pt (the
converged model), backprops the predicted class logit, and overlays the
resulting CAM on three real test-set examples selected from a real
inference pass (cached at /tmp/accum_long_roc_data.npz): a true-positive
"removed", a true-negative "present", and a misclassified example.
"""
import time
t0 = time.perf_counter()

import cv2
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.train import resolve_device
from src.models.classifier import WatermarkClassifier

OUT = Path("results/figures/mechanism")
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open("configs/v2_accum_long.yaml"))
device = resolve_device(cfg["train"]["device"])
model = WatermarkClassifier.from_config(cfg["model"]).to(device)
model.load_state_dict(torch.load("checkpoints_v2_accum_long/best_model.pt", map_location=device))
model.eval()

activations = {}
gradients = {}

def fwd_hook(module, inp, out):
    activations["value"] = out

def bwd_hook(module, grad_in, grad_out):
    gradients["value"] = grad_out[0]

target_layer = model.backbone.stages[-1]
h1 = target_layer.register_forward_hook(fwd_hook)
h2 = target_layer.register_full_backward_hook(bwd_hook)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def load_input(path, img_size=224):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_size, img_size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - MEAN) / STD
    x = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return x.to(device), rgb

def gradcam(path):
    x, rgb = load_input(path)
    x.requires_grad_(False)
    logits = model(x)
    pred_class = logits.argmax(1).item()
    prob = torch.softmax(logits, dim=1)[0, pred_class].item()

    model.zero_grad()
    logits[0, pred_class].backward()

    act = activations["value"][0]      # (C, H, W)
    grad = gradients["value"][0]       # (C, H, W)
    weights = grad.mean(dim=(1, 2))    # (C,)
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return rgb, cam, pred_class, prob

# Real examples, selected from an actual inference pass on the test set
# (cached at /tmp/accum_long_roc_data.npz), not hand-picked/illustrative.
d = np.load("/tmp/accum_long_roc_data.npz")
probs_all, labels_all = d["probs"], d["labels"]
df = pd.read_csv("data/splits_v2/test.csv")
df["prob"] = probs_all
df["pred"] = (probs_all > 0.5).astype(int)

tp = df[(df.label == 1) & (df.pred == 1)].sort_values("prob", ascending=False).iloc[0]
tn = df[(df.label == 0) & (df.pred == 0)].sort_values("prob").iloc[0]
mis = df[df.label != df.pred].iloc[0]

examples = [
    (tp["path"], "True positive: removed", tp["label"]),
    (tn["path"], "True negative: present", tn["label"]),
    (mis["path"], "Misclassified", mis["label"]),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
for ax, (path, tag, true_label) in zip(axes, examples):
    rgb, cam, pred_class, prob = gradcam(path)
    ax.imshow(rgb)
    ax.imshow(cam, cmap="jet", alpha=0.45)
    pred_name = "removed" if pred_class == 1 else "present"
    true_name = "removed" if true_label == 1 else "present"
    ax.set_title(f"{tag}\ntrue={true_name}, pred={pred_name} (p={prob:.3f})\n{Path(path).name}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

h1.remove(); h2.remove()

fig.suptitle("Grad-CAM (ConvNeXt-Tiny, last stage): checkpoints_v2_accum_long/best_model.pt", fontsize=12, fontweight="bold")
fig.text(0.5, 0.02,
          "Three real test-set examples selected from an actual inference pass (highest-confidence correct removed, "
          "highest-confidence correct present, first misclassified by index). Overlay = Grad-CAM w.r.t. the predicted class.",
          ha="center", fontsize=8, style="italic", wrap=True)
fig.subplots_adjust(bottom=0.2, top=0.78, left=0.03, right=0.98, wspace=0.15)
fig.savefig(OUT / "mech5_gradcam.png", dpi=300)
plt.close(fig)

elapsed = time.perf_counter() - t0
print(f"mech5 done in {elapsed:.2f}s")
