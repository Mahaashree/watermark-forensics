"""
End-to-end visual inference demo: Original -> Watermarked -> Attacked ->
Classifier verdict, for real test-set examples.

Examples are selected from the real, already-scored test set (cached
inference at /tmp/accum_long_roc_data.npz, cross-referenced with
data/splits_v2/test.csv and results/attack_breakdown/attack_type_map.json)
-- no new attacks are run. "Attacked" panels are the real saved artifacts
from data/watermarked_v2 / data/removed_v2. "Watermarked" (pre-attack)
panels are regenerated via the real deterministic DWT_DCT_SVD.embed() with
the exact params build_verified_dataset.py used (alpha=0.02, block_size=8,
dwt_level=2, watermark_seed=42) -- same pattern as mech_fig1_spectral.py,
since that intermediate was never persisted to disk.

Every verify() correlation score and softmax confidence shown is real,
computed against the real image data -- nothing illustrative.

Outputs -> results/inference_demo/
"""
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.watermark.embed import DWT_DCT_SVD, generate_watermark

OUT = Path("results/inference_demo")
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
ALPHA = 0.02
THRESHOLD = 0.5
method = DWT_DCT_SVD(dwt_level=2, block_size=8)
watermark = generate_watermark((IMG_SIZE // 4 // 8,) * 2, seed=42)

# ---- Real cached inference + real per-image labels, cross-referenced ----
d = np.load("/tmp/accum_long_roc_data.npz")
probs, labels = d["probs"], d["labels"]
df = pd.read_csv("data/splits_v2/test.csv")
assert (df.label.values == labels).all(), "cached npz / test.csv order mismatch"
df["prob"] = probs
df["pred"] = (probs > 0.5).astype(int)
amap = json.load(open("results/attack_breakdown/attack_type_map.json"))
df["fname"] = df.path.apply(lambda p: Path(p).name)
df["attack_type"] = df.fname.map(amap)

# Hand-picked by real-inference criteria (see docstring), not illustrative:
# 2 successful removals correctly classified, 2 failed attacks correctly
# classified, 2 real misclassifications (one false-positive, one
# false-negative) -- covering both attack types in the dataset.
selection = [
    ("data/removed_v2/0424.png", "Attack succeeded, correctly classified"),
    ("data/removed_v2/0300.png", "Attack succeeded, correctly classified"),
    ("data/watermarked_v2/0418.png", "Attack failed, correctly classified"),
    ("data/watermarked_v2/0804.png", "Attack failed, correctly classified"),
    ("data/watermarked_v2/0012.png", "Misclassified (false positive)"),
    ("data/removed_v2/0018.png", "Misclassified (false negative)"),
]

def load_rgb(path, size=IMG_SIZE):
    bgr = cv2.imread(path)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def make_panel_figure(row, tag, save_path):
    fname = row["fname"]
    raw_path = f"data/raw/{fname}"
    raw_rgb = load_rgb(raw_path)

    # Real deterministic re-embed (never persisted by the pipeline)
    wm_rgb = method.embed(raw_rgb, watermark, ALPHA)
    wm_extracted = method.extract(wm_rgb, raw_rgb, ALPHA)
    wm_present, wm_corr = method.verify(wm_extracted, watermark, THRESHOLD)

    # Real saved post-attack artifact
    attacked_rgb = load_rgb(row["path"])
    att_extracted = method.extract(attacked_rgb, raw_rgb, ALPHA)
    att_present, att_corr = method.verify(att_extracted, watermark, THRESHOLD)
    # Sanity: verify() must reproduce the same present/removed label the
    # dataset was built with (label=0 <-> present, label=1 <-> removed).
    expected_present = row["label"] == 0
    assert att_present == expected_present, (
        f"{fname}: verify()={att_present} but dataset label implies {expected_present}"
    )

    pred_name = "Removed" if row["pred"] == 1 else "Present"
    true_name = "Removed" if row["label"] == 1 else "Present"
    correct = row["pred"] == row["label"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))

    axes[0].imshow(raw_rgb)
    axes[0].set_title(f"Original\ndata/raw/{fname}", fontsize=9)

    axes[1].imshow(wm_rgb)
    axes[1].set_title(
        f"Watermarked (pre-attack)\nregenerated via embed()\n"
        f"verify(): present={wm_present}, corr={wm_corr:.3f}",
        fontsize=9,
    )

    axes[2].imshow(attacked_rgb)
    axes[2].set_title(
        f"Attacked ({row['attack_type']})\n{row['path']}\n"
        f"verify(): present={att_present}, corr={att_corr:.3f}",
        fontsize=9,
    )

    for ax in axes[:3]:
        ax.set_xticks([]); ax.set_yticks([])

    axes[3].axis("off")
    verdict_color = "#d4f4dd" if correct else "#f4d4d4"
    verdict_text = "Correct" if correct else "MISCLASSIFIED"
    axes[3].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[3].transAxes, color=verdict_color, zorder=0))
    axes[3].text(
        0.5, 0.5,
        f"Classifier verdict\n\n"
        f"Predicted: {pred_name}\n"
        f"P(removed) = {row['prob']:.3f}\n\n"
        f"True: {true_name}\n\n"
        f"{verdict_text}",
        ha="center", va="center", fontsize=11, fontweight="bold", transform=axes[3].transAxes,
    )
    axes[3].set_title("Verdict", fontsize=9)

    fig.suptitle(f"{tag} -- checkpoints_v2_accum_long/best_model.pt", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.72, bottom=0.06, left=0.02, right=0.98, wspace=0.12)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return raw_rgb, wm_rgb, attacked_rgb, wm_present, wm_corr, att_present, att_corr, pred_name, true_name, correct

saved_files = []
grid_data = []
for path, tag in selection:
    row = df[df.path == path].iloc[0]
    out_name = f"demo_{row['fname'].replace('.png', '')}_{'success' if row['label']==1 and row['pred']==1 else 'fail' if row['label']==0 and row['pred']==0 else 'misclass'}.png"
    save_path = OUT / out_name
    result = make_panel_figure(row, tag, save_path)
    saved_files.append(str(save_path))
    grid_data.append((row, tag, result))
    print(f"Saved {save_path}")

# ---- Combined grid: one row per example, same 4 columns ----
fig, axes = plt.subplots(len(grid_data), 4, figsize=(16, 4.2 * len(grid_data)))
for row_i, (row, tag, (raw_rgb, wm_rgb, attacked_rgb, wm_present, wm_corr, att_present, att_corr, pred_name, true_name, correct)) in enumerate(grid_data):
    ax_row = axes[row_i]
    ax_row[0].imshow(raw_rgb)
    ax_row[0].set_title(f"Original\n{row['fname']}", fontsize=8)

    ax_row[1].imshow(wm_rgb)
    ax_row[1].set_title(f"Watermarked (pre-attack)\nverify: present={wm_present}, corr={wm_corr:.3f}", fontsize=8)

    ax_row[2].imshow(attacked_rgb)
    ax_row[2].set_title(f"Attacked ({row['attack_type']})\nverify: present={att_present}, corr={att_corr:.3f}", fontsize=8)

    for ax in ax_row[:3]:
        ax.set_xticks([]); ax.set_yticks([])

    ax_row[3].axis("off")
    verdict_color = "#d4f4dd" if correct else "#f4d4d4"
    verdict_text = "Correct" if correct else "MISCLASSIFIED"
    ax_row[3].add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_row[3].transAxes, color=verdict_color, zorder=0))
    ax_row[3].text(
        0.5, 0.5,
        f"{tag}\n\nPred: {pred_name} (p={row['prob']:.3f})\nTrue: {true_name}\n{verdict_text}",
        ha="center", va="center", fontsize=9, fontweight="bold", transform=ax_row[3].transAxes,
    )

fig.suptitle(
    "End-to-end inference demo -- checkpoints_v2_accum_long/best_model.pt\n"
    "6 real test-set examples: 2 successful removals, 2 failed attacks, 2 misclassifications (both directions)",
    fontsize=12, fontweight="bold",
)
fig.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.45, wspace=0.12)
grid_path = OUT / "demo_grid_all.png"
fig.savefig(grid_path, dpi=180)
plt.close(fig)
saved_files.append(str(grid_path))
print(f"Saved {grid_path}")

print("\nAll files:")
for f in saved_files:
    print(" -", f)
