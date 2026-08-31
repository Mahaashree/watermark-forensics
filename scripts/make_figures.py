"""
Generate showcase figures from existing results/*.json and results/logs*/train.log.

Read-only over the pipeline: does not retrain, does not touch src/ or configs/.
Run after src.train / src.evaluate have produced results/*.json and results/logs*/train.log.

CLI: python -m scripts.make_figures
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

# Colors from the project's validated categorical palette (references/palette.md
# in the dataviz skill): fixed hue order, slot 1 = blue, slot 2 = orange.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

FIG_DIR = Path("results/figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "sans-serif"],
    "axes.edgecolor": AXIS,
    "axes.labelcolor": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "grid.color": GRID,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_train_log(path: Path):
    """Split a train.log into per-run blocks (one per 'Device:' line).

    Returns a list of dicts: {"splits_dir": str|None, "completed": bool,
    "epochs": [{"epoch": int, "train_loss": float, "train_acc": float,
    "val_loss": float, "val_acc": float}]}.
    """
    text = path.read_text()
    lines = text.splitlines()

    runs = []
    current = None
    for line in lines:
        if " - INFO - Device: " in line:
            if current is not None:
                runs.append(current)
            current = {"splits_dir": None, "completed": False, "epochs": []}
            continue
        if current is None:
            continue
        if "Config: {" in line:
            m = re.search(r"'splits_dir':\s*'([^']+)'", line)
            if m:
                current["splits_dir"] = m.group(1)
        m = re.search(
            r"Epoch (\d+) \| Train: ([\d.]+)/([\d.]+) \| Val: ([\d.]+)/([\d.]+)",
            line,
        )
        if m:
            current["epochs"].append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_acc": float(m.group(3)),
                "val_loss": float(m.group(4)),
                "val_acc": float(m.group(5)),
            })
        if "Training complete" in line:
            current["completed"] = True
    if current is not None:
        runs.append(current)
    return runs


def last_completed_run(runs, splits_dir=None):
    candidates = [r for r in runs if r["completed"] and r["epochs"]]
    if splits_dir is not None:
        matching = [r for r in candidates if r["splits_dir"] == splits_dir]
        if matching:
            candidates = matching
    return candidates[-1] if candidates else None


def plot_training_curves(run, title, out_path):
    epochs = [e["epoch"] for e in run["epochs"]]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax_loss.plot(epochs, [e["train_loss"] for e in run["epochs"]],
                 color=BLUE, linewidth=2, marker="o", markersize=4, label="Train")
    ax_loss.plot(epochs, [e["val_loss"] for e in run["epochs"]],
                 color=ORANGE, linewidth=2, marker="o", markersize=4, label="Val")
    ax_loss.set_title("Loss", color=TEXT_PRIMARY, fontsize=10, loc="left")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend(frameon=False)

    ax_acc.plot(epochs, [e["train_acc"] for e in run["epochs"]],
                color=BLUE, linewidth=2, marker="o", markersize=4, label="Train")
    ax_acc.plot(epochs, [e["val_acc"] for e in run["epochs"]],
                color=ORANGE, linewidth=2, marker="o", markersize=4, label="Val")
    ax_acc.set_title("Accuracy", color=TEXT_PRIMARY, fontsize=10, loc="left")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(frameon=False)

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(cm_left, cm_right, label_left, label_right, out_path):
    class_names = ["watermarked", "removed"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, cm, title in zip(axes, [cm_left, cm_right], [label_left, label_right]):
        cm_arr = cm
        vmax = max(max(row) for row in cm_arr)
        ax.imshow(cm_arr, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1], class_names)
        ax.set_yticks([0, 1], class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                val = cm_arr[i][j]
                color = "white" if val > vmax * 0.6 else TEXT_PRIMARY
                ax.text(j, i, str(val), ha="center", va="center",
                         color=color, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_comparison(metrics_left, metrics_right, label_left, label_right, out_path):
    keys = ["accuracy", "auroc", "precision", "recall", "f1"]
    display = ["Accuracy", "AUROC", "Precision", "Recall", "F1"]
    left_vals = [metrics_left[k] for k in keys]
    right_vals = [metrics_right[k] for k in keys]

    x = range(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - width / 2 for i in x], left_vals, width, color=BLUE, label=label_left)
    ax.bar([i + width / 2 for i in x], right_vals, width, color=ORANGE, label=label_right)
    ax.set_xticks(list(x), display)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Test-set metrics: label-leak run vs. verification-labeled run",
                 color=TEXT_PRIMARY, fontsize=11, loc="left")
    for i, v in enumerate(left_vals):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=TEXT_SECONDARY)
    for i, v in enumerate(right_vals):
        ax.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_alpha_sweep(out_path):
    # Recorded in phase2.md / current_status.txt (2026-08-26 session):
    # alpha sweep run to balance the verification-labeled dataset.
    alphas = [0.20, 0.12, 0.05, 0.03, 0.02]
    present = [835, 821, 748, 589, 441]
    removed = [65, 79, 152, 311, 459]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, present, color=BLUE, linewidth=2, marker="o", markersize=5, label="Present (watermark survives)")
    ax.plot(alphas, removed, color=ORANGE, linewidth=2, marker="o", markersize=5, label="Removed (watermark lost)")
    ax.axvline(0.02, color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.02, 900, "alpha used\nfor v2 dataset", fontsize=8, color=TEXT_SECONDARY, ha="center", va="top")
    ax.invert_xaxis()
    ax.set_xlabel("Watermark strength (alpha)")
    ax.set_ylabel("Image count (of 900)")
    ax.set_title("Alpha sweep: finding a balanced present/removed split",
                 color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    default_metrics = json.loads(Path("results/test_metrics.json").read_text())
    v2_metrics = json.loads(Path("results/v2/test_metrics.json").read_text())

    default_runs = parse_train_log(Path("results/logs/train.log"))
    v2_runs = parse_train_log(Path("results/logs_v2/train.log"))

    default_run = last_completed_run(default_runs, splits_dir="data/splits")
    v2_run = last_completed_run(v2_runs, splits_dir="data/splits_v2")

    if default_run:
        plot_training_curves(
            default_run,
            "Phase 1 training curve (construction-based labels — the leaky run)",
            FIG_DIR / "training_curves_default.png",
        )
    if v2_run:
        plot_training_curves(
            v2_run,
            "Phase 2 (v2) training curve (verification-based labels — the honest run)",
            FIG_DIR / "training_curves_v2.png",
        )

    plot_confusion_matrices(
        default_metrics["confusion_matrix"], v2_metrics["confusion_matrix"],
        "Leaky (construction-labeled)", "Honest (verification-labeled)",
        FIG_DIR / "confusion_matrices.png",
    )

    plot_metrics_comparison(
        default_metrics, v2_metrics,
        "Leaky (construction-labeled)", "Honest (verification-labeled)",
        FIG_DIR / "metrics_comparison.png",
    )

    plot_alpha_sweep(FIG_DIR / "alpha_sweep.png")

    print(f"Wrote figures to {FIG_DIR}/")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" -", p)


if __name__ == "__main__":
    main()
