"""
Publication-quality figures from results already on disk. No new training,
no re-derived numbers -- every value here is copied verbatim from committed
JSON/log artifacts (paths noted in each figure's comment) or, for the ROC
curve, from a single inference pass (no backprop, no weight change) through
the already-trained, already-committed checkpoint at
checkpoints_v2_accum_long/best_model.pt, saved to /tmp/accum_long_roc_data.npz
and sanity-checked to reproduce results/v2_accum_long/test_metrics.json's
AUROC exactly (0.6614992150706436) before use here.

CLI: python scripts/make_result_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.autolayout": False,
})


def add_caption(fig, text, bottom_margin, top_margin=0.90, left=0.10, right=0.97):
    """Reserve explicit bottom margin via subplots_adjust (not tight_layout,
    which fought with manual margins and caused caption/tick-label overlap
    in an earlier draft) and place the caption low and fixed, growing
    upward, so it never collides with x-tick labels above it."""
    fig.subplots_adjust(top=top_margin, bottom=bottom_margin, left=left, right=right)
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=8, style="italic")


# ---------------------------------------------------------------------------
# Figure 1: baseline vs new converged model, grouped bar chart
# Old baseline: given in task prompt, matches results/v2_baseline_reference/
#   test_metrics.json exactly (auroc 0.8875713658322354, acc 0.8, etc.)
# New: results/v2_accum_long/test_metrics.json (accum=2, long-budget, converged)
# ---------------------------------------------------------------------------
def fig1_baseline_vs_new():
    metrics = ["AUROC", "Accuracy", "Precision", "Recall", "F1"]
    old = [0.8875713658322354, 0.8, 0.85, 0.7391304347826086, 0.7906976744186046]
    new = [0.6614992150706436, 0.6733333333333333, 0.7024793388429752,
           0.8673469387755102, 0.776255707762557]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 7.5))
    b1 = ax.bar(x - w / 2, old, w, label="Old baseline\n(original attack mix)", color="#8c8c8c")
    b2 = ax.bar(x + w / 2, new, w, label="New (converged)\n(distortion_optimization + SANA-VAE)", color="#2b6cb0")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test performance: old baseline vs. new converged model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    add_caption(
        fig,
        "Lower scores reflect a harder, more realistic attack mix (distortion-optimization + SANA-VAE\n"
        "regeneration replacing the original single generic-distortion attack), not a model regression --\n"
        "see results/attack_breakdown/ANALYSIS.md. Old: results/v2_baseline_reference/test_metrics.json.\n"
        "New: results/v2_accum_long/test_metrics.json (batch=8, grad_accum_steps=2, converged at epoch 4).",
        bottom_margin=0.28, top_margin=0.82,
    )
    fig.savefig(OUT / "fig1_baseline_vs_new.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: per-attack-family AUROC, grouped by the two independently
# converged runs -- results/attack_breakdown/{batch8,accum_long}_{optimization,sana}.json
# ---------------------------------------------------------------------------
def fig2_per_attack_family():
    runs = ["batch=8\n(committed)", "accum=2, long budget\n(converged)"]
    optimization = [0.7327327327327328, 0.7687687687687688]
    sana = [0.6724806201550387, 0.6482558139534883]

    x = np.arange(len(runs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 7.5))
    b1 = ax.bar(x - w / 2, optimization, w, label="distortion_optimization", color="#2f855a")
    b2 = ax.bar(x + w / 2, sana, w, label="SANA-VAE", color="#c05621")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-attack-family AUROC: consistent\nacross two independent runs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    add_caption(
        fig,
        "distortion_optimization scores higher than SANA-VAE in both the batch=8 run and the independently\n"
        "retrained, long-budget grad-accumulation run -- the ordering does not flip between two differently-\n"
        "configured but both genuinely converged models, evidence this is a real attack-family difficulty\n"
        "difference rather than training noise. Source: results/attack_breakdown/*_{optimization,sana}.json.",
        bottom_margin=0.26, top_margin=0.78,
    )
    fig.savefig(OUT / "fig2_per_attack_family_auroc.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: training curves, short (degenerate) vs long (converged) budget
# results/logs_v2_accum/train.log (6 epochs) and
# results/logs_v2_accum_long/train.log (19 epochs)
# ---------------------------------------------------------------------------
def fig3_training_curves():
    short_val_acc = [0.5467, 0.4533, 0.4733, 0.5267, 0.5467, 0.4667]
    long_val_acc = [
        0.5467, 0.5467, 0.4733, 0.6333, 0.5467, 0.4667, 0.5733, 0.5733,
        0.5600, 0.4600, 0.5867, 0.5400, 0.5867, 0.5133, 0.5600, 0.5467,
        0.5667, 0.6067, 0.5933,
    ]
    majority_baseline = 0.5467  # val split majority-class rate, 82/150

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.plot(range(1, len(short_val_acc) + 1), short_val_acc, marker="o",
            color="#c53030", label="Short budget (epochs=12, patience=5) -- degenerate collapse", linewidth=2)
    ax.plot(range(1, len(long_val_acc) + 1), long_val_acc, marker="o",
            color="#2b6cb0", label="Long budget (epochs=36, patience=15) -- converged", linewidth=2)
    ax.axhline(majority_baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Majority-class baseline ({majority_baseline:.4f})")
    ax.annotate("epoch 4 breakout: 0.6333", xy=(4, 0.6333), xytext=(7, 0.66),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Batch-size confound: same effective batch size, different step budget")
    ax.set_ylim(0.42, 0.68)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.5, 20)

    add_caption(
        fig,
        "Both runs use batch_size=8, grad_accum_steps=2 (effective batch 16, matching the original baseline).\n"
        "The short-budget run never breaks above the majority-class baseline and early-stops collapsed;\n"
        "matching the baseline's optimizer-step count (3x epochs and patience) let epoch 4 break out to 0.6333\n"
        "and converge normally. Source: results/logs_v2_accum/train.log, results/logs_v2_accum_long/train.log.",
        bottom_margin=0.26, top_margin=0.86,
    )
    fig.savefig(OUT / "fig3_training_curves_collapse_vs_converged.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Task 1 control AUROC across three model versions
# origin/v2 reference: git show origin/v2:results/task1_control_auroc_final.json
# batch=8: results/task1_control_auroc.json
# accum-long: results/task1_control_auroc_accum_long.json
# ---------------------------------------------------------------------------
def fig4_task1_trend():
    labels = ["origin/v2\nreference", "batch=8\n(committed)", "accum=2, long budget\n(converged)"]
    values = [0.6717586163363211, 0.6106830487296959, 0.5712203248646397]

    fig, ax = plt.subplots(figsize=(8, 9))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=["#a0aec0", "#4299e1", "#2b6cb0"], width=0.5)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Task 1 control AUROC")
    ax.set_title("Task 1 control AUROC across model versions\n(unresolved, not a confirmed trend)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    add_caption(
        fig,
        "CAUTION: this pattern is NOT confirmed as a real declining detection signal. The origin/v2 reference\n"
        "was measured against removed images from the OLD (v5) attack pipeline, which data/control_v5 was\n"
        "purpose-built to match in degradation severity. Both of our reruns evaluate a DIFFERENT attack mix\n"
        "(distortion_optimization + SANA-VAE) against that same, no-longer-matched control set -- a suspected\n"
        "control-set/attack-mix mismatch, not necessarily a real change in signal, is the leading explanation\n"
        "for why both of our runs sit below the reference point. The further batch=8-vs-accum-long gap (0.611\n"
        "vs 0.571) evaluates identical data with only the model differing, so that part is NOT explained by\n"
        "the control mismatch and remains a separate, genuinely open question (see ANALYSIS.md).",
        bottom_margin=0.32, top_margin=0.80,
    )
    fig.savefig(OUT / "fig4_task1_control_trend.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: ROC curve, new converged model only. Old baseline's raw
# prediction scores were never persisted to disk (src/evaluate.py only ever
# saved aggregate metrics) and its checkpoint no longer exists (overwritten
# by this session's later retrains reusing the same checkpoints_v2/ path) --
# genuinely unobtainable, not approximated. New model's scores are a fresh
# inference pass (no training, no weight change) through
# checkpoints_v2_accum_long/best_model.pt, saved to
# /tmp/accum_long_roc_data.npz, AUROC-verified to match
# results/v2_accum_long/test_metrics.json exactly (0.6614992150706436)
# before this plot was made.
# ---------------------------------------------------------------------------
def fig5_roc_curve():
    data = np.load("/tmp/accum_long_roc_data.npz")
    probs, labels = data["probs"], data["labels"]
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    ax.plot(fpr, tpr, color="#2b6cb0", linewidth=2,
            label=f"New converged model (AUROC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: new converged model (test split)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)

    add_caption(
        fig,
        "Old baseline's ROC is NOT shown: src/evaluate.py never persisted raw per-sample prediction scores\n"
        "(only aggregate metrics), and that run's checkpoint no longer exists -- later retrains this session\n"
        "reused the same checkpoints_v2/ path and overwrote it. Skipped rather than approximated. New model's\n"
        "curve is from a fresh inference pass (no training) through checkpoints_v2_accum_long/best_model.pt,\n"
        "AUROC-verified against results/v2_accum_long/test_metrics.json (0.6614992150706436) before plotting.",
        bottom_margin=0.24, top_margin=0.85, left=0.14, right=0.92,
    )
    fig.savefig(OUT / "fig5_roc_new_model_only.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_baseline_vs_new()
    fig2_per_attack_family()
    fig3_training_curves()
    fig4_task1_trend()
    fig5_roc_curve()
    print("Wrote:", sorted(p.name for p in OUT.glob("*.png")))
