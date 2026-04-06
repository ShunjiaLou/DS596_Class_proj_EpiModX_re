"""
plot_results.py — Generate Figures 2a-c and 3a-b from test output
==================================================================
Reproduces:
  Figure 2a-c: Per-patient AUROC, AUPRC, F1, ACC for each histone mark
  Figure 3a-b: Cross-disease group precision and recall bubble plots

Usage (run after all three test jobs complete on SCC):
  python plot_results.py

Output:
  figures/fig2_per_patient_metrics.png
  figures/fig3ab_cross_disease_precision_recall.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, accuracy_score, precision_score,
                             recall_score)

# ── Configuration ──────────────────────────────────────────────────────────────

HISTONES    = ["H3K27ac", "H3K4me3", "H3K27me3"]
MODEL_NAME  = "LLM_Moe"
RESULTS_DIR = "./test_results"
DATASETS_DIR = "./Datasets"
FIGURES_DIR = "./figures"

# Patient column order matches generate_dataset.py output:
# NCI×6, MCI×5, CI×4, AD×4, ADCI×3 = 22 total
GROUPS = {
    "NCI":  list(range(0, 6)),
    "MCI":  list(range(6, 11)),
    "CI":   list(range(11, 15)),
    "AD":   list(range(15, 19)),
    "ADCI": list(range(19, 22)),
}
N_PATIENTS = 22
GROUP_ORDER = ["NCI", "MCI", "CI", "AD", "ADCI"]
GROUP_COLORS = {"NCI": "#4C9BE8", "MCI": "#82C893", "CI": "#F5A623",
                "AD": "#E85454", "ADCI": "#9B59B6"}

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def load_test_results(histone):
    """Load prediction pickle and ground-truth CSV for test chromosomes."""
    result_path = os.path.join(RESULTS_DIR, f"{histone}_{MODEL_NAME}_test_result")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Missing: {result_path}\n"
                                f"Run sbatch_test.sh {histone} first.")
    with open(result_path, "rb") as f:
        preds_list = pickle.load(f)

    preds = np.concatenate(preds_list, axis=0)   # (N_samples, 22)

    csv_path = os.path.join(DATASETS_DIR, f"{histone}_all_data.csv")
    df = pd.read_csv(csv_path)
    test_df = df[df["chrom"].isin(["chr8", "chr9"])].reset_index(drop=True)
    labels = test_df.iloc[:, 3:].values.astype(float)   # (N_samples, 22)

    assert preds.shape == labels.shape, \
        f"Shape mismatch: preds {preds.shape} vs labels {labels.shape}"
    return preds, labels


def per_patient_metrics(preds, labels):
    """Compute AUROC, AUPRC, F1, ACC for each of the 22 patients."""
    metrics = {"AUROC": [], "AUPRC": [], "F1": [], "ACC": []}
    for i in range(N_PATIENTS):
        y_true = labels[:, i]
        y_score = preds[:, i]
        y_pred = (y_score >= 0.5).astype(int)
        if len(np.unique(y_true)) < 2:
            metrics["AUROC"].append(np.nan)
            metrics["AUPRC"].append(np.nan)
        else:
            metrics["AUROC"].append(roc_auc_score(y_true, y_score))
            metrics["AUPRC"].append(average_precision_score(y_true, y_score))
        metrics["F1"].append(f1_score(y_true, y_pred, zero_division=0))
        metrics["ACC"].append(accuracy_score(y_true, y_pred))
    return metrics


def per_group_precision_recall(preds, labels):
    """Compute per-group precision and recall (for Fig 3a-b)."""
    results = {}
    for group, indices in GROUPS.items():
        y_true = labels[:, indices].flatten()
        y_score = preds[:, indices].flatten()
        y_pred = (y_score >= 0.5).astype(int)
        if len(np.unique(y_true)) < 2:
            results[group] = {"precision": np.nan, "recall": np.nan, "p_value": 1.0}
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        # Approximate p-value via proportion z-test (simplified)
        from scipy.stats import mannwhitneyu
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            _, p = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        else:
            p = 1.0
        results[group] = {"precision": prec, "recall": rec, "p_value": p}
    return results


# ── Figure 2a-c: Per-patient metrics ──────────────────────────────────────────

def plot_fig2(all_metrics):
    """
    Box plots of AUROC, AUPRC, F1, ACC per histone mark.
    Reproduces paper Figure 2a-c layout.
    """
    metric_names = ["AUROC", "AUPRC", "F1", "ACC"]
    fig, axes = plt.subplots(len(HISTONES), len(metric_names),
                             figsize=(14, 10), sharey=False)
    fig.suptitle("Per-patient Prediction Performance (Figure 2a-c)", fontsize=14)

    for row, histone in enumerate(HISTONES):
        metrics = all_metrics[histone]
        for col, metric in enumerate(metric_names):
            ax = axes[row][col]
            values = [v for v in metrics[metric] if not np.isnan(v)]

            # Color each dot by disease group
            colors = []
            for g, idx in GROUPS.items():
                colors.extend([GROUP_COLORS[g]] * len(idx))

            ax.boxplot(values, patch_artist=True,
                       boxprops=dict(facecolor="lightgrey"),
                       medianprops=dict(color="black", linewidth=2))

            # Overlay individual dots colored by group
            x_vals = np.ones(len(metrics[metric])) + np.random.uniform(-0.1, 0.1, N_PATIENTS)
            for i, (v, c) in enumerate(zip(metrics[metric], colors)):
                if not np.isnan(v):
                    ax.scatter(x_vals[i], v, color=c, zorder=3, s=40, alpha=0.8)

            ax.set_title(f"{histone}" if col == 0 else "", fontsize=10)
            ax.set_ylabel(metric if col == 0 else "", fontsize=9)
            ax.set_xlabel(metric if row == len(HISTONES) - 1 else "", fontsize=9)
            ax.set_xticks([])
            ax.set_ylim(0, 1.05)

    # Legend for disease groups
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=GROUP_COLORS[g], markersize=8, label=g)
               for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               title="Disease Group", bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIGURES_DIR, "fig2_per_patient_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3a-b: Cross-disease precision/recall bubble plots ──────────────────

def plot_fig3ab(all_group_metrics):
    """
    Bubble plots of precision (3a) and recall (3b) across disease groups and histones.
    Bubble size = -log10(p-value). Reproduces paper Figure 3a-b layout.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-disease Group Analysis (Figure 3a-b)", fontsize=13)

    for ax, metric_key, title in zip(axes,
                                     ["precision", "recall"],
                                     ["(a) Precision", "(b) Recall"]):
        for hi, histone in enumerate(HISTONES):
            gm = all_group_metrics[histone]
            for gi, group in enumerate(GROUP_ORDER):
                val = gm[group][metric_key]
                p   = gm[group]["p_value"]
                size = max(20, -np.log10(p + 1e-300) * 30)
                color = GROUP_COLORS[group]
                ax.scatter(gi, hi, s=size, color=color, alpha=0.8, edgecolors="black", lw=0.5)
                ax.text(gi, hi + 0.2, f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, color="black")

        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels(GROUP_ORDER, fontsize=10)
        ax.set_yticks(range(len(HISTONES)))
        ax.set_yticklabels(HISTONES, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Disease Group", fontsize=10)
        ax.set_ylabel("Histone Mark" if metric_key == "precision" else "")
        ax.set_xlim(-0.5, len(GROUP_ORDER) - 0.5)
        ax.set_ylim(-0.5, len(HISTONES) - 0.5)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Bubble size legend
    for size_p, label in [(0.05, "p=0.05"), (0.01, "p=0.01"), (0.001, "p=0.001")]:
        axes[1].scatter([], [], s=max(20, -np.log10(size_p) * 30),
                        color="grey", label=label, alpha=0.7)
    axes[1].legend(title="Bubble size\n(-log10 p-value)", loc="upper right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig3ab_cross_disease_precision_recall.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_metrics       = {}
    all_group_metrics = {}

    for histone in HISTONES:
        print(f"\nLoading results for {histone}...")
        try:
            preds, labels = load_test_results(histone)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Predictions: {preds.shape}, Labels: {labels.shape}")
        all_metrics[histone]       = per_patient_metrics(preds, labels)
        all_group_metrics[histone] = per_group_precision_recall(preds, labels)

        # Print summary
        for m in ["AUROC", "AUPRC", "F1", "ACC"]:
            vals = [v for v in all_metrics[histone][m] if not np.isnan(v)]
            print(f"  {m}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    if not all_metrics:
        print("\nNo results found. Run the test jobs on SCC first.")
        return

    print("\nGenerating Figure 2a-c...")
    plot_fig2(all_metrics)

    print("Generating Figure 3a-b...")
    plot_fig3ab(all_group_metrics)

    print("\nDone. Figures saved to ./figures/")


if __name__ == "__main__":
    main()
