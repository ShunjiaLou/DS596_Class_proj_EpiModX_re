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
    """Load prediction pickle (or CSV fallback) and ground-truth CSV for test chromosomes."""
    result_path = os.path.join(RESULTS_DIR, f"{histone}_{MODEL_NAME}_test_result")
    csv_path_pred = os.path.join("./results", f"{histone}_predictions.csv")

    if os.path.exists(result_path):
        with open(result_path, "rb") as f:
            preds_list = pickle.load(f)
        preds = np.concatenate(preds_list, axis=0)   # (N_samples, 22)
    elif os.path.exists(csv_path_pred):
        print(f"  Loading predictions from CSV: {csv_path_pred}")
        preds = pd.read_csv(csv_path_pred).values.astype(float)
    else:
        raise FileNotFoundError(f"Missing: {result_path}\n"
                                f"Run sbatch_test.sh {histone} first.")

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


def compute_cross_patient_metrics(preds, labels):
    """
    For each (train_patient_i, pred_patient_j) pair:
      - Use predictions from patient i's output head
      - Evaluate against patient j's ground-truth labels
    Returns precision_matrix, recall_matrix, pval_matrix — all (22, 22).
    """
    from scipy.stats import mannwhitneyu
    n = N_PATIENTS
    prec_mat = np.zeros((n, n))
    rec_mat  = np.zeros((n, n))
    pval_mat = np.ones((n, n))

    for i in range(n):          # train patient (y-axis)
        y_pred_i  = (preds[:, i] >= 0.5).astype(int)
        y_score_i = preds[:, i]
        for j in range(n):      # pred patient (x-axis)
            y_true_j = labels[:, j]
            prec_mat[i, j] = precision_score(y_true_j, y_pred_i, zero_division=0)
            rec_mat[i, j]  = recall_score(y_true_j,  y_pred_i, zero_division=0)
            pos = y_score_i[y_true_j == 1]
            neg = y_score_i[y_true_j == 0]
            if len(pos) > 1 and len(neg) > 1:
                _, p = mannwhitneyu(pos, neg, alternative="greater")
                pval_mat[i, j] = p

    return prec_mat, rec_mat, pval_mat


# ── Figure 2: Per-patient metrics heatmap ─────────────────────────────────────

def plot_fig2(all_metrics):
    """
    Heatmap: rows = patients (22), columns = metrics (AUROC, AUPRC, F1, ACC).
    Single heatmap per histone on the left; 4 stacked colorbars on the right,
    each scaled to that metric's actual value range.
    """
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    metric_names = ["AUROC", "AUPRC", "F1", "ACC"]
    cmaps = ["Blues", "Purples", "Oranges", "Greens"]
    available = [h for h in HISTONES if h in all_metrics]

    patient_groups = []
    for g, indices in GROUPS.items():
        patient_groups.extend([g] * len(indices))
    row_labels = [f"{patient_groups[i]} P{i+1}" for i in range(N_PATIENTS)]

    # Per-metric value range (shared across histones)
    metric_ranges = {}
    for m in metric_names:
        all_vals = [v for h in available for v in all_metrics[h][m] if not np.isnan(v)]
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        step = 0.05
        lo_r = max(0.0, np.floor(lo / step) * step)
        hi_r = min(1.0, np.ceil(hi  / step) * step)
        metric_ranges[m] = (lo_r, hi_r)

    n_h = len(available)
    fig_h = max(8, n_h * 7) + 1.5   # tall enough; +1.5 for legend

    fig = plt.figure(figsize=(8, fig_h))
    # outer grid: one row per histone
    outer = gridspec.GridSpec(n_h, 1, figure=fig, hspace=0.45,
                              top=0.94, bottom=0.08)

    fig.suptitle("Per-patient Prediction Performance (Figure 2a-c)", fontsize=12)

    for hi, histone in enumerate(available):
        metrics = all_metrics[histone]
        matrix = np.array([metrics[m] for m in metric_names]).T  # (22, 4)

        # inner grid: heatmap | colorbars
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[hi],
            width_ratios=[5, 1], wspace=0.35
        )
        ax_heat = fig.add_subplot(inner[0])
        ax_cbars = fig.add_subplot(inner[1])
        ax_cbars.set_visible(False)   # placeholder; we'll use inset axes

        # ── Heatmap (use first metric's cmap just to set up; we'll overlay) ──
        # Draw each column separately so each has its own colour scale
        col_imgs = []
        for ci, (m, cmap) in enumerate(zip(metric_names, cmaps)):
            lo, hv = metric_ranges[m]
            col_data = matrix[:, ci].reshape(-1, 1)
            # We can't do per-column colormaps on a single imshow, so we
            # composite them side-by-side using imshow with extent
            im = ax_heat.imshow(
                col_data, aspect="auto", cmap=cmap,
                vmin=lo, vmax=hv, interpolation="nearest",
                extent=[ci - 0.5, ci + 0.5, N_PATIENTS - 0.5, -0.5]
            )
            col_imgs.append(im)

        ax_heat.set_xlim(-0.5, len(metric_names) - 0.5)
        ax_heat.set_ylim(N_PATIENTS - 0.5, -0.5)
        ax_heat.set_xticks(range(len(metric_names)))
        ax_heat.set_xticklabels(metric_names, fontsize=9)
        ax_heat.set_yticks(range(N_PATIENTS))
        ax_heat.set_yticklabels(row_labels, fontsize=7)
        for i, lbl in enumerate(ax_heat.get_yticklabels()):
            lbl.set_color(GROUP_COLORS[patient_groups[i]])
        ax_heat.set_title(histone, fontsize=10, loc="left", pad=4)

        # ── 4 stacked colorbars on the right ──
        cbar_h = 0.18      # fraction of axes height per colorbar
        cbar_gap = 0.06
        total = len(metric_names) * cbar_h + (len(metric_names) - 1) * cbar_gap
        y_start = (1.0 - total) / 2   # centre vertically

        for ci, (m, cmap, im) in enumerate(zip(metric_names, cmaps, col_imgs)):
            y0 = y_start + ci * (cbar_h + cbar_gap)
            cax = inset_axes(ax_cbars, width="60%", height=f"{cbar_h*100:.0f}%",
                             loc="lower left",
                             bbox_to_anchor=(0.1, y0, 1, 1),
                             bbox_transform=ax_cbars.transAxes,
                             borderpad=0)
            cb = plt.colorbar(im, cax=cax)
            lo, hv = metric_ranges[m]
            cb.set_ticks(np.linspace(lo, hv, 3))
            cb.ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
            cb.ax.tick_params(labelsize=7)
            cb.set_label(m, fontsize=8, labelpad=3)

    # Disease group legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=GROUP_COLORS[g], markersize=8, label=g)
               for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               title="Disease Group  (y-axis label color)",
               bbox_to_anchor=(0.5, 0.0), fontsize=8, title_fontsize=8,
               frameon=True)

    out = os.path.join(FIGURES_DIR, "fig2_per_patient_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: Cross-patient precision/recall bubble plot ──────────────────────

def plot_fig3ab(all_cross_metrics):
    """
    22×22 bubble plot: y-axis = train patient, x-axis = pred patient.
    Dot color = deviation from global mean (diverging blue→white→red).
    Dot size  = -log10(p-value).
    Groups separated by lines; group labels on both axes.
    One figure per histone mark (precision left, recall right).
    """
    available = [h for h in HISTONES if h in all_cross_metrics]
    if not available:
        print("  No cross-patient metrics — skipping Fig 3.")
        return

    patient_groups = []
    for g in GROUP_ORDER:
        patient_groups.extend([g] * len(GROUPS[g]))

    # Group boundary x/y positions
    boundaries = [0]
    for g in GROUP_ORDER:
        boundaries.append(boundaries[-1] + len(GROUPS[g]))
    group_centers = [(boundaries[k] + boundaries[k+1]) / 2 - 0.5
                     for k in range(len(GROUP_ORDER))]

    for histone in available:
        prec_mat, rec_mat, pval_mat = all_cross_metrics[histone]

        # Deviation from global mean (centres colormap at 0)
        prec_dev = prec_mat - np.nanmean(prec_mat)
        rec_dev  = rec_mat  - np.nanmean(rec_mat)
        vmax = max(np.percentile(np.abs(prec_dev), 97),
                   np.percentile(np.abs(rec_dev),  97))

        # Dot sizes: -log10(p), clipped and scaled
        logp = np.clip(-np.log10(pval_mat + 1e-300), 0, 3)
        size_max_pt = 120          # max dot area in points²
        sizes = logp / 3.0 * size_max_pt + 6

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        fig.suptitle(f"Cross-disease Precision & Recall — {histone} (Figure 3a-b)",
                     fontsize=12)
        cmap = cm.RdBu_r
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        for ax, dev_mat, panel in zip(axes, [prec_dev, rec_dev],
                                      ["(a) Precision", "(b) Recall"]):
            for i in range(N_PATIENTS):       # train patient  → y-axis (top = last group)
                for j in range(N_PATIENTS):   # pred patient   → x-axis
                    y_pos = N_PATIENTS - 1 - i    # flip so AD+CI at top
                    ax.scatter(j, y_pos,
                               s=sizes[i, j],
                               color=cmap(norm(dev_mat[i, j])),
                               alpha=0.85, linewidths=0)

            # Group separator lines
            for b in boundaries[1:-1]:
                ax.axvline(x=b - 0.5, color="black", lw=0.8, alpha=0.6)
                ax.axhline(y=N_PATIENTS - b - 0.5, color="black", lw=0.8, alpha=0.6)

            # Axis ticks at group centres
            ax.set_xticks(group_centers)
            ax.set_xticklabels(GROUP_ORDER, fontsize=10)
            ax.set_yticks([N_PATIENTS - 1 - c for c in group_centers])
            ax.set_yticklabels(list(reversed(GROUP_ORDER)), fontsize=10)
            ax.set_xlabel("Pred Group", fontsize=11)
            ax.set_ylabel("Train Group", fontsize=11)
            ax.set_title(panel, fontsize=11)
            ax.set_xlim(-0.5, N_PATIENTS - 0.5)
            ax.set_ylim(-0.5, N_PATIENTS - 0.5)
            ax.set_aspect("equal")

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.7)
            cb.set_label(panel.split()[-1] + " deviation", fontsize=9)
            cb.ax.tick_params(labelsize=8)

        # Dot size legend (top-right of right panel)
        for logp_val, label in [(0.5, "0.5"), (1.0, "1.0"), (1.5, "1.5"), (2.0, "2.0")]:
            s = logp_val / 3.0 * size_max_pt + 6
            axes[1].scatter([], [], s=s, color="grey", alpha=0.8, label=label)
        axes[1].legend(title="-log₁₀ p", fontsize=8, title_fontsize=8,
                       loc="lower right", frameon=True)

        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "fig3ab_cross_disease_precision_recall.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_metrics       = {}
    all_cross_metrics = {}

    for histone in HISTONES:
        print(f"\nLoading results for {histone}...")
        try:
            preds, labels = load_test_results(histone)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Predictions: {preds.shape}, Labels: {labels.shape}")
        all_metrics[histone] = per_patient_metrics(preds, labels)

        print(f"  Computing 22×22 cross-patient metrics...")
        all_cross_metrics[histone] = compute_cross_patient_metrics(preds, labels)

        for m in ["AUROC", "AUPRC", "F1", "ACC"]:
            vals = [v for v in all_metrics[histone][m] if not np.isnan(v)]
            print(f"  {m}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    if not all_metrics:
        print("\nNo results found. Run the test jobs on SCC first.")
        return

    print("\nGenerating Figure 2 heatmap...")
    plot_fig2(all_metrics)

    print("Generating Figure 3 bubble plot...")
    plot_fig3ab(all_cross_metrics)

    print("\nDone. Figures saved to ./figures/")


if __name__ == "__main__":
    main()
