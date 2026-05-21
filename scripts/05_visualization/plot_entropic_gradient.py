"""
Entropic Gradient — Sorted Condition Raincloud Plots
=====================================================
Generates Figure 4 (SE dSW) and Supplementary Figure 1 (SE dFC).

All 12 conditions (excluding ayahuasca) are ordered by their group mean
SE value, producing a continuous entropic gradient from deep anaesthesia
to schizophrenia.

Usage:
    python plot_entropic_gradient.py          # generates both dSW and dFC
    python plot_entropic_gradient.py dSW      # Figure 4 only
    python plot_entropic_gradient.py dFC      # Supplementary Figure 1 only
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# ============================================================
# Plot settings
# ============================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ============================================================
# Paths
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))
out_dir = os.path.join(main_dir, "results/paper_figures")
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Colour palette (one colour per dataset family)
# ============================================================
dataset_color = {
    "anestesia":  "#E69F00",
    "dmt":        "#CC79A7",
    "lsd":        "#009E73",
    "modafinil":  "#F0E442",
    "ucla":       "#0072B2",
}

def get_family(dataset_label):
    """Map a condition label to its dataset family key."""
    for key in dataset_color:
        if key in dataset_label:
            return key
    return "ucla"

# ============================================================
# Condition labels used in the AAL CSVs
# ============================================================
# Mapping: raw CSV label -> display label
dsw_label_map = {
    "anestesia_block1": "Wake",
    "anestesia_block2": "Sedation",
    "anestesia_block3": "Deep",
    "anestesia_block4": "Recovery",
    "dmt_dmt":          "DMT",
    "dmt_pcb":          "Placebo",
    "lsd":              "LSD",
    "lsd_plcb":         "Placebo",
    "modafinil_condition1": "Placebo",
    "modafinil_condition2": "Modafinil",
    "ucla_schz":        "Schizophrenia",
    "ucla_control":     "Control",
}

dfc_label_map = {
    "anestesia_block1": "Wake",
    "anestesia_block2": "Sedation",
    "anestesia_block3": "Deep",
    "anestesia_block4": "Recovery",
    "dmt_dmt":          "DMT",
    "dmt_plcb":         "Placebo",
    "lsd_lsd":          "LSD",
    "lsd_plcb":         "Placebo",
    "modafinil":        "Modafinil",
    "modafinil_placebo": "Placebo",
    "ucla_schz":        "Schizophrenia",
    "ucla_control":     "Control",
}


def load_data(metric):
    """Load the appropriate AAL CSV and filter to the 12 conditions."""
    if metric == "dSW":
        path = os.path.join(main_dir, "data/AAL/all_data.csv")
        label_map = dsw_label_map
    else:
        path = os.path.join(main_dir, "data/AAL/dFC_all_data.csv")
        label_map = dfc_label_map

    df = pd.read_csv(path)
    # Keep only the 12 conditions (exclude ayahuasca)
    df = df[df["dataset"].isin(label_map.keys())].copy()
    return df, label_map


def plot_gradient(metric):
    """Generate the sorted-condition raincloud plot for a given metric."""
    df, label_map = load_data(metric)

    # Sort conditions by group mean
    means_per_cond = df.groupby("dataset")["SampEn"].mean().sort_values()
    conditions = means_per_cond.index.tolist()

    series_list = []
    labels = []
    colors = []

    for cond in conditions:
        s = df[df["dataset"] == cond]["SampEn"]
        series_list.append(s)
        labels.append(label_map[cond])
        colors.append(dataset_color[get_family(cond)])

    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)

    # Half-violin (right side only)
    parts = ax.violinplot(series_list, positions=x, widths=0.8,
                          showmeans=False, showextrema=False)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.3)
        body.set_edgecolor("none")
        # Cut left half
        verts = body.get_paths()[0].vertices
        xm = verts[:, 0].mean()
        verts[:, 0] = np.maximum(verts[:, 0], xm)

    # Jittered dots
    np.random.seed(42)
    for i, s in enumerate(series_list):
        if len(s) == 0:
            continue
        jitter = (np.random.rand(len(s)) - 0.5) * 0.15
        ax.scatter(x[i] + jitter, s, s=40, alpha=0.6,
                   color=colors[i], edgecolors="none", zorder=2)

    # Mean + SEM
    m = [s.mean() for s in series_list]
    se = [s.std() / np.sqrt(len(s)) for s in series_list]
    ax.errorbar(x, m, yerr=se, fmt="none", ecolor="black",
                capsize=8, lw=2, zorder=3)
    ax.scatter(x, m, s=80, color="black", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("SE" if metric == "dSW" else "SE (dFC)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend (colour per family)
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=dataset_color["anestesia"], label="Anaesthesia"),
        Patch(facecolor=dataset_color["modafinil"], label="Modafinil"),
        Patch(facecolor=dataset_color["lsd"],       label="LSD"),
        Patch(facecolor=dataset_color["dmt"],       label="DMT"),
        Patch(facecolor=dataset_color["ucla"],       label="Schizophrenia"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False, fontsize=11)

    # Save
    tag = "Figure4_entropic_gradient" if metric == "dSW" else "FigureS1_entropic_gradient_dFC"
    for ext in ["png", "svg"]:
        ax.figure.savefig(os.path.join(out_dir, f"{tag}.{ext}"),
                          dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {tag} (.png, .svg)")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    metrics = sys.argv[1:] if len(sys.argv) > 1 else ["dSW", "dFC"]
    for m in metrics:
        plot_gradient(m)
