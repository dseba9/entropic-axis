"""
Per-Dataset Raincloud Plots of Differences — Delta SE by condition
=====================================================================
Generates Figure 2 (SE dSW) and Figure 3 (SE dFC) using difference metrics
relative to control.

Each panel shows the distribution of condition-specific changes in sample
entropy (active minus control) to normalize baseline scanner differences.

Usage:
    python plot_dataset_rainclouds.py          # both dSW and dFC
    python plot_dataset_rainclouds.py dSW      # Figure 2 only
    python plot_dataset_rainclouds.py dFC      # Figure 3 only
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Plot settings
# ============================================================
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 15
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# ============================================================
# Paths
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))
out_dir = os.path.join(main_dir, "results/paper_figures")
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Colour palette
# ============================================================
dataset_color = {
    "anestesia":  "#E69F00",
    "dmt":        "#CC79A7",
    "lsd":        "#009E73",
    "modafinil":  "#F0E442",
    "ucla":       "#0072B2",
}

def get_family(label):
    for key in dataset_color:
        if key in label:
            return key
    return "ucla"

def significance_symbol(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

panels_dsw = [
    {
        "title": "Anesthesia",
        "conditions": ["anestesia_block1", "anestesia_block2",
                        "anestesia_block3", "anestesia_block4"],
        "labels": ["Sedation", "Deep", "Recovery"],
        "paired": True,
        "panel_letter": "A",
    },
    {
        "title": "Modafinil",
        "conditions": ["modafinil_condition1", "modafinil_condition2"],
        "labels": ["Modafinil"],
        "paired": True,
        "panel_letter": "B",
    },
    {
        "title": "LSD",
        "conditions": ["lsd_plcb", "lsd"],
        "labels": ["LSD"],
        "paired": True,
        "panel_letter": "C",
    },
    {
        "title": "DMT",
        "conditions": ["dmt_pcb", "dmt_dmt"],
        "labels": ["DMT"],
        "paired": True,
        "panel_letter": "D",
    },
    {
        "title": "Schizophrenia",
        "conditions": ["ucla_control", "ucla_schz"],
        "labels": ["Schizophrenia"],
        "paired": False,
        "panel_letter": "E",
    },
]

panels_dfc = [
    {
        "title": "Anesthesia",
        "conditions": ["anestesia_block1", "anestesia_block2",
                        "anestesia_block3", "anestesia_block4"],
        "labels": ["Sedation", "Deep", "Recovery"],
        "paired": True,
        "panel_letter": "A",
    },
    {
        "title": "Modafinil",
        "conditions": ["modafinil_placebo", "modafinil"],
        "labels": ["Modafinil"],
        "paired": True,
        "panel_letter": "B",
    },
    {
        "title": "LSD",
        "conditions": ["lsd_plcb", "lsd_lsd"],
        "labels": ["LSD"],
        "paired": True,
        "panel_letter": "C",
    },
    {
        "title": "DMT",
        "conditions": ["dmt_plcb", "dmt_dmt"],
        "labels": ["DMT"],
        "paired": True,
        "panel_letter": "D",
    },
    {
        "title": "Schizophrenia",
        "conditions": ["ucla_control", "ucla_schz"],
        "labels": ["Schizophrenia"],
        "paired": False,
        "panel_letter": "E",
    },
]


def plot_panel_diff(ax, df, panel):
    """Draw a single panel showing differences from control."""
    conditions = panel["conditions"]
    labels = panel["labels"]
    is_paired = panel["paired"]

    ctrl_cond = conditions[0]
    series_list = []

    # Calculate differences for each active condition
    for i in range(1, len(conditions)):
        act_cond = conditions[i]
        ctrl_df = df[df["dataset"] == ctrl_cond].set_index("Subject")["SampEn"]
        act_df = df[df["dataset"] == act_cond].set_index("Subject")["SampEn"]
        common = ctrl_df.index.intersection(act_df.index)
        
        if len(common) > 0 and is_paired:
            diffs = act_df.loc[common] - ctrl_df.loc[common]
        else:
            ctrl_mean = df[df["dataset"] == ctrl_cond]["SampEn"].mean()
            diffs = act_df.reset_index(drop=True) - ctrl_mean
            
        diffs = diffs.dropna()
        series_list.append(diffs)

    x = np.arange(len(series_list))

    # Reference line at 0
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    # Half-violins
    parts = ax.violinplot(series_list, positions=x, widths=0.6 if len(series_list) > 1 else 0.4,
                          showmeans=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        col = dataset_color[get_family(conditions[i+1])]
        body.set_facecolor(col)
        body.set_alpha(0.25)
        body.set_edgecolor("none")
        verts = body.get_paths()[0].vertices
        xm = verts[:, 0].mean()
        verts[:, 0] = np.maximum(verts[:, 0], xm)

    # Jittered dots
    np.random.seed(42)
    for i, s in enumerate(series_list):
        if len(s) == 0:
            continue
        col = dataset_color[get_family(conditions[i+1])]
        jitter = (np.random.rand(len(s)) - 0.5) * (0.12 if len(series_list) > 1 else 0.08)
        ax.scatter(x[i] + jitter, s, s=60, alpha=0.6,
                   color=col, edgecolors="none", zorder=2)

    # Mean + SEM
    means = [s.mean() for s in series_list]
    sems = [s.std() / np.sqrt(len(s)) for s in series_list]
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="black",
                capsize=8, lw=2, zorder=3)
    ax.scatter(x, means, s=100, color="black", zorder=4)

    # Statistics and annotations
    for i in range(len(series_list)):
        s = series_list[i]
        s0 = df[df["dataset"] == ctrl_cond]["SampEn"].values
        si = df[df["dataset"] == conditions[i+1]]["SampEn"].values
        try:
            if is_paired:
                n = min(len(s0), len(si))
                _, p = wilcoxon(s0[:n], si[:n])
            else:
                _, p = mannwhitneyu(s0, si, alternative="two-sided")
        except Exception:
            p = 1.0
            
        if p < 0.05:
            sym = significance_symbol(p)
            max_val = max(s.max(), 0)
            y_pos = max_val + 0.08 * (s.max() - s.min() if s.max() != s.min() else 1.0)
            ax.text(i, y_pos, sym, ha="center", va="bottom",
                    color="black", fontsize=18, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_title(panel['panel_letter'], loc='left',
                 fontsize=19, fontweight="bold", pad=12)
    ax.set_ylabel(r"$\Delta$SE", fontsize=17, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(series_list) == 1:
        ax.set_xlim(-0.6, 0.6)
    else:
        ax.set_xlim(-0.5, len(series_list) - 0.5)


def generate_figure(metric):
    """Generate the 5-panel differences figure."""
    if metric == "dSW":
        path = os.path.join(main_dir, "data/AAL/all_data.csv")
        panels = panels_dsw
    else:
        path = os.path.join(main_dir, "data/AAL/dFC_all_data.csv")
        panels = panels_dfc

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 5, figsize=(18, 6), constrained_layout=True)
    for i, panel in enumerate(panels):
        plot_panel_diff(axes[i], df, panel)

    tag = f"Figure{2 if metric == 'dSW' else 3}_rainclouds_{metric}"
    for ext in ["png", "svg"]:
        fig.savefig(os.path.join(out_dir, f"{tag}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {tag} (.png, .svg)")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    metrics = sys.argv[1:] if len(sys.argv) > 1 else ["dSW", "dFC"]
    for m in metrics:
        generate_figure(m)
