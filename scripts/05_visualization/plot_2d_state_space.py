"""
2D State Space — SE dSW vs SE dFC
===================================
Generates Figure 5 (AAL parcellation) and Supplementary Figure 2
(Tian-Schaefer parcellation).

Each condition's mean SE values are z-scored relative to the wakefulness
baseline within each dataset. Condition means are plotted as coloured
markers with SEM error bars and displacement vectors from the origin
(wakefulness).

Usage:
    python plot_2d_state_space.py             # both parcellations
    python plot_2d_state_space.py AAL         # Figure 5 only
    python plot_2d_state_space.py Tian        # Supplementary Figure 2 only
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

# ============================================================
# Study definitions
# ============================================================
# Each study: control condition, list of active conditions to plot,
# and label offsets for annotation placement.
studies_dsw = {
    "Anesthesia": {
        "control": "anestesia_block1",
        "conditions": {
            "anestesia_block2": {"label": "Sedation", "dx": -0.5, "dy": -0.2},
            "anestesia_block3": {"label": "Deep",     "dx": -0.4, "dy": -0.3},
        },
    },
    "LSD": {
        "control": "lsd_plcb",
        "conditions": {
            "lsd": {"label": "LSD", "dx": -0.4, "dy": 0.2},
        },
    },
    "DMT": {
        "control": "dmt_pcb",
        "conditions": {
            "dmt_dmt": {"label": "DMT", "dx": 0.1, "dy": -0.1},
        },
    },
    "Modafinil": {
        "control": "modafinil_condition1",
        "conditions": {
            "modafinil_condition2": {"label": "Modafinil", "dx": 0.15, "dy": 0.15},
        },
    },
    "Schizophrenia": {
        "control": "ucla_control",
        "conditions": {
            "ucla_schz": {"label": "Schizophrenia", "dx": 0.1, "dy": 0.1},
        },
    },
}

studies_dfc = {
    "Anesthesia": {
        "control": "anestesia_block1",
        "conditions": {
            "anestesia_block2": {"label": "Sedation", "dx": -0.5, "dy": -0.2},
            "anestesia_block3": {"label": "Deep",     "dx": -0.4, "dy": -0.3},
        },
    },
    "LSD": {
        "control": "lsd_plcb",
        "conditions": {
            "lsd_lsd": {"label": "LSD", "dx": -0.4, "dy": 0.2},
        },
    },
    "DMT": {
        "control": "dmt_plcb",
        "conditions": {
            "dmt_dmt": {"label": "DMT", "dx": 0.1, "dy": -0.1},
        },
    },
    "Modafinil": {
        "control": "modafinil_placebo",
        "conditions": {
            "modafinil": {"label": "Modafinil", "dx": 0.15, "dy": 0.15},
        },
    },
    "Schizophrenia": {
        "control": "ucla_control",
        "conditions": {
            "ucla_schz": {"label": "Schizophrenia", "dx": 0.1, "dy": 0.1},
        },
    },
}

# Tian-Schaefer labels (dSW)
studies_tian_dsw = {
    "Anesthesia": {
        "control": "anestesia_baseline",
        "conditions": {
            "anestesia_light": {"label": "Sedation", "dx": -0.5, "dy": -0.2},
            "anestesia_deep":  {"label": "Deep",     "dx": -0.4, "dy": -0.3},
        },
    },
    "LSD": {
        "control": "lsd_plcb",
        "conditions": {
            "lsd_lsd": {"label": "LSD", "dx": -0.4, "dy": 0.2},
        },
    },
    "DMT": {
        "control": "dmt_plcb",
        "conditions": {
            "dmt_dmt": {"label": "DMT", "dx": 0.1, "dy": -0.1},
        },
    },
    "Modafinil": {
        "control": "placebo_modafinil",
        "conditions": {
            "modafinil": {"label": "Modafinil", "dx": 0.15, "dy": 0.15},
        },
    },
    "Schizophrenia": {
        "control": "ucla_control",
        "conditions": {
            "ucla_schz": {"label": "Schizophrenia", "dx": 0.1, "dy": 0.1},
        },
    },
}

# Tian-Schaefer labels (dFC)
studies_tian_dfc = {
    "Anesthesia": {
        "control": "anestesia_block1",
        "conditions": {
            "anestesia_block2": {"label": "Sedation", "dx": -0.5, "dy": -0.2},
            "anestesia_block3": {"label": "Deep",     "dx": -0.4, "dy": -0.3},
        },
    },
    "LSD": {
        "control": "lsd_plcb",
        "conditions": {
            "lsd_lsd": {"label": "LSD", "dx": -0.4, "dy": 0.2},
        },
    },
    "DMT": {
        "control": "dmt_plcb",
        "conditions": {
            "dmt_dmt": {"label": "DMT", "dx": 0.1, "dy": -0.1},
        },
    },
    "Modafinil": {
        "control": "modafinil_placebo",
        "conditions": {
            "modafinil": {"label": "Modafinil", "dx": 0.15, "dy": 0.15},
        },
    },
    "Schizophrenia": {
        "control": "ucla_control",
        "conditions": {
            "ucla_schz": {"label": "Schizophrenia", "dx": 0.1, "dy": 0.1},
        },
    },
}


def load_and_zscore(dsw_path, dfc_path, studies_dsw_map, studies_dfc_map):
    """Load two CSVs, z-score each condition within its study, return merged stats."""
    df_dsw = pd.read_csv(dsw_path)
    df_dfc = pd.read_csv(dfc_path)

    records = []

    for study_name in studies_dsw_map:
        s_dsw = studies_dsw_map[study_name]
        s_dfc = studies_dfc_map[study_name]

        ctrl_dsw = s_dsw["control"]
        ctrl_dfc = s_dfc["control"]

        # Z-score parameters from control
        ctrl_dsw_vals = df_dsw[df_dsw["dataset"] == ctrl_dsw]["SampEn"]
        ctrl_dfc_vals = df_dfc[df_dfc["dataset"] == ctrl_dfc]["SampEn"]

        if len(ctrl_dsw_vals) == 0 or len(ctrl_dfc_vals) == 0:
            print(f"  Warning: no control data for {study_name}, skipping")
            continue

        mu_dsw, std_dsw = ctrl_dsw_vals.mean(), ctrl_dsw_vals.std()
        mu_dfc, std_dfc = ctrl_dfc_vals.mean(), ctrl_dfc_vals.std()

        if std_dsw == 0:
            std_dsw = 1.0
        if std_dfc == 0:
            std_dfc = 1.0

        for cond_dsw, info in s_dsw["conditions"].items():
            label = info["label"]
            dx = info["dx"]
            dy = info["dy"]

            # Find matching dFC condition
            cond_dfc = None
            for c in s_dfc["conditions"]:
                if s_dfc["conditions"][c]["label"] == label:
                    cond_dfc = c
                    break
            if cond_dfc is None:
                continue

            vals_dsw = df_dsw[df_dsw["dataset"] == cond_dsw]["SampEn"]
            vals_dfc = df_dfc[df_dfc["dataset"] == cond_dfc]["SampEn"]

            if len(vals_dsw) == 0 or len(vals_dfc) == 0:
                continue

            z_dsw = (vals_dsw - mu_dsw) / std_dsw
            z_dfc = (vals_dfc - mu_dfc) / std_dfc

            records.append({
                "label": label,
                "family": get_family(cond_dsw),
                "mean_dsw": z_dsw.mean(),
                "sem_dsw":  z_dsw.std() / np.sqrt(len(z_dsw)),
                "mean_dfc": z_dfc.mean(),
                "sem_dfc":  z_dfc.std() / np.sqrt(len(z_dfc)),
                "dx": dx,
                "dy": dy,
            })

    return pd.DataFrame(records)


def plot_2d(stats_df, output_tag):
    """Plot the 2D state space scatter."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Reference lines
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.4, linewidth=1)

    # Baseline point
    ax.errorbar(0, 0, xerr=0, yerr=0, fmt="o", color="black", ms=12, zorder=10)
    ax.text(0.1, 0.1, "Wake", fontsize=12, fontweight="bold", ha="left", va="bottom")

    for _, row in stats_df.iterrows():
        x = row["mean_dfc"]
        y = row["mean_dsw"]
        xe = row["sem_dfc"]
        ye = row["sem_dsw"]
        color = dataset_color[row["family"]]
        label = row["label"]

        # Arrow from origin
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.4, lw=2.5))

        # Point with error bars
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color=color,
                    ms=12, capsize=5, elinewidth=2, markeredgecolor="none")

        # Label
        ax.text(x + row["dx"], y + row["dy"], label,
                color=color, fontweight="bold", fontsize=12)

    ax.set_xlabel("SE dFC (Z-Score)", fontsize=14)
    ax.set_ylabel("SE dSW (Z-Score)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.set_xlim(-1.8, 1.5)
    ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(os.path.join(out_dir, f"{output_tag}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_tag} (.png, .svg)")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["AAL", "Tian"]

    if "AAL" in targets:
        dsw_path = os.path.join(main_dir, "data/AAL/all_data.csv")
        dfc_path = os.path.join(main_dir, "data/AAL/dFC_all_data.csv")
        stats = load_and_zscore(dsw_path, dfc_path, studies_dsw, studies_dfc)
        plot_2d(stats, "Figure5_2D_state_space")

    if "Tian" in targets:
        dsw_path = os.path.join(main_dir, "data/Tian_Schaefer/dSW_all_data.csv")
        dfc_path = os.path.join(main_dir, "data/Tian_Schaefer/dFC_all_data.csv")
        # Check Tian-Schaefer condition labels
        tian_dsw = pd.read_csv(dsw_path)
        tian_dfc = pd.read_csv(dfc_path)
        print("Tian dSW conditions:", sorted(tian_dsw["dataset"].unique()))
        print("Tian dFC conditions:", sorted(tian_dfc["dataset"].unique()))
        stats = load_and_zscore(dsw_path, dfc_path, studies_tian_dsw, studies_tian_dfc)
        if len(stats) > 0:
            plot_2d(stats, "FigureS2_2D_state_space_Tian")
        else:
            print("Warning: no data matched for Tian-Schaefer. Check condition labels.")
