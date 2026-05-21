import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))

NETWORKS = ["Visual", "Somatomotor", "DorsalAttn", "SalVentAttn",
            "Limbic", "FrontoParietal", "DefaultMode"]
NET_SHORT = ["VIS", "SM", "DAN", "SAL", "LIM", "FP", "DM"]

# Define datasets and conditions
# Each tuple: (label, dataset_key, active_cond, control_cond, paired)
DATASETS = [
    ("LSD",          "LSD",          "LSD",          "PLB",   True,  "#c0392b"),
    ("DMT",          "DMT",          "DMT",          "PLB",   True,  "#e67e22"),
    ("Anesthesia (U)", "Anestesia", "Unconscious",  "Awake", True,  "#8e44ad"),
    ("Anesthesia (S)", "Anestesia", "Sedation",     "Awake", True,  "#9b59b6"),
    ("Schizophrenia", "Schiz",      "SCHZ",         "CTRL",  False, "#2c3e50"),
    ("Modafinil",     "Modafinil",  "MOD",          "PLB",   True,  "#27ae60"),
]

# Paths for dSW (called SE in filenames) and dFC
BASE_PATH = os.path.join(main_dir, "data/LONO_CSVs")
PATHS_DSW = {
    "LSD":       os.path.join(BASE_PATH, "LSD_SE_LONO_SubjectCondition.csv"),
    "DMT":       os.path.join(BASE_PATH, "DMT_SE_LONO_SubjectCondition.csv"),
    "Modafinil": os.path.join(BASE_PATH, "Modafinil_SE_LONO_SubjectCondition.csv"),
    "Anestesia": os.path.join(BASE_PATH, "Anestesia_SE_LONO_SubjectCondition.csv"),
    "Schiz":     os.path.join(BASE_PATH, "Schizophrenia_SE_LONO_SubjectCondition.csv"),
}

PATHS_DFC = {
    "LSD":       os.path.join(BASE_PATH, "LSD_dFC_SE_LONO_SubjectCondition.csv"),
    "DMT":       os.path.join(BASE_PATH, "DMT_dFC_SE_LONO_SubjectCondition.csv"),
    "Modafinil": os.path.join(BASE_PATH, "Modafinil_dFC_SE_LONO_SubjectCondition.csv"),
    "Anestesia": os.path.join(BASE_PATH, "Anestesia_dFC_SE_LONO_SubjectCondition.csv"),
    "Schiz":     os.path.join(BASE_PATH, "Schizophrenia_dFC_SE_LONO_SubjectCondition.csv"),
}

# --- Statistics Functions ---

def fdr_bh(ps):
    """Benjamini-Hochberg FDR correction."""
    ps = np.array(ps)
    n = len(ps)
    order = np.argsort(ps)
    p_adj = np.empty(n)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(1.0, ps[idx] * n / (rank + 1))
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj

def rank_biserial_wilcoxon(d):
    """Effect size for Wilcoxon signed-rank test."""
    d = d[d != 0]
    if len(d) == 0: return 0
    ranks = stats.rankdata(np.abs(d))
    pos_sum = np.sum(ranks[d > 0])
    neg_sum = np.sum(ranks[d < 0])
    n = len(d)
    total_sum = n * (n + 1) / 2
    return (pos_sum - neg_sum) / total_sum

def rank_biserial_mannwhitney(a, b):
    """Effect size for Mann-Whitney U test."""
    u, p = stats.mannwhitneyu(a, b)
    n1, n2 = len(a), len(b)
    return 1 - (2 * u / (n1 * n2))

def load_and_prep(path):
    """Loads CSV and calculates SDI = Complete - Network."""
    if not os.path.exists(path):
        print(f"Warning: File {path} not found.")
        return None
    df = pd.read_csv(path, index_col=0).dropna(how="all")
    parts = [idx.split("_") for idx in df.index]
    df["subject"]   = ["_".join(p[:2]) for p in parts]
    df["condition"] = ["_".join(p[2:]) for p in parts]
    for net in NETWORKS:
        df[f"SDI_{net}"] = df["Complete"] - df[net]
    return df

def analyze_dataset(df, cond_a, cond_b, paired):
    """Computes stats and deltas for all networks."""
    deltas, sems, raw_ps, effect_sizes = [], [], [], []
    
    for net in NETWORKS:
        col = f"SDI_{net}"
        if paired:
            a = df[df.condition == cond_a].set_index("subject")[col]
            b = df[df.condition == cond_b].set_index("subject")[col]
            common = a.index.intersection(b.index)
            d = (a - b).loc[common]
            stat, p = stats.wilcoxon(d)
            r = rank_biserial_wilcoxon(d)
            deltas.append(d.mean())
            sems.append(d.std(ddof=1) / np.sqrt(len(d)))
            raw_ps.append(p)
            effect_sizes.append(r)
        else:
            a = df[df.condition == cond_a][col].dropna().values
            b = df[df.condition == cond_b][col].dropna().values
            stat, p = stats.mannwhitneyu(a, b)
            r = rank_biserial_mannwhitney(a, b)
            deltas.append(np.mean(a) - np.mean(b))
            sems.append(np.sqrt(np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b)))
            raw_ps.append(p)
            effect_sizes.append(r)
            
    return {
        "deltas": np.array(deltas),
        "sems": np.array(sems),
        "p_raw": np.array(raw_ps),
        "p_fdr": fdr_bh(raw_ps),
        "r": np.array(effect_sizes)
    }

def main():
    # Process dSW
    print("\n--- ANALYZING dSW (Topological Complexity) ---")
    results_dsw = []
    for label, ds_key, cond_a, cond_b, paired, color in DATASETS:
        df = load_and_prep(PATHS_DSW[ds_key])
        if df is not None:
            # Check if conditions exist
            if cond_a not in df.condition.unique() or cond_b not in df.condition.unique():
                print(f"Skipping {label}: conditions {cond_a} or {cond_b} not found.")
                continue
                
            stats_res = analyze_dataset(df, cond_a, cond_b, paired)
            n = df[df.condition == cond_a].subject.nunique()
            res = {"label": label, "color": color, "n": n, **stats_res}
            results_dsw.append(res)
            
            print(f"\n>> {label} ({cond_a} vs {cond_b}, n={n})")
            print(f"  {'Network':<15} {'ΔSDI':>10} {'p_raw':>10} {'p_fdr':>10} {'r':>10}")
            for i, net in enumerate(NETWORKS):
                p_fdr = stats_res['p_fdr'][i]
                p_raw = stats_res['p_raw'][i]
                sig = "***" if p_fdr < .001 else "**" if p_fdr < .01 else "*" if p_fdr < .05 else "†" if p_raw < .05 else ""
                print(f"  {net:<15} {stats_res['deltas'][i]:>+10.4f} {p_raw:>10.4f} {p_fdr:>10.4f} {stats_res['r'][i]:>10.4f} {sig}")

    # Process dFC
    print("\n\n--- ANALYZING dFC (Connectivity Magnitude) ---")
    results_dfc = []
    for label, ds_key, cond_a, cond_b, paired, color in DATASETS:
        df = load_and_prep(PATHS_DFC[ds_key])
        if df is not None:
            # Check if conditions exist
            if cond_a not in df.condition.unique() or cond_b not in df.condition.unique():
                print(f"Skipping {label}: conditions {cond_a} or {cond_b} not found.")
                continue
                
            stats_res = analyze_dataset(df, cond_a, cond_b, paired)
            n = df[df.condition == cond_a].subject.nunique()
            res = {"label": label, "color": color, "n": n, **stats_res}
            results_dfc.append(res)
            
            print(f"\n>> {label} ({cond_a} vs {cond_b}, n={n})")
            print(f"  {'Network':<15} {'ΔSDI':>10} {'p_raw':>10} {'p_fdr':>10} {'r':>10}")
            for i, net in enumerate(NETWORKS):
                p_fdr = stats_res['p_fdr'][i]
                p_raw = stats_res['p_raw'][i]
                sig = "***" if p_fdr < .001 else "**" if p_fdr < .01 else "*" if p_fdr < .05 else "†" if p_raw < .05 else ""
                print(f"  {net:<15} {stats_res['deltas'][i]:>+10.4f} {p_raw:>10.4f} {p_fdr:>10.4f} {stats_res['r'][i]:>10.4f} {sig}")


if __name__ == "__main__":
    main()
