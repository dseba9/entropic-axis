import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300

metric = sys.argv[1] if len(sys.argv) > 1 else "dSW"

script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))
csv_dir = os.path.join(main_dir, "data/LONO_CSVs/")
out_dir = os.path.join(main_dir, "results/paper_figures/")

if metric == "dSW":
    prefix = ""
    output_pdf = os.path.join(out_dir, "sdi_rainclouds_dSW.pdf")
    output_png = os.path.join(out_dir, "sdi_rainclouds_dSW.png")
    y_label = "ΔSE (dSW)  (active − control)"
else:
    prefix = "_dFC"
    output_pdf = os.path.join(out_dir, "sdi_rainclouds_dFC.pdf")
    output_png = os.path.join(out_dir, "sdi_rainclouds_dFC.png")
    y_label = "ΔSE (dFC)  (active − control)"

PATHS = {
    "LSD":       f"{csv_dir}LSD{prefix}_SE_LONO_SubjectCondition.csv",
    "DMT":       f"{csv_dir}DMT{prefix}_SE_LONO_SubjectCondition.csv",
    "Anestesia": f"{csv_dir}Anestesia{prefix}_SE_LONO_SubjectCondition.csv",
    "Schiz":     f"{csv_dir}Schizophrenia{prefix}_SE_LONO_SubjectCondition.csv",
    "Modafinil": f"{csv_dir}Modafinil{prefix}_SE_LONO_SubjectCondition.csv",
}

NETWORKS  = ["Visual", "Somatomotor", "DorsalAttn", "SalVentAttn", "Limbic", "FrontoParietal", "DefaultMode"]
NET_SHORT = ["VIS", "SM", "DAN", "SAL", "LIM", "FP", "DM"]

PANELS = [
    ("LSD",       "LSD",          "PLB",   True,  "#009E73", "A  LSD"),
    ("DMT",       "DMT",          "PLB",   True,  "#CC79A7", "B  DMT"),
    ("Anestesia", "Unconscious",  "Awake", True,  "#E69F00", "C  Anesthesia"),
    ("Schiz",     "SCHZ",         "CTRL",  False, "#0072B2", "D  Schizophrenia"),
    ("Modafinil", "MOD",          "PLB",   True,  "#F0E442", "E  Modafinil"),
]

EXCLUSIONS = {
    "DMT": ["Sub_1", "Sub_6", "Sub_7", "Sub_12", "Sub_14", "Sub_18"]
}

def load(ds, path):
    df = pd.read_csv(path, index_col=0).dropna(how="all").dropna(how="any")
    parts = [idx.split("_") for idx in df.index]
    df["subject"]   = ["_".join(p[:2]) for p in parts]
    df["condition"] = ["_".join(p[2:]) for p in parts]
    if ds in EXCLUSIONS:
        df = df[~df["subject"].isin(EXCLUSIONS[ds])]
    for net in NETWORKS:
        df[f"SDI_{net}"] = df["Complete"] - df[net]
    return df

def fdr_bh(ps):
    ps = np.array(ps)
    n = len(ps)
    order = np.argsort(ps)
    p_adj = np.empty(n)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(1.0, ps[idx] * n / (rank + 1))
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj

def plot_custom_raincloud(ax, df_plot, color):
    series_list = []
    labels = NET_SHORT
    
    for net in NET_SHORT:
        s = df_plot[df_plot['Network'] == net]['Delta'].values
        series_list.append(s)
        
    x = np.arange(len(NET_SHORT))
    
    valid_series = [s for s in series_list if len(s) > 0]
    if not valid_series: return
        
    # 1. Half-Violin (Right)
    parts = ax.violinplot(series_list, positions=x, widths=0.7, showmeans=False, showextrema=False)
    
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(color)
        b.set_alpha(0.4)
        b.set_edgecolor("none")
        verts = b.get_paths()[0].vertices
        xm = verts[:,0].mean()
        verts[:,0] = np.maximum(verts[:,0], xm)

    # 2. Jitter Dots (Center/Leftish)
    for i, s in enumerate(series_list):
        if len(s) == 0: continue
        jitter = (np.random.rand(len(s)) - 0.5) * 0.15 - 0.1
        ax.scatter(x[i] + jitter, s, s=25, alpha=0.6, color=color, edgecolors='none', zorder=2)

    # 3. Mean + ErrorBar (Black)
    means = [s.mean() if len(s)>0 else np.nan for s in series_list]
    sems  = [s.std(ddof=1)/np.sqrt(len(s)) if len(s)>0 else np.nan for s in series_list]
    
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black', capsize=4, lw=1.5, zorder=3)
    ax.scatter(x, means, s=30, color='black', zorder=4)

def main():
    dfs = {key: load(key, path) for key, path in PATHS.items()}
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 7), sharey=False)
    fig.subplots_adjust(wspace=0.4, left=0.06, right=0.98, top=0.8, bottom=0.25)
    
    for ax, (ds, cond_a, cond_b, paired, color, label) in zip(axes, PANELS):
        df = dfs[ds]
        n = df[df.condition == cond_a].subject.nunique()
        
        plot_data = []
        ps_raw = []
        means = []
        
        for net, net_short in zip(NETWORKS, NET_SHORT):
            col = f"SDI_{net}"
            if paired:
                a = df[df.condition == cond_a].set_index("subject")[col]
                b = df[df.condition == cond_b].set_index("subject")[col]
                common = a.index.intersection(b.index)
                if len(common) == 0:
                    means.append(np.nan)
                    ps_raw.append(1.0)
                    continue
                d = (a - b).loc[common]
                stat, p = stats.wilcoxon(d)
                means.append(d.mean())
                for val in d.values:
                    plot_data.append({"Network": net_short, "Delta": val})
            else:
                a = df[df.condition == cond_a][col].values
                b = df[df.condition == cond_b][col].values
                if len(a) == 0 or len(b) == 0:
                    means.append(np.nan)
                    ps_raw.append(1.0)
                    continue
                stat, p = stats.mannwhitneyu(a, b)
                mean_diff = np.mean(a) - np.mean(b)
                means.append(mean_diff)
                for val in a:
                    plot_data.append({"Network": net_short, "Delta": val - np.mean(b)})
            ps_raw.append(p)
            
        ps_fdr = fdr_bh(ps_raw)
        df_plot = pd.DataFrame(plot_data)
        
        plot_custom_raincloud(ax, df_plot, color)
        
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        
        ymax = df_plot["Delta"].max() * 1.3 if len(df_plot) > 0 and df_plot["Delta"].max() > 0 else 0.1
        ymin = df_plot["Delta"].min() * 1.3 if len(df_plot) > 0 and df_plot["Delta"].min() < 0 else -0.1
        ax.set_ylim(ymin, ymax)
        
        for i, (p_raw, p_fdr, m) in enumerate(zip(ps_raw, ps_fdr, means)):
            if pd.isna(m): continue
            if   p_fdr < 0.001: marker, bold = "***", True
            elif p_fdr < 0.01:  marker, bold = "**",  True
            elif p_fdr < 0.05:  marker, bold = "*",   True
            elif p_raw < 0.05:  marker, bold = "†",   False
            else: continue
            
            sub_df = df_plot[df_plot["Network"] == NET_SHORT[i]]
            if len(sub_df) == 0: continue
            ypos = sub_df["Delta"].max() + (ymax-ymin)*0.05
            if m < 0 and ypos < 0: ypos = sub_df["Delta"].min() - (ymax-ymin)*0.08
            
            ax.text(i, ypos, marker, ha="center", va="bottom" if m>=0 else "top",
                    fontsize=13 if bold else 12, fontweight="bold" if bold else "normal", color="black")
            
        ax.set_title(f"{label}\n(n={n})", fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
        else:
            ax.set_ylabel("")
            
        ax.spines[["top", "right"]].set_visible(False)
        x = np.arange(len(NET_SHORT))
        ax.set_xticks(x)
        ax.set_xticklabels(NET_SHORT, rotation=45, ha="right", fontsize=11)

    legend_elements = [
        Line2D([0],[0], marker="$***$", color="black", markersize=14, label="p$_{FDR}$<0.001", linestyle="none"),
        Line2D([0],[0], marker="$**$",  color="black", markersize=12, label="p$_{FDR}$<0.01",  linestyle="none"),
        Line2D([0],[0], marker="$*$",   color="black", markersize=10, label="p$_{FDR}$<0.05",  linestyle="none"),
        Line2D([0],[0], marker="$†$",   color="gray",  markersize=10, label="p$_{raw}$<0.05 (uncorr.)", linestyle="none"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=11, framealpha=0.0, bbox_to_anchor=(0.5, -0.02))
    
    title_metric = "dSW" if metric == "dSW" else "dFC"
    fig.suptitle(f"Network contribution to whole-brain SampEn ({title_metric}) — Raincloud Distribution", fontsize=14, fontweight="bold", y=1.02)
    
    plt.savefig(os.path.join(out_dir, f"sdi_rainclouds_{metric}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"sdi_rainclouds_{metric}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"sdi_rainclouds_{metric}.svg"), bbox_inches='tight')
    print(f"\nFigures saved: {output_pdf}, {output_png}")

if __name__ == "__main__":
    main()
