import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300

script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))

atlas_nii_path = os.path.join(main_dir, 'scripts/01_parcelling/masks/1000_Schaefer/1000_Schaefer.nii')
lut_path = os.path.join(main_dir, 'scripts/01_parcelling/masks/1000_Schaefer/Schaefer2018_1000Parcels_7Networks_order.lut')
csv_dir = os.path.join(main_dir, 'data/LONO_CSVs')
out_dir = os.path.join(main_dir, 'results/paper_figures')
os.makedirs(out_dir, exist_ok=True)

NETWORKS = ["Visual", "Somatomotor", "DorsalAttn", "SalVentAttn", "Limbic", "FrontoParietal", "DefaultMode"]
LUT_NET_SHORT = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
PLOT_NET_SHORT = ["VIS", "SM", "DAN", "SAL", "LIM", "FP", "DM"]

# Column configurations in left-to-right order:
# (Label, CSV prefix, Active condition, Control condition, is_paired, color)
COLS = [
    ("Anaesthesia",   "Anestesia",     "Unconscious", "Awake", True,  "#E69F00"),
    ("Modafinil",     "Modafinil",     "MOD",         "PLB",   True,  "#F0E442"),
    ("LSD",         "LSD",           "LSD",         "PLB",   True,  "#009E73"),
    ("DMT",         "DMT",           "DMT",         "PLB",   True,  "#CC79A7"),
    ("Schizophrenia","Schizophrenia", "SCHZ",        "CTRL",  False, "#0072B2"),
]

# Exclusions are not applied for DMT in dSW analysis to match original paper statistics and summary CSV
DMT_EXCLUSIONS = []

# Approved individual limits for dSW (with symmetric limits to highlight shared networks)
DSW_LIMITS = {
    "Anaesthesia":   (-0.054, 0.054),
    "Modafinil":     (-0.045, 0.045),
    "LSD":           (-0.040, 0.040),
    "DMT":           (-0.033, 0.033),
    "Schizophrenia": (-0.092, 0.092),
}

# Load atlas and network mapping
atlas_img = nib.load(atlas_nii_path)
atlas_data = atlas_img.get_fdata()

roi_to_net = {}
with open(lut_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:
            roi_id = int(parts[0])
            name = parts[4]
            name_parts = name.split('_')
            if len(name_parts) >= 3:
                net_name = name_parts[2]
                if net_name in LUT_NET_SHORT:
                    roi_to_net[roi_id] = LUT_NET_SHORT.index(net_name)

def plot_custom_raincloud(ax, df_plot, color):
    series_list = []
    for net in PLOT_NET_SHORT:
        s = df_plot[df_plot['Network'] == net]['Delta'].values
        series_list.append(s)
        
    x = np.arange(len(PLOT_NET_SHORT))
    
    # 1. Half-Violin (Right)
    parts = ax.violinplot(series_list, positions=x, widths=0.7, showmeans=False, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(color)
        b.set_alpha(0.4)
        b.set_edgecolor("none")
        verts = b.get_paths()[0].vertices
        xm = verts[:,0].mean()
        verts[:,0] = np.maximum(verts[:,0], xm)

    # 2. Jitter Dots (Left)
    for i, s in enumerate(series_list):
        if len(s) == 0: continue
        jitter = (np.random.rand(len(s)) - 0.5) * 0.15 - 0.1
        ax.scatter(x[i] + jitter, s, s=25, alpha=0.6, color=color, edgecolors='none', zorder=2)

    # 3. Mean + ErrorBar (Black)
    means = [s.mean() if len(s)>0 else np.nan for s in series_list]
    sems  = [s.std(ddof=1)/np.sqrt(len(s)) if len(s)>0 else np.nan for s in series_list]
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black', capsize=4, lw=2.0, zorder=3)
    ax.scatter(x, means, s=35, color='black', zorder=4)

def main():
    # Set up matplotlib figure
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    
    # Top section for Panel A (glass brains) and bottom section for Panel B (rainclouds)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.75], hspace=0.35, left=0.06, right=0.98, top=0.95, bottom=0.08)
    
    gs_top = gs[0].subgridspec(3, 5, wspace=0.3, hspace=0.08, height_ratios=[1, 1, 0.12])
    gs_bottom = gs[1].subgridspec(1, 5, wspace=0.35)
    
    axes_l = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
    axes_z = [fig.add_subplot(gs_top[1, i]) for i in range(5)]
    axes_cb = [fig.add_subplot(gs_top[2, i]) for i in range(5)]
    
    axes_rc = [fig.add_subplot(gs_bottom[0, i]) for i in range(5)]
    
    for i, (label, csv_key, cond_a, cond_b, is_paired, color) in enumerate(COLS):
        # 1. Load data
        csv_path = os.path.join(csv_dir, f"{csv_key}_SE_LONO_SubjectCondition.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path, index_col=0).dropna(how="all").dropna(how="any")
        parts = [idx.split("_") for idx in df.index]
        df["subject"]   = ["_".join(p[:2]) for p in parts]
        df["condition"] = ["_".join(p[2:]) for p in parts]
        
        if label == "DMT" and len(DMT_EXCLUSIONS) > 0:
            df = df[~df["subject"].isin(DMT_EXCLUSIONS)]
            
        for net in NETWORKS:
            df[f"SDI_{net}"] = df["Complete"] - df[net]
            
        # 2. Compute LONO deltas & statistics
        delta_se = []
        ps_raw = []
        plot_data = []
        
        for net, net_short in zip(NETWORKS, PLOT_NET_SHORT):
            col = f"SDI_{net}"
            if is_paired:
                a = df[df.condition == cond_a].set_index("subject")[col]
                b = df[df.condition == cond_b].set_index("subject")[col]
                common = a.index.intersection(b.index)
                d = (a - b).loc[common]
                delta_se.append(d.mean())
                stat, p = stats.wilcoxon(d)
                ps_raw.append(p)
                for val in d.values:
                    plot_data.append({"Network": net_short, "Delta": val})
            else:
                a = df[df.condition == cond_a][col].dropna().values
                b = df[df.condition == cond_b][col].dropna().values
                delta_se.append(np.mean(a) - np.mean(b))
                stat, p = stats.mannwhitneyu(a, b)
                ps_raw.append(p)
                for val in a:
                    plot_data.append({"Network": net_short, "Delta": val - np.mean(b)})
                    
        # Apply FDR correction across the 7 networks
        n_tests = len(ps_raw)
        order = np.argsort(ps_raw)
        ps_fdr = np.empty(n_tests)
        for rank, idx in enumerate(order):
            ps_fdr[idx] = min(1.0, ps_raw[idx] * n_tests / (rank + 1))
        for j in range(n_tests - 2, -1, -1):
            ps_fdr[order[j]] = min(ps_fdr[order[j]], ps_fdr[order[j + 1]])
            
        # 3. Plot Panel A (Glass Brains) - with Fading Applied!
        sdi_brain_data = np.zeros_like(atlas_data)
        for roi_id, net_idx in roi_to_net.items():
            val = delta_se[net_idx]
            if ps_raw[net_idx] >= 0.05:
                val = val * 0.15  # Fade out non-significant networks (p >= 0.05)
            sdi_brain_data[atlas_data == roi_id] = val
            
        sdi_img = nib.Nifti1Image(sdi_brain_data, atlas_img.affine)
        
        vmin, vmax = DSW_LIMITS[label]
        
        plotting.plot_glass_brain(sdi_img, display_mode='l', axes=axes_l[i],
                                   cmap='RdBu_r', vmin=vmin, vmax=vmax, plot_abs=False, colorbar=False)
        plotting.plot_glass_brain(sdi_img, display_mode='z', axes=axes_z[i],
                                   cmap='RdBu_r', vmin=vmin, vmax=vmax, plot_abs=False, colorbar=False)
        
        # Colorbar below each brain column
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                          cax=axes_cb[i], orientation='horizontal')
        cb.ax.tick_params(labelsize=13)
        cb.set_ticks([vmin, vmax])
        cb.ax.set_xticklabels([f"[{vmin:.2f}", f"{vmax:.2f}]"])
        cb.set_label(r'$\Delta$SE(dSW)', fontsize=15, fontweight='bold', labelpad=4)
        
        # Set clean title (no parenthesized names)
        axes_l[i].set_title(label, fontsize=19, fontweight='bold', pad=10)
        
        # 4. Plot Panel B (Rainclouds)
        df_plot = pd.DataFrame(plot_data)
        plot_custom_raincloud(axes_rc[i], df_plot, color)
        
        axes_rc[i].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        
        # Align y-axis limits symmetric to show distribution cleanly
        ymax = df_plot["Delta"].max() * 1.3 if len(df_plot) > 0 and df_plot["Delta"].max() > 0 else 0.1
        ymin = df_plot["Delta"].min() * 1.3 if len(df_plot) > 0 and df_plot["Delta"].min() < 0 else -0.1
        axes_rc[i].set_ylim(ymin, ymax)
        
        # Add significance markers with enhanced visibility
        for j, (p_r, p_f, m) in enumerate(zip(ps_raw, ps_fdr, delta_se)):
            if pd.isna(m): continue
            if   p_f < 0.001: marker, bold = "***", True
            elif p_f < 0.01:  marker, bold = "**",  True
            elif p_f < 0.05:  marker, bold = "*",   True
            elif p_r < 0.05:  marker, bold = "†",   False
            else: continue
            
            sub_df = df_plot[df_plot["Network"] == PLOT_NET_SHORT[j]]
            if len(sub_df) == 0: continue
            ypos = sub_df["Delta"].max() + (ymax-ymin)*0.06
            if m < 0 and ypos < 0: ypos = sub_df["Delta"].min() - (ymax-ymin)*0.10
            
            axes_rc[i].text(j, ypos, marker, ha="center", va="bottom" if m>=0 else "top",
                            fontsize=18 if bold else 16, fontweight="bold", color="black")
            
        # Subplot labels and tick parameters
        axes_rc[i].set_xlabel("")
        axes_rc[i].spines[["top", "right"]].set_visible(False)
        axes_rc[i].set_xticks(np.arange(len(PLOT_NET_SHORT)))
        axes_rc[i].set_xticklabels(PLOT_NET_SHORT, rotation=45, ha="right", fontsize=15)
        axes_rc[i].tick_params(width=1.5)
        
        if i == 0:
            axes_rc[0].set_ylabel(r"$\Delta$SE(dSW)", fontsize=17, fontweight="bold")
        else:
            axes_rc[i].set_ylabel("")
            
    # Add A & B panel labels (moved closer to the plots)
    fig.text(0.025, 0.96, 'A', fontsize=32, fontweight='bold', ha='left', va='top')
    fig.text(0.025, 0.44, 'B', fontsize=32, fontweight='bold', ha='left', va='top')
    
    # Save composite figure
    out_png = os.path.join(out_dir, "Figure6.png")
    out_svg = os.path.join(out_dir, "Figure6.svg")
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, bbox_inches='tight')
    plt.close()
    
    print(f"Composite Figure 6 saved successfully to:")
    print(f"  {out_png}")
    print(f"  {out_svg}")

if __name__ == "__main__":
    main()
