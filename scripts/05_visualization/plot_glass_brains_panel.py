import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))
atlas_nii_path = os.path.join(main_dir, 'scripts/01_parcelling/masks/1000_Schaefer/1000_Schaefer.nii')
lut_path = os.path.join(main_dir, 'scripts/01_parcelling/masks/1000_Schaefer/Schaefer2018_1000Parcels_7Networks_order.lut')
csv_dir = os.path.join(main_dir, 'data/LONO_CSVs')
out_dir = os.path.join(main_dir, 'results/paper_figures')
os.makedirs(out_dir, exist_ok=True)

NETWORKS = ["Visual", "Somatomotor", "DorsalAttn", "SalVentAttn", "Limbic", "FrontoParietal", "DefaultMode"]
NET_SHORT = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

# Order matches approved figures (Anaesthesia, Modafinil, LSD, DMT, Schizophrenia)
DATASETS = [
    ("Anaesthesia", "Anestesia",     "Unconscious", "Awake", True,  "A"),
    ("Modafinil",   "Modafinil",     "MOD",         "PLB",   True,  "B"),
    ("LSD",         "LSD",           "LSD",         "PLB",   True,  "C"),
    ("DMT",         "DMT",           "DMT",         "PLB",   True,  "D"),
    ("Schizophrenia","Schizophrenia","SCHZ",        "CTRL",  False, "E"),
]
# DMT subjects to exclude (as used during original analysis)
DMT_EXCLUSIONS = ["Sub_1", "Sub_6", "Sub_7", "Sub_12", "Sub_14", "Sub_18"]

metric = sys.argv[1] if len(sys.argv) > 1 else "dSW"
prefix = "" if metric == "dSW" else "_dFC"

# Approved dSW colorbar limits from the interactive slider session
DSW_LIMITS = {
    "Anaesthesia":   (-0.054, 0.054),
    "Modafinil":     (-0.045, 0.065),
    "LSD":           (-0.060, 0.060),
    "DMT":           (-0.033, 0.003),
    "Schizophrenia": (-0.092, -0.010),
}

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
                if net_name in NET_SHORT:
                    roi_to_net[roi_id] = NET_SHORT.index(net_name)

# For dSW, load pre-computed deltas from summary CSV (same as original interactive_plot.py)
summary_df = None
if metric == "dSW":
    summary_path = os.path.join(csv_dir, 'lono_stats_results_summary.csv')
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print("Loaded pre-computed dSW deltas from summary CSV.")

dataset_images = {}
dataset_limits = {}

for label, csv_key, cond_a, cond_b, is_paired, letter in DATASETS:
    # ── dSW: use pre-computed deltas from summary CSV ────────────────────
    if metric == "dSW" and summary_df is not None:
        csv_ds_name = "Anesthesia" if label == "Anaesthesia" else label
        cond_str = f"{cond_a} vs {cond_b}"
        ds_df = summary_df[(summary_df['Dataset'] == csv_ds_name) &
                           (summary_df['Condition'] == cond_str)]
        if len(ds_df) == 0:
            print(f"  [SKIP] No summary entry for {label} / '{cond_str}'")
            continue
        delta_se = [ds_df[ds_df['Network'] == net]['Delta_SDI'].values[0]
                    for net in NETWORKS]

    # ── dFC: compute from per-subject CSVs ───────────────────────────────
    else:
        csv_path = os.path.join(csv_dir, f"{csv_key}{prefix}_SE_LONO_SubjectCondition.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col=0).dropna(how="all").dropna(how="any")
        if len(df) == 0:
            continue

        parts = [idx.split("_") for idx in df.index]
        df["subject"]   = ["_".join(p[:2]) for p in parts]
        df["condition"] = ["_".join(p[2:]) for p in parts]

        if label == "DMT":
            df = df[~df["subject"].isin(DMT_EXCLUSIONS)]

        for net in NETWORKS:
            df[f"SDI_{net}"] = df["Complete"] - df[net]

        delta_se = []
        for net in NETWORKS:
            col = f"SDI_{net}"
            if is_paired:
                a = df[df.condition == cond_a].set_index("subject")[col]
                b = df[df.condition == cond_b].set_index("subject")[col]
                common = a.index.intersection(b.index)
                delta_se.append((a - b).loc[common].mean() if len(common) > 0 else 0.0)
            else:
                a = df[df.condition == cond_a][col].dropna().values
                b = df[df.condition == cond_b][col].dropna().values
                delta_se.append(np.mean(a) - np.mean(b) if len(a) > 0 and len(b) > 0 else 0.0)

    # ── Build brain map ──────────────────────────────────────────────────
    sdi_brain_data = np.zeros_like(atlas_data)
    for roi_id, net_idx in roi_to_net.items():
        sdi_brain_data[atlas_data == roi_id] = delta_se[net_idx]

    sdi_img = nib.Nifti1Image(sdi_brain_data, atlas_img.affine)
    dataset_images[label] = sdi_img

    # Save intermediate NIfTI
    map_out_dir = os.path.join(main_dir, "results/LONO")
    os.makedirs(map_out_dir, exist_ok=True)
    map_key = "Schiz" if label == "Schizophrenia" else label
    nib.save(sdi_img, os.path.join(map_out_dir, f"map_{map_key}_{metric}.nii.gz"))

    # ── Colorbar limits ──────────────────────────────────────────────────
    if metric == "dSW":
        vmin, vmax = DSW_LIMITS[label]
    else:
        # Symmetric around zero (approved behavior for dFC)
        vm = max(np.abs(delta_se)) if len(delta_se) > 0 else 0.05
        if vm < 0.02:
            vm = 0.02
        vmin, vmax = -vm, vm

    dataset_limits[label] = (vmin, vmax)
    print(f"{label}: vmin={vmin:.4f}, vmax={vmax:.4f}, values={np.round(delta_se, 4)}")

# ── Render Panel ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 10))
gs = fig.add_gridspec(3, 5, wspace=0.4, hspace=0.1, height_ratios=[1, 1, 0.15])

axes_l  = [fig.add_subplot(gs[0, i]) for i in range(5)]
axes_z  = [fig.add_subplot(gs[1, i]) for i in range(5)]
axes_cb = [fig.add_subplot(gs[2, i]) for i in range(5)]

for i, (label, csv_key, cond_a, cond_b, is_paired, letter) in enumerate(DATASETS):
    if label not in dataset_images:
        axes_l[i].axis('off'); axes_z[i].axis('off'); axes_cb[i].axis('off')
        continue

    img = dataset_images[label]
    vmin, vmax = dataset_limits[label]

    plotting.plot_glass_brain(img, display_mode='l', axes=axes_l[i],
                               cmap='RdBu_r', vmin=vmin, vmax=vmax, plot_abs=False, colorbar=False)
    plotting.plot_glass_brain(img, display_mode='z', axes=axes_z[i],
                               cmap='RdBu_r', vmin=vmin, vmax=vmax, plot_abs=False, colorbar=False)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                      cax=axes_cb[i], orientation='horizontal')
    cb.ax.tick_params(labelsize=10)
    cb.set_ticks([vmin, vmax])
    cb.ax.set_xticklabels([f"[{vmin:.3f}", f"{vmax:.3f}]"])
    cb.set_label(f'ΔSE {metric}', fontsize=12, fontweight='bold', labelpad=5)

    axes_l[i].set_title(label, fontsize=18, fontweight='bold', pad=10)

fig.suptitle(f"Network Driver Impact (ΔSE {metric}) — Per-Dataset Scaling",
             fontsize=22, fontweight='bold', y=1.02)

out_png = os.path.join(out_dir, f"glassbrains_panel_{metric}.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, f"glassbrains_panel_{metric}.svg"), bbox_inches='tight')
print(f"Panel saved: {out_png}")
