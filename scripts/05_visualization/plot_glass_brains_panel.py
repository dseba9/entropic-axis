import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))
atlas_nii_path = os.path.join(main_dir, 'scripts/01_parcelling/masks/1000_Schaefer/1000_Schaefer.nii')
lut_path = os.path.join(main_dir, 'scripts/01_parcelling/masks/1000_Schaefer/Schaefer2018_1000Parcels_7Networks_order.lut')
csv_dir = os.path.join(main_dir, 'data/LONO_CSVs')
out_dir = os.path.join(main_dir, 'results/paper_figures')

NETWORKS = ["Visual", "Somatomotor", "DorsalAttn", "SalVentAttn", "Limbic", "FrontoParietal", "DefaultMode"]
NET_SHORT = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

DATASETS = [
    ("Anaesthesia", "Unconscious", "Awake", True, "A"),
    ("Modafinil", "MOD", "PLB", True, "B"),
    ("LSD", "LSD", "PLB", True, "C"),
    ("DMT", "DMT", "PLB", True, "D"),
    ("Schizophrenia", "SCHZ", "CTRL", False, "E")
]

metric = sys.argv[1] if len(sys.argv) > 1 else "dSW"
prefix = "" if metric == "dSW" else "_dFC"

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

dataset_images = {}
dataset_vmax = {}

for ds_name, cond_a, cond_b, is_paired, letter in DATASETS:
    csv_path = os.path.join(csv_dir, f"{ds_name}{prefix}_SE_LONO_SubjectCondition.csv")
    if not os.path.exists(csv_path): continue
    
    df = pd.read_csv(csv_path, index_col=0).dropna(how="all").dropna(how="any")
    if len(df) == 0: continue
    
    parts = [idx.split("_") for idx in df.index]
    df["subject"] = ["_".join(p[:2]) for p in parts]
    df["condition"] = ["_".join(p[2:]) for p in parts]
    
    # No exclusions to match the interactive_plot.py summary CSV behavior
    
    for net in NETWORKS:
        df[f"SDI_{net}"] = df["Complete"] - df[net]
        
    delta_se = []
    for net in NETWORKS:
        col = f"SDI_{net}"
        if is_paired:
            a = df[df.condition == cond_a].set_index("subject")[col]
            b = df[df.condition == cond_b].set_index("subject")[col]
            common = a.index.intersection(b.index)
            if len(common) > 0:
                delta_se.append((a - b).loc[common].mean())
            else: delta_se.append(0)
        else:
            a = df[df.condition == cond_a][col].values
            b = df[df.condition == cond_b][col].values
            if len(a) > 0 and len(b) > 0:
                delta_se.append(np.mean(a) - np.mean(b))
            else: delta_se.append(0)
            
    sdi_brain_data = np.zeros_like(atlas_data)
    for roi_id, net_idx in roi_to_net.items():
        sdi_brain_data[atlas_data == roi_id] = delta_se[net_idx]
        
    sdi_img = nib.Nifti1Image(sdi_brain_data, atlas_img.affine)
    dataset_images[ds_name] = sdi_img
    
    # Save intermediate NIfTI map to results/LONO/
    map_out_dir = os.path.join(main_dir, "results/LONO")
    os.makedirs(map_out_dir, exist_ok=True)
    map_name = "Schiz" if ds_name == "Schizophrenia" else ds_name
    nib.save(sdi_img, os.path.join(map_out_dir, f"map_{map_name}_{metric}.nii.gz"))
    
    # Set user-approved limits for dSW
    if metric == "dSW":
        limits = {
            "LSD": (-0.060, 0.060),
            "DMT": (-0.033, 0.003),
            "Anaesthesia": (-0.054, 0.054),
            "Schizophrenia": (-0.092, -0.010),
            "Modafinil": (-0.045, 0.065)
        }
        vmin, vmax = limits[ds_name]
    else:
        vmin = np.min(delta_se) if len(delta_se) > 0 else -0.05
        vmax = np.max(delta_se) if len(delta_se) > 0 else 0.05
        if abs(vmin) < 0.02 and abs(vmax) < 0.02:
            vmin, vmax = -0.02, 0.02
        
    dataset_vmax[ds_name] = (vmin, vmax)
    print(f"{ds_name}: vmin={vmin:.4f}, vmax={vmax:.4f}, values={np.round(delta_se, 4)}")
 
fig = plt.figure(figsize=(28, 10))
# Grid: 3 rows (Brains Top, Brains Bottom, Colorbars) x 5 columns
gs = fig.add_gridspec(3, 5, wspace=0.4, hspace=0.1, height_ratios=[1, 1, 0.15])
 
axes_l = [fig.add_subplot(gs[0, i]) for i in range(5)]
axes_z = [fig.add_subplot(gs[1, i]) for i in range(5)]
axes_cb = [fig.add_subplot(gs[2, i]) for i in range(5)]
 
for i, (ds_name, cond_a, cond_b, is_paired, letter) in enumerate(DATASETS):
    if ds_name not in dataset_images:
        axes_l[i].axis('off'); axes_z[i].axis('off'); axes_cb[i].axis('off')
        continue
    
    img = dataset_images[ds_name]
    vmin, vmax = dataset_vmax[ds_name]
    
    plotting.plot_glass_brain(img, display_mode='l', axes=axes_l[i],
                               cmap='RdBu_r', vmin=vmin, vmax=vmax, plot_abs=False, colorbar=False)
                              
    plotting.plot_glass_brain(img, display_mode='z', axes=axes_z[i],
                               cmap='RdBu_r', vmin=vmin, vmax=vmax, plot_abs=False, colorbar=False)
 
    # Individual Colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                 cax=axes_cb[i], orientation='horizontal')
    cb.ax.tick_params(labelsize=10)
    cb.set_ticks([vmin, vmax])
    cb.ax.set_xticklabels([f"[{vmin:.3f}", f"{vmax:.3f}]"])
    cb.set_label(f'ΔSE {metric}', fontsize=12, fontweight='bold', labelpad=5)
    
    # Dataset Label
    axes_l[i].set_title(f"{ds_name}", fontsize=18, fontweight='bold', pad=10)
 
fig.suptitle(f"Network Driver Impact (ΔSE {metric}) — Individual Dataset Scaling", fontsize=22, fontweight='bold', y=1.02)
 
out_png = os.path.join(out_dir, f"glassbrains_panel_{metric}.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, f"glassbrains_panel_{metric}.svg"), bbox_inches='tight')
print(f"Panel generated with individual scaling: {out_png}")
