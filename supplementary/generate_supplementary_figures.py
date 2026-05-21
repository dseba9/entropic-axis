import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_prob_points(*args, facecolors=None, alpha=None, capsize=10, elinewidth=10, capthick=6, dot_size=400, yticks=None, ytickslabels=None, ymin=None, ymax=None, random_state=10, figsize=(10,10)):
    np.random.seed(random_state)
    SPREAD_LENGTH = 0.1
    
    x_list = np.arange(1.0, 1.0+0.5*len(args), 0.5)
    y_list = list(map(lambda x: x.mean(), args))
    error_list = list(map(lambda x: x.std()/np.sqrt(len(x)), args))
    spread_list = x_list - SPREAD_LENGTH/2.0
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.errorbar(x_list, y_list, error_list, ls="none", capsize=capsize, elinewidth=elinewidth, capthick=capthick, ecolor="black")
    ax.scatter(x_list, y_list, s=dot_size, c="black", zorder=10)
    
    if facecolors is None:
        facecolors = np.repeat("#cccccc", len(args))
    if alpha is None:
        alpha = np.repeat(0.5, len(args))

    if type(facecolors) == str:
        facecolors = np.repeat(facecolors, len(args))
    if type(alpha) == float or type(alpha) == int:
        alpha = np.repeat(alpha, len(args))
    
    for i, arg in enumerate(args):
        ax.scatter(
            np.random.rand(len(arg))*SPREAD_LENGTH + spread_list[i], 
            arg, 
            c=facecolors[i], 
            edgecolors='none', 
            s=dot_size, 
            alpha=alpha[i],
            zorder=0
        )

    ax.set_xlim([1.0-0.4, max(x_list)+0.4])
    ax.set_xticks(x_list, np.repeat("", len(args)))
    ax.spines[['right', 'top']].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=5)
    
    if ymin and ymax:
        ax.set_ylim([ymin, ymax])
    
    if yticks:
        if ytickslabels:
            ax.set_yticks(yticks, ytickslabels)
        else:
            ax.set_yticks(yticks)
    return fig

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(script_dir)
output_dir_fig = os.path.join(main_dir, 'supplementary/figures/')
os.makedirs(output_dir_fig, exist_ok=True)

pairs = [
    ('lsd_plcb', 'lsd_lsd', 'LSD'),
    ('dmt_plcb', 'dmt_dmt', 'DMT'),
    ('placebo_modafinil', 'modafinil', 'Modafinil'),
    ('ucla_control', 'ucla_schz', 'Schizophrenia')
]

# Process dSW Data
print("Generating figures for Tian Schaefer dSW...")
data_dSW = pd.read_csv(os.path.join(main_dir, 'data/Tian_Schaefer/dSW_all_data.csv'))

for plcb_name, drug_name, label in pairs:
    plcb = data_dSW[data_dSW['dataset'] == plcb_name]["SampEn"].values
    drug = data_dSW[data_dSW['dataset'] == drug_name]["SampEn"].values
    
    plcb = plcb[~np.isnan(plcb)]
    drug = drug[~np.isnan(drug)]
    
    if len(plcb) > 0 and len(drug) > 0:
        fig = plot_prob_points(
            plcb, drug,
            figsize=(5,5),
            facecolors=["#880000", "#000088"],
            alpha=0.4,
            capsize=10, elinewidth=5, capthick=5,
            dot_size=200, random_state=10    
        )
        plt.rcParams.update({'font.size': 20})
        plt.tight_layout(pad=2)
        plt.xlabel('PLCB            DRUG')
        plt.ylabel('SampEn Values')
        plt.title(f'dSW - {label}')
        fig.savefig(f"{output_dir_fig}dSW_{label}.png")
        plt.close(fig)

# Process dFC Data
print("Generating figures for Tian Schaefer dFC...")
data_dFC = pd.read_csv(os.path.join(main_dir, 'data/Tian_Schaefer/dFC_all_data.csv'))

for plcb_name, drug_name, label in pairs:
    plcb = data_dFC[data_dFC['dataset'] == plcb_name]["SampEn"].values
    drug = data_dFC[data_dFC['dataset'] == drug_name]["SampEn"].values
    
    plcb = plcb[~np.isnan(plcb)]
    drug = drug[~np.isnan(drug)]
    
    if len(plcb) > 0 and len(drug) > 0:
        fig = plot_prob_points(
            plcb, drug,
            figsize=(5,5),
            facecolors=["#880000", "#000088"],
            alpha=0.4,
            capsize=10, elinewidth=5, capthick=5,
            dot_size=200, random_state=10    
        )
        plt.rcParams.update({'font.size': 20})
        plt.tight_layout(pad=2)
        plt.xlabel('PLCB            DRUG')
        plt.ylabel('SampEn Values')
        plt.title(f'dFC - {label}')
        fig.savefig(f"{output_dir_fig}dFC_{label}.png")
        plt.close(fig)

print("Finished generating figures.")
