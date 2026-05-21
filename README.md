# A Shared Entropic Axis Spans States of Consciousness Across Pharmacological and Clinical Conditions

![Analysis Pipeline](results/paper_figures/Figure1.png)

This repository contains the processing scripts and processed datasets to reproduce all statistical analyses and figures for the manuscript investigating the shared entropic axis across altered states of consciousness (Psychedelics, Anesthesia, Modafinil, and Schizophrenia).

---

## 📂 Repository Structure

* **`data/`**: Processed tables (`all_data.csv`, LONO CSVs, and summaries) needed to reproduce statistics and figures immediately without requiring raw fMRI NIfTI files.
* **`requirements/`**:
  * **`versions_info.txt`**: Detailed system, MATLAB, Python, and package version specifications. See [versions_info.txt](requirements/versions_info.txt).
* **`scripts/`**:
  * **`01_parcelling/`**: Extraction of regional BOLD time-series from fMRI data. Contains the standardized script `run_parcelling.m` (with configuration options at the top), helper function `bold_to_networks.m`, and atlas templates (`1000_Schaefer.nii`, `AAL.nii`, and `Tian_Schaefer_combinada.nii`). Requires SPM12.
  * **`02_correlations/`**: Functional connectivity correlation matrix computation. Contains `run_correlations.m`.
  * **`03_graph_analysis/`**: Dynamic Small-World Propensity and Leave-One-Network-Out (LONO) calculation. Contains `run_sliding_window_LONO.m`.
  * **`04_statistics/`**: Linear mixed-effects (LME) models, Wald tests, and LONO group-level stats (Python / R).
    * `final_mixed_models_stats.ipynb`: Interactive Jupyter Notebook to step through the global LME models for dSW and dFC.
  * **`05_visualization/`**: Generates the main manuscript figures (Glass Brains, Raincloud Plots, and composite figures) (Python).
* **`supplementary/`**:
  * Contains the code to generate supplementary figure panels (`generate_supplementary_figures.py`).
  * Output figures are saved in `supplementary/figures/`.
* **`results/`**: Directory where main outputs and figures are saved.

---

## 🛠️ Installation & Requirements

### Python Environment
Install all required Python packages via pip:
```bash
pip install -r requirements.txt
```

### MATLAB Environment
The preprocessing pipeline requires **MATLAB** alongside the following external third-party dependencies (which should be downloaded and added to your MATLAB path):

1. **SPM12** (Statistical Parametric Mapping): Required for parcellation. [Download from SPM Website](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/).
2. **Brain Connectivity Toolbox (BCT)** (Release 2019_03_03): Required for graph metrics. [Download from BCT Website](https://sites.google.com/site/bctnet/).
3. **BrainNet Viewer** (Version 20191031): Required for 3D visualization. [Download from NITRC](https://www.nitrc.org/projects/bnv/).
4. **Small-World Propensity (SWP)** (Muldoon et al., 2016): Required for dSW. [Download from GitHub](https://github.com/akhannap/small-world-propensity).
5. **Physionet Sample Entropy (sampen)**: Required for entropy. [Download from PhysioNet](https://physionet.org/content/sampen/).

Add these directories to your MATLAB path (`pathtool` or `addpath`) before executing the unified `run_*.m` scripts.

---

## 📊 Reproducing Figures & Statistics

To reproduce the main results of the manuscript:

1. **Global Mixed-Effects Statistics (Shared Entropic Axis):**
   ```bash
   python scripts/04_statistics/create_final_stats.py
   python scripts/04_statistics/generate_contrast_results.py
   ```
   *Alternatively, open `scripts/04_statistics/final_mixed_models_stats.ipynb` to inspect and run LME models interactively.*

2. **LONO Statistics (Leave-One-Network-Out Wilcoxon Tests):**
   ```bash
   python scripts/04_statistics/process_lono.py
   ```

3. **Generate Main Manuscript Figures:**
   ```bash
   # Generate 3D brain map projections (Figure 6A & S3A)
   python scripts/05_visualization/plot_glass_brains_panel.py dSW
   python scripts/05_visualization/plot_glass_brains_panel.py dFC
   
   # Generate Continuous Entropic Gradients (Figure 4 & S1)
   python scripts/05_visualization/plot_entropic_gradient.py
   
   # Generate 2D State Space Maps (Figure 5 & S2)
   python scripts/05_visualization/plot_2d_state_space.py
   ```

4. **Generate Distribution Raincloud Plots:**
   ```bash
   # Generate Per-Dataset Subject-Level Rainclouds (Figure 2 & 3)
   python scripts/05_visualization/plot_dataset_rainclouds.py
   
   # Generate LONO Network-Level Delta Rainclouds (Figure 6B & S3B)
   python scripts/05_visualization/plot_sdi_rainclouds.py dSW
   python scripts/05_visualization/plot_sdi_rainclouds.py dFC
   ```

5. **Generate Supplementary Figures (Strip-Plots):**
   ```bash
   python supplementary/generate_supplementary_figures.py
   ```

For further details regarding exact software versions and computational configurations, please refer to [requirements/versions_info.txt](requirements/versions_info.txt).

---

## 📚 Acknowledgements & Atlas Citations

This repository includes derived NIfTI atlas masks (in `scripts/01_parcelling/masks/`) strictly for reproducibility purposes. If you use this pipeline, please ensure you cite the original creators of these open-science resources:

1. **AAL Atlas**: Tzourio-Mazoyer, N., et al. (2002). Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain. *Neuroimage*, 15(1), 273-289.
2. **Schaefer 1000 Parcellation**: Schaefer, A., et al. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex*, 28(9), 3095-3114.
3. **Tian Subcortical Atlas**: Tian, Y., et al. (2020). Topographic organization of the human subcortex unveiled with functional connectivity gradients. *Nature Neuroscience*, 23(11), 1421-1432.
4. **Brain Connectivity Toolbox**: Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: uses and interpretations. *Neuroimage*, 52(3), 1059-1069.
