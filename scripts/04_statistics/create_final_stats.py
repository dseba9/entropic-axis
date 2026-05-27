import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))

datasets = {
    'AAL dSW': os.path.join(main_dir, 'data/AAL/all_data.csv'),
    'AAL dFC': os.path.join(main_dir, 'data/AAL/dFC_all_data.csv'),
    'Tian Schaefer dSW': os.path.join(main_dir, 'data/Tian_Schaefer/dSW_all_data.csv'),
    'Tian Schaefer dFC': os.path.join(main_dir, 'data/Tian_Schaefer/dFC_all_data.csv')
}

results = {}

for name, path in datasets.items():
    try:
        df = pd.read_csv(path)
        if 'ayah' in df['dataset'].values or df['dataset'].str.contains('ayah', case=False).any():
            df = df[~df['dataset'].str.contains('ayah', case=False)]
        
        m_mixed = smf.mixedlm('SampEn ~ dataset', df, groups=df['Subject']).fit()
        f_test = m_mixed.f_test(np.eye(len(m_mixed.params))[1:-1])
        
        results[name] = {
            'F_val': float(np.squeeze(f_test.fvalue)),
            'p_val': float(np.squeeze(f_test.pvalue)),
            'df_num': f_test.df_num,
            'df_denom': f_test.df_denom,
            'summary': m_mixed.summary().as_text()
        }
    except Exception as e:
        results[name] = {'error': str(e)}

# Generate exact strings for text.md
print("=== FOR TEXT.MD ===")
aal_dsw = results['AAL dSW']
print(f"AAL dSW: F{int(aal_dsw['df_num'])},{int(aal_dsw['df_denom'])} = {aal_dsw['F_val']:.2f}, p = {aal_dsw['p_val']:.1e}")

aal_dfc = results['AAL dFC']
print(f"AAL dFC: F{int(aal_dfc['df_num'])},{int(aal_dfc['df_denom'])} = {aal_dfc['F_val']:.2f}, p = {aal_dfc['p_val']:.1e}")

print("\n=== NOTEBOOK CELL CODE ===")
notebook_code = """import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(script_dir))

datasets = {
    'AAL dSW': os.path.join(main_dir, 'data/AAL/all_data.csv'),
    'AAL dFC': os.path.join(main_dir, 'data/AAL/dFC_all_data.csv'),
    'Tian Schaefer dSW': os.path.join(main_dir, 'data/Tian_Schaefer/dSW_all_data.csv'),
    'Tian Schaefer dFC': os.path.join(main_dir, 'data/Tian_Schaefer/dFC_all_data.csv')
}

for name, path in datasets.items():
    print("="*60)
    print(f"Model: {name}")
    print("="*60)
    try:
        df = pd.read_csv(path)
        df = df[~df['dataset'].str.contains('ayah', case=False)]
        
        m_mixed = smf.mixedlm('SampEn ~ dataset', df, groups=df['Subject']).fit()
        f_test = m_mixed.f_test(np.eye(len(m_mixed.params))[1:-1])
        
        fval = float(np.squeeze(f_test.fvalue))
        pval = float(np.squeeze(f_test.pvalue))
        
        print(f"F-test (Omnibus main effect of state):")
        print(f"F({int(f_test.df_num)}, {int(f_test.df_denom)}) = {fval:.2f}, p = {pval:.1e}")
        print("\\n")
        print(m_mixed.summary())
    except Exception as e:
        print(f"Failed to run model: {e}")
    print("\\n\\n")
"""

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Final Mixed-Effects Models Statistics\n",
    "This notebook runs the final linear mixed-effects models (`statsmodels.formula.api.mixedlm`) for the 12 experimental datasets (excluding ayahuasca), comparing the main effect of state across dynamic networking metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": notebook_code.splitlines(True)
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open(os.path.join(main_dir, 'scripts/04_statistics/final_mixed_models_stats.ipynb'), 'w') as f:
    json.dump(nb, f, indent=1)

print("\nNotebook final_mixed_models_stats.ipynb created.")

print("\n=== TIAN SCHAEFER TABLES FOR SUPPLEMENTARY ===")
for metric in ['dSW', 'dFC']:
    key = f'Tian Schaefer {metric}'
    res = results[key]
    print(f"## {key} Mixed-Effects Model Summary")
    print("```text")
    print(f"Omnibus F-test: F({int(res['df_num'])}, {int(res['df_denom'])}) = {res['F_val']:.2f}, p = {res['p_val']:.1e}\n")
    print(res['summary'])
    print("```\n")

