"""Perform sensitivity analysis on the results of the model runs."""
from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from scipy import stats
from tqdm import tqdm

from swmmanywhere.logging import logger
from swmmanywhere_paper.src import experimenter
from swmmanywhere_paper.src import plotting as swplt
from swmmanywhere.preprocessing import check_bboxes
from swmmanywhere.swmmanywhere import load_config

# %% [markdown]
# ## Initialise directories and load results
# %%
# Load the configuration file and extract relevant data
projects = ["cranbrook_node_1439.1",
                "bellinge_G62F060_G61F180_l1",
                "bellinge_G72F550_G72F010_l1",
                "bellinge_G60F61Y_G60F390_l1",
                "bellinge_G74F150_G74F140_l1",

                "bellinge_G80F390_G80F380_l1",
                "bellinge_G72F800_G72F050_l1",
                "bellinge_G73F000_G72F120_l1",
                
                #'cranbrook_formatted_largesample',
                ]
dfp = []
for project in projects:

    base_dir = Path.home() / "Documents" / "data" / "swmmanywhere"
    config_path = base_dir / project / f'config.yml'
    config = load_config(config_path, validation = False)
    config['base_dir'] = base_dir / project
    objectives = config['metric_list']
    parameters = config['parameters_to_sample']

    # Load the results
    bbox = check_bboxes(config['bbox'], config['base_dir'])
    results_dir = config['base_dir'] / f'bbox_{bbox}' / 'results'
    fids = list(results_dir.glob('*_metrics.csv'))
    dfs = [pd.read_csv(fid) for fid in tqdm(fids, total = len(fids))]

    # Concatenate the results
    df = pd.concat(dfs).reset_index(drop=True)
    df['project'] = project

    df = df.loc[:,~df.columns.str.contains('subcatchment')]
    df = df.drop('bias_flood_depth',axis=1)
    df[df == np.inf] = None
    df = df.sort_values(by = 'iter')

    dfp.append(df)

df = pd.concat(dfp)
df = df.reset_index(drop=True)
plot_fid = results_dir.parent / 'plots'

n_panels = len(parameters)
n_cols = int(n_panels**0.5)
if n_cols * (n_cols + 1) >= n_panels:
    n_rows = n_cols + 1
else:
    n_rows = n_cols

# Perform bootstrapping (for 'outlet_nse_flow')
n_bootstraps = 50
frac = 0.1
objective = 'outlet_nse_flow'
bootstrap_results = {p : [] for p in parameters}
for n in tqdm(range(n_bootstraps)):
    ix = df.sample(frac=frac, replace=True).index
    if 'nse' in objective:
        weights = df.loc[ix,objective].clip(lower=0)
    elif 'kge' in objective:
        weights = df.loc[ix,objective].clip(lower=-0.41) + 0.41
    elif 'relerror' in objective:
        weights = df.loc[ix,objective].abs()

    for parameter in parameters:
        kde = stats.gaussian_kde(df.loc[ix,parameter], weights = weights)
    
        x = np.linspace(df[parameter].min(), df[parameter].max(), 100)
        bootstrap_results[parameter].append(kde(x))

# Calculate bounds
bounds = {k : (np.array(x).min(axis=0), np.array(x).max(axis=0)) 
            for k,x in bootstrap_results.items()}

# By objective - all projects
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
hl = {}
for objective in ['outlet_nse_flow', 
                  'outlet_kge_flow', 
                  'outlet_relerror_flow',
                  'outlet_relerror_diameter']:
    if 'nse' in objective:
        weights = df[objective].clip(lower=0)
    elif 'kge' in objective:
        weights = df[objective].clip(lower=-0.41) + 0.41
    elif 'relerror' in objective:
        weights = df[objective].abs()

    for parameter,ax in zip(parameters, axs.flat):
        kde = stats.gaussian_kde(df[parameter], weights = weights)
    
        x = np.linspace(df[parameter].min(), df[parameter].max(), 100)
        handle = ax.plot(x, kde(x),label=objective)

        ax.fill_between(x,
                         bounds[parameter][0],
                         bounds[parameter][1],
                         color = 'gray',
                         alpha = 0.1)
        ax.set_title(parameter.replace('_','\n'))
        ax.grid(True)
        hl[objective] = handle

axs[-1,-1].legend(handles=[x[0] for x in hl.values()])
fig.tight_layout()
fig.savefig(plot_fid / 'parameter_distributions_byobjective.png')
# By objective - one projects
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
hl = {}
for project in projects:
    for objective in ['outlet_nse_flow', 
                    'outlet_kge_flow', 
                    'outlet_relerror_flow',
                    'outlet_relerror_diameter',
                    'outlet_nse_flooding',
                    'outlet_kge_flooding',
                    'grid_nse_flooding',
                    'grid_kge_flooding']:
        df_ = df.loc[df.project == project]
        if 'nse' in objective:
            weights = df_[objective].clip(lower=0)
        elif 'kge' in objective:
            weights = df_[objective].clip(lower=-0.41) + 0.41
        elif 'relerror' in objective:
            weights = df_[objective].abs()
        
        if weights.isna().all():
            continue
        weights = weights.fillna(0)
            
        for parameter,ax in zip(parameters, axs.flat):
            kde = stats.gaussian_kde(df_[parameter], weights = weights)
        
            x = np.linspace(df_[parameter].min(), df_[parameter].max(), 100)
            handle = ax.plot(x, kde(x),label=objective)
           # ax.fill_between(x,
           #              bounds[parameter][0],
           #              bounds[parameter][1],
           #              color = 'gray',
           #              alpha = 0.1)
            ax.set_title(parameter.replace('_','\n'))
            ax.grid(True)
            hl[objective] = handle

    axs[-1,-1].legend(handles=[x[0] for x in hl.values()])
    fig.tight_layout()
    fig.savefig(plot_fid / f'parameter_distributions_byobjective_{project}.png')

# By project - outlet_nse_flow
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
hl = {}

objective = 'outlet_nse_flow'
for project in projects:
    df_ = df.loc[df.project == project]
    if 'nse' in objective:
        weights = df_[objective].clip(lower=0)
    elif 'kge' in objective:
        weights = df_[objective].clip(lower=-0.41) + 0.41
    elif 'relerror' in objective:
        weights = df_[objective].abs()

    for parameter,ax in zip(parameters, axs.flat):
        kde = stats.gaussian_kde(df_[parameter], weights = weights)
    
        x = np.linspace(df_[parameter].min(), df_[parameter].max(), 100)
        handle = ax.plot(x, kde(x),label=project)
        ax.fill_between(x,
                         bounds[parameter][0],
                         bounds[parameter][1],
                         color = 'gray',
                         alpha = 0.1)
        ax.set_title(parameter.replace('_','\n'))
        ax.grid(True)
        hl[project] = handle

axs[-1,-1].legend(handles=[x[0] for x in hl.values()])
fig.tight_layout()
fig.savefig(plot_fid / f'parameter_distributions_byproject_{objective}.png')