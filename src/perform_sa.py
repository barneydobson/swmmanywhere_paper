"""Perform sensitivity analysis on the results of the model runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from tqdm import tqdm

from swmmanywhere.logging import logger
from swmmanywhere_paper.src import experimenter
from swmmanywhere_paper.src import plotting as swplt
from swmmanywhere.filepaths import check_bboxes
from swmmanywhere.swmmanywhere import load_config
from swmmanywhere.metric_utilities import metrics
# %% [markdown]
# ## Initialise directories and load results
# %%
# Load the configuration file and extract relevant data
projects = [ "cranbrook_node_1439.1",
             "bellinge_G60F61Y_G60F390_l1",
             "bellinge_G72F800_G72F050_l1"
            ]
for project in projects:

    base_dir = Path.home() / "Documents" / "data" / "swmmanywhere" / 'notrim_experiment'
    config_path = base_dir / project / f'config.yml'
    config = load_config(config_path, validation = False)
    config['base_dir'] = base_dir / project
    parameters = config['parameters_to_sample']

    # Load the results
    bbox = check_bboxes(config['bbox'], config['base_dir'])
    results_dir = config['base_dir'] / f'bbox_{bbox}' / 'results'
    fids = list(results_dir.glob('*_metrics.csv'))
    dfs = [pd.read_csv(fid) for fid in tqdm(fids, total = len(fids))]

    # Calculate how many processors were used
    nprocs = len(fids)

    # Concatenate the results
    df = pd.concat(dfs).reset_index(drop=True)

    df = df.loc[:,~df.columns.str.contains('subcatchment')]
    df = df.loc[:,~df.columns.str.contains('grid')]
    df = df.drop('bias_flood_depth',axis=1)
    df[df == np.inf] = None
    df = df.sort_values(by = 'iter')
    
    # Clip anoms
    for obj in ['outfall_kge_flooding', 'outfall_nse_flooding', 'outfall_nse_flow', 'outfall_kge_flow']:
        df.loc[df[obj] < -5, obj] = -5


    # Format order
    objectives = df.columns.intersection(metrics.keys())
    obj_grps = ['flow','flooding','outfall']
    objectives = pd.Series(objectives.rename('objective')).reset_index()
    objectives['group'] = 'graph'
    for ix, obj in objectives.iterrows():
        for grp in obj_grps:
            if grp in obj['objective']:
                objectives.loc[ix,'group'] = grp
                break
    objectives = objectives.sort_values(by=['group','objective']).objective

    problem = experimenter.formulate_salib_problem(parameters)
    parameters_order = pd.DataFrame([problem['names'],problem['groups']]).T
    parameters_order.columns = ['parameter','group']
    parameters_order = parameters_order.sort_values(by=['group','parameter']).parameter
    
    # Make a directory to store plots in
    plot_fid = results_dir.parent / 'plots'
    plot_fid.mkdir(exist_ok=True, parents=True)

    # %% [markdown]
    # ## Plot the objectives
    # %%
    # Highlight the behavioural indices 
    # (i.e., KGE, NSE, PBIAS are in some preferred range)
    behavioral_indices = swplt.create_behavioral_indices(df,
                                                         objectives)


    # Plot the objectives
    swplt.plot_objectives(df, 
                            ['node_merge_distance'], 
                            objectives, 
                            pd.Series([False] * df.shape[0]),
                            plot_fid)


    # %% [markdown]
    # ## Perform Sensitivity Analysis
    # %%

    # Formulate the SALib problem
    problem = experimenter.formulate_salib_problem(parameters)

    # Calculate any missing samples
    n_ideal = pd.DataFrame(
        experimenter.generate_samples(parameters_to_select=parameters,
        N=2**config['sample_magnitude'])
        ).iter.nunique()
    missing_iters = set(range(n_ideal)).difference(df.iter)
    if missing_iters:
        logger.warning(f"Missing {len(missing_iters)} iterations")

    # Perform the sensitivity analysis for groups
    problem['outputs'] = objectives
    rg = {objective: sobol.analyze(
                problem, 
                (
                    df[objective]
                    .iloc[0:
                                    (2**(config['sample_magnitude'] + 1) *\
                                      (len(set(problem['groups'])) + 1))]
                    .fillna(df[objective].median())
                    .values
                ),
                print_to_console=False
            ) 
            for objective in objectives}

    # Perform the sensitivity analysis for parameters
    problemi = problem.copy()
    del problemi['groups']
    ri = {objective: sobol.analyze(problemi, 
                        (
                            df[objective]
                            .fillna(df[objective].median())
                            .values
                        ),
                        print_to_console=False) 
                        for objective in objectives}

    # Barplot of sensitvitiy indices
    for r_, groups in zip([rg,ri],  ['groups','parameters']):
        swplt.plot_sensitivity_indices(r_, 
                                     objectives, 
                                     plot_fid / f'{groups}_indices.png')

    # Heatmap of sensitivity indices    
    for r_, groups in zip([rg,ri],  ['groups','parameters']):
        swplt.heatmaps(r_, plot_fid / f'heatmap_{groups}_indices.png',problem)
