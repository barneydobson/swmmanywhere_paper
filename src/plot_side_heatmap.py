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
from swmmanywhere.preprocessing import check_bboxes
from swmmanywhere.swmmanywhere import load_config
from swmmanywhere.metric_utilities import metrics
# %% [markdown]
# ## Initialise directories and load results
# %%
# Load the configuration file and extract relevant data
projects = [ "cranbrook_node_1439.1",
            "bellinge_G73F000_G72F120_l1",
            ]
ris = []
for project in projects:

    base_dir = Path.home() / "Documents" / "data" / "swmmanywhere"
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
    for obj in ['outlet_kge_flooding', 'outlet_nse_flooding', 'outlet_nse_flow', 'outlet_kge_flow']:
        df.loc[df[obj] < -5, obj] = -5


    # Format order
    objectives = df.columns.intersection(metrics.keys())
    obj_grps = ['flow','flooding','outlet']
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
    
    ris.append(ri)

swplt.heatmaps(ris[0], plot_fid / f'heatmap_side.png',problem,ris[1], projects)