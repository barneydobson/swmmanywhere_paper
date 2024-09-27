"""Perform sensitivity analysis on the results of the model runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from swmmanywhere.filepaths import check_bboxes
from swmmanywhere.metric_utilities import metrics
from swmmanywhere.swmmanywhere import load_config
from tqdm import tqdm

from swmmanywhere_paper import experimenter


def plot_fig6():
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
        config_path = base_dir / project / 'config.yml'
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
        plot_objectives(df, 
                                ['node_merge_distance'], 
                                objectives, 
                                pd.Series([False] * df.shape[0]),
                                plot_fid)

def setup_axes(ax: plt.Axes, 
               df: pd.DataFrame,
               parameter: str, 
               objective: str, 
               behavioral_indices: pd.Series
               ):
    """Set up the axes for plotting.

    Args:
        ax (plt.Axes): The axes to plot on.
        df (pd.DataFrame): A dataframe containing the results.
        parameter (list[str]): The parameter to plot.
        objective (list[str]): The objective to plot.
        behavioral_indices (pd.Series): A tuple of two series
            see create_behavioral_indices.
    """
    ax.scatter(df[parameter], df[objective],c = 'k',s=1,marker='.',linewidths=0.1,edgecolors='face')
    ax.scatter(df.loc[behavioral_indices, parameter], 
               df.loc[behavioral_indices, objective], s=2, c='r')
    
    #ax.set_yscale('symlog')
    ax.set_ylabel(metric_mapping[objective])
    ax.set_xlabel(param_mapping[parameter])
    ax.grid(True)
    if 'nse' in objective:
        ax.set_ylim([0, 1])
    if 'kge' in objective:
        ax.set_ylim([-0.41,1])

def plot_objectives(df: pd.DataFrame, 
                    parameters: list[str], 
                    objectives: list[str], 
                    behavioral_indices: pd.Series,
                    plot_fid: Path):
    """Plot the objectives.

    Args:
        df (pd.DataFrame): A dataframe containing the results.
        parameters (list[str]): A list of parameters to plot.
        objectives (list[str]): A list of objectives to plot.
        behavioral_indices (pd.Series): A tuple of two series
            see create_behavioral_indices.
        plot_fid (Path): The directory to save the plots to.
    """
    n_panels = len(objectives)
    n_cols = int(n_panels**0.5)
    if n_cols * n_cols < n_panels:
        n_rows = n_cols + 1
    else:
        n_rows = n_cols

    n_cols = 3
    n_rows = 6
    
    col_mapping = {
        0: ["outlet_relerror_length", "outlet_relerror_npipes", "outlet_relerror_nmanholes"],
        1: ["nc_deltacon0","nc_laplacian_dist","nc_vertex_edge_distance"],
        2: ["kstest_edge_betweenness","kstest_betweenness", None],
        3: ["outlet_relerror_diameter","outlet_kstest_diameters", None],
        4: ["outlet_nse_flow","outlet_kge_flow","outlet_relerror_flow"],
        5: ["outlet_nse_flooding","outlet_kge_flooding","outlet_relerror_flooding"],
    }

    
def add_threshold_lines(ax, objective, xmin, xmax):
    """Add threshold lines to the axes.

    Args:
        ax (plt.Axes): The axes to plot on.
        objective (list[str]): The objective to plot.
        xmin (float): The minimum x value.
        xmax (float): The maximum x value.
    """
    thresholds = {
        'relerror': [-0.1, 0.1],
        'nse': [0.7],
        'kge': [0.7]
    }
    for key, values in thresholds.items():
        if key in objective:
            for value in values:
                ax.plot([xmin, xmax], [value, value], 'k--')

    for parameter in param_mapping.keys():
        if parameter not in parameters:
            continue
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        for idx, objs in col_mapping.items():
            for ax, objective in zip(axs[idx], objs):
                if objective is None:
                    ax.axis('off')
                    continue
                setup_axes(ax, df, parameter, objective, behavioral_indices)
                add_threshold_lines(ax, 
                                    objective, 
                                    df[parameter].min(), 
                                    df[parameter].max())
                ax.set_xlabel('')
        
        fig.suptitle(f"{parameter.replace('_',' ').title()} [m]")
        fig.tight_layout()
        fig.savefig(plot_fid / f"{parameter.replace('_', '-')}.png", dpi=500)
        plt.close(fig)