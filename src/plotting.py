"""Plotting SWMManywhere.

A module with some built in plotting for SWMManywhere.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from paretoset import paretoset
import seaborn as sns
from SALib.plotting.bar import plot as barplot
from scipy import stats

from swmmanywhere import metric_utilities
from swmmanywhere.geospatial_utilities import graph_to_geojson
from swmmanywhere.graph_utilities import load_graph
from swmmanywhere.parameters import MetricEvaluation
from swmmanywhere.filepaths import filepaths_from_yaml
from swmmanywhere.swmmanywhere import load_config
from swmmanywhere_paper.src import utilities
from swmmanywhere_paper.src.mappings import metric_mapping, param_mapping

class ResultsPlotter():
    """Plotter object."""
    def __init__(self, 
                 address_path: Path,
                 real_dir: Path,
                 ):
        """Initialise results plotter.
        
        This plotter loads the results, graphs and subcatchments from the two
        yaml files. It provides a central point for plotting the results without
        needing to reload data.

        Args:
            address_path (Path): The path to the address yaml file.
            real_dir (Path): The path to the directory containing the real data.
        """
        # Load the addresses
        self.addresses = filepaths_from_yaml(address_path)
        self.config = load_config(self.addresses.project / 'config.yml',
                                  validation=False)
        # Create the plot directory
        self.plotdir = self.addresses.model / 'plots'
        self.plotdir.mkdir(exist_ok = True)

        # Load synthetic and real results
        self._synthetic_results = pd.read_parquet(
            self.addresses.model / 'results.parquet')
        self._synthetic_results.id = self._synthetic_results.id.astype(str)

        self._real_results = pd.read_parquet(real_dir / 'real_results.parquet')
        self._real_results.id = self._real_results.id.astype(str)

        # Load the synthetic and real graphs
        self._synthetic_G = load_graph(self.addresses.graph)
        self._synthetic_G = nx.relabel_nodes(self._synthetic_G,
                         {x : str(x) for x in self._synthetic_G.nodes})
        nx.set_node_attributes(self._synthetic_G,
            {u : str(d.get('outlet',None)) for u,d 
             in self._synthetic_G.nodes(data=True)},
            'outlet')

        self._real_G = load_graph(real_dir / 'graph.json')
        self._real_G = nx.relabel_nodes(self._real_G,
                        {x : str(x) for x in self._real_G.nodes})
        
        # Calculate the slope
        calculate_slope(self._synthetic_G)
        calculate_slope(self._real_G)

        # Load the subcatchments
        self._synthetic_subcatchments = gpd.read_file(self.addresses.subcatchments)
        self._real_subcatchments = gpd.read_file(real_dir / 'subcatchments.geojson')

        # Calculate outlets
        self.sg_syn, self.syn_outlet = metric_utilities.best_outlet_match(self.synthetic_G, 
                                                                self.real_subcatchments)
        self.sg_real, self.real_outlet = metric_utilities.dominant_outlet(self.real_G, 
                                                                self.real_results)
        
        # Calculate travel times
        self._real_G = utilities.calc_flowtimes(self.real_G, 
                                 self.real_results,
                                 self.real_outlet)
        self._synthetic_G = utilities.calc_flowtimes(self.synthetic_G,
                                 self.synthetic_results,
                                 self.syn_outlet)

    def __getattr__(self, name):
        """Because these are large datasets, return a copy."""
        if name in dir(self):
            return getattr(self, name)
        elif f'_{name}' in dir(self): 
            return getattr(self, f'_{name}').copy()
        raise AttributeError(f"'ResultsPlotter' object has no attribute '{name}'")

    def make_all_plots(self):
        """make_all_plots."""
        f,axs = plt.subplots(2,3,figsize = (10,7.5))
        self.outlet_plot('flow', ax_ = axs[0,0])
        self.outlet_plot('flooding', ax_ = axs[0,1])
        self.shape_relerror_plot('grid')
        self.shape_relerror_plot('subcatchment')
        self.design_distribution(value='diameter', ax_ = axs[0,2])
        self.design_distribution(value='chamber_floor_elevation', ax_ = axs[1,0])
        self.design_distribution(value='slope', ax_ = axs[1,1])
        self.design_distribution(value='travel_time', ax_ = axs[1,2])
        self.annotate_flows_and_depths()
        f.tight_layout()
        f.savefig(self.plotdir / 'all_plots.png')

    def annotate_flows_and_depths(self):
        """annotate_flows_and_depths.
        
        Annotate maximum flow and flood values on the edges/nodes of the graph.
        Save these in the plotdir.
        """
        synthetic_max = self.synthetic_results.groupby(['id','variable']).max()
        real_max = self.real_results.groupby(['id','variable']).max()

        syn_G = self.synthetic_G
        for u,v,d in syn_G.edges(data=True):
            d['flow'] = synthetic_max.loc[(d['id'],'flow'),'value']
        real_G = self.real_G
        for u,v,d in real_G.edges(data=True):
            d['flow'] = real_max.loc[(d['id'],'flow'),'value']

        for u,d in syn_G.nodes(data=True):
            d['flood'] = synthetic_max.loc[(u,'flooding'),'value']
        for u,d in real_G.nodes(data=True):
            d['flood'] = real_max.loc[(u,'flooding'),'value']

        graph_to_geojson(syn_G, 
                         self.plotdir / 'synthetic_graph_nodes.geojson',
                         self.plotdir / 'synthetic_graph_edges.geojson',
                         syn_G.graph['crs'])
        graph_to_geojson(real_G, 
                         self.plotdir / 'real_graph_nodes.geojson',
                         self.plotdir / 'real_graph_edges.geojson',
                         real_G.graph['crs'])

    def outlet_plot(self, 
                    var: str = 'flow',
                    fid: Path | None = None,
                    ax_ = None,
                    cutoff = pd.to_datetime('2000-01-01 03:00:00')):
        """Plot flow/flooding at outlet.

        If an ax is provided, plot on that ax, otherwise create a new figure and
        save it to the provided fid (or plot directory if not provided).
        
        Args:
            var (str, optional): The variable to plot (flow or flooding). 
                Defaults to 'flow'.
            fid (Path, optional): The file to save the plot to. Defaults to None.
            ax_ ([type], optional): The axes to plot on. Defaults to None.
        """            
        if var == 'flow':
            # Identify synthetic and real arcs that flow into the best outlet node
            syn_arc = [d['id'] for u,v,d in self.synthetic_G.edges(data=True)
                        if v == self.syn_outlet]
            real_arc = [d['id'] for u,v,d in self.real_G.edges(data=True)
                    if v == self.real_outlet]
        elif var == 'flooding':
            # Use all nodes in the outlet match subgraphs
            syn_arc = list(self.sg_syn.nodes)
            real_arc = list(self.sg_real.nodes)
        df = metric_utilities.align_by_id(self.synthetic_results,
                                           self.real_results,
                                           var,
                                           syn_arc,
                                           real_arc
                                           )
        if not ax_:
            f, ax = plt.subplots()
        else:
            ax = ax_
        df.value_real.plot(ax=ax, color = 'b', linestyle = '-')
        df.value_syn.plot(ax=ax, color = 'r', linestyle = '--')
        plt.legend(['synthetic','real'])
        ax.set_xlabel('time')
        ax.set_xlim([df.index.min(),cutoff])
        ax.grid(True)
        if var == 'flow':
            unit = 'l/s'
        elif var == 'flooding':
            unit = 'l'
        ax.set_ylabel(f'{var.title()} ({unit})')
        if not ax_:
            f.savefig(self.plotdir / f'outlet-{var}.png')

    def shape_relerror_plot(self, shape: str = 'grid'):
        """shape_relerror_plot.
        
        Plot the relative error of the shape. Either at 'grid' or 'subcatchment' 
        scale. Saves results to the plotdir.
        
        Args:
            shape (str, optional): The shape to plot. Defaults to 'grid'.
        """            
        variable = 'flooding'
        if shape == 'grid':
            scale = self.config.get('metric_evaluation', {}).get('grid_scale',1000)
            shapes = metric_utilities.create_grid(self.real_subcatchments.total_bounds,
                                                scale)
            shapes.crs = self.real_subcatchments.crs
        elif shape == 'subcatchment':
            shapes = self.real_subcatchments
            shapes = shapes.rename(columns={'id':'sub_id'})
        else:
            raise ValueError("shape must be 'grid' or 'subcatchment'")
        # Align the results
        results = metric_utilities.align_by_shape(variable,
                                synthetic_results = self.synthetic_results,
                                real_results = self.real_results,
                                shapes = shapes,
                                synthetic_G = self.synthetic_G,
                                real_G = self.real_G)
        # Calculate the relative error
        val = (
            results
            .groupby('sub_id')
            .apply(lambda x: metric_utilities.relerror(x.value_real, x.value_syn))
            .rename('relerror')
            .reset_index()
        )
        total = (
            results
            .groupby('sub_id')
            [['value_real','value_syn']]
            .sum()
        )
        # Merge with shapes
        shapes = pd.merge(shapes[['geometry','sub_id']],
                          val,
                          on ='sub_id')
        shapes = pd.merge(shapes, 
                          total,
                          on = 'sub_id')
        shapes.to_file(self.plotdir / f'{shape}-relerror.geojson',
                       driver='GeoJSON')

    def recalculate_metrics(self, metric_list: list[str] | None = None):
        """recalculate_metrics.
        
        Recalculate the metrics for the synthetic and real results, if no
        metric_list is provided, use the default metric_list from the config.

        Args:
            metric_list (list[str], optional): The metrics to recalculate. 
                Defaults to None.

        Returns:
            dict: A dictionary of the recalculated metrics.
        """
        if not metric_list:
            metric_list_ = self.config['metric_list']
        else:
            metric_list_ = metric_list
        if 'metric_evaluation' in self.config.get('parameter_overrides', {}):
            metric_evaluation = MetricEvaluation(
                **self.config['parameter_overrides']['metric_evaluation'])
        else:
            metric_evaluation = MetricEvaluation()

        return metric_utilities.iterate_metrics(self.synthetic_results, 
                                  self.synthetic_subcatchments,
                                  self.synthetic_G,
                                  self.real_results,
                                  self.real_subcatchments,
                                  self.real_G,
                                  metric_list_,
                                  metric_evaluation
                                  )

    def design_distribution(self, 
                            value: str = 'diameter',
                            weight: str='length',
                            ax_ = None):
        """design_distribution.

        Plot the distribution of a value in the graph. Saves the plot to the
        provided axes, if not, saves to plotdir.

        Args:
            value (str, optional): The value to plot. Defaults to 'diameter'.
            weight (str, optional): The weight to use. Defaults to 'length'.
            ax_ ([type], optional): The axes to plot on. Defaults to None.
        """
        syn_v, syn_cdf = weighted_cdf(self.synthetic_G,value,weight)
        real_v, real_cdf = weighted_cdf(self.real_G,value,weight)
        if not ax_:
            f, ax = plt.subplots()
        else:
            ax = ax_
        ax.plot(real_v,real_cdf, 'b')
        ax.plot(syn_v,syn_cdf, '--r')
        if value == 'slope':
            unit = 'm/m'
            ax.set_xlim([min([x for x in syn_v]), 
                         max([x for x in syn_v])])
            ax.plot([-1/100,-1/100],[0,1],':c')
            ax.plot([10/100,10/100],[0,1],':c')
        elif value == 'chamber_floor_elevation':
            unit = 'mASL'
        elif value == 'travel_time':
            unit = 's'
            ax.set_xscale('symlog')
        else:
            unit = 'm'
        ax.set_xlabel(f'{value.title()} ({unit})')
        ax.set_ylabel('P(X <= x)')
        plt.legend(['real','synthetic'])
        ax.grid(True)
        if not ax_:
            f.savefig(self.plotdir / f'{value}_{weight}_distribution.png')  
        
def calculate_slope(G: nx.Graph):
    """calculate_slope.
    
    Calculate the slope of the edges in the graph in place.
    
    Args:
        G (nx.Graph): The graph to calculate the slope for.
    """
    nx.set_edge_attributes(
        G,
        {
            (u, v, k): (G.nodes[u]['chamber_floor_elevation'] - \
                        G.nodes[v]['chamber_floor_elevation']) / d['length']
            for u, v, k, d in G.edges(data=True, keys=True)
        },
        'slope'
    )
    
def weighted_cdf(G: nx.Graph, value: str = 'diameter', weight: str = 'length'):
    """weighted_cdf.
    
    Calculate the weighted cumulative distribution function of a value in the
    graph.
    
    Args:
        G (nx.Graph): The graph to calculate the cdf for.
        value (str, optional): The value to calculate the cdf for. Defaults to 
            'diameter'.
        weight (str, optional): The weight to use. Defaults to 'length'.

    Returns:
        tuple[list, list]: The values and the cdf.
    """
    # Create a DataFrame from the provided lists
    if value in ['diameter','slope']:
        data = pd.DataFrame([
            {value: d[value], 'weight': d.get(weight,1)}
            for u,v,d in G.edges(data=True)
        ])
    elif value in ['chamber_floor_elevation','travel_time']:
        data = pd.DataFrame([
                    {value: d[value], 'weight': d.get(weight,1)}
                    for u,d in G.nodes(data=True)
        ])        

    # Sort by diameter
    data_sorted = data.sort_values(by=value)

    # Calculate cumulative weights
    cumulative_weights = data_sorted['weight'].cumsum()

    # Normalize the cumulative weights to form the CDF
    cumulative_weights /= cumulative_weights.iloc[-1]

    return data_sorted[value].tolist(), cumulative_weights.tolist()

def create_behavioral_indices(df: pd.DataFrame,
                              objectives) -> pd.Series:
    """Create behavioral indices for a dataframe.

    Args:
        df (pd.DataFrame): A dataframe containing the results.

    Returns:
        tuple [pd.Series]: A tuple of two series, the first is the
            behavioural indices for 'strict' objectives (KGE/NSE), the second 
            is the behavioural indices for less strict objectives (relerror).
    """
    behavioural_ind_nse = ((df.loc[:, df.columns.str.contains('nse')] > 0.7) & \
                           (df.loc[:, df.columns.str.contains('nse')] < 1)).any(axis=1)
    behavioural_ind_kge = ((df.loc[:, df.columns.str.contains('kge')] > 0.7) &\
                            (df.loc[:, df.columns.str.contains('kge')] < 1)).any(axis=1)
    behavioural_ind_relerror = (df.loc[:, 
                                   df.columns.str.contains('relerror')].abs() < 0.1
                            ).any(axis=1)
    
    df_ = df[objectives].copy()

    for objective in objectives:
        if 'relerror' in objective:
            df_[objective] = df_[objective].abs()
    priority_objs = ['outlet_kge_flow',
          'outlet_kge_flooding',
        #  'grid_kge_flooding',
         # 'grid_nse_flooding',
          'outlet_nse_flow',
          'outlet_nse_flooding',
          'outlet_relerror_flow',
          'outlet_relerror_flooding',
          #'grid_relerror_flooding'
          ]
    mask = paretoset(df_[priority_objs],
                     sense = ['max' 
                              if any([s in o for s in ['kge','nse']])
                              else 'min' for o in priority_objs])

    combined = behavioural_ind_nse & behavioural_ind_kge & behavioural_ind_relerror & mask
    
    return mask

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

    for parameter in parameters:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        for ax, objective in zip(axs.flat, objectives):
            setup_axes(ax, df, parameter, objective, behavioral_indices)
            add_threshold_lines(ax, 
                                objective, 
                                df[parameter].min(), 
                                df[parameter].max())
        
        fig.suptitle(parameter)
        fig.tight_layout()
        fig.savefig(plot_fid / f"{parameter.replace('_', '-')}.png", dpi=500)
        plt.close(fig)
    
def plot_distributions(df: pd.DataFrame, 
                    parameters: list[str], 
                    objectives: list[str], 
                    plot_fid: Path,
                    axs= None):
    """Plot the objectives.

    Args:
        df (pd.DataFrame): A dataframe containing the results.
        parameters (list[str]): A list of parameters to plot.
        behavioral_indices (pd.Series): A tuple of two series
            see create_behavioral_indices.
        plot_fid (Path): The directory to save the plots to.
        axs (plt.Axes, optional): The axes to plot on. Defaults to None.
    """
    n_panels = len(parameters)
    n_cols = int(n_panels**0.5)
    if n_cols * (n_cols + 1) >= n_panels:
        n_rows = n_cols + 1
    else:
        n_rows = n_cols
    if not isinstance(axs,np.ndarray):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        sf = True
    else:
        sf = False
        fig = plt.gcf()
    from sklearn.preprocessing import StandardScaler
    df_ = df.copy()
    df_ = df_.dropna(axis=1,how='all').dropna(axis=0,how='any')
    
    scaler = StandardScaler()
    
    df_ = pd.DataFrame(data=scaler.fit_transform(df_),
                       columns = df_.columns,
                       index = df_.index)
#    objs = set(objectives).intersection(df_.columns)
#    for objective in objs:
#        if 'relerror' in objective:
#            df_[objective] = df_[objective].abs()

#    weights = pd.concat([df_[x].rank(ascending = False)
#                         if any([s in x for s in ['kge','nse']])
#                         else df_[x].rank(ascending = True)
#                         for x in objs], 
#                        axis = 1)
#    weights = df_[[x for x in objs if 'nse' in x or 'kge' in x]]
#    weights = weights.dropna(axis=1, how='all').mean(axis=1)
    weights = df_['outlet_nse_flow']
    weights[weights < 0] = 0
    #weights = weights.max() - weights
    for parameter,ax in zip(parameters, axs.flat):
        column_index = df_.columns.get_loc(parameter)
        # Fit the column scaler to the original data column (not the scaled data)
        column_scaler = StandardScaler()

        column_scaler.mean_ = scaler.mean_[column_index]
        column_scaler.scale_ = scaler.scale_[column_index]        
        
        kde = stats.gaussian_kde(df_[parameter], weights = weights)
        x = np.linspace(df_[parameter].min(), df_[parameter].max(), 100)
        ax.plot(column_scaler.inverse_transform(x.reshape(-1,1)), kde(x))
        ax.set_title(parameter.replace('_','\n'))
        ax.grid(True)
    
    fig.tight_layout()
    fig.savefig(plot_fid / f"parameter_distributions.png", dpi=500)
    if sf:
        plt.close(fig)
    
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
    ax.scatter(df[parameter], df[objective], s=0.5, c='b')
    ax.scatter(df.loc[behavioral_indices, parameter], 
               df.loc[behavioral_indices, objective], s=2, c='r')
    
    #ax.set_yscale('symlog')
    ax.set_ylabel(objective)
    ax.set_xlabel(parameter)
    ax.grid(True)
    if 'nse' in objective:
        ax.set_ylim([0, 1])
    if 'kge' in objective:
        ax.set_ylim([-0.41,1])

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

def plot_sensitivity_indices(r_: dict[str, pd.DataFrame],
                             objectives: list[str],
                             plot_fid: Path):
    """Plot the sensitivity indices.

    Args:
        r_ (dict[str, pd.DataFrame]): A dictionary containing the sensitivity 
            indices as produced by SALib.analyze.
        objectives (list[str]): A list of objectives to plot.
        plot_fid (Path): The directory to save the plots to.
    """
    f,axs = plt.subplots(len(objectives),1,figsize=(10,10))
    for ix, ax, (objective, r) in zip(range(len(objectives)), axs, r_.items()):
        total, first, second = r.to_df()
        barplot(total,ax=ax)
        if ix != len(objectives) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([x.replace('_','\n') for x in total.index], 
                                    rotation = 0)
            
        ax.set_ylabel(objective.replace('_','\n'),rotation = 0,labelpad=20)
        ax.get_legend().remove()
    f.tight_layout()
    f.savefig(plot_fid)  
    plt.close(f)

def heatmaps(rs: list[dict[str, pd.DataFrame]],
                             plot_fid: Path,
                             problem = None,
                             sups = ['']):
    """Plot heatmap of sensitivity indices.

    Args:
        r_ (dict[str, pd.DataFrame]): A dictionary containing the sensitivity 
            indices as produced by SALib.analyze.
        plot_fid (Path): The directory to save the plots to.
    """
    if isinstance(rs, list):
        f,axs_ = plt.subplots(2,len(rs),figsize=(14,10))
        axs_ = axs_.T
    else:
        rs = [r_]
        f,axs_ = plt.subplots(2,1,figsize=(10,10))
        axs_ = [axs_]
    for rd, axs, sup in zip(rs,axs_, sups):
        totals = []
        interactions = []
        firsts = []
        for (objective,r) in rd.items():
            total, first, second = r.to_df()
            interaction = total['ST'] - first['S1']
            
            total = total['ST'].to_dict()
            total['objective'] = objective
            totals.append(total)

            interaction = interaction.to_dict()
            interaction['objective'] = objective
            interactions.append(interaction)

            first = first['S1'].to_dict()
            first['objective'] = objective
            firsts.append(first)

        totals = pd.DataFrame(totals).set_index('objective')
        interactions = pd.DataFrame(interactions).set_index('objective')
        firsts = pd.DataFrame(firsts).set_index('objective')

        if set(problem['names']) == set(totals.columns):

            df = pd.DataFrame([problem['names'],problem['groups']]).T
            df.columns = ['parameter','group']
            df = df.sort_values(by=['group','parameter'])
            totals = totals[df.parameter]
            interactions = interactions[df.parameter]
            firsts = firsts[df.parameter]
        
        obj_grps = ['flow','flooding','outlet']
        objectives = totals.reset_index()[['objective']]
        objectives['group'] = 'graph'
        for ix, obj in objectives.iterrows():
            for grp in obj_grps:
                if grp in obj['objective']:
                    objectives.loc[ix,'group'] = grp
                    break
        objectives = objectives.sort_values(by=['group','objective'])
        totals = totals.loc[objectives['objective']]
        totals.index.rename('ST', inplace=True)
        interactions = interactions.loc[objectives['objective']]
        firsts = firsts.loc[objectives['objective']]
        firsts.index.rename('S1' , inplace=True)
        

        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        cmap.set_bad(color='grey')  # Color for NaN values
        cmap.set_under(color='#d5f5eb')  # Color for 0.0 values
        

        sns.heatmap(firsts.rename(columns = param_mapping,
                                  index = metric_mapping), 
                                  vmin = 1/100, 
                                  linewidth=0.5,
                                  ax=axs[0],
                                  cmap=cmap,
                                  cbar = False,
                                  vmax = 1.0)
        axs[0].set_xticklabels([])
        sns.heatmap(totals.rename(columns = param_mapping,
                                  index = metric_mapping), vmin = 1/100, linewidth=0.5,ax=axs[1],cmap=cmap,cbar = False,vmax = 1.0)
        axs[0].set_title(sup)
        if rd is not rs[0]:
            axs[0].set_yticklabels([])
            axs[1].set_yticklabels([])
            axs[0].set_ylabel('')
            axs[1].set_ylabel('')
    cbar_ax = f.add_axes([0.9, 0.15, 0.02, 0.7])  # position [left, bottom, width, height]
    plt.subplots_adjust(right=0.85)
    f.colorbar(axs[1].collections[0], cax=cbar_ax)
    # f.tight_layout()
    f.savefig(plot_fid)
    plt.close(f)
