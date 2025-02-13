"""Perform sensitivity analysis on the results of the model runs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from SALib.analyze import sobol
from swmmanywhere.filepaths import check_bboxes
from swmmanywhere.logging import logger
from swmmanywhere.metric_utilities import metrics
from swmmanywhere.swmmanywhere import load_config
from tqdm import tqdm

from swmmanywhere_paper import experimenter
from swmmanywhere_paper.mappings import metric_mapping, param_mapping

# %% [markdown]
# ## Initialise directories and load results
# %%
# Load the configuration file and extract relevant data


def plot_fig5(base_dir):
    """Create fig 5 and supplement of heatmaps."""
    projects = [
        {
            "cranbrook_node_1439.1": "Cran Brook",
            "bellinge_G60F61Y_G60F390_l1": "Bellinge 1 (largest)",
            "bellinge_G72F800_G72F050_l1": "Bellinge 7 (smallest)",
        },
        {
            "bellinge_G73F000_G72F120_l1": "Bellinge 2",
            "bellinge_G80F390_G80F380_l1": "Bellinge 3",
            "bellinge_G72F550_G72F010_l1": "Bellinge 4",
            "bellinge_G62F060_G61F180_l1": "Bellinge 5",
            "bellinge_G74F150_G74F140_l1": "Bellinge 6",
        },
    ]
    fids = ["fig5.svg", "fig_sup.svg"]
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    for projects_, fid_ in zip(projects, fids):
        fid = plot_dir / fid_
        make_heatmap_subplots(base_dir, projects_, fid)


def make_heatmap_subplots(base_dir, projects, plot_fid):
    """Create heatmap of sensitivity indices."""
    ris = []
    for project, tlab in projects.items():
        config_path = base_dir / project / "config.yml"
        config = load_config(config_path, validation=False)
        config["base_dir"] = base_dir / project
        parameters = config["parameters_to_sample"]

        # Load the results
        bbox = check_bboxes(config["bbox"], config["base_dir"])
        results_dir = config["base_dir"] / f"bbox_{bbox}" / "results"
        fids = list(results_dir.glob("*_metrics.csv"))
        dfs = [pd.read_csv(fid) for fid in tqdm(fids, total=len(fids))]

        # Concatenate the results
        df = pd.concat(dfs).reset_index(drop=True)

        df = df.loc[:, ~df.columns.str.contains("subcatchment")]
        df = df.loc[:, ~df.columns.str.contains("grid")]
        df = df.drop("bias_flood_depth", axis=1)

        df[df == np.inf] = None
        df = df.sort_values(by="iter")

        # Clip anoms
        for obj in [
            "outfall_kge_flooding",
            "outfall_nse_flooding",
            "outfall_nse_flow",
            "outfall_kge_flow",
        ]:
            df.loc[df[obj] < -5, obj] = -5

        # Format order
        objectives = df.columns.intersection(metrics.keys())
        obj_grps = ["flow", "flooding", "outfall"]
        objectives = pd.Series(objectives.rename("objective")).reset_index()
        objectives["group"] = "graph"
        for ix, obj in objectives.iterrows():
            for grp in obj_grps:
                if grp in obj["objective"]:
                    objectives.loc[ix, "group"] = grp
                    break
        objectives = objectives.sort_values(by=["group", "objective"]).objective

        problem = experimenter.formulate_salib_problem(parameters)
        parameters_order = pd.DataFrame([problem["names"], problem["groups"]]).T
        parameters_order.columns = ["parameter", "group"]
        parameters_order = parameters_order.sort_values(
            by=["group", "parameter"]
        ).parameter

        # Formulate the SALib problem
        problem = experimenter.formulate_salib_problem(parameters)

        # Calculate any missing samples
        n_ideal = pd.DataFrame(
            experimenter.generate_samples(
                parameters_to_select=parameters, N=2 ** config["sample_magnitude"]
            )
        ).iter.nunique()

        missing_iters = set(range(n_ideal)).difference(df.iter)
        if missing_iters:
            logger.warning(f"Missing {len(missing_iters)} iterations")

        # Perform the sensitivity analysis for groups
        problem["outputs"] = objectives

        # Perform the sensitivity analysis for parameters
        problemi = problem.copy()
        del problemi["groups"]
        ri = {
            objective: sobol.analyze(
                problemi,
                (df[objective].fillna(df[objective].median()).values),
                print_to_console=False,
            )
            for objective in objectives
        }

        ris.append(ri)

    heatmaps(ris, plot_fid, problem, projects.values())


def heatmaps(
    rs: list[dict[str, pd.DataFrame]], plot_fid: Path, problem=None, sups=[""]
):
    """Plot heatmap of sensitivity indices.

    Args:
        rs (list[dict[str, pd.DataFrame]]): A list of dictionaries containing
            the sensitivity indices as produced by SALib.analyze.
        plot_fid (Path): The file to save the plots to.
        problem (dict): The problem formulation.
        sups (list[str]): The titles for the subplots.
    """
    f, axs_ = plt.subplots(2, len(rs), figsize=(4 * len(rs), 10))
    axs_ = axs_.T
    for rd, axs, sup in zip(rs, axs_, sups):
        totals_ = []
        interactions_ = []
        firsts_ = []
        for objective, r in rd.items():
            total, first, second = r.to_df()
            interaction = total["ST"] - first["S1"]

            total = total["ST"].to_dict()
            total["objective"] = objective
            totals_.append(total)

            interaction = interaction.to_dict()
            interaction["objective"] = objective
            interactions_.append(interaction)

            first = first["S1"].to_dict()
            first["objective"] = objective
            firsts_.append(first)

        totals = pd.DataFrame(totals_).set_index("objective")
        interactions = pd.DataFrame(interactions_).set_index("objective")
        firsts = pd.DataFrame(firsts_).set_index("objective")

        if set(problem["names"]) == set(totals.columns):
            df = pd.DataFrame([problem["names"], problem["groups"]]).T
            df.columns = ["parameter", "group"]
            df = df.sort_values(by=["group", "parameter"])
            totals = totals[df.parameter]
            interactions = interactions[df.parameter]
            firsts = firsts[df.parameter]

        obj_grps = ["flow", "flooding", "outfall"]
        objectives = totals.reset_index()[["objective"]]
        objectives["group"] = "graph"
        for ix, obj in objectives.iterrows():
            for grp in obj_grps:
                if grp in obj["objective"]:
                    objectives.loc[ix, "group"] = grp
                    break
        objectives = objectives.sort_values(by=["group", "objective"])
        totals = totals.loc[objectives["objective"]]
        totals.index.rename("ST", inplace=True)
        interactions = interactions.loc[objectives["objective"]]
        firsts = firsts.loc[objectives["objective"]]
        firsts.index.rename("S1", inplace=True)

        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        cmap.set_bad(color="grey")  # Color for NaN values
        cmap.set_under(color="#d5f5eb")  # Color for 0.0 values

        sns.heatmap(
            firsts.rename(columns=param_mapping, index=metric_mapping).loc[
                metric_mapping.values(), param_mapping.values()
            ],
            vmin=1 / 100,
            linewidth=0.5,
            ax=axs[0],
            cmap=cmap,
            cbar=False,
            vmax=1.0,
        )
        axs[0].set_xticklabels([])
        sns.heatmap(
            totals.rename(columns=param_mapping, index=metric_mapping).loc[
                metric_mapping.values(), param_mapping.values()
            ],
            vmin=1 / 100,
            linewidth=0.5,
            ax=axs[1],
            cmap=cmap,
            cbar=False,
            vmax=1.0,
        )
        axs[0].set_title(sup)
        axs[0].set_aspect("equal", adjustable="box")
        axs[1].set_aspect("equal", adjustable="box")
        if rd is not rs[0]:
            axs[0].set_yticklabels([])
            axs[1].set_yticklabels([])
            axs[0].set_ylabel("")
            axs[1].set_ylabel("")
    cbar_ax = f.add_axes(
        [0.9, 0.15, 0.02, 0.7]
    )  # position [left, bottom, width, height]
    plt.subplots_adjust(right=0.85)
    f.colorbar(axs[1].collections[0], cax=cbar_ax)
    # f.tight_layout()
    f.savefig(plot_fid)
    plt.close(f)
