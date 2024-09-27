"""Perform sensitivity analysis on the results of the model runs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.integrate import cumulative_trapezoid as cumtrapz
from sklearn.preprocessing import MinMaxScaler
from swmmanywhere.filepaths import check_bboxes
from swmmanywhere.metric_utilities import metrics
from swmmanywhere.swmmanywhere import load_config
from tqdm import tqdm

from swmmanywhere_paper.mappings import metric_mapping, param_mapping


def plot_fig78(base_dir):
    """Create CDF estimates of the parameter values."""
    # %% [markdown]
    # ## Initialise directories and load results
    # %%
    # Load the configuration file and extract relevant data
    projects = [
        "cranbrook_node_1439.1",
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
        config_path = base_dir / project / "config.yml"
        config = load_config(config_path, validation=False)
        config["base_dir"] = base_dir / project
        objectives = config["metric_list"]
        parameters = config["parameters_to_sample"]

        # Load the results
        bbox = check_bboxes(config["bbox"], config["base_dir"])
        results_dir = config["base_dir"] / f"bbox_{bbox}" / "results"
        results_dir.make_dir(exist_ok=True, parents=True)
        fids = list(results_dir.glob("*_metrics.csv"))
        dfs = [pd.read_csv(fid) for fid in tqdm(fids, total=len(fids))]

        # Concatenate the results
        df = pd.concat(dfs).reset_index(drop=True)
        df["project"] = project

        df = df.loc[:, ~df.columns.str.contains("subcatchment")]
        df = df.loc[:, ~df.columns.str.contains("grid")]
        df = df.drop("bias_flood_depth", axis=1)
        df[df == np.inf] = None
        df = df.sort_values(by="iter")

        dfp.append(df)

    df = pd.concat(dfp)
    df = df.reset_index(drop=True)
    for obj in [
        "outfall_kge_flooding",
        "outfall_nse_flooding",
        "outfall_nse_flow",
        "outfall_kge_flow",
    ]:
        df.loc[df[obj] < -5, obj] = -5

    objectives = df.columns.intersection(metrics.keys())
    obj_grps = ["flow", "flooding", "outfall"]
    objectives = pd.Series(objectives.rename("objective")).reset_index()
    objectives["group"] = "graph"
    for ix, obj in objectives.iterrows():
        for grp in obj_grps:
            if grp in obj["objective"]:
                objectives.loc[ix, "group"] = grp
                break
    objectives = objectives.sort_values(by=["group", "objective"])

    plot_fid = results_dir.parent / "plots"
    plot_fid.mkdir(exist_ok=True)

    n_panels = len(parameters)
    n_cols = int(n_panels**0.5)
    if n_cols * n_cols < n_panels:
        n_rows = n_cols + 1
    else:
        n_rows = n_cols

    n_cols = 4
    n_rows = 5

    par_mapping = {
        0: [
            "node_merge_distance",
            "outfall_length",
            "max_street_length",
            "river_buffer_distance",
        ],
        1: [
            "chahinian_slope_scaling",
            "chahinian_angle_scaling",
            "length_scaling",
            "contributing_area_scaling",
        ],
        2: [
            "chahinian_slope_exponent",
            "chahinian_angle_exponent",
            "length_exponent",
            "contributing_area_exponent",
        ],
        3: ["max_fr", "min_v", "max_v", "min_depth"],
        4: ["max_depth", "precipitation", None, None],
    }

    # By objective - all projects
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    hl = {}
    for objective in [
        "outfall_nse_flow",
        "outfall_kge_flow",
        "outfall_relerror_flow",
        "outfall_relerror_diameter",
    ]:
        if "nse" in objective:
            weights = df[objective].clip(lower=0)
        elif "kge" in objective:
            weights = df[objective].clip(lower=-0.41) + 0.41
        elif "relerror" in objective:
            weights = df[objective].abs()
        for idx, objs in par_mapping.items():
            for ax, parameter in zip(axs[idx], objs):
                if parameter is None:
                    ax.axis("off")
                    continue
                kde = stats.gaussian_kde(df[parameter], weights=weights)

                x = np.linspace(df[parameter].min(), df[parameter].max(), 100)
                handle = ax.plot(x, kde(x), label=metric_mapping[objective])

                # ax.fill_between(x,
                #                 bounds[parameter][0],
                #                 bounds[parameter][1],
                #                 color = 'gray',
                #                 alpha = 0.1)
                ax.set_title(param_mapping[parameter])
                ax.grid(True)
                hl[objective] = handle

    axs[-1, -1].legend(handles=[x[0] for x in hl.values()])
    fig.tight_layout()
    fig.savefig(plot_fid / "parameter_distributions_byobjective.png")

    # By objective - one projects
    for project in projects:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        hl = {}
        cols = {"flooding": "b", "flow": "r", "graph": "k", "outfall": "g"}
        for objective, grp in objectives.set_index("objective").group.items():
            col = cols[grp]
            if objective == "outfall_relerror_diameter":
                ls = "-."
                lab = "Diameter (RE)"
            else:
                ls = "-"
                if grp == "outfall":
                    lab = "Design metric (RE/KS)"
                elif grp == "flow":
                    lab = "Flow (NSE/KGE/RE)"
                elif grp == "flooding":
                    lab = "Flooding (NSE/KGE/RE)"
                else:
                    lab = "Graph metric (-)"
            df_ = df.loc[df.project == project]
            if "nse" in objective:
                weights = df_[objective].clip(lower=0)
            elif "kge" in objective:
                weights = df_[objective].clip(lower=-0.41) + 0.41
            elif "relerror" in objective:
                weights = 1 - df_[objective].abs().clip(upper=1.0)
            else:
                weights = pd.Series(
                    [x[0] for x in MinMaxScaler().fit_transform(-df_[[objective]])]
                )

            if weights.isna().all() or weights.var() == 0:
                continue
            weights = weights.fillna(0)

            for idx, objs in par_mapping.items():
                for ax, parameter in zip(axs[idx], objs):
                    if parameter is None:
                        ax.axis("off")
                        continue
                    kde = stats.gaussian_kde(df_[parameter], weights=weights)
                    x = np.linspace(df_[parameter].min(), df_[parameter].max(), 100)

                    # handle = ax.plot(x, kde(x),label=lab,color = col,ls=ls,alpha=0.7)
                    handle = ax.plot(
                        x[1:],
                        cumtrapz(kde(x), x),
                        label=lab,
                        color=col,
                        ls=ls,
                        alpha=0.7,
                    )
                    # ax.fill_between(x,
                    #              bounds[parameter][0],
                    #              bounds[parameter][1],
                    #              color = 'gray',
                    #              alpha = 0.1)
                    ax.set_xlabel(param_mapping[parameter])
                    ax.grid(True)
                    if ax is not axs[idx][0]:
                        ax.set_yticklabels([])
                    else:
                        ax.set_ylabel("CDF(x)")
                    hl[lab] = handle

        axs[-1, -1].legend(handles=[x[0] for x in hl.values()])
        fig.tight_layout()
        fig.savefig(plot_fid / f"parameter_distributions_byobjective_{project}.png")

    # By project - outfall_nse_flow
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    hl = {}

    objective = "outfall_nse_flow"
    colls = {
        "cranbrook_node_1439.1": {"col": "k", "ls": "-", "lw": 1},
        "bellinge_G62F060_G61F180_l1": {"col": "r", "ls": "-.", "lw": 0.5},
        "bellinge_G72F550_G72F010_l1": {"col": "r", "ls": "-", "lw": 0.5},
        "bellinge_G60F61Y_G60F390_l1": {"col": "b", "ls": "-.", "lw": 0.5},
        "bellinge_G74F150_G74F140_l1": {"col": "b", "ls": "-", "lw": 0.5},
        "bellinge_G80F390_G80F380_l1": {"col": "g", "ls": "-.", "lw": 0.5},
        "bellinge_G72F800_G72F050_l1": {"col": "g", "ls": "-", "lw": 0.5},
        "bellinge_G73F000_G72F120_l1": {"col": "c", "ls": "-", "lw": 0.5},
    }
    for project in projects:
        df_ = df.loc[df.project == project]
        if objective == "outfall_relerror_diameter":
            lab = "Diameter (RE)"
        else:
            if grp == "outfall":
                lab = "Design metric (RE/KS)"
            elif grp == "flow":
                lab = "Flow (NSE/KGE/RE)"
            elif grp == "flooding":
                lab = "Flooding (NSE/KGE/RE)"
            else:
                lab = "Graph metric (-)"
        df_ = df.loc[df.project == project]
        if "nse" in objective:
            weights = df_[objective].clip(lower=0)
        elif "kge" in objective:
            weights = df_[objective].clip(lower=-0.41) + 0.41
        elif "relerror" in objective:
            weights = 1 - df_[objective].abs().clip(upper=1.0)
        else:
            weights = pd.Series(
                [x[0] for x in MinMaxScaler().fit_transform(-df_[[objective]])]
            )

        if weights.isna().all() or weights.var() == 0:
            continue
        weights = weights.fillna(0)
        for idx, objs in par_mapping.items():
            for ax, parameter in zip(axs[idx], objs):
                if parameter is None:
                    ax.axis("off")
                    continue
                kde = stats.gaussian_kde(df_[parameter], weights=weights)

                x = np.linspace(df_[parameter].min(), df_[parameter].max(), 100)
                handle = ax.plot(
                    x[1:],
                    cumtrapz(kde(x), x),
                    label=project,
                    color=colls[project]["col"],
                    ls=colls[project]["ls"],
                    lw=1,
                )
                # ax.fill_between(x,
                #                 bounds[parameter][0],
                #                 bounds[parameter][1],
                #                 color = 'gray',
                #                 alpha = 0.1)
                ax.set_xlabel(param_mapping[parameter])
                ax.grid(True)
                if ax is not axs[idx][0]:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel("CDF(x)")
                hl[project] = handle

    axs[-1, -1].legend(handles=[x[0] for x in hl.values()])
    fig.tight_layout()
    fig.savefig(plot_fid / f"parameter_distributions_byproject_{objective}.png")
