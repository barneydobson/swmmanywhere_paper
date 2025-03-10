"""The experimenter module is used to sample and run SWMManywhere.

This module is designed to be run in parallel as a jobarray. It generates
parameter samples and runs the SWMManywhere model for each sample. The results
are saved to a csv file in a results directory.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from SALib.sample import sobol

# Set the number of threads to 1 to avoid conflicts with parallel processing
# for pysheds (at least I think that is what is happening)
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from swmmanywhere import swmmanywhere  # noqa: E402
from swmmanywhere.logging import logger  # noqa: E402
from swmmanywhere.parameters import get_full_parameters_flat  # noqa: E402

os.environ["SWMMANYWHERE_VERBOSE"] = "true"


def formulate_salib_problem(
    parameters_to_select: list[str | dict] | None = None,
) -> dict:
    """Formulate a SALib problem for a sensitivity analysis.

    Args:
        parameters_to_select (list, optional): List of parameters to include in
            the analysis, if a list entry is a dictionary, the value is the
            bounds, otherwise the bounds are taken from the parameters file.
            Defaults to None.

    Returns:
        dict: A dictionary containing the problem formulation.
    """
    # Set as empty by default
    parameters_to_select = [] if parameters_to_select is None else parameters_to_select

    # Get all parameters schema
    parameters = get_full_parameters_flat()
    names = []
    bounds = []
    dists = []
    groups = []

    for parameter in parameters_to_select:
        if isinstance(parameter, dict):
            bound = next(iter(parameter.values()))
            parameter = next(iter(parameter))
        else:
            bound = [parameters[parameter]["minimum"], parameters[parameter]["maximum"]]

        names.append(parameter)
        bounds.append(bound)
        dists.append(parameters[parameter].get("dist", "unif"))
        groups.append(parameters[parameter]["category"])
    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
        "dists": dists,
        "groups": groups,
    }


def generate_samples(
    N: int | None = None,
    parameters_to_select: list[str | dict] = [],
    seed: int = 1,
    groups: bool = False,
    calc_second_order: bool = True,
) -> list[dict]:
    """Generate samples for a sensitivity analysis.

    Args:
        N (int, optional): Number of samples to generate. Defaults to None.
        parameters_to_select (list, optional): List of parameters to include in
            the analysis, if a list entry is a dictionary, the value is the
            bounds, otherwise the bounds are taken from the parameters file.
            Defaults to [].
        seed (int, optional): Random seed. Defaults to 1.
        groups (bool, optional): Whether to sample by group, True, or by
            parameter, False (significantly changes how many samples are taken).
            Defaults to False.
        calc_second_order (bool, optional): Whether to calculate second order
            indices. Defaults to True.

    Returns:
        list: A list of dictionaries containing the parameter values.
    """
    problem = formulate_salib_problem(parameters_to_select)

    if N is None:
        N = 2 ** (problem["num_vars"] - 1)

    # If we are not grouping, we need to remove the groups from the problem to
    # pass to SAlib, but we retain the groups information for the output
    # regardless
    problem_ = problem.copy()

    if not groups:
        del problem_["groups"]

    # Sample
    param_values = sobol.sample(
        problem_, N, calc_second_order=calc_second_order, seed=seed
    )
    # Store samples
    X = [
        {"param": y, "value": z, "iter": ix, "group": x}
        for ix, params in enumerate(param_values)
        for x, y, z in zip(problem["groups"], problem["names"], params, strict=True)
    ]
    return X


def process_parameters(
    jobid: int, nproc: int | None, config_base: dict
) -> tuple[dict[int, dict], Path]:
    """Generate and run parameter samples for the sensitivity analysis.

    This function generates parameter samples and runs the swmmanywhere model
    for each sample. It is designed to be run in parallel as a jobarray. It
    selects parameters values from the generated ones based on the jobid and
    the number of processors. It copies the config file and passes these
    parameters into swmmanywhere via the `parameter_overrides` property. Existing
    overrides that are not being sampled are retained, existing overrides that
    are being sampled are overwritten by the sampled value.

    Args:
        jobid (int): The job id.
        nproc (int | None): The number of processors to use. If None, the number
            of samples is used (i.e., only one model is simulated).
        config_base (dict): The base configuration dictionary.

    Returns:
        dict[dict]: A dict (keys as models) of dictionaries containing the results.
        Path: The path to the inp file.
    """
    # Generate samples
    X = generate_samples(
        parameters_to_select=config_base["parameters_to_sample"],
        N=2 ** config_base["sample_magnitude"],
    )

    df = pd.DataFrame(X)
    gb = df.groupby("iter")
    n_iter = len(gb)
    logger.info(f"{n_iter} samples created")
    flooding_results = {}
    nproc = nproc if nproc is not None else n_iter

    # Assign jobs based on jobid
    if jobid >= nproc:
        raise ValueError("Jobid should be less than the number of processors.")
    job_idx = range(jobid, n_iter, nproc)

    config = config_base.copy()

    # Iterate over the samples, running the model when the jobid matches the
    # processor number
    for ix in job_idx:
        config = config_base.copy()
        params_ = gb.get_group(ix)

        # Update the parameters
        overrides: defaultdict = defaultdict(dict)
        for grp, param, val in params_[["group", "param", "value"]].itertuples(
            index=False, name=None
        ):
            # Experimenter overrides take precedence over the config file
            if grp in config.get("parameter_overrides", {}):
                overrides[grp] = config["parameter_overrides"][grp]
            elif grp not in overrides:
                overrides[grp] = {}

            overrides[grp][param] = val
        config["parameter_overrides"] = dict(overrides)

        # Run the model
        config["model_number"] = ix
        logger.info(f"Running swmmanywhere for model {ix}")

        address, metrics = swmmanywhere.swmmanywhere(config)
        if metrics is None:
            metrics = {}
            logger.warning(f"Model run {ix} failed.")

        # Save the results
        flooding_results[ix] = {
            "iter": ix,
            **metrics,
            **params_.set_index("param").value.to_dict(),
        }
    return flooding_results, address


def save_results(jobid: int, results: dict[int, dict], address: Path) -> None:
    """Save the results of the sensitivity analysis.

    A results directory is created in the addresses.bbox directory, and the
    results are saved to a csv file there, labelled by jobid.

    Args:
        jobid (int): The job id.
        results (dict[str, dict]): A list of dictionaries containing the results.
        address (Path): The path to the inp file
    """
    results_fid = address.parent.parent / "results"
    results_fid.mkdir(parents=True, exist_ok=True)
    fid_flooding = results_fid / f"{jobid}_metrics.csv"
    df = pd.DataFrame(results).T
    df["jobid"] = jobid
    df.to_csv(fid_flooding, index=False)


def parse_arguments() -> tuple[int, int | None, Path]:
    """Parse the command line arguments.

    Returns:
        tuple: A tuple containing the job id, number of processors, and the
            configuration file path.
    """
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument("--jobid", type=int, default=0, help="Job ID")
    parser.add_argument("--nproc", type=int, default=None, help="Number of processors")
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path(__file__).parent.parent.parent
        / "tests"
        / "test_data"
        / "demo_config.yml",
        help="Configuration file path",
    )

    args = parser.parse_args()

    return args.jobid, args.nproc, args.config_path


def check_parameters_to_sample(config: dict):
    """Check the parameters to sample in the config.

    Args:
        config (dict): The configuration.

    Raises:
        ValueError: If a parameter to sample is not in the parameters
            dictionary.
    """
    params = get_full_parameters_flat()
    for param in config.get("parameters_to_sample", {}):
        # If the parameter is a dictionary, the values are bounds, all we are
        # checking here is that the parameter exists, we only need the first
        # entry.
        if isinstance(param, dict):
            if len(param) > 1:
                raise ValueError(
                    """If providing new bounds in the config, a dict 
                                 of len 1 is required, where the key is the 
                                 parameter to change and the values are 
                                 (new_lower_bound, new_upper_bound)."""
                )
            param = list(param.keys())[0]

        # Check that the parameter is available
        if param not in params:
            raise ValueError(f"{param} not found in parameters dictionary.")

        # Check that the parameter is sample-able
        required_attrs = set(["minimum", "maximum", "default", "category"])
        correct_attrs = required_attrs.intersection(params[param])
        missing_attrs = required_attrs.difference(correct_attrs)
        if any(missing_attrs):
            raise ValueError(f"{param} missing {missing_attrs} so cannot be sampled.")

    return config


if __name__ == "__main__":
    # Get args
    jobid, nproc, config_path = parse_arguments()

    # Set up logging
    logger.add(config_path.parent / f"experimenter_{jobid}.log")

    # Load the configuration
    config_base = swmmanywhere.load_config(
        config_path, schema_fid=Path(__file__).parent / "schema.yml"
    )

    # Check the parameters to sample
    config_base = check_parameters_to_sample(config_base)

    # Ensure the parameter overrides are set, since these are the way the
    # sampled parameter values are implemented
    config_base["parameter_overrides"] = config_base.get("parameter_overrides") or {}

    # Sample and run
    flooding_results, address = process_parameters(jobid, nproc, config_base)

    # Save the results
    save_results(jobid, flooding_results, address)
