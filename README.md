# SWMManywhere sensitivity analysis

This repository is to reproduce the experiments and plots from [this paper](https://doi.org/10.1016/j.envsoft.2025.106358):


> Dobson, Barnaby, et al. "SWMManywhere: A Workflow for Generation and Sensitivity Analysis of Synthetic Urban Drainage Models, Anywhere." (2025). doi: 10.1016/j.envsoft.2025.106358

## Installation

Clone the repository:

```bash
git clone https://github.com/barneydobson/swmmanywhere_paper.git
```

Navigate to the repository and install:

```bash
pip install -e .
pip install -r dev-requirements.txt
```

## Running the experiments

The intended use is via `config` file. We extend its behaviour with two new options:

- `parameters_to_sample`: provides a list of parameters to be sampled. For example,

```bash
parameters_to_sample:
- min_v
- max_v
```

- `sample_magnitude`: provides the amount of sampling effort to perform. The total number
of samples to be evaluated is equal to:

```bash
2**(sample_magnitude + 1) * (n_parameters_to_sample + 1)
```

This new `config` file should be passed to the `experimenter`. For example,

```bash
python experimenter.py --config_path=/path/to/config.yml
```

You are likely to need to run such an experiment on HPC. The `experimenter` is set up
to parallelise as a PBS jobarray - with an example submit file in `submit_icl_example`.

## Recreating plots

The results of the experiments used in the paper are contained in `tests/test_data` in this repository.
Only those required to create the plots are retained to avoid overwhelming the storage on this repository.
All figures can be reproduced in the `tests/test_data/plots` directory by running `tests/test_figs.py` locally:

```bash
pytest tests/test_figs.py
```
