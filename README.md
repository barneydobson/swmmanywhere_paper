# SWMManywhere sensitivity analysis
<!-- markdown-link-check-disable -->
[![Test and build](https://github.com/ImperialCollegeLondon/SWMManywhere/actions/workflows/ci.yml/badge.svg)](https://github.com/ImperialCollegeLondon/SWMManywhere/actions/workflows/ci.yml)
<!-- markdown-link-check-enable -->

This repository is to reproduce the experiments and plots from [ref].

## Running the experiments

The intended use is via `config` file. We extend its behaviour with two new options:

- `parameters_to_sample`: provides a list of parameters to be sampled. For example,
```
parameters_to_sample:
- min_v
- max_v
```
- `sample_magnitude`: provides the amount of sampling effort to perform. The total number
of samples to be evaluated is equal to:
```
2**(sample_magnitude + 1) * (n_parameters_to_sample + 1)
```

This new `config` file should be passed to the `experimenter`. For example,
```
python experimenter.py --config_path=/path/to/config
```

You are likely to need to run such an experiment on HPC. The `experimenter` is set up
to parallelise as a PBS jobarray - with an example in `submit_icl_example`.
