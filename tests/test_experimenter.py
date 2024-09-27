"""Tests for the main experimenter."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml
from swmmanywhere import parameters, swmmanywhere

from swmmanywhere_paper import experimenter

schema_fid = Path(__file__).parent.parent / "swmmanywhere_paper" / "schema.yml"


def assert_close(a: float, b: float, rtol: float = 1e-3) -> None:
    """Assert that two floats are close."""
    assert np.isclose(a, b, rtol=rtol).all()


def test_formulate_salib_problem():
    """Test the formulate_salib_problem function."""
    problem = experimenter.formulate_salib_problem([{"min_v": [0.5, 1.5]}, "max_v"])
    assert problem["num_vars"] == 2
    max_v = parameters.HydraulicDesign().model_json_schema()["properties"]["max_v"]
    assert problem["names"] == ["min_v", "max_v"]
    assert problem["bounds"] == [[0.5, 1.5], [max_v["minimum"], max_v["maximum"]]]


def test_generate_samples():
    """Test the generate_samples function."""
    samples = experimenter.generate_samples(
        N=2,
        parameters_to_select=["min_v", "max_v", "chahinian_slope_scaling"],
        seed=1,
        groups=False,
    )
    assert len(samples) == 48
    assert set([x["param"] for x in samples]) == {
        "min_v",
        "max_v",
        "chahinian_slope_scaling",
    }
    assert_close(samples[0]["value"], 0.31093)

    samples = experimenter.generate_samples(
        N=2,
        parameters_to_select=["min_v", "max_v", "chahinian_slope_scaling"],
        seed=1,
        groups=True,
    )
    assert len(samples) == 36


def test_process_parameters():
    """Test process_parameters."""
    config = {
        "parameters_to_sample": ["min_v", "max_v"],
        "sample_magnitude": 3,
    }

    # Test standard
    with mock.patch(
        "swmmanywhere_paper.experimenter.swmmanywhere.swmmanywhere",
        return_value=("fake_path", {"fake_metric": 1}),
    ) as mock_sa:
        result = experimenter.process_parameters(0, 1, config)

    assert len(result[0]) == 48
    assert_close(result[0][0]["min_v"], 0.310930)

    # Test experimenter takes precedence over overrides
    config["parameter_overrides"] = {"hydraulic_design": {"min_v": 1.0}}
    with mock.patch(
        "swmmanywhere_paper.experimenter.swmmanywhere.swmmanywhere",
        return_value=("fake_path", {"fake_metric": 1}),
    ) as mock_sa:
        result = experimenter.process_parameters(0, 1, config)

    assert len(result[0]) == 48
    assert_close(result[0][0]["min_v"], 0.310930)

    # Test non experimenter overrides still work
    config["parameter_overrides"] = {"hydraulic_design": {"max_fr": 0.5}}
    with mock.patch(
        "swmmanywhere_paper.experimenter.swmmanywhere.swmmanywhere",
        return_value=("fake_path", {"fake_metric": 1}),
    ) as mock_sa:
        result = experimenter.process_parameters(0, 1, config)

    for call in mock_sa.mock_calls:
        assert call.args[0]["parameter_overrides"]["hydraulic_design"]["max_fr"] == 0.5

    assert len(result[0]) == 48
    assert_close(result[0][0]["min_v"], 0.310930)


def test_check_parameters_to_sample():
    """Test the check_parameters_to_sample validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Load the config
        config = swmmanywhere.load_config(validation=False)

        # Make an edit that should fail
        config["base_dir"] = str(base_dir)
        del config["real"]
        with open(base_dir / "test_config.yml", "w") as f:
            yaml.dump(config, f)

        # Test parameter_overrides invalid category
        config["parameter_overrides"] = {"fake_category": {"fake_parameter": 0}}
        with pytest.raises(ValueError) as exc_info:
            swmmanywhere.check_parameter_overrides(config)
        assert "fake_category not a category" in str(exc_info.value)

        # Test parameter_overrides invalid parameter
        config["parameter_overrides"] = {"hydraulic_design": {"fake_parameter": 0}}
        with pytest.raises(ValueError) as exc_info:
            swmmanywhere.check_parameter_overrides(config)
        assert "fake_parameter not found" in str(exc_info.value)

        # Test parameter_overrides valid
        config["parameter_overrides"] = {"hydraulic_design": {"min_v": 1.0}}
        _ = swmmanywhere.check_parameter_overrides(config)
