import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m7_radio.models import PhysicsModels, ResultAnalyzer, RollingSolver


@pytest.fixture
def low_noise_params():
    return {
        "duration_s": 0.01,
        "sample_rate_hz": 1_000_000.0,
        "carrier_freq_hz": 200_000.0,
        "modulation_index": 0.7,
        "seed": 42,
        "noise_relative_amplitude": 0.02,
        "noise_components": 10,
        "noise_min_freq_hz": 350_000.0,
        "noise_max_freq_hz": 900_000.0,
        "rlc_resistance_ohm": 8.0,
        "rlc_inductance_h": 1e-3,
        "envelope_scale": 1.0,
        "lowpass_tau_s": 6e-5,
    }


def test_channel_without_noise_equals_transmitted(low_noise_params):
    params = dict(low_noise_params)
    params["noise_relative_amplitude"] = 0.0

    model, model_name = PhysicsModels().get_model(params)
    solution = RollingSolver().solve(model, params)
    results = ResultAnalyzer.analyze(solution, params, model_name)

    assert np.allclose(solution["received_signal"], solution["transmitted_signal"], atol=1e-12)
    assert np.all(solution["envelope"] >= 0.0)
    assert results["mse"] >= 0.0


def test_recovery_quality_in_low_noise_case(low_noise_params):
    model, model_name = PhysicsModels().get_model(low_noise_params)
    solution = RollingSolver().solve(model, low_noise_params)
    results = ResultAnalyzer.analyze(solution, low_noise_params, model_name)

    n_expected = int(low_noise_params["duration_s"] * low_noise_params["sample_rate_hz"])
    assert abs(solution["time"].size - n_expected) <= 1
    assert solution["source_signal"].shape == solution["recovered_signal"].shape
    assert results["corrcoef"] > 0.7
    assert results["mse"] < 0.2
