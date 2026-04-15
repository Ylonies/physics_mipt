import builtins
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m7_radio.models.input import InputHandler
from m7_radio.models.models import PhysicsModels


def _run_input_with_answers(answers, monkeypatch):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda _: next(it))
    return InputHandler.get_parameters()


def test_input_defaults_are_accepted(monkeypatch):
    params = _run_input_with_answers([""] * 13, monkeypatch)

    assert params["carrier_freq_hz"] == 200_000.0
    assert params["noise_components"] == 20
    assert params["rlc_inductance_h"] == 1e-3
    assert params["lowpass_tau_s"] == 6e-5


def test_input_rejects_out_of_range_duration(monkeypatch):
    with pytest.raises(ValueError, match="диапазоне"):
        _run_input_with_answers(["-1"], monkeypatch)


def test_input_rejects_invalid_noise_band(monkeypatch):
    answers = [
        "",  # duration_s
        "",  # sample_rate_hz
        "",  # carrier_freq_hz
        "",  # modulation_index
        "",  # seed
        "",  # noise_relative_amplitude
        "",  # noise_components
        "1000000",  # noise_min_freq_hz
        "1000000",  # noise_max_freq_hz (must be > min)
    ]
    with pytest.raises(ValueError, match="должна быть больше"):
        _run_input_with_answers(answers, monkeypatch)


def test_model_handles_upper_edge_values():
    params = {
        "duration_s": 0.002,
        "sample_rate_hz": 400_000.0,
        "carrier_freq_hz": 100_000.0,
        "modulation_index": 1.0,
        "seed": 1_000_000,
        "noise_relative_amplitude": 2.0,
        "noise_components": 50,
        "noise_min_freq_hz": 150_000.0,
        "noise_max_freq_hz": 350_000.0,
        "rlc_resistance_ohm": 0.01,
        "rlc_inductance_h": 10.0,
        "envelope_scale": 100.0,
        "lowpass_tau_s": 1e-7,
    }

    model, _ = PhysicsModels().get_model(params)
    solution = model(params)

    assert solution["time"].size > 100
    assert solution["received_signal"].shape == solution["time"].shape
    assert solution["recovered_signal"].shape == solution["time"].shape
