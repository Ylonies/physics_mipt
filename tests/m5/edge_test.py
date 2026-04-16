import builtins
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m5.models.input import InputHandler
from m5.models.models import PhysicsModels


def _run_input_with_answers(answers, monkeypatch):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda _: next(it))
    return InputHandler.get_parameters()


def test_defaults_are_accepted(monkeypatch):
    params = _run_input_with_answers([""] * 8, monkeypatch)
    assert params["epsilon_v"] == 1.2
    assert params["r_ohm"] == 8.0
    assert params["n_points"] == 6000


def test_negative_resistance_rejected(monkeypatch):
    with pytest.raises(ValueError, match="диапазоне"):
        _run_input_with_answers(["", "-1"], monkeypatch)


def test_extreme_but_valid_parameters_do_not_crash():
    params = {
        "epsilon_v": 3.0,
        "r_ohm": 1.0,
        "l_h": 1e-4,
        "c_f": 5e-8,
        "u0_v": 0.0,
        "i0_a": 0.0,
        "t_end_s": 0.0005,
        "n_points": 3000,
        "a_a_per_v3": 4e-3,
        "b_a_per_v2": -16e-3,
        "c_a_per_v": 17e-3,
    }
    model, _ = PhysicsModels().get_model(params)
    solution = model(params)
    assert solution["time"].size == params["n_points"]
