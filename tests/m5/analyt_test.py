import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m5.models import PhysicsModels, ResultAnalyzer, RollingSolver


def _base_params():
    return {
        "epsilon_v": 1.2,
        "r_ohm": 8.0,
        "l_h": 1e-3,
        "c_f": 2e-6,
        "u0_v": 0.2,
        "i0_a": 0.0,
        "t_end_s": 0.02,
        "n_points": 4000,
        "a_a_per_v3": 4e-3,
        "b_a_per_v2": -16e-3,
        "c_a_per_v": 17e-3,
    }


def test_diode_piecewise_current():
    p = _base_params()
    model = PhysicsModels()
    assert model.diode_current(-0.5, p) == 0.0
    assert model.diode_current(0.0, p) == 0.0
    assert model.diode_current(0.5, p) > 0.0


def test_solution_shapes_and_finite_values():
    params = _base_params()
    model_func, name = PhysicsModels().get_model(params)
    solution = RollingSolver().solve(model_func, params)
    metrics = ResultAnalyzer.analyze(solution, params, name)

    t = solution["time"]
    assert t.ndim == 1
    assert solution["u_diode_v"].shape == t.shape
    assert solution["i_inductor_a"].shape == t.shape
    assert np.all(np.isfinite(solution["u_diode_v"]))
    assert np.all(np.isfinite(solution["i_inductor_a"]))
    assert metrics["u_max_abs"] >= 0.0
    assert 0.0 <= metrics["sine_purity"] <= 1.0
