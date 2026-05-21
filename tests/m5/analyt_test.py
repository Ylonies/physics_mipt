import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m5.models import PhysicsModels, ResultAnalyzer, RollingSolver
from m5.models.input import DEFAULTS


def _base_params(**kw):
    p = {**DEFAULTS}
    p.update(kw)
    return p


def test_diode_piecewise_current():
    model = PhysicsModels()
    assert model.diode_current(-0.5, _base_params()) == 0.0
    assert model.diode_current(0.0, _base_params()) == 0.0
    assert model.diode_current(0.5, _base_params()) > 0.0


def test_dc_point_in_negative_resistance_branch():
    p = _base_params()
    u0, i0 = PhysicsModels.find_dc_operating_point(p)
    g = PhysicsModels.diode_conductance(u0, p)
    assert 0.73 < u0 < 1.93
    assert g < 0
    assert i0 > 0


def test_linear_criterion_oscillation_for_defaults():
    p = _base_params()
    lin = PhysicsModels.linear_oscillation_criterion(p)
    assert lin["can_oscillate_linear"]


def test_defaults_show_self_oscillation():
    p = _base_params()
    model_func, name = PhysicsModels().get_model(p)
    solution = RollingSolver().solve(model_func, p)
    metrics = ResultAnalyzer.analyze(solution, p, name)
    assert metrics["self_oscillation"]
    assert metrics["u_amplitude_steady"] > 0.05
    assert metrics["fundamental_freq_hz"] > 100


def test_high_r_damps_oscillation():
    p = _base_params(r_ohm=50.0, u0_v=1.15)
    lin = PhysicsModels.linear_oscillation_criterion(p)
    assert not lin["can_oscillate_linear"]
    model_func, name = PhysicsModels().get_model(p)
    metrics = ResultAnalyzer.analyze(RollingSolver().solve(model_func, p), p, name)
    assert not metrics["self_oscillation"]
