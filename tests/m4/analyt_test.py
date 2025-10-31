import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m4.models.result_analyzer import ResultAnalyzer
from m4.models import PhysicsModels


@pytest.fixture
def base_params():
    return {
        'm': 1.0,
        'R': 0.1,
        'mu': 0.2,
        'alpha': math.radians(30),
        's0': 0.0,
        'v0': 0.0,
        'omega0': 0.0,
        'x0': 0.0,
        'F': 1.0,
        't_end': 2.0
    }


@pytest.fixture
def models():
    return PhysicsModels()


# === 1) Тест для НАКЛОНА БЕЗ ПРОСКАЛЬЗЫВАНИЯ ===
def test_incline_no_slip(models, base_params):
    base_params['choice'] = '1'
    model, name, y0, t_span, t_eval = models.get_model(base_params)

    sol = solve_ivp(model, t_span, y0, t_eval=t_eval)
    results = ResultAnalyzer.analyze(sol, base_params, name)

    g = 9.81
    m, R = base_params['m'], base_params['R']
    alpha = base_params['alpha']
    I = 2/5 * m * R**2
    a_analytical = g * math.sin(alpha) / (1 + I/(m*R**2))

    t = results['time']
    s_a = 0.5 * a_analytical * t**2
    v_a = a_analytical * t
    omega_a = v_a / R

    assert np.allclose(results['s'], s_a, atol=1e-3)
    assert np.allclose(results['v'], v_a, atol=1e-3)
    assert np.allclose(results['omega'], omega_a, atol=1e-3)
    assert results['slip_fraction'] < 1e-6 


# === 2) Тест для НАКЛОНА С ВОЗМОЖНЫМ ПРОСКАЛЬЗЫВАНИЕМ ===
def test_incline_with_slip(models, base_params):
    base_params['choice'] = '1b'
    base_params['mu'] = 0.05
    model, name, y0, t_span, t_eval = models.get_model(base_params)

    sol = solve_ivp(model, t_span, y0, t_eval=t_eval)
    results = ResultAnalyzer.analyze(sol, base_params, name)

    assert results['slip_fraction'] > 0.0
    assert np.all(results['v'] >= 0.0)


# === 3) Тест для ГОРИЗОНТАЛЬНОЙ ПЛОСКОСТИ ===
def test_horizontal_force(models, base_params):
    base_params['choice'] = '2'
    base_params['F'] = 1.0
    base_params['mu'] = 0.0 
    model, name, y0, t_span, t_eval = models.get_model(base_params)

    sol = solve_ivp(model, t_span, y0, t_eval=t_eval)
    results = ResultAnalyzer.analyze(sol, base_params, name)

    m = base_params['m']
    F = base_params['F']

    t = results['time']
    x_a = 0.5 * (F/m) * t**2
    v_a = (F/m) * t

    assert np.allclose(results['x'], x_a, atol=1e-3)
    assert np.allclose(results['vx'], v_a, atol=1e-3)

    E = results['energy_total']
    assert np.all(np.diff(E) >= -1e-6)
