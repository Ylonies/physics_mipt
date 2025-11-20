import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m5.models import PhysicsModels, ResultAnalyzer


def _complete_params(params):
    a = params['a']
    b = params['b']
    m = params['m']
    I_cm = 0.25 * m * (a**2 + b**2)
    I = I_cm + m * b**2
    params['I'] = I
    params['l'] = b
    return params


def _natural_frequency(params, g=9.81):
    m = params['m']
    l = params['l']
    I = params['I']
    return math.sqrt(m * g * l / I)


def _theta_linear_undamped(t, theta0, omega0, omega_n):
    # theta(t) = theta0 cos(ω t) + (omega0/ω) sin(ω t)
    return theta0 * np.cos(omega_n * t) + (omega0 / omega_n) * np.sin(omega_n * t)


def _theta_linear_damped(t, theta0, omega0, omega_n, I, gamma):
    # beta = gamma / (2I), ωd = sqrt(ω0^2 - beta^2)
    beta = gamma / (2.0 * I)
    omega_d2 = max(omega_n**2 - beta**2, 0.0)
    omega_d = math.sqrt(omega_d2)
    if omega_d == 0.0:
        # critically damped (rare in our tests) - fall back to numerical
        return np.full_like(t, theta0)
    exp_term = np.exp(-beta * t)
    return exp_term * (
        theta0 * np.cos(omega_d * t) + (omega0 + beta * theta0) / omega_d * np.sin(omega_d * t)
    )


@pytest.fixture
def models():
    return PhysicsModels()


@pytest.fixture
def base_params_small():
    return {
        'a': 0.10,            # m (minor axis)
        'b': 0.20,            # m (major axis)
        'm': 1.0,             # kg
        'theta0': math.radians(1.0),   # small angle
        'omega0': 0.2,        # rad/s
        't_end': 5.0,
        'gamma': 0.0
    }


def _solve(models, params):
    model, y0, t_span, t_eval = models.ellipse_pendulum_model(params)
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol


def test_small_angle_undamped_matches_linear(models, base_params_small):
    params = _complete_params(dict(base_params_small))
    params['gamma'] = 0.0

    sol = _solve(models, params)
    results = ResultAnalyzer.analyze(sol, params)

    t = results['time']
    theta_num = results['theta']

    omega_n = _natural_frequency(params)
    theta_lin = _theta_linear_undamped(t, params['theta0'], params['omega0'], omega_n)

    # accuracy for small-angle, undamped
    assert np.allclose(theta_num, theta_lin, atol=1e-3)

    # energy conservation (relative drift small)
    E = results['energy_total']
    assert (E.max() - E.min()) / max(E.max(), 1e-9) < 1e-3


def test_small_angle_damped_matches_linear(models, base_params_small):
    params = _complete_params(dict(base_params_small))

    # choose damping well below critical: beta = 0.1 * omega_n
    omega_n = _natural_frequency(params)
    I = params['I']
    params['gamma'] = 0.2 * I * omega_n  # beta = gamma/(2I) = 0.1 * omega_n

    sol = _solve(models, params)
    results = ResultAnalyzer.analyze(sol, params)

    t = results['time']
    theta_num = results['theta']

    theta_lin = _theta_linear_damped(t, params['theta0'], params['omega0'], omega_n, I, params['gamma'])
    assert np.allclose(theta_num, theta_lin, atol=5e-3)

    # energy should be non-increasing overall (allow tiny numerical ups)
    E = results['energy_total']
    dE = np.diff(E)
    assert np.sum(dE > 1e-6) <= 2


def test_random_small_angle_undamped_sets(models):
    rng = np.random.default_rng(123)
    for _ in range(5):
        a = rng.uniform(0.05, 0.3)
        b = rng.uniform(0.05, 0.3)
        m = rng.uniform(0.5, 3.0)
        theta0 = math.radians(rng.uniform(-2.0, 2.0))
        omega0 = rng.uniform(-0.5, 0.5)
        t_end = rng.uniform(1.0, 3.0)
        params = _complete_params({
            'a': a, 'b': b, 'm': m, 'theta0': theta0, 'omega0': omega0, 't_end': t_end, 'gamma': 0.0
        })

        sol = _solve(models, params)
        results = ResultAnalyzer.analyze(sol, params)

        t = results['time']
        theta_num = results['theta']

        omega_n = _natural_frequency(params)
        theta_lin = _theta_linear_undamped(t, theta0, omega0, omega_n)

        assert np.allclose(theta_num, theta_lin, atol=2e-3)

