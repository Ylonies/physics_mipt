import numpy as np
import math
import pytest
from scipy.optimize import fsolve
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m1_stone_flight.models.solver import TrajectorySolver
from m1_stone_flight.models.result_analyzer import ResultAnalyzer

def viscous_model(t, state, params):
    x, y, vx, vy = state
    g = 9.81
    gamma = params['gamma']
    m = params['m']
    ax = -gamma/m * vx
    ay = -g - gamma/m * vy
    return [vx, vy, ax, ay]

# аналитическое решение для линейного сопротивления
def linear_drag_analytical(v0, alpha, gamma, m, g=9.81):
    v0x = v0 * np.cos(alpha)
    v0y = v0 * np.sin(alpha)

    def y_func(t):
        return (m/gamma)*(v0y + m*g/gamma)*(1 - np.exp(-gamma/m * t)) - (m*g/gamma)*t

    t_guess = 2*v0y/g
    t_flight = fsolve(y_func, t_guess)[0]
    x_flight = (m/gamma)*v0x*(1 - np.exp(-gamma/m*t_flight))
    
    t_max_height = (m/gamma) * np.log(1 + gamma*v0y/(m*g))
    y_max = y_func(t_max_height)
    
    return t_flight, x_flight, y_max

@pytest.fixture
def basic_params():
    return {'v0': 20, 'alpha': math.radians(45), 'gamma': 0.5, 'm': 1.0}

@pytest.fixture
def solver():
    return TrajectorySolver()

def test_trajectory_shape_and_properties(solver, basic_params):
    solution = solver.solve(lambda t, y: viscous_model(t, y, basic_params), basic_params)
    results = ResultAnalyzer.analyze(solution, basic_params)

    x, y, t = results['trajectory'][0], results['trajectory'][1], results['time']

    assert len(x) == len(y) == len(t)
    assert x.shape == y.shape == t.shape
    assert abs(x[0]) < 1e-10
    assert abs(y[0]) < 1e-10
    assert t[0] == 0.0
    assert t[-1] <= 10  

def test_linear_drag_results(solver, basic_params):
    solution = solver.solve(lambda t, y: viscous_model(t, y, basic_params), basic_params)
    results = ResultAnalyzer.analyze(solution, basic_params)
    
    t_flight_a, x_flight_a, y_max_a = linear_drag_analytical(
        basic_params['v0'], basic_params['alpha'], basic_params['gamma'], basic_params['m']
    )
    
    # сравниваем основные параметры численно и аналитически
    assert abs(results['flight_time'] - t_flight_a) < 1e-3
    assert abs(results['flight_distance'] - x_flight_a) < 1e-3
    assert abs(results['max_height'] - y_max_a) < 1e-1

def test_ground_impact_detection(solver, basic_params):
    solution = solver.solve(lambda t, y: viscous_model(t, y, basic_params), basic_params)
    results = ResultAnalyzer.analyze(solution, basic_params)
    
    y_vals = results['trajectory'][1]
    assert abs(y_vals[-1]) < 1e-3
    assert np.all(y_vals >= -1e-3)
