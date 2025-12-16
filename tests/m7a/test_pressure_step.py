import math

import numpy as np

from m7a.models import PhysicsModels, ResultAnalyzer, SimulationConfig


def test_pressure_step_final_state_close_to_theory():
    cfg = SimulationConfig
    model = PhysicsModels()

    N = 300
    T0 = 1.0
    V0 = cfg.Lx * cfg.Ly
    P0 = N * cfg.kB * T0 / V0

    params = {
        "N": N,
        "T0": T0,
        "mode": 3,
        "P_ext_initial": P0,
        "P_ext_final": 2.0 * P0,
        "t_pressure_step": 0.02,
        "max_steps": 20000,
        "diag_interval": 50,
        "nbins": 60,
    }
    results = model.run(params)

    temps = np.asarray(results["temps"], dtype=float)
    piston_xs = np.asarray(results["piston_xs"], dtype=float)
    assert temps.size > 10
    assert piston_xs.size == temps.size

    tail = min(50, temps.size)
    T_sim = float(np.mean(temps[-tail:]))
    V_sim = float(np.mean(piston_xs[-tail:]) * cfg.Ly)

    th = ResultAnalyzer.theoretical_final_state_pressure_step(
        N=N, T0=T0, V0=V0, P_ext=params["P_ext_final"]
    )
    V_th = th["V_f"]
    T_th = th["T_f"]

    assert math.isfinite(V_sim) and math.isfinite(T_sim)
    assert V_th > 0 and T_th > 0

    # It's a noisy finite-N MD model, so use a relatively soft tolerance.
    assert abs(V_sim - V_th) / V_th < 0.30
    assert abs(T_sim - T_th) / T_th < 0.30


def test_pressure_step_sound_speed_order_of_magnitude():
    cfg = SimulationConfig
    model = PhysicsModels()

    N = 400
    T0 = 1.0
    V0 = cfg.Lx * cfg.Ly
    P0 = N * cfg.kB * T0 / V0

    params = {
        "N": N,
        "T0": T0,
        "mode": 3,
        "P_ext_initial": P0,
        "P_ext_final": 2.0 * P0,
        "t_pressure_step": 0.02,
        "max_steps": 25000,
        "diag_interval": 50,
        "nbins": 80,
    }
    results = model.run(params)

    temps = np.asarray(results["temps"], dtype=float)
    tail = min(50, temps.size)
    T_mean = float(np.mean(temps[-tail:])) if tail else float("nan")
    c_th = ResultAnalyzer.theoretical_sound_speed(T_mean)

    est = ResultAnalyzer.estimate_sound_speed_from_density(
        density_x=results["density_x"],
        density_profiles=results["density_profiles"],
        times=results["times"],
        t0=results["t_event"],
        piston_xs=results["piston_xs"],
        min_points=10,
    )
    c_est = est["c"]

    assert math.isfinite(c_th) and c_th > 0
    assert math.isfinite(c_est) and c_est > 0
    assert est["fit_points"] >= 10

    # Allow wide tolerance: we only need a sane agreement with theory.
    assert 0.4 * c_th <= c_est <= 2.5 * c_th


