import numpy as np

from m7a.models import PhysicsModels, SimulationConfig


def _run_driven(piston_target: float):
    cfg = SimulationConfig
    drive_speed = 1.0
    steps_needed = int(abs(cfg.Lx - piston_target) / (drive_speed * cfg.dt)) + 2000

    model = PhysicsModels()
    params = {
        "N": 200,
        "T0": 1.0,
        "mode": 2,
        "piston_target": piston_target,
        "max_steps": steps_needed,
        "diag_interval": 25,
        "nbins": 0,
    }
    return model.run(params)


def test_driven_piston_target_min_edge():
    cfg = SimulationConfig
    res = _run_driven(0.4)
    assert abs(res["x_p_final"] - 0.4) < 1e-3
    assert np.isfinite(res["temps"]).all()
    assert res["x_p_final"] >= 2.0 * cfg.radius


def test_driven_piston_target_max_edge():
    res = _run_driven(1.0)
    assert abs(res["x_p_final"] - 1.0) < 1e-3
    assert np.isfinite(res["temps"]).all()


