import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m9.models import PhysicsModels, ResultAnalyzer, RollingSolver
from m9.models.input import DEFAULTS
from m9.models.models import Ray2D, ThinLens


def _scope_params(lens_mode: str = "1", **overrides):
    params = {
        **DEFAULTS,
        "lens_mode": lens_mode,
        "lens1_pos_m": DEFAULTS["object_distance_m"],
        "aperture_obj_m": 0.03,
        "aperture_eye_m": 0.02,
    }
    params.update(overrides)
    params["lens1_pos_m"] = params["object_distance_m"]
    return params


def test_thin_lens_formula():
    f, s = 0.2, 0.5
    sp = PhysicsModels.thin_lens_image_distance(s, f)
    m = PhysicsModels.thin_lens_magnification(s, sp)
    assert abs(1 / s + 1 / sp - 1 / f) < 1e-9
    assert abs(m + sp / s) < 1e-9


def test_lensmaker_radius_matches_focal_length():
    f, t, n = 0.2, 0.012, 1.5
    r = PhysicsModels.radius_from_lensmaker(f, t, n)
    inv_f = (n - 1) * (2 / r - (n - 1) * t / (n * r * r))
    assert abs(1 / inv_f - f) / f < 0.05


def test_spotting_scope_thin_magnification():
    params = _scope_params("1")
    model, name = PhysicsModels().get_model(params)
    sol = RollingSolver().solve(model, params)
    res = ResultAnalyzer.analyze(sol, params, name)
    assert sol["geometry_ok"]
    assert res["rel_err_magnification"] < 0.02
    assert abs(res["m_theory"] - res["optics"]["m1"] * res["optics"]["m2"]) < 1e-9


def test_single_lens_ray_matches_theory():
    params = _scope_params("1")
    s, f = params["object_distance_m"], params["f_obj_m"]
    sp = PhysicsModels.thin_lens_image_distance(s, f)
    m_th = -sp / s
    lens = ThinLens(params["lens1_pos_m"], f, params["aperture_obj_m"])
    plane = lens.position + sp
    pm = PhysicsModels()
    r0 = pm.trace_thin(Ray2D(0, -0.01, 1, 0, 1, 1), [lens], plane)[0]
    r1 = pm.trace_thin(Ray2D(0, 0.01, 1, 0, 1, 1), [lens], plane)[0]
    m_ray = (r1.y - r0.y) / 0.02
    assert abs(m_ray - m_th) / abs(m_th) < 0.01


def test_thick_lens_trace_produces_image():
    params = _scope_params("2")
    model, name = PhysicsModels().get_model(params)
    sol = RollingSolver().solve(model, params)
    assert sol["image"].max() > 0.02
    assert len(sol["ray_paths"]) > 0
