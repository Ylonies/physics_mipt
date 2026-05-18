import builtins
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m9.models.input import DEFAULTS, InputHandler
from m9.models.models import PhysicsModels


def _run_input(answers, monkeypatch):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda _: next(it))
    return InputHandler.get_parameters()


def test_thin_defaults(monkeypatch):
    answers = ["1", "", "", "", "", "", ""]
    params = _run_input(answers, monkeypatch)
    assert params["lens_mode"] == "1"
    assert params["f_obj_m"] == DEFAULTS["f_obj_m"]


def test_retry_on_invalid_f_eye(monkeypatch):
    answers = ["1", "", "", "1", "0.05", "", "", ""]
    params = _run_input(answers, monkeypatch)
    assert params["f_eye_m"] == 0.05


def test_invalid_mode_retries(monkeypatch):
    answers = ["3", "1", "", "", "", "", "", ""]
    params = _run_input(answers, monkeypatch)
    assert params["lens_mode"] == "1"


def test_spotting_scope_builds():
    params = {
        **DEFAULTS,
        "lens_mode": "1",
        "lens1_pos_m": 0.25,
        "aperture_obj_m": 0.03,
        "aperture_eye_m": 0.02,
    }
    lenses, plane, optics = PhysicsModels.build_spotting_scope_thin(params)
    assert len(lenses) == 2
    assert plane > lenses[1].position
    assert abs(optics["m_total"]) > 0
    assert optics["s2"] > 0
