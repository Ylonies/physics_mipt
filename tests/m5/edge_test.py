import builtins
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m5.models.input import DEFAULTS, InputHandler


def _run(answers, monkeypatch):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda _: next(it))
    return InputHandler.get_parameters()


def test_defaults(monkeypatch):
    p = _run([""] * 8, monkeypatch)
    assert p["r_ohm"] == DEFAULTS["r_ohm"]


def test_retry_invalid(monkeypatch):
    p = _run(["100", "1.2", "2", "", "", "", "", "", ""], monkeypatch)
    assert p["r_ohm"] == 2.0
