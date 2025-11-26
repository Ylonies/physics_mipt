import builtins
import math
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m5.models import InputHandler


def run_with_inputs(inputs, monkeypatch):
    it = iter(inputs)
    monkeypatch.setattr(builtins, "input", lambda _: next(it))
    return InputHandler.get_parameters()


@pytest.mark.parametrize("inputs", [
    # valid without damping
    ["0.10", "0.20", "1.0", "5.0", "0.2", "3.0", "n"],
    # valid with damping
    ["0.10", "0.20", "1.5", "1.0", "0.0", "2.0", "y", "0.05"],
])
def test_valid_inputs(inputs, monkeypatch):
    params = run_with_inputs(inputs, monkeypatch)
    assert params['a'] > 0 and params['b'] > 0 and params['m'] > 0
    assert params['t_end'] > 0
    assert 'I' in params and 'l' in params
    a, b, m = params['a'], params['b'], params['m']
    I_expected = 0.25 * m * (a*a + b*b) + m * b*b
    assert abs(params['I'] - I_expected) < 1e-12
    assert params['l'] == pytest.approx(b, abs=0.0)
    assert params['theta0'] == pytest.approx(math.radians(params['theta0_deg']))
    assert params['gamma'] >= 0.0


def test_reprompt_a(monkeypatch, capsys):
    # a invalid (0.0), then corrected to 0.10
    inputs = [
        "0.0", "0.10",    # a
        "0.20",           # b
        "1.0",            # m
        "5.0",            # theta0_deg
        "0.2",            # omega0
        "3.0",            # t_end
        "n"               # no damping
    ]
    params = run_with_inputs(inputs, monkeypatch)
    out = capsys.readouterr().out
    assert "Значение должно быть >=" in out
    assert params['a'] == pytest.approx(0.10)


def test_reprompt_b(monkeypatch, capsys):
    inputs = [
        "0.10",           # a
        "0.0", "0.20",    # b invalid then valid
        "1.0",
        "5.0",
        "0.2",
        "3.0",
        "n"
    ]
    params = run_with_inputs(inputs, monkeypatch)
    out = capsys.readouterr().out
    assert "Значение должно быть >=" in out
    assert params['b'] == pytest.approx(0.20)


def test_reprompt_m(monkeypatch, capsys):
    inputs = [
        "0.10",
        "0.20",
        "0.0", "1.0",     # m invalid then valid
        "5.0",
        "0.2",
        "3.0",
        "n"
    ]
    params = run_with_inputs(inputs, monkeypatch)
    out = capsys.readouterr().out
    assert "Значение должно быть >=" in out
    assert params['m'] == pytest.approx(1.0)


def test_reprompt_theta0(monkeypatch, capsys):
    inputs = [
        "0.10",
        "0.20",
        "1.0",
        "100", "10",      # theta0_deg out of range then valid
        "0.2",
        "3.0",
        "n"
    ]
    params = run_with_inputs(inputs, monkeypatch)
    out = capsys.readouterr().out
    assert "Значение должно быть <=" in out
    assert params['theta0_deg'] == pytest.approx(10.0)


def test_reprompt_t_end(monkeypatch, capsys):
    inputs = [
        "0.10",
        "0.20",
        "1.0",
        "5.0",
        "0.2",
        "0.0", "2.0",     # t_end invalid then valid
        "n"
    ]
    params = run_with_inputs(inputs, monkeypatch)
    out = capsys.readouterr().out
    assert "Значение должно быть >=" in out
    assert params['t_end'] == pytest.approx(2.0)


def test_reprompt_yes_no(monkeypatch, capsys):
    inputs = [
        "0.10",
        "0.20",
        "1.0",
        "5.0",
        "0.2",
        "3.0",
        "k", "n"          # invalid then 'n'
    ]
    params = run_with_inputs(inputs, monkeypatch)
    out = capsys.readouterr().out
    assert "Пожалуйста, введите 'y' или 'n'." in out
    assert params['gamma'] == pytest.approx(0.0)


def test_reprompt_gamma(monkeypatch, capsys):
    inputs = [
        "0.10",
        "0.20",
        "1.0",
        "5.0",
        "0.2",
        "3.0",
        "y",
        "-0.01", "0.05"   # gamma invalid then valid
    ]
    params = run_with_inputs(inputs, monkeypatch)
    out = capsys.readouterr().out
    assert "Значение должно быть >=" in out
    assert params['gamma'] == pytest.approx(0.05)

