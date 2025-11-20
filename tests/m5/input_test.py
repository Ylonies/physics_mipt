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


@pytest.mark.parametrize("inputs", [
    # non-positive a
    ["0.0", "0.2", "1.0", "5.0", "0.0", "1.0", "n"],
    ["-0.1", "0.2", "1.0", "5.0", "0.0", "1.0", "n"],
    # non-positive b
    ["0.1", "0.0", "1.0", "5.0", "0.0", "1.0", "n"],
    ["0.1", "-0.2", "1.0", "5.0", "0.0", "1.0", "n"],
    # non-positive m
    ["0.1", "0.2", "0.0", "5.0", "0.0", "1.0", "n"],
    ["0.1", "0.2", "-1.0", "5.0", "0.0", "1.0", "n"],
    # non-positive t_end
    ["0.1", "0.2", "1.0", "5.0", "0.0", "0.0", "n"],
    ["0.1", "0.2", "1.0", "5.0", "0.0", "-1.0", "n"],
    # negative gamma when enabled
    ["0.1", "0.2", "1.0", "5.0", "0.0", "2.0", "y", "-0.01"],
])
def test_invalid_inputs(inputs, monkeypatch):
    with pytest.raises(ValueError):
        run_with_inputs(inputs, monkeypatch)

