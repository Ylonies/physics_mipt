"""Проверка М5 на краевых значениях параметров: согласованность теории и моделирования."""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from m5.models import PhysicsModels, ResultAnalyzer, RollingSolver
from m5.models.input import DEFAULTS


def _run(**overrides):
    p = {**DEFAULTS, **overrides}
    pm = PhysicsModels()
    try:
        lin = PhysicsModels.linear_oscillation_criterion(p)
    except ValueError as e:
        return {"error": str(e), "params": p}
    sol = pm._tunnel_diode_generator(p)
    m = ResultAnalyzer.analyze(sol, p, "test")
    return {"lin": lin, "m": m, "params": p, "finite": np.all(np.isfinite(sol["u_diode_v"]))}


def _label(case: dict) -> str:
    p = case["params"]
    return f"ε={p['epsilon_v']} R={p['r_ohm']} L={p['l_h']} C={p['c_f']}"


EDGE_CASES = [
    ("R_min_osc", {"r_ohm": 0.5}),
    ("R_max_damp", {"r_ohm": 50.0}),
    ("L_min", {"l_h": 1e-4}),
    ("L_max", {"l_h": 0.05}),
    ("C_min", {"c_f": 1e-7}),
    ("C_max", {"c_f": 1e-4}),
    ("eps_min", {"epsilon_v": 0.5}),
    ("eps_max", {"epsilon_v": 3.5}),
    ("u0_zero", {"u0_v": 0.0}),
    ("u0_high", {"u0_v": 2.5}),
    ("i0_neg", {"i0_a": -0.1}),
    ("i0_pos", {"i0_a": 0.1}),
    ("t_short", {"t_end_s": 0.005, "n_points": 2000}),
    ("t_long", {"t_end_s": 0.2, "n_points": 20000}),
]


@pytest.mark.parametrize("name,overrides", EDGE_CASES)
def test_edge_finite_and_consistent(name, overrides):
    case = _run(**overrides)
    if "error" in case:
        pytest.skip(f"{name}: нет DC точки — {case['error']}")
    assert case["finite"], f"{name}: NaN/Inf в решении"
    lin, m = case["lin"], case["m"]
    # Если линейный критерий запрещает автогенерацию — устойчивых колебаний быть не должно
    if not lin["can_oscillate_linear"]:
        assert not m["self_oscillation"], (
            f"{name}: теория=нет, симуляция=да — {_label(case)}"
        )
    # Если критерий допускает автогенерацию — при достаточном t_end ожидаем колебания
    elif case["params"].get("t_end_s", DEFAULTS["t_end_s"]) >= 0.01:
        assert m["self_oscillation"], (
            f"{name}: теория=да, симуляция=нет — {_label(case)}"
        )
    # Частота при колебаниях — порядок f_LC (допуск ×5 из-за нелинейности)
    if m["self_oscillation"] and m["fundamental_freq_hz"] > 0:
        f_lc = lin["f_lc_hz"]
        ratio = m["fundamental_freq_hz"] / f_lc
        assert 0.2 < ratio < 5.0, f"{name}: f={m['fundamental_freq_hz']:.0f} vs f_LC={f_lc:.0f}"


def test_report_edge_table(capsys):
    """Печать сводки для ручной проверки (pytest -s)."""
    rows = []
    for name, overrides in EDGE_CASES:
        case = _run(**overrides)
        if "error" in case:
            rows.append((name, "ERR", "-", "-", "-", case["error"][:40]))
            continue
        lin, m = case["lin"], case["m"]
        p = case["params"]
        if not lin["can_oscillate_linear"]:
            ok = not m["self_oscillation"]
        elif p.get("t_end_s", DEFAULTS["t_end_s"]) < 0.01:
            ok = True  # короткое время — отдельная проверка
        else:
            ok = m["self_oscillation"]
        rows.append(
            (
                name,
                "OK" if ok else "MISMATCH",
                "Y" if lin["can_oscillate_linear"] else "N",
                "Y" if m["self_oscillation"] else "N",
                f"{m['u_amplitude_steady']:.3f}",
                f"THD={m['thd']:.2f}",
            )
        )
    print("\n=== Краевые значения М5 ===")
    print(f"{'case':<14} {'match':<8} {'lin':<4} {'sim':<4} {'amp':<8} note")
    for r in rows:
        print(f"{r[0]:<14} {r[1]:<8} {r[2]:<4} {r[3]:<4} {r[4]:<8} {r[5]}")
    mismatches = [r for r in rows if r[1] == "MISMATCH"]
    assert not mismatches, f"Расхождения теория/симуляция: {mismatches}"
