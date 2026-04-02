"""Аналитические проверки физики M3 (без длительных стресс-прогонов).

Покрытие: п.7.1 — B_z на оси (Biot–Савар vs формула); симметрия B_⊥=0 на оси;
п.7.2 — однородное B: радиус гироорбиты и период ω=|q|B/m; п.7.3 — сохранение E_k
в поле зеркала с интерполяцией сетки.
"""

import numpy as np
import pytest

from src.m3_magnetic_mirror.models.field import MagneticField, bz_axis_two_rings
from src.m3_magnetic_mirror.models import integrator


@pytest.fixture
def ring_params():
    return {"R": 1.0, "d": 2.0, "I": 1000.0, "n_seg": 120}


def test_bz_axis_biot_savart_matches_analytic(ring_params):
    p = ring_params
    field = MagneticField(p["R"], p["d"], p["I"], p["n_seg"])
    zs = np.linspace(field._z_lo + 0.02, field._z_hi - 0.02, 50)
    for z in zs:
        b_num = field.B_direct(np.array([0.0, 0.0, z]))[2]
        b_ana = bz_axis_two_rings(z, p["R"], p["d"], p["I"])
        assert abs(b_num - b_ana) / (abs(b_ana) + 1e-20) < 0.01


def test_rk4_uniform_b_gyration_radius_and_period():
    """Однородное B‖z: окружность через начало, r_max ≈ 2 r_L, ω = |q|B/m."""
    bz = 0.05
    q, m = 1.602176634e-19, 1.67262192369e-27

    def b_fn(_r):
        return np.array([0.0, 0.0, bz])

    v_perp = 1e5
    r0 = np.zeros(3)
    v0 = np.array([0.0, v_perp, 0.0])
    omega = abs(q) * bz / m
    r_l = m * v_perp / (abs(q) * bz)
    dt = (2 * np.pi / omega) / 100
    n_steps = int(round(2 * 2 * np.pi / (omega * dt)))
    hist = integrator.integrate(r0, v0, dt, n_steps, q / m, b_fn, escape_z=None)
    x, y = hist["r"][:, 0], hist["r"][:, 1]
    r_xy = np.hypot(x, y)
    assert np.max(r_xy) < 2 * r_l * 1.05
    assert np.max(r_xy) > 2 * r_l * 0.85
    ek = 0.5 * m * np.sum(hist["v"] * hist["v"], axis=1)
    assert np.max(np.abs(ek - ek[0]) / ek[0]) < 1e-4
    vx = hist["v"][:, 0]
    peaks = np.where((vx[1:-1] > vx[:-2]) & (vx[1:-1] > vx[2:]))[0] + 1
    if len(peaks) >= 2:
        t_per = np.mean(np.diff(hist["t"][peaks]))
        assert abs(t_per - 2 * np.pi / omega) / (2 * np.pi / omega) < 0.02


def test_kinetic_energy_conserved_in_mirror_field(ring_params):
    """Поле зеркала через интерполяцию; E_k = const (магнитное поле не совершает работы)."""
    p = ring_params
    q, m = 1.602176634e-19, 1.67262192369e-27
    field = MagneticField(
        p["R"],
        p["d"],
        p["I"],
        p["n_seg"],
        grid_nr=36,
        grid_nz=72,
    )
    b0 = abs(bz_axis_two_rings(0.0, p["R"], p["d"], p["I"]))
    dt = (2 * np.pi * m / (abs(q) * b0)) / 64
    t_max = 4 * 2 * np.pi * m / (abs(q) * b0)
    n_steps = int(t_max / dt)
    r0 = np.array([0.05, 0.0, 0.0])
    v0 = np.array([0.0, 8e4, 5e3])
    hist = integrator.integrate(r0, v0, dt, n_steps, q / m, field.B, escape_z=None)
    ek = 0.5 * m * np.sum(hist["v"] * hist["v"], axis=1)
    rel = np.max(np.abs(ek - ek[0]) / (ek[0] + 1e-30))
    assert rel < 0.02


def test_transverse_field_vanishes_on_axis(ring_params):
    """Аксиальная симметрия: на оси B_x ≈ B_y ≈ 0 (численный Biot–Савар)."""
    p = ring_params
    field = MagneticField(p["R"], p["d"], p["I"], p["n_seg"])
    for z in np.linspace(field._z_lo + 0.05, field._z_hi - 0.05, 25):
        b = field.B_direct(np.array([0.0, 0.0, z]))
        assert np.hypot(b[0], b[1]) < 1e-9 * (abs(b[2]) + 1e-15)
