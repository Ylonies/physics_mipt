"""Сборка поля, шаг по времени, одиночная / пакетная траектория."""

from __future__ import annotations

import numpy as np

from src.m3_magnetic_mirror.models.field import MagneticField, bz_axis_two_rings
from src.m3_magnetic_mirror.models import integrator


def b_ref_default(R: float, d: float, I: float) -> float:
    return abs(bz_axis_two_rings(0.0, R, d, I))


def cyclotron_period(q: float, m: float, b: float) -> float:
    return 2 * np.pi * m / (abs(q) * b)


def run_single(
    R: float,
    d: float,
    I: float,
    q: float,
    m: float,
    r0: np.ndarray,
    v0: np.ndarray,
    *,
    n_seg: int,
    grid_nr: int,
    grid_nz: int,
    dt_frac_tcyc: float,
    b_ref: float | None,
    t_max_tcyc: float,
    escape_z: float | None,
    r_max_factor: float = 3.0,
    z_margin_factor: float = 1.0,
) -> dict:
    field = MagneticField(
        R,
        d,
        I,
        n_seg,
        grid_nr=grid_nr,
        grid_nz=grid_nz,
        r_max_factor=r_max_factor,
        z_margin_factor=z_margin_factor,
    )
    b0 = b_ref if b_ref is not None else b_ref_default(R, d, I)
    dt = cyclotron_period(q, m, b0) / dt_frac_tcyc
    t_max = t_max_tcyc * cyclotron_period(q, m, b0)
    n_steps = max(1, int(t_max / dt))
    esc = escape_z if escape_z is not None else d / 2 + R
    qm = q / m
    hist = integrator.integrate(r0, v0, dt, n_steps, qm, field.B, escape_z=esc)
    hist["dt"] = dt
    hist["escape_z"] = esc
    hist["field"] = field
    hist["b_ref"] = b0
    return hist


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def run_batch(
    R: float,
    d: float,
    I: float,
    q: float,
    m: float,
    n_particles: int,
    rng: np.random.Generator,
    *,
    n_seg: int,
    grid_nr: int,
    grid_nz: int,
    dt_frac_tcyc: float,
    b_ref: float | None,
    t_max_tcyc: float,
    escape_z: float | None,
    r0_max: float,
    z0_half: float,
    v_mag: float,
    r_max_factor: float = 3.0,
    z_margin_factor: float = 1.0,
) -> list[dict]:
    dirs = random_unit_vectors(n_particles, rng)
    out = []
    for i in range(n_particles):
        r0 = np.array(
            [
                rng.uniform(-r0_max, r0_max),
                rng.uniform(-r0_max, r0_max),
                rng.uniform(-z0_half, z0_half),
            ]
        )
        v0 = dirs[i] * v_mag
        out.append(
            run_single(
                R,
                d,
                I,
                q,
                m,
                r0,
                v0,
                n_seg=n_seg,
                grid_nr=grid_nr,
                grid_nz=grid_nz,
                dt_frac_tcyc=dt_frac_tcyc,
                b_ref=b_ref,
                t_max_tcyc=t_max_tcyc,
                escape_z=escape_z,
                r_max_factor=r_max_factor,
                z_margin_factor=z_margin_factor,
            )
        )
    return out
