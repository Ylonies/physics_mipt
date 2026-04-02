"""Графики поля и траектории."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.m3_magnetic_mirror.models.field import bz_axis_two_rings


def plot_bz_axis_vs_numeric(ax, field, zs: np.ndarray) -> None:
    b_num = np.array([field.B_direct(np.array([0.0, 0.0, z]))[2] for z in zs])
    b_ana = np.array([bz_axis_two_rings(z, field.R, field.d, field.I) for z in zs])
    ax.plot(zs, b_ana, label=r"$B_z$ analytic")
    ax.plot(zs, b_num, "--", label=r"$B_z$ Biot–Savart")
    ax.set_xlabel("z, m")
    ax.set_ylabel(r"$B_z$, T")
    ax.legend()


def plot_trajectory_overview(fig, hist: dict, mass: float) -> None:
    r, t = hist["r"], hist["t"]
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    cyl_r = np.hypot(x, y)

    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot(x, y, z, lw=0.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(222)
    ax2.plot(cyl_r, z, lw=0.8)
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel("z")

    ax3 = fig.add_subplot(223)
    ax3.plot(t, cyl_r, label=r"$r(t)$")
    ax3.plot(t, z, label=r"$z(t)$")
    ax3.set_xlabel("t, s")
    ax3.legend()

    v = hist["v"]
    ek = 0.5 * mass * np.sum(v * v, axis=1)
    ax4 = fig.add_subplot(224)
    ax4.plot(t, (ek - ek[0]) / (ek[0] + 1e-30), lw=0.8)
    ax4.set_xlabel("t, s")
    ax4.set_ylabel(r"$(E_k - E_{k0})/E_{k0}$")


def plot_field_magnitude_rz(ax, field, nr: int = 28, nz: int = 40) -> None:
    rs = np.linspace(0, field._r_max, nr)
    zs = np.linspace(field._z_lo, field._z_hi, nz)
    Rg, Zg = np.meshgrid(rs, zs, indexing="ij")
    flat = np.column_stack([Rg.ravel(), np.zeros(Rg.size), Zg.ravel()])
    bm = np.array([np.linalg.norm(field.B_direct(flat[k])) for k in range(flat.shape[0])])
    bm = bm.reshape(nr, nz)
    cs = ax.contourf(Zg, Rg, bm, levels=28)
    ax.set_xlabel("z, m")
    ax.set_ylabel("r, m")
    plt.colorbar(cs, ax=ax, label=r"$|B|$, T")
