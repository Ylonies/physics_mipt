"""Biot–Savart field from two coaxial rings; optional (r,z) grid interpolation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

MU0_OVER_4PI = 1e-7  # μ₀/(4π)


def _ring_wire_elements(ring_radius: float, z_plane: float, n_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """CCW current when viewed from +z: dl along increasing φ."""
    dphi = 2 * np.pi / n_seg
    positions = []
    dls = []
    for k in range(n_seg):
        phi = (k + 0.5) * dphi
        c, s = np.cos(phi), np.sin(phi)
        positions.append([ring_radius * c, ring_radius * s, z_plane])
        dls.append([-ring_radius * dphi * s, ring_radius * dphi * c, 0.0])
    return np.asarray(positions), np.asarray(dls)


def biot_savart_sum(obs: np.ndarray, wire_pos: np.ndarray, wire_dl: np.ndarray, current: float) -> np.ndarray:
    """Sum μ₀I/(4π) dl×r/|r|³ over segments; r from wire element to obs."""
    b = np.zeros(3)
    for p, dl in zip(wire_pos, wire_dl):
        r = obs - p
        dist = np.linalg.norm(r)
        if dist < 1e-15:
            continue
        b += MU0_OVER_4PI * current * np.cross(dl, r) / (dist**3)
    return b


def bz_axis_single_ring(z: float, z_ring: float, R: float, I: float) -> float:
    dz = z - z_ring
    den = R**2 + dz**2
    return MU0_OVER_4PI * 2 * np.pi * I * R**2 / den**1.5  # μ₀ I R² / (2 den^{3/2})


def bz_axis_two_rings(z: float, R: float, d: float, I: float) -> float:
    h = d / 2
    return bz_axis_single_ring(z, h, R, I) + bz_axis_single_ring(z, -h, R, I)


class MagneticField:
    def __init__(
        self,
        R: float,
        d: float,
        I: float,
        n_seg: int,
        *,
        grid_nr: int | None = None,
        grid_nz: int | None = None,
        r_max_factor: float = 3.0,
        z_margin_factor: float = 1.0,
    ):
        self.R = R
        self.d = d
        self.I = I
        self.n_seg = n_seg
        z_half = d / 2
        self._z_lo = -(z_half + z_margin_factor * R)
        self._z_hi = z_half + z_margin_factor * R
        self._r_max = r_max_factor * R

        self._wire_up_pos, self._wire_up_dl = _ring_wire_elements(R, z_half, n_seg)
        self._wire_dn_pos, self._wire_dn_dl = _ring_wire_elements(R, -z_half, n_seg)

        self._interp_br: RegularGridInterpolator | None = None
        self._interp_bz: RegularGridInterpolator | None = None
        if grid_nr is not None and grid_nz is not None:
            self._build_grid(grid_nr, grid_nz)

    def B_direct(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(3)
        b = biot_savart_sum(x, self._wire_up_pos, self._wire_up_dl, self.I)
        b += biot_savart_sum(x, self._wire_dn_pos, self._wire_dn_dl, self.I)
        return b

    def _build_grid(self, nr: int, nz: int) -> None:
        rs = np.linspace(0.0, self._r_max, nr)
        zs = np.linspace(self._z_lo, self._z_hi, nz)
        br = np.zeros((nr, nz))
        bz = np.zeros((nr, nz))
        for i, r in enumerate(rs):
            for j, z in enumerate(zs):
                if r < 1e-12:
                    bz[i, j] = bz_axis_two_rings(z, self.R, self.d, self.I)
                    br[i, j] = 0.0
                else:
                    bvec = self.B_direct(np.array([r, 0.0, z]))
                    br[i, j] = bvec[0]
                    bz[i, j] = bvec[2]
        self._interp_br = RegularGridInterpolator(
            (rs, zs), br, bounds_error=False, fill_value=None
        )
        self._interp_bz = RegularGridInterpolator(
            (rs, zs), bz, bounds_error=False, fill_value=None
        )

    def B(self, x: np.ndarray) -> np.ndarray:
        """Return B in Cartesian coordinates at position x."""
        if self._interp_br is None:
            return self.B_direct(x)
        x = np.asarray(x, dtype=float).reshape(3)
        r = float(np.hypot(x[0], x[1]))
        z = float(x[2])
        if r > self._r_max or z < self._z_lo or z > self._z_hi:
            return self.B_direct(x)
        if r < 1e-12:
            bz = float(self._interp_bz([[0.0, z]])[0])
            return np.array([0.0, 0.0, bz])
        phi = np.arctan2(x[1], x[0])
        br = float(self._interp_br([[r, z]])[0])
        bz = float(self._interp_bz([[r, z]])[0])
        return np.array([br * np.cos(phi), br * np.sin(phi), bz])
