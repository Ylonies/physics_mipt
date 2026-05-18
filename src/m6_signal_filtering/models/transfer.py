from __future__ import annotations

import numpy as np


def rc_omega_c(R: float, C: float) -> float:
    return 1.0 / (R * C)


def rc_H(omega: float | np.ndarray, R: float, C: float) -> np.ndarray:
    w = np.asarray(omega, dtype=float)
    return 1.0 / (1.0 + 1j * w * R * C)


def rc_H_mag(omega: float | np.ndarray, R: float, C: float) -> np.ndarray:
    return np.abs(rc_H(omega, R, C))


def rc_H_phase(omega: float | np.ndarray, R: float, C: float) -> np.ndarray:
    return np.angle(rc_H(omega, R, C))


def rlc_omega_0(L: float, C: float) -> float:
    return 1.0 / np.sqrt(L * C)


def rlc_Q(R: float, L: float, C: float) -> float:
    return (1.0 / R) * np.sqrt(L / C)


def rlc_H(omega: float | np.ndarray, R: float, L: float, C: float) -> np.ndarray:
    w = np.asarray(omega, dtype=float)
    den = 1.0 - w**2 * L * C + 1j * w * R * C
    return (1j * w * R * C) / den


def rlc_H_mag(omega: float | np.ndarray, R: float, L: float, C: float) -> np.ndarray:
    return np.abs(rlc_H(omega, R, L, C))


def rlc_H_phase(omega: float | np.ndarray, R: float, L: float, C: float) -> np.ndarray:
    return np.angle(rlc_H(omega, R, L, C))
