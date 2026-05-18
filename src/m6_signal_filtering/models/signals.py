from __future__ import annotations

import numpy as np


def harmonic(t: np.ndarray, u0: float, omega: float) -> np.ndarray:
    return u0 * np.sin(omega * t)


def square_meander(t: np.ndarray, u0: float, period: float) -> np.ndarray:
    phase = np.mod(t, period) / period
    return np.where(phase < 0.5, u0, -u0)


def square_fourier_amplitudes(u0: float, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    ks = np.arange(1, k_max + 1, 2)
    return ks, (4.0 * u0 / np.pi) / ks
