from __future__ import annotations

import numpy as np

from src.m6_signal_filtering.models.signals import square_fourier_amplitudes


def suggest_dt(tau: float, omega_max: float, *, n_per_period: int = 50) -> float:
    return min(tau / 200, 2 * np.pi / (n_per_period * omega_max))


def measure_harmonic(t: np.ndarray, u: np.ndarray, omega: float, t_settle: float) -> tuple[float, float]:
    m = t >= t_settle
    tw, uw = t[m], u[m]
    a_sin = 2.0 * np.mean(uw * np.sin(omega * tw))
    a_cos = 2.0 * np.mean(uw * np.cos(omega * tw))
    return float(np.hypot(a_sin, a_cos)), float(np.arctan2(a_cos, a_sin))


def fft_harmonic_amplitudes(t: np.ndarray, u: np.ndarray, period: float, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    t_end = t[-1]
    m = (t >= t_end - period) & (t <= t_end)
    seg = u[m]
    w0 = 2 * np.pi / period
    t_seg = t[m] - (t_end - period)
    ks = np.arange(1, k_max + 1, 2)
    amps = []
    for k in ks:
        s = np.sin(k * w0 * t_seg)
        c = np.cos(k * w0 * t_seg)
        a = 2.0 * np.dot(seg, s) / len(seg)
        b = 2.0 * np.dot(seg, c) / len(seg)
        amps.append(np.hypot(a, b))
    return ks, np.array(amps)


def synthesize_from_harmonics(t: np.ndarray, ks: np.ndarray, amps_in: np.ndarray, h_fn, omega0: float) -> np.ndarray:
    u = np.zeros_like(t)
    for k, a in zip(ks, amps_in):
        w = k * omega0
        h = h_fn(w)
        u += a * np.abs(h) * np.sin(w * t + np.angle(h))
    return u


def analytical_square_spectrum(u0: float, period: float, k_max: int, h_fn) -> tuple[np.ndarray, np.ndarray]:
    ks, amps_in = square_fourier_amplitudes(u0, k_max)
    w0 = 2 * np.pi / period
    return ks, amps_in * np.abs(h_fn(ks * w0))
