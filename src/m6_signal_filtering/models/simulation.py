from __future__ import annotations

import numpy as np

from src.m6_signal_filtering.models import integrator, signals, spectrum, transfer


def _harmonic_t_max(tau: float, omega: float, t_floor: float) -> float:
    return max(t_floor, 5.0 * tau, 8.0 * (2 * np.pi / omega))


def run_rc_harmonic(R: float, C: float, u0: float, omega: float, t_max: float, dt: float) -> dict:
    tau = R * C
    t_max = _harmonic_t_max(tau, omega, t_max)
    t_settle = max(10.0 * tau, 5.0 * (2 * np.pi / omega))
    u_in_fn = lambda t: signals.harmonic(np.asarray(t), u0, omega)
    hist = integrator.integrate_rc(u_in_fn, R, C, 0.0, 0.0, t_max, dt)
    amp, phase = spectrum.measure_harmonic(hist["t"], hist["u_out"], omega, t_settle)
    h = transfer.rc_H(omega, R, C)
    return {
        **hist,
        "omega": omega,
        "amp_num": amp,
        "phase_num": phase,
        "amp_theor": u0 * np.abs(h),
        "phase_theor": np.angle(h),
        "t_settle": t_settle,
    }


def sweep_rc_harmonic(R: float, C: float, u0: float, omega_ratios: np.ndarray, t_max: float, dt: float) -> list[dict]:
    wc = transfer.rc_omega_c(R, C)
    return [run_rc_harmonic(R, C, u0, r * wc, t_max, dt) for r in omega_ratios]


def run_rc_square(R: float, C: float, u0: float, period: float, t_max: float, dt: float, k_max: int) -> dict:
    t_max = max(period, int(np.ceil(t_max / period)) * period)
    u_in_fn = lambda t: signals.square_meander(np.asarray(t), u0, period)
    hist = integrator.integrate_rc(u_in_fn, R, C, 0.0, 0.0, t_max, dt)
    ks_fft, amps_fft = spectrum.fft_harmonic_amplitudes(hist["t"], hist["u_out"], period, k_max)
    h_fn = lambda w: transfer.rc_H(w, R, C)
    ks_ana, amps_ana = spectrum.analytical_square_spectrum(u0, period, k_max, h_fn)
    _, amps_in = signals.square_fourier_amplitudes(u0, k_max)
    u_syn = spectrum.synthesize_from_harmonics(hist["t"], ks_ana, amps_in, h_fn, 2 * np.pi / period)
    return {**hist, "ks_fft": ks_fft, "amps_fft": amps_fft, "ks_ana": ks_ana, "amps_ana": amps_ana, "u_syn": u_syn}


def run_rlc_harmonic(R: float, L: float, C: float, u0: float, omega: float, t_max: float, dt: float) -> dict:
    w0 = transfer.rlc_omega_0(L, C)
    q = transfer.rlc_Q(R, L, C)
    tau = 2 * q / w0
    t_max = _harmonic_t_max(tau, omega, t_max)
    t_settle = max(5.0 * tau, 5.0 * (2 * np.pi / omega))
    u_in_fn = lambda t: signals.harmonic(np.asarray(t), u0, omega)
    hist = integrator.integrate_rlc(u_in_fn, R, L, C, 0.0, 0.0, 0.0, t_max, dt)
    amp, phase = spectrum.measure_harmonic(hist["t"], hist["u_out"], omega, t_settle)
    h = transfer.rlc_H(omega, R, L, C)
    return {
        **hist,
        "omega": omega,
        "amp_num": amp,
        "phase_num": phase,
        "amp_theor": u0 * np.abs(h),
        "phase_theor": np.angle(h),
        "t_settle": t_settle,
    }


def sweep_rlc_harmonic(R: float, L: float, C: float, u0: float, omega_ratios: np.ndarray, t_max: float, dt: float) -> list[dict]:
    w0 = transfer.rlc_omega_0(L, C)
    return [run_rlc_harmonic(R, L, C, u0, r * w0, t_max, dt) for r in omega_ratios]


def run_rlc_square(R: float, L: float, C: float, u0: float, period: float, t_max: float, dt: float, k_max: int) -> dict:
    t_max = max(period, int(np.ceil(t_max / period)) * period)
    u_in_fn = lambda t: signals.square_meander(np.asarray(t), u0, period)
    hist = integrator.integrate_rlc(u_in_fn, R, L, C, 0.0, 0.0, 0.0, t_max, dt)
    ks_fft, amps_fft = spectrum.fft_harmonic_amplitudes(hist["t"], hist["u_out"], period, k_max)
    h_fn = lambda w: transfer.rlc_H(w, R, L, C)
    ks_ana, amps_ana = spectrum.analytical_square_spectrum(u0, period, k_max, h_fn)
    _, amps_in = signals.square_fourier_amplitudes(u0, k_max)
    u_syn = spectrum.synthesize_from_harmonics(hist["t"], ks_ana, amps_in, h_fn, 2 * np.pi / period)
    return {**hist, "ks_fft": ks_fft, "amps_fft": amps_fft, "ks_ana": ks_ana, "amps_ana": amps_ana, "u_syn": u_syn}
