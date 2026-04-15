"""Numerical simulation pipeline for the radio transmitter-receiver pair."""

from __future__ import annotations

import math

import numpy as np

from m7.models.input import SimulationConfig
from m7.models.models import SimulationResult


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(signal)))
    if max_abs == 0.0:
        return signal.copy()
    return signal / max_abs


def _generate_source_signal(t: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low_component = (
        0.55 * np.sin(2.0 * math.pi * 1_200.0 * t)
        + 0.35 * np.sin(2.0 * math.pi * 2_100.0 * t + 0.2)
        + 0.25 * np.sin(2.0 * math.pi * 3_000.0 * t + 1.1)
    )
    burst = 0.15 * np.sign(np.sin(2.0 * math.pi * 450.0 * t))
    tremolo = 1.0 + 0.1 * np.sin(2.0 * math.pi * 70.0 * t)
    noisy_audio_like = tremolo * low_component + burst + 0.03 * rng.standard_normal(t.size)
    return _normalize_signal(noisy_audio_like)


def _generate_high_freq_noise(t: np.ndarray, config: SimulationConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 1)
    frequencies = rng.uniform(
        low=config.noise_min_freq_hz,
        high=config.noise_max_freq_hz,
        size=config.noise_components,
    )
    phases = rng.uniform(0.0, 2.0 * math.pi, size=config.noise_components)
    amplitudes = rng.uniform(0.2, 1.0, size=config.noise_components)

    noise = np.zeros_like(t)
    for f_hz, phase, amp in zip(frequencies, phases, amplitudes, strict=True):
        noise += amp * np.sin(2.0 * math.pi * f_hz * t + phase)
    noise = _normalize_signal(noise)
    noise += 0.1 * rng.standard_normal(t.size)
    return _normalize_signal(noise)


def _simulate_rlc_response(input_signal: np.ndarray, dt: float, config: SimulationConfig) -> np.ndarray:
    # Analytic RLC model in frequency domain for voltage on R (band-pass behavior).
    w0 = 2.0 * math.pi * config.carrier_freq_hz
    q_factor = w0 * config.rlc_inductance_h / config.rlc_resistance_ohm

    freqs_hz = np.fft.rfftfreq(input_signal.size, d=dt)
    omega = 2.0 * math.pi * freqs_hz
    ratio = omega / w0

    numerator = 1j * ratio / q_factor
    denominator = 1.0 - ratio**2 + 1j * ratio / q_factor
    transfer = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    spectrum = np.fft.rfft(input_signal)
    output_spectrum = spectrum * transfer
    return np.fft.irfft(output_spectrum, n=input_signal.size)


def _rectifier_envelope_detector(signal: np.ndarray, scale: float) -> np.ndarray:
    # Full-wave ideal rectification is more robust to phase shifts in the carrier path.
    return scale * np.abs(signal)


def _single_pole_lowpass(signal: np.ndarray, dt: float, tau: float) -> np.ndarray:
    alpha = dt / (tau + dt)
    filtered = np.zeros_like(signal)
    for idx in range(1, signal.size):
        filtered[idx] = filtered[idx - 1] + alpha * (signal[idx] - filtered[idx - 1])
    return filtered


def _align_by_max_correlation(reference: np.ndarray, candidate: np.ndarray, max_shift: int) -> np.ndarray:
    best_shift = 0
    best_corr = -np.inf
    n = reference.size
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            ref_slice = reference[shift:]
            cand_slice = candidate[: n - shift]
        else:
            ref_slice = reference[: n + shift]
            cand_slice = candidate[-shift:]
        if ref_slice.size < 32:
            continue
        corr = float(np.corrcoef(ref_slice, cand_slice)[0, 1])
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    aligned = np.zeros_like(candidate)
    if best_shift >= 0:
        aligned[best_shift:] = candidate[: n - best_shift]
    else:
        aligned[: n + best_shift] = candidate[-best_shift:]
    return aligned


def run_simulation(config: SimulationConfig) -> SimulationResult:
    dt = 1.0 / config.sample_rate_hz
    t = np.arange(0.0, config.duration_s, dt)

    source = _generate_source_signal(t, config.seed)
    transmitted = (1.0 + config.modulation_index * source) * np.cos(
        2.0 * math.pi * config.carrier_freq_hz * t
    )

    channel_noise = config.noise_relative_amplitude * _generate_high_freq_noise(t, config)
    received = transmitted + channel_noise

    rlc_output = _simulate_rlc_response(received, dt, config)
    envelope = _rectifier_envelope_detector(rlc_output, config.envelope_scale)
    lowpassed = _single_pole_lowpass(envelope, dt, config.lowpass_tau_s)

    # Remove DC component and normalize to compare with the source envelope.
    recovered = _normalize_signal(lowpassed - np.mean(lowpassed))
    max_shift = int(0.002 * config.sample_rate_hz)
    recovered = _align_by_max_correlation(source, recovered, max_shift=max_shift)
    if np.corrcoef(source, recovered)[0, 1] < 0:
        recovered = -recovered

    # Ignore early transient while the resonant contour is building up oscillation.
    warmup_idx = max(1, int(0.1 * recovered.size))
    mse = float(np.mean((recovered[warmup_idx:] - source[warmup_idx:]) ** 2))
    corrcoef = float(np.corrcoef(source[warmup_idx:], recovered[warmup_idx:])[0, 1])

    return SimulationResult(
        t=t,
        source_signal=source,
        transmitted_signal=transmitted,
        channel_noise=channel_noise,
        received_signal=received,
        rlc_output=rlc_output,
        envelope=envelope,
        recovered_signal=recovered,
        mse=mse,
        corrcoef=corrcoef,
    )
