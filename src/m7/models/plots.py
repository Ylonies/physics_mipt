"""Visualization for all main stages of the radio-link simulation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from m7.models.input import SimulationConfig
from m7.models.models import SimulationResult


def _slice_for_visibility(t: np.ndarray, max_time: float = 0.002) -> slice:
    idx = int(np.searchsorted(t, max_time))
    return slice(0, max(100, idx))


def plot_simulation(result: SimulationResult, config: SimulationConfig) -> None:
    view = _slice_for_visibility(result.t)
    t_ms = result.t[view] * 1_000.0

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), constrained_layout=True)

    axes[0].plot(t_ms, result.source_signal[view], label="Source (baseband)", lw=1.2)
    axes[0].set_title("Original digital/audio-like message signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t_ms, result.transmitted_signal[view], label="AM transmitted", lw=1.0)
    axes[1].set_title(f"AM modulation, carrier = {config.carrier_freq_hz / 1_000:.0f} kHz")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t_ms, result.received_signal[view], label="After channel + HF noise", lw=1.0)
    axes[2].set_title("Received noisy RF signal")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    axes[3].plot(t_ms, result.rlc_output[view], label="RLC output", lw=1.0)
    axes[3].plot(t_ms, result.envelope[view], label="Rectified envelope", lw=1.0)
    axes[3].set_title("Resonant selection and ideal diode detector")
    axes[3].set_ylabel("Amplitude")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    axes[4].plot(result.t * 1_000.0, result.source_signal, label="Source", lw=1.2)
    axes[4].plot(result.t * 1_000.0, result.recovered_signal, label="Recovered after RC LPF", lw=1.2)
    axes[4].set_title("Message recovery quality")
    axes[4].set_xlabel("Time, ms")
    axes[4].set_ylabel("Amplitude")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="upper right")

    plt.show()
