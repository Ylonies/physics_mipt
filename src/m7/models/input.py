"""Input layer for radio-link simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    duration_s: float = 0.02
    sample_rate_hz: float = 5_000_000.0
    carrier_freq_hz: float = 200_000.0
    modulation_index: float = 0.7
    seed: int = 42

    # Channel noise: high-frequency interference.
    noise_relative_amplitude: float = 0.25
    noise_components: int = 20
    noise_min_freq_hz: float = 350_000.0
    noise_max_freq_hz: float = 1_200_000.0

    # High-Q resonant series RLC receiver stage.
    rlc_resistance_ohm: float = 8.0
    rlc_inductance_h: float = 1e-3

    # Envelope detector and low-pass filter.
    envelope_scale: float = 1.0
    lowpass_tau_s: float = 6e-5


class InputHandler:
    """Provides simulation parameters, similar to other modules."""

    @staticmethod
    def get_parameters() -> SimulationConfig:
        return SimulationConfig()
