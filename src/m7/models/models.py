"""Domain models for the radio-link simulation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    t: np.ndarray
    source_signal: np.ndarray
    transmitted_signal: np.ndarray
    channel_noise: np.ndarray
    received_signal: np.ndarray
    rlc_output: np.ndarray
    envelope: np.ndarray
    recovered_signal: np.ndarray
    mse: float
    corrcoef: float
