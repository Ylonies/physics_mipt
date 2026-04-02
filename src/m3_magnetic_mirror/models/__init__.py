"""Ядро M3: поле, интегратор, симуляция, графики."""

from src.m3_magnetic_mirror.models.field import (
    MagneticField,
    biot_savart_sum,
    bz_axis_single_ring,
    bz_axis_two_rings,
)
from src.m3_magnetic_mirror.models.integrator import integrate, rk4_step

__all__ = [
    "MagneticField",
    "biot_savart_sum",
    "bz_axis_single_ring",
    "bz_axis_two_rings",
    "integrate",
    "rk4_step",
]
