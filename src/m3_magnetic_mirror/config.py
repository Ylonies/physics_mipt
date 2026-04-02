"""Editable defaults for M3. Change values here or override in code."""

import numpy as np

# --- geometry & currents ---
RING_RADIUS = 1.0  # m
RING_SEPARATION = 2.0  # m, distance between parallel planes (rings at ±d/2)
RING_CURRENT = 1000.0  # A, same direction in both rings (CCW when viewed from +z)

# --- particle (proton) ---
PARTICLE_CHARGE = 1.602176634e-19  # C
PARTICLE_MASS = 1.67262192369e-27  # kg

# --- field discretization ---
N_SEGMENTS_PER_RING = 100
FIELD_GRID_NR = 50
FIELD_GRID_NZ = 100
FIELD_R_MAX_FACTOR = 3.0  # r grid: [0, FACTOR * R] (запас под гиро-радиус)
# extend z beyond ring planes to cover escape boundary (d/2 + R) and a margin
FIELD_Z_MARGIN_FACTOR = 1.0  # ±(d/2 + FACTOR * R)

# --- trajectory ---
DT_FRACTION_OF_TCYC = 50.0  # dt = T_cyc / this, T_cyc = 2π m / (|q| B_ref)
B_REF_FOR_DT = None  # if None, use |B_z(0,0)| on axis
# ~35 периодов — разумное время прогона для графиков; для длинной статистики увеличьте
T_MAX_TCYC = 35.0
ESCAPE_Z = None  # if None, use RING_SEPARATION / 2 + RING_RADIUS

# --- single run initial state (Cartesian, m and m/s) ---
# Небольшое r₀ и преобладание v⊥ дают дольше удержание в пробке (наглядные графики)
R0 = np.array([0.05, 0.0, 0.0])
V0 = np.array([0.0, 8.0e4, 2.5e3])

# --- batch (*): random IC ranges ---
BATCH_N = 80
BATCH_SEED = 42
BATCH_R0_MAX = 0.8 * RING_RADIUS
BATCH_Z0_RANGE = 0.4 * RING_SEPARATION  # uniform in [-range, range]
BATCH_V_MAG = 1.01e5  # sample directions on sphere with fixed |v|

RUN_BATCH = False  # set True in main for histogram / confinement stats
