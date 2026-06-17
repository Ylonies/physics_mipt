import numpy as np
import pytest

from src.m1_electrostatics.models.grid_generator import GridGenerator
from src.m1_electrostatics.models.physics_model import PhysicsModel


@pytest.fixture
def physics():
    return PhysicsModel()

@pytest.fixture
def generator():
    return GridGenerator()

def test_input_handler_boundaries_max_load(physics, generator):
    R, d, V, n = 10.0, 0.1, 1e6, 50

    pts, areas, pots = generator.create_system(R, d, V, n)
    charges = physics.calculate_charges(pts, areas, pots)
    c_num = physics.calculate_capacitance(charges, V, len(pts)//2)

    assert np.isfinite(c_num), "Расчет упал при максимальных параметрах R и n"
    eps0 = 8.8541878128e-12
    c_ideal = eps0 * (np.pi * R**2) / d
    assert c_num > c_ideal, "Даже при R=10 эффект краев должен быть положительным"

def test_input_handler_boundaries_min_geometry(physics, generator):
    R, d, V, n = 0.001, 5.0, -1e6, 5

    pts, areas, pots = generator.create_system(R, d, V, n)
    charges = physics.calculate_charges(pts, areas, pots)
    c_num = physics.calculate_capacitance(charges, V, len(pts)//2)

    assert c_num > 0, "Емкость должна быть положительной"

def test_input_handler_extreme_aspect_ratio(physics, generator):

    # 1. Почти слиплись
    R1, d1, n = 10.0, 0.0001, 20
    pts1, areas1, pots1 = generator.create_system(R1, d1, 100, n)
    c1 = physics.calculate_capacitance(physics.calculate_charges(pts1, areas1, pots1), 100, len(pts1)//2)

    # 2. Почти точки
    R2, d2 = 0.001, 5.0
    pts2, areas2, pots2 = generator.create_system(R2, d2, 100, n)
    c2 = physics.calculate_capacitance(physics.calculate_charges(pts2, areas2, pots2), 100, len(pts2)//2)

    assert np.isfinite(c1) and np.isfinite(c2), "Система не справилась с экстремальным соотношением R/d"

def test_non_negative_edge_effect_at_all_steps(physics, generator):
    R, d, V = 0.1, 0.05, 100.0
    for n in [5, 15, 30, 50]:
        pts, areas, pots = generator.create_system(R, d, V, n)
        charges = physics.calculate_charges(pts, areas, pots)
        c_num = physics.calculate_capacitance(charges, V, len(pts)//2)

        eps0 = 8.8541878128e-12
        c_ideal = eps0 * (np.pi * R**2) / d

        assert c_num >= c_ideal, f"При n={n} емкость ниже идеальной! Нужно подправить самодействие A_ii."