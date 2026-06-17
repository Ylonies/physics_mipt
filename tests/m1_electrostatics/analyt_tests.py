import pytest
import numpy as np
from src.m1_electrostatics.models.physics_model import PhysicsModel
from src.m1_electrostatics.models.grid_generator import GridGenerator

@pytest.fixture
def model():
    return PhysicsModel()

@pytest.fixture
def generator():
    return GridGenerator()

# ТЕСТ 1: Проверка на нулевой потенциал
def test_zero_potential(model, generator):
    R = 1.0
    d = 0.1
    V = 0.0
    n = 10
    pts, areas, pots = generator.create_system(R, d, V, n)
    charges = model.calculate_charges(pts, areas, pots)

    assert np.allclose(charges, 0.0), "При V=0 заряды должны быть нулевыми"

# ТЕСТ 2: Симметрия зарядов
def test_charge_symmetry(model, generator):
    R = 0.5
    d = 0.05
    V = 100.0
    n = 15
    pts, areas, pots = generator.create_system(R, d, V, n)
    charges = model.calculate_charges(pts, areas, pots)

    q_sum = np.sum(charges)
    assert abs(q_sum) < 1e-12, f"Суммарный заряд системы должен быть 0, получено: {q_sum}"

# ТЕСТ 3: Эффект расстояния
def test_distance_effect(model, generator):
    R = 0.2
    V = 100
    n = 10

    pts1, areas1, pots1 = generator.create_system(R, 0.01, V, n)
    c1 = model.calculate_capacitance(model.calculate_charges(pts1, areas1, pots1), V, len(pts1)//2)

    pts2, areas2, pots2 = generator.create_system(R, 0.1, V, n)
    c2 = model.calculate_capacitance(model.calculate_charges(pts2, areas2, pots2), V, len(pts2)//2)

    assert c1 > c2, "Емкость должна уменьшаться при увеличении расстояния d"

# ТЕСТ 4: Сходимость при измельчении сетки
def test_grid_convergence(model, generator):
    R = 0.1
    d = 0.02
    V = 50.0

    pts_low, areas_low, pots_low = generator.create_system(R, d, V, 30)
    c_low = model.calculate_capacitance(model.calculate_charges(pts_low, areas_low, pots_low), V, len(pts_low)//2)

    pts_high, areas_high, pots_high = generator.create_system(R, d, V, 40)
    c_high = model.calculate_capacitance(model.calculate_charges(pts_high, areas_high, pots_high), V, len(pts_high)//2)

    # Разница не должна быть огромной (сетка сходится)
    rel_diff = abs(c_high - c_low) / c_high
    assert rel_diff < 0.05, f"Слишком большая разница при изменении сетки: {rel_diff:.2%}"