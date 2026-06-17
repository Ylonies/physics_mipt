import numpy as np
from scipy import linalg

class PhysicsModel:
    def __init__(self, eps0=8.8541878128e-12):
        self.eps0 = eps0

    def build_potential_matrix(self, points, areas):
        n = len(points)

        # 1. Векторизованный расчет расстояний
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        # 2. Матрица взаимных потенциалов
        with np.errstate(divide='ignore'):
            a_matrix = 1.0 / (4.0 * np.pi * self.eps0 * dist_matrix)

        # Для кругового элемента средний потенциал Phi = q / (eps0 * sqrt(S) * const)
        r_eff = np.sqrt(areas / np.pi)

        # A_ii = 8 / (3 * pi * eps0 * r_eff)
        diag_values = 8.0 / (3.0 * np.pi**2 * self.eps0 * r_eff)

        np.fill_diagonal(a_matrix, diag_values)
        return a_matrix

    def calculate_charges(self, points, areas, potentials):
        a_matrix = self.build_potential_matrix(points, areas)
        charges = linalg.solve(a_matrix, potentials)
        return charges

    def calculate_capacitance(self, charges, v_diff, half_n):
        q_total = np.sum(charges[:half_n])
        return abs(q_total / v_diff) * 2