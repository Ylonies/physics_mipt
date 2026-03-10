import numpy as np

from src.m1_electrostatics.models.grid_generator import GridGenerator
from src.m1_electrostatics.models.input import InputHandler
from src.m1_electrostatics.models.physics_model import PhysicsModel
from src.m1_electrostatics.models.result_visualizer import ResultVisualizer


class ElectrodeSystem:
    def __init__(self):
        self.model = PhysicsModel()
        self.grid_generator = GridGenerator()

    def run(self):
        try:
            params = InputHandler.get_parameters()

            points, areas, potentials = self.grid_generator.create_system(
                params['R'], params['d'], params['V'], params['n_rings']
            )

            print("\nВыполняется расчет матрицы и решение СЛАУ...")
            charges = self.model.calculate_charges(points, areas, potentials)

            results = self._analyze_results(charges, points, params)
            self._print_results(results)

            viz = ResultVisualizer()
            viz.show_all_plots(points, charges, areas, params["R"], params["d"])

        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except Exception as e:
            print(f"Ошибка вычислений: {e}")

    def _analyze_results(self, charges, points, params):
        """Собирает физические показатели системы"""
        n_half = len(points) // 2

        c_numeric = self.model.calculate_capacitance(charges, params['V'], n_half)

        eps0 = self.model.eps0
        c_ideal = (eps0 * np.pi * params['R']**2) / params['d']

        return {
            'c_numeric': c_numeric,
            'c_ideal': c_ideal,
            'diff_percent': (c_numeric / c_ideal - 1) * 100,
            'total_nodes': len(points)
        }

    def _print_results(self, results):
        print(f"\n=== РЕЗУЛЬТАТЫ РАСЧЕТА ===")
        print(f"Всего элементов сетки: {results['total_nodes']}")
        print(f"Ёмкость (численно): {results['c_numeric'] * 1e12:.2f} пФ")
        print(f"Ёмкость (идеальная): {results['c_ideal'] * 1e12:.2f} пФ")
        print(f"Эффект краев увеличил ёмкость на: {results['diff_percent']:.1f}%")

