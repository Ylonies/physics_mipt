from .models import InputHandler, PhysicsModels, RollingSolver, ResultAnalyzer, ResultVisualizer


class RollingSphereSim:
    def __init__(self, g: float = 9.81):
        self.g = g
        self.physics_models = PhysicsModels(g)
        self.solver = RollingSolver(g)

    def run(self):
        try:
            params = InputHandler.get_parameters()
            model, model_name, initial_state, t_span, t_eval = self.physics_models.get_model(params)
            solution = self.solver.solve(model, initial_state, t_span, t_eval)
            results = ResultAnalyzer.analyze(solution, params, model_name)

            print("\n=== РЕЗУЛЬТАТЫ ===")
            print(f"Модель: {model_name}")
            if 'energy_total' in results:
                e0 = results['energy_total'][0]
                e_end = results['energy_total'][-1]
                print(f"Энергия начальная: {e0:.6f} Дж, конечная: {e_end:.6f} Дж")
                print(f"ΔE: {e_end - e0:.6e} Дж")
            if 'slip_fraction' in results:
                print(f"Доля времени в проскальзывании: {results['slip_fraction']*100:.2f}%")

            # plots
            ResultVisualizer.plot(results, params, model_name)

        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except Exception as e:
            print(f"Ошибка вычислений: {e}")


if __name__ == "__main__":
    RollingSphereSim().run()

