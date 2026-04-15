from .models import InputHandler, PhysicsModels, RollingSolver, ResultAnalyzer, ResultVisualizer


class RadioCommSim:
    def __init__(self):
        self.physics_models = PhysicsModels()
        self.solver = RollingSolver()

    def run(self) -> None:
        try:
            params = InputHandler.get_parameters()
            config, model_name = self.physics_models.get_model(params)
            solution = self.solver.solve(config)
            metrics = ResultAnalyzer.analyze(solution, model_name)

            print("\n=== РЕЗУЛЬТАТЫ ===")
            print(f"Модель: {metrics['model_name']}")
            print(f"MSE(source, recovered): {metrics['mse']:.6f}")
            print(f"Correlation(source, recovered): {metrics['corrcoef']:.4f}")

            ResultVisualizer.plot(solution, config, model_name)
        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except Exception as e:
            print(f"Ошибка вычислений: {e}")


if __name__ == "__main__":
    RadioCommSim().run()
