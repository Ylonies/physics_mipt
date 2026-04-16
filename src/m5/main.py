from .models import InputHandler, PhysicsModels, RollingSolver, ResultAnalyzer, ResultVisualizer


class TunnelDiodeGeneratorSim:
    def __init__(self):
        self.physics_models = PhysicsModels()
        self.solver = RollingSolver()

    def run(self) -> None:
        try:
            params = InputHandler.get_parameters()
            model, model_name = self.physics_models.get_model(params)
            solution = self.solver.solve(model, params)
            metrics = ResultAnalyzer.analyze(solution, params, model_name)

            print("\n=== РЕЗУЛЬТАТЫ ===")
            print(f"Модель: {metrics['model_name']}")
            print(f"Автогенерация: {'да' if metrics['self_oscillation'] else 'нет'}")
            print(f"Максимальная амплитуда напряжения на диоде: {metrics['u_max_abs']:.4f} В")
            print(f"Оценка основной частоты: {metrics['fundamental_freq_hz']:.2f} Гц")
            print(f"Коэффициент гармонических искажений THD: {metrics['thd']:.4f}")
            print(f"Коэффициент синусоидальности: {metrics['sine_purity']:.4f}")

            ResultVisualizer.plot(metrics, params, model_name)
        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except Exception as e:
            print(f"Ошибка вычислений: {e}")


if __name__ == "__main__":
    TunnelDiodeGeneratorSim().run()
