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

            print("\n=== РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ ===")
            print(f"Модель: {metrics['model_name']}")
            print(f"Рабочая точка (DC): U₀={metrics['dc_u0_v']:.3f} В, I₀={metrics['dc_i0_a']*1e3:.3f} мА")
            print(f"Автогенерация (симуляция): {'да' if metrics['self_oscillation'] else 'нет'}")
            print(f"Амплитуда U (установившаяся): {metrics['u_amplitude_steady']:.4f} В")
            print(f"Частота основной гармоники: {metrics['fundamental_freq_hz']:.1f} Гц "
                  f"(f₀≈{metrics['linear']['f_lc_hz']:.1f} Гц)")
            print(f"THD={metrics['thd']:.4f}, синусоидальность={metrics['sine_purity']:.4f}")

            print("\nПеребор типичных R, L, C (ε=1.2 В)...")
            survey = PhysicsModels.survey_parameter_ranges(params["epsilon_v"])
            print(ResultAnalyzer.format_assignment_report(metrics, survey))

            ResultVisualizer.plot(metrics, params, model_name)
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    TunnelDiodeGeneratorSim().run()
