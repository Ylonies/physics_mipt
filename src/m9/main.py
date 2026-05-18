from .models import InputHandler, PhysicsModels, RollingSolver, ResultAnalyzer, ResultVisualizer


class RayTracingSim:
    def __init__(self):
        self.physics_models = PhysicsModels()
        self.solver = RollingSolver()

    def run(self) -> None:
        try:
            params = InputHandler.get_parameters()
            model, model_name = self.physics_models.get_model(params)
            solution = self.solver.solve(model, params)
            results = ResultAnalyzer.analyze(solution, params, model_name)
            o = results["optics"]

            print("\n=== РЕЗУЛЬТАТЫ ===")
            print(f"Режим: {results['model_name']}")
            print(f"Объектив: s={params['object_distance_m']:.3f} м, s'={o['sp1']:.3f} м, M₁={o['m1']:.3f}")
            print(f"Промежуточное изображение: x={o['intermediate_x']:.3f} м")
            print(f"Окуляр: s₂={o['s2']:.3f} м, s'={o['sp2']:.3f} м, M₂={o['m2']:.3f}")
            print(f"M (теория, M₁·M₂) = {results['m_theory']:.3f}")
            print(f"M (трассировка)     = {results['m_ray']:.3f}")
            if params["lens_mode"] == "1":
                print(f"Ошибка M (тонкая vs теория): {results['rel_err_magnification'] * 100:.2f}%")
            if not solution.get("geometry_ok", True):
                print("⚠ Окуляр стоит левее промежуточного изображения (s₂<0) — необычная геометрия.")

            ResultVisualizer.plot(results, params, model_name)
        except Exception as e:
            print(f"Ошибка вычислений: {e}")


if __name__ == "__main__":
    RayTracingSim().run()
