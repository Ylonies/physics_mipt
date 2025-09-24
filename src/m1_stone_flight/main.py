from .models import InputHandler, PhysicsModels, TrajectorySolver, ResultAnalyzer, ResultVisualizer

class StoneFlight:
    def __init__(self, g=9.81):
        self.g = g
        self.physics_models = PhysicsModels(g)
        self.solver = TrajectorySolver(g)
    
    def run(self):
        try:
            params = InputHandler.get_parameters()
            model, model_name = self.physics_models.get_model(params)
            solution = self.solver.solve(model, params)
            results = ResultAnalyzer.analyze(solution, params)
            self._print_results(results, model_name)
            ResultVisualizer.plot(results, params, model_name)

        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except Exception as e:
            print(f"Ошибка вычислений: {e}")
    
    def _print_results(self, results, model_name):
        print(f"\n=== РЕЗУЛЬТАТЫ ===")
        print(f"Модель: {model_name}")
        print(f"Время полета: {results['flight_time']:.2f} с")
        print(f"Дальность полета: {results['flight_distance']:.2f} м")
        print(f"Максимальная высота: {results['max_height']:.2f} м")

if __name__ == "__main__":
    simulator = StoneFlight()
    simulator.run()