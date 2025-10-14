from .models import InputHandler, PhysicsModels, ResultVisualizer

class CollisionSimulator:
    def __init__(self):
        self.physics_models = PhysicsModels()
    
    def run(self):
        try:
            print("=== Моделирование столкновений шаров ===")
            params = InputHandler.get_parameters()
            results, model_name = self.physics_models.get_model(params)
            self._print_results(results, params, model_name)
            ResultVisualizer.plot(results, params, model_name)

        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except Exception as e:
            print(f"Ошибка вычислений: {e}")
    
    def _print_results(self, results, params, model_name):
        print(f"\n=== РЕЗУЛЬТАТЫ ===")
        print(f"Модель: {model_name}")
        
        if params['object_choice'] == '1':
            if params['model_choice'] == '1': 
                print(f"Скорость после столкновения: {results['v_after']:.2f} м/с")
                print(f"Угол отражения: {np.degrees(results['alpha_after']):.1f}°")
            else:
                print(f"Время контакта: {results['contact_time']:.4f} с")
                print(f"Максимальная деформация: {results['max_deformation']:.4f} м")
                print(f"Максимальная сила: {results['max_force']:.2f} Н")
        else:
            if params['model_choice'] == '1':
                print(f"Скорость первого шара после: {results['v1_after']:.2f} м/с")
                print(f"Скорость второго шара после: {results['v2_after']:.2f} м/с")
            else:
                print(f"Время контакта: {results['contact_time']:.4f} с")
                print(f"Максимальная деформация: {results['max_deformation']:.4f} м")
                print(f"Максимальная сила: {results['max_force']:.2f} Н")
                print(f"Скорость первого шара после: {results['v1_after']:.2f} м/с")
                print(f"Скорость второго шара после: {results['v2_after']:.2f} м/с")

if __name__ == "__main__":
    import numpy as np
    simulator = CollisionSimulator()
    simulator.run()