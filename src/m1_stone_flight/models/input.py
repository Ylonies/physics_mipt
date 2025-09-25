import math

class InputHandler:
    
    @staticmethod
    def get_parameters():
        print("=== Моделирование полета камня ===")
        
        params = {}
        
        params['v0'] = float(input("Начальная скорость (м/с): "))
        if not (0 < params['v0'] <= 340):
            raise ValueError("Начальная скорость должна быть в диапазоне (0, 340] м/с.")

        alpha_deg = float(input("Угол броска в градусах: "))
        if not (0 < alpha_deg <= 90):
            raise ValueError("Угол броска должен быть в пределах (0, 90] градусов.")
        params['alpha'] = math.radians(alpha_deg)

        params['gamma'] = float(input("Коэффициент сопротивления: "))
        if params['gamma'] < 0:
            raise ValueError("Коэффициент сопротивления не может быть отрицательным.")
        if params['gamma'] >= 1:
            raise ValueError("Коэффициент сопротивления должен быть меньше 1.")
        params['m'] = float(input("Масса тела (кг): "))
        if params['m'] <= 0:
            raise ValueError("Масса должна быть положительной.")

        print("\nВыберите модель сопротивления:")
        print("1 - Вязкое трение (F ∼ v)")
        print("2 - Лобовое сопротивление (F ∼ v²)")
        params['model_choice'] = input("Введите 1 или 2: ").strip()
        
        return params