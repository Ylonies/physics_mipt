import math

class InputHandler:
    @staticmethod
    def get_parameters():
        print("=== М4. Шар на столе ===")
        params = {}

        print("Выберите сценарий:")
        print("1  - Наклон: качение без проскальзывания")
        print("1b - Наклон: с возможным проскальзыванием")
        print("2  - Горизонталь: произвольное качение с μ")
        choice = input("Введите 1 / 1b / 2: ").strip()
        if choice not in ("1", "1b", "2"):
            raise ValueError("Выберите корректный сценарий: 1, 1b или 2")
        params['choice'] = choice

        m = float(input("Масса шара m (кг) (0 .. 1000]: "))
        if not (0.01 <= m <= 1000):
            raise ValueError("Масса должна быть в диапазоне 0.01–1000 кг.")
        params['m'] = m

        R = float(input("Радиус шара R (м) (0 .. 10]: "))
        if not (0 < R <= 10):
            raise ValueError("Радиус должен быть в диапазоне 0.01–10 м.")
        params['R'] = R

        mu = float(input("Коэффициент сухого трения μ [0 .. 10]: "))
        if not (0 <= mu <= 10):
            raise ValueError("μ должно быть в диапазоне 0–10.")
        params['mu'] = mu

        if choice in ("1", "1b"):
            alpha_deg = float(input("Угол наклона α (градусы) [0 .. 90]: "))
            if not (0 <= alpha_deg <= 90):
                raise ValueError("α должно быть в диапазоне 0–90 градусов.")
            params['alpha'] = math.radians(alpha_deg)
            params['s0'] = 0.0

            v0 = float(input("Начальная скорость вдоль плоскости v0 (м/с) [0 .. 300]: "))
            if not (0 <= v0 <= 300):
                raise ValueError("v0 должно быть в диапазоне 0–300 м/с.")
            params['v0'] = v0
            
        elif choice == "1b":
            omega0 = float(input("Начальная угловая скорость ω0 (рад/с) [0 .. 300]: "))
            if not (0 <= omega0 <= 300):
                raise ValueError("ω0 должно быть в диапазоне 0–300 рад/с.")
            params['omega0'] = omega0
        else:  
            params['x0'] = 0.0

            v0 = float(input("Начальная поступательная скорость v0 (м/с) [0 .. 300]: "))
            if not (0 <= v0 <= 300):
                raise ValueError("v0 должно быть в диапазоне 0–100 м/с.")
            params['v0'] = v0

            omega0 = float(input("Начальная угловая скорость ω0 (рад/с) [0 .. 300]: "))
            if not (0 <= omega0 <= 300):
                raise ValueError("ω0 должно быть в диапазоне 0–300 рад/с.")
            params['omega0'] = omega0

            F = float(input("Постоянная горизонтальная сила F (Н) [-10000 .. 10000]: "))
            if not (-10000 <= F <= 100000):
                raise ValueError("F должно быть в диапазоне -1000 .. 1000 Н.")
            params['F'] = F

        t_end = float(input("Время моделирования (с) [0 .. 100]: "))
        if not (0 < t_end <= 100):
            raise ValueError("Время моделирования должно быть в диапазоне 0–100 с.")
        
        params['t_end'] = t_end

        
        return params
