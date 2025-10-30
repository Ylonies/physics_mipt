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
        params['choice'] = choice

        params['m'] = float(input("Масса шара m (кг): "))
        if params['m'] <= 0:
            raise ValueError("Масса должна быть положительной.")
        params['R'] = float(input("Радиус шара R (м): "))
        if params['R'] <= 0:
            raise ValueError("Радиус должен быть положительным.")

        params['mu'] = float(input("Коэффициент сухого трения μ: "))
        if params['mu'] < 0:
            raise ValueError("μ не может быть отрицательным.")

        if choice in ("1", "1b"):
            alpha_deg = float(input("Угол наклона α (градусы): "))
            if not (0 <= alpha_deg <= 90):
                raise ValueError("α в пределах [0, 90].")
            params['alpha'] = math.radians(alpha_deg)
            params['s0'] = 0.0
            params['v0'] = float(input("Начальная скорость вдоль плоскости v0 (м/с): "))
            params['omega0'] = float(input("Начальная угл. скорость ω0 (рад/с): "))
            params['t_end'] = float(input("Время моделирования (с): "))
        else:
            params['x0'] = 0.0
            params['y0'] = 0.0
            params['vx0'] = float(input("Начальная скорость vx0 (м/с): "))
            params['vy0'] = float(input("Начальная скорость vy0 (м/с): "))
            params['omega_x0'] = float(input("Начальная угл. скорость ωx0 (рад/с): "))
            params['omega_y0'] = float(input("Начальная угл. скорость ωy0 (рад/с): "))
            params['Fx'] = float(input("Постоянная внешняя сила Fx (Н): "))
            params['Fy'] = float(input("Постоянная внешняя сила Fy (Н): "))
            params['t_end'] = float(input("Время моделирования (с): "))

        return params

