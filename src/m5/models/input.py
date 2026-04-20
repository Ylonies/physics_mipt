import math

class InputHandler:
    @staticmethod
    def get_float(prompt, min_value=None, max_value=None):
        while True:
            try:
                value = float(input(prompt))
                if min_value is not None and value < min_value:
                    print(f"Значение должно быть >= {min_value}. Попробуйте снова.")
                    continue
                if max_value is not None and value > max_value:
                    print(f"Значение должно быть <= {max_value}. Попробуйте снова.")
                    continue
                return value
            except ValueError:
                print("Неверный ввод, пожалуйста, введите число.")

    @staticmethod
    def get_yes_no(prompt):
        while True:
            value = input(prompt).strip().lower()
            if value in ('y', 'n'):
                return value
            print("Пожалуйста, введите 'y' или 'n'.")

    @staticmethod
    def get_parameters():
        print("=== М5. Физический маятник: эллипс ===")
        params = {}

        params['a'] = InputHandler.get_float("Малая ось a (м): ", min_value=0.0001)
        params['b'] = InputHandler.get_float("Большая ось b (м): ", min_value=0.0001)
        params['m'] = InputHandler.get_float("Масса эллипса m (кг): ", min_value=0.0001)

        # Ограничение угла (малые колебания)
        params['theta0_deg'] = InputHandler.get_float(
            "Начальное отклонение θ0 (градусы): ",
            min_value=-60,
            max_value=60
        )
        params['theta0'] = math.radians(params['theta0_deg'])

        params['omega0'] = InputHandler.get_float(
            "Начальная угловая скорость ω0 (рад/с): ",
            min_value=-10,
            max_value=10
        )

        params['t_end'] = InputHandler.get_float("Время моделирования (с): ", min_value=0.1)

        if InputHandler.get_yes_no("Добавить трение? (y/n): ") == 'y':
            params['gamma'] = InputHandler.get_float("Коэффициент трения γ: ", min_value=0)
        else:
            params['gamma'] = 0.0

        a, b, m = params['a'], params['b'], params['m']
        I_cm = 0.25 * m * (a**2 + b**2)
        I = I_cm + m * b**2
        params['I'] = I
        params['l'] = b

        print(f"Автоматически рассчитано: I = {I:.4f} кг·м², l = {b:.4f} м")
        return params
