import math 

class InputHandler:
    @staticmethod
    def get_parameters():
        print("=== М5. Физический маятник: эллипс ===")
        params = {}
        params['a'] = float(input("Малая ось a (м): "))
        params['b'] = float(input("Большая ось b (м): "))
        params['m'] = float(input("Масса эллипса m (кг): "))
        if params['a'] <= 0 or params['b'] <= 0 or params['m'] <= 0:
            raise ValueError("Все значения должны быть положительными.")

        params['theta0_deg'] = float(input("Начальное отклонение θ0 (градусы): "))
        params['theta0'] = math.radians(params['theta0_deg'])
        params['omega0'] = float(input("Начальная угловая скорость ω0 (рад/с): "))
        params['t_end'] = float(input("Время моделирования (с): "))
        if params['t_end'] <= 0:
            raise ValueError("Время моделирования должно быть положительным.")
        gamma = input("Добавить трение? (y/n): ").strip().lower()
        if gamma == 'y':
            params['gamma'] = float(input("Коэффициент трения γ: "))
            if params['gamma'] < 0:
                raise ValueError("Коэффициент трения γ должен быть неотрицательным.")
        else:
            params['gamma'] = 0.0

        a = params['a']
        b = params['b']
        m = params['m']
        I_cm = 0.25 * m * (a**2 + b**2)
        I = I_cm + m * b**2  
        params['I'] = I
        params['l'] = b 
        print(f"Автоматически рассчитано: I = {params['I']:.4f} кг·м², l = {params['l']:.4f} м")
        return params