import math

class InputHandler:
    
    @staticmethod
    def get_mass(prompt):
        value = float(input(prompt))
        if value <= 0:
            raise ValueError("Масса должна быть положительным числом.")
        return value

    @staticmethod
    def get_speed(prompt):
        value = float(input(prompt))
        if value < 0:
            raise ValueError("Скорость не может быть отрицательной.")
        return value

    @staticmethod
    def get_parameters():
        params = {}

        print("\nВыберите c чем сталкивается шар:")
        print("1 - с бесконечно тяжелой стеной")
        print("2 - с другим шаром")
        params['object_choice'] = input("Введите 1 или 2: ").strip()

        if params['object_choice'] == '1':
            params['m1'] = InputHandler.get_mass("Масса шара (кг): ")
            params['v0'] = InputHandler.get_speed("Скорость шара до столкновения (м/с): ")
            
            alpha_deg = float(input("Угол полёта относительно x (0-90°): "))
            if not (0 <= alpha_deg <= 90):
                raise ValueError("Угол полёта должен быть в пределах [0, 90] градусов.")
            params['alpha'] = math.radians(alpha_deg)   

        elif params['object_choice'] == '2':
            params['m1'] = InputHandler.get_mass("Масса первого шара (кг): ")
            params['v0'] = InputHandler.get_speed("Скорость первого шара до столкновения (м/с): ")
            
            params['m2'] = InputHandler.get_mass("Масса второго шара (кг): ")
            params['u0'] = InputHandler.get_speed("Скорость второго шара до столкновения (м/с): ")

        else:
            raise ValueError("Выбор должен быть 1 или 2.")
        
        print("\nВыберите модель столкновения:")
        print("1 - абсолютно упругий удар")
        print("2 - шар деформируется согласно закону Гука: F ~ -delta_x")

        params['model_choice'] = input("Введите 1 или 2: ").strip()
        if (params['model_choice'] == 2):
            params['k'] = float(input("Коэффициент упругости (Н/м): "))
            if params['k'] < 0:
                raise ValueError("Коэффициент упругости не может быть отрицательным.")
        
        return params
