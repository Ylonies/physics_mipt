import math

class InputHandler:

    @staticmethod
    def get_float(prompt, default=None, min_value=None, max_value=None):
        while True:
            try:
                s = input(f"{prompt} [default={default}]: ").strip()
                val = float(s) if s else default
                if val is None:
                    print("Это поле обязательно для заполнения.")
                    continue
                if min_value is not None and val < min_value:
                    print(f"Значение должно быть >= {min_value}. Попробуйте снова.")
                    continue
                if max_value is not None and val > max_value:
                    print(f"Значение должно быть <= {max_value}. Попробуйте снова.")
                    continue
                return val
            except ValueError:
                print("Ошибка: введите числовое значение.")
                continue

    @staticmethod
    def get_int(prompt, default=None, min_value=None, max_value=None):
        while True:
            try:
                s = input(f"{prompt} [default={default}]: ").strip()
                val = int(s) if s else default
                if val is None:
                    print("Это поле обязательно для заполнения.")
                    continue
                if min_value is not None and val < min_value:
                    print(f"Значение должно быть >= {min_value}. Попробуйте снова.")
                    continue
                if max_value is not None and val > max_value:
                    print(f"Значение должно быть <= {max_value}. Попробуйте снова.")
                    continue
                return val
            except ValueError:
                print("Ошибка: введите целое число.")
                continue

    @staticmethod
    def get_parameters():
        params = {}

        params["R"] = InputHandler.get_float(
            "Радиус электродов R (м)", default=0.1, min_value=0.001, max_value=10.0
        )

        params["d"] = InputHandler.get_float(
            "Расстояние между пластинами d (м)", default=0.05, min_value=0.0001, max_value=5.0
        )

        params["V"] = InputHandler.get_float(
            "Разность потенциалов V (В)", default=100.0, min_value=-1e6, max_value=1e6
        )

        params["n_rings"] = InputHandler.get_int(
            "Количество концентрических колец (точность расчёта)", default=30, min_value=5, max_value=60
        )

        return params