class InputHandler:
    @staticmethod
    def _read_float(prompt: str, default: float, low: float, high: float) -> float:
        raw = input(f"{prompt} [{default}]: ").strip()
        value = default if raw == "" else float(raw)
        if not (low <= value <= high):
            raise ValueError(f"Значение должно быть в диапазоне [{low} .. {high}]")
        return value

    @staticmethod
    def _read_int(prompt: str, default: int, low: int, high: int) -> int:
        raw = input(f"{prompt} [{default}]: ").strip()
        value = default if raw == "" else int(raw)
        if not (low <= value <= high):
            raise ValueError(f"Значение должно быть в диапазоне [{low} .. {high}]")
        return value

    @staticmethod
    def get_parameters():
        print("=== М5. Автогенератор на туннельном диоде ===")
        params = {}

        params["epsilon_v"] = InputHandler._read_float(
            "ЭДС источника ε, В", default=1.2, low=0.0, high=10.0
        )
        params["r_ohm"] = InputHandler._read_float(
            "Сопротивление R, Ом", default=8.0, low=0.01, high=10_000.0
        )
        params["l_h"] = InputHandler._read_float(
            "Индуктивность L, Гн", default=1e-3, low=1e-7, high=10.0
        )
        params["c_f"] = InputHandler._read_float(
            "Емкость C, Ф", default=2e-6, low=1e-10, high=1.0
        )

        print("\n--- Начальные условия ---")
        params["u0_v"] = InputHandler._read_float(
            "Начальное напряжение на диоде U(0), В", default=0.2, low=-5.0, high=5.0
        )
        params["i0_a"] = InputHandler._read_float(
            "Начальный ток через индуктивность I(0), А", default=0.0, low=-5.0, high=5.0
        )

        print("\n--- Параметры расчета ---")
        params["t_end_s"] = InputHandler._read_float(
            "Время моделирования, с", default=0.05, low=1e-4, high=10.0
        )
        params["n_points"] = InputHandler._read_int(
            "Количество точек по времени", default=6000, low=500, high=200_000
        )

        # Коэффициенты аппроксимации ВАХ диода (из условия), в СИ.
        params["a_a_per_v3"] = 4e-3
        params["b_a_per_v2"] = -16e-3
        params["c_a_per_v"] = 17e-3

        return params
