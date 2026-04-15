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
        print("=== М7. Радиосвязь: передатчик-приемник ===")
        params = {}

        params["duration_s"] = InputHandler._read_float(
            "Время моделирования, с", default=0.02, low=0.001, high=1.0
        )
        params["sample_rate_hz"] = InputHandler._read_float(
            "Частота дискретизации, Гц", default=5_000_000.0, low=200_000.0, high=50_000_000.0
        )
        params["carrier_freq_hz"] = InputHandler._read_float(
            "Несущая частота, Гц", default=200_000.0, low=20_000.0, high=5_000_000.0
        )
        params["modulation_index"] = InputHandler._read_float(
            "Индекс амплитудной модуляции (AM)", default=0.7, low=0.05, high=1.0
        )
        params["seed"] = InputHandler._read_int(
            "Зерно генератора шума (seed)", default=42, low=0, high=1_000_000
        )

        print("\n--- Канальные ВЧ помехи ---")
        params["noise_relative_amplitude"] = InputHandler._read_float(
            "Отн. амплитуда помех", default=0.25, low=0.0, high=2.0
        )
        params["noise_components"] = InputHandler._read_int(
            "Количество синус-компонент помех", default=20, low=1, high=400
        )
        params["noise_min_freq_hz"] = InputHandler._read_float(
            "Нижняя граница частот помех, Гц", default=350_000.0, low=1_000.0, high=20_000_000.0
        )
        params["noise_max_freq_hz"] = InputHandler._read_float(
            "Верхняя граница частот помех, Гц", default=1_200_000.0, low=2_000.0, high=30_000_000.0
        )
        if params["noise_max_freq_hz"] <= params["noise_min_freq_hz"]:
            raise ValueError("Верхняя граница частот помех должна быть больше нижней.")

        print("\n--- Резонансный RLC-контур приемника ---")
        params["rlc_resistance_ohm"] = InputHandler._read_float(
            "Сопротивление R, Ом", default=8.0, low=0.01, high=10_000.0
        )
        params["rlc_inductance_h"] = InputHandler._read_float(
            "Индуктивность L, Гн", default=1e-3, low=1e-7, high=10.0
        )

        print("\n--- Детектор + RC НЧ-фильтр ---")
        params["envelope_scale"] = InputHandler._read_float(
            "Коэффициент усиления детектора", default=1.0, low=0.01, high=100.0
        )
        params["lowpass_tau_s"] = InputHandler._read_float(
            "Постоянная времени RC, с", default=6e-5, low=1e-7, high=1.0
        )

        return params
