import math
from .config import SimulationConfig

class InputHandler:

    @staticmethod
    def get_float(prompt, default=None, min_value=None, max_value=None):
        while True:
            try:
                s = input(f"{prompt} [default={default}]: ")
                val = float(s) if s.strip() else default
                if min_value is not None and val < min_value:
                    print(f"Значение должно быть >= {min_value}. Попробуйте снова.")
                    continue
                if max_value is not None and val > max_value:
                    print(f"Значение должно быть <= {max_value}. Попробуйте снова.")
                    continue
                return val
            except:
                continue

    @staticmethod
    def get_int(prompt, default=None, min_value=None, max_value=None):
        while True:
            try:
                s = input(f"{prompt} [default={default}]: ")
                val = int(s) if s.strip() else default
                if min_value is not None and val < min_value:
                    print(f"Значение должно быть >= {min_value}. Попробуйте снова.")
                    continue
                if max_value is not None and val > max_value:
                    print(f"Значение должно быть <= {max_value}. Попробуйте снова.")
                    continue
                return val
            except:
                continue

    @staticmethod
    def get_parameters():
        params = {}
        params["N"] = InputHandler.get_int("Число частиц N", default=1000, min_value=100, max_value=5000)
        T_C = InputHandler.get_float(
            "Начальная температура T0 (°C)", default=27.0, min_value=-100.0, max_value=1000.0
        )
        T_K = T_C + 273.15
        cfg = SimulationConfig
        T_sim = T_K * cfg.kB / cfg.m
        params["T0"] = T_sim


        params["piston_target"] = InputHandler.get_float("Конечное положение поршня x_final", default=0.7, min_value=0.4, max_value=1.0)

        print("Выберите режим процесса:")
        print("1 — Квазистатический (медленное движение поршня)")
        print("2 — Резкий переход (неквазистатический)")
        mode = int(input("Ваш выбор [1/2]: ").strip())
        params["mode"] = mode
        return params