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

        # Сначала выбираем режим
        print("Выберите режим процесса:")
        print("0 - Без поршня (фиксированный объём)")
        print("1 — Квазистатический (медленное движение поршня)")
        print("2 — Резкий переход (быстрое движение поршня)")
        while True:
            try:
                mode = int(input("Ваш выбор [0/1/2]: ").strip())
                if mode not in (0, 1, 2):
                    print("Выберите 0, 1 или 2.")
                    continue
                break
            except:
                continue
        params["mode"] = mode

        if mode in (1, 2):
            params["piston_target"] = InputHandler.get_float(
                "Минимальное положение поршня x_final (x_start = 1)", default=0.7, min_value=0.4, max_value=1.0
            )
        else:
            params["piston_target"] = None 

        return params