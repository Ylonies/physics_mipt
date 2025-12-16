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
        print("3 — Резкий скачок внешнего давления (динамический поршень)")
        while True:
            try:
                mode = int(input("Ваш выбор [0/1/2/3]: ").strip())
                if mode not in (0, 1, 2, 3):
                    print("Выберите 0, 1, 2 или 3.")
                    continue
                break
            except:
                continue
        params["mode"] = mode

        if mode in (1, 2):
            params["piston_target"] = InputHandler.get_float(
                "Минимальное положение поршня x_final (x_start = 1)", default=0.7, min_value=0.4, max_value=1.0
            )
        elif mode == 3:
            V0 = cfg.Lx * cfg.Ly
            P0 = params["N"] * cfg.kB * params["T0"] / V0
            factor = InputHandler.get_float(
                "Во сколько раз изменить внешнее давление (P_ext_final = factor * P0)",
                default=2.0, min_value=0.1, max_value=20.0
            )
            params["P_ext_initial"] = P0
            params["P_ext_final"] = factor * P0
            params["t_pressure_step"] = InputHandler.get_float(
                "Момент скачка давления t_step (с)", default=0.05, min_value=0.0, max_value=10.0
            )
            params["piston_target"] = None
        else:
            params["piston_target"] = None 

        return params