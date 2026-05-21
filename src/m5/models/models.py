"""М5: автогенератор на туннельном диоде (нелинейный RLC-контур)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# Коэффициенты ВАХ из условия (СИ)
A_DIODE = 4e-3
B_DIODE = -16e-3
C_DIODE = 17e-3

# Область отрицательной дифференциальной проводимости (из g_D=0)
U_NEG_LO = 0.732
U_NEG_HI = 1.934


class PhysicsModels:
    @staticmethod
    def diode_current(u_v: float, params: dict | None = None) -> float:
        if u_v <= 0.0:
            return 0.0
        a = A_DIODE if params is None else params["a_a_per_v3"]
        b = B_DIODE if params is None else params["b_a_per_v2"]
        c = C_DIODE if params is None else params["c_a_per_v"]
        return a * u_v**3 + b * u_v**2 + c * u_v

    @staticmethod
    def diode_conductance(u_v: float, params: dict | None = None) -> float:
        if u_v <= 0.0:
            return 0.0
        a = A_DIODE if params is None else params["a_a_per_v3"]
        b = B_DIODE if params is None else params["b_a_per_v2"]
        c = C_DIODE if params is None else params["c_a_per_v"]
        return 3.0 * a * u_v**2 + 2.0 * b * u_v + c

    @staticmethod
    def find_dc_operating_point(params: dict) -> tuple[float, float]:
        """Рабочая точка: ε = R·I_D(U) + U."""
        eps = params["epsilon_v"]
        r = params["r_ohm"]

        def load(u: float) -> float:
            return eps - r * PhysicsModels.diode_current(u, params) - u

        roots: list[float] = []
        grid = np.linspace(1e-4, 3.5, 12000)
        vals = [load(u) for u in grid]
        for k in range(len(grid) - 1):
            if vals[k] == 0.0:
                roots.append(grid[k])
            elif vals[k] * vals[k + 1] < 0.0:
                lo, hi = grid[k], grid[k + 1]
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if load(lo) * load(mid) <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                roots.append(0.5 * (lo + hi))

        if not roots:
            raise ValueError(
                "Не найдена рабочая точка постоянного тока (пересечение ε=R·I_D(U)+U с ВАХ). "
                f"При ε={eps:.2f} В и R={r:.1f} Ом рабочее напряжение может быть >3.5 В "
                "или нагрузочная прямая не пересекает кривую диода. "
                "Попробуйте ε≈1.0–1.5 В и R≈1–10 Ом."
            )

        # Предпочитаем точку в области отрицательной проводимости
        in_neg = [u for u in roots if U_NEG_LO < u < U_NEG_HI]
        u0 = in_neg[0] if in_neg else roots[-1]
        i0 = PhysicsModels.diode_current(u0, params)
        return float(u0), float(i0)

    @staticmethod
    def linear_oscillation_criterion(params: dict, u0: float | None = None) -> dict:
        if u0 is None:
            u0, _ = PhysicsModels.find_dc_operating_point(params)
        g0 = PhysicsModels.diode_conductance(u0, params)
        r, l, c = params["r_ohm"], params["l_h"], params["c_f"]
        damping = r * c + l * g0
        threshold = -r * c / l
        in_neg_branch = U_NEG_LO < u0 < U_NEG_HI
        can_oscillate = bool(in_neg_branch and g0 < 0.0 and damping < 0.0)
        f0 = 1.0 / (2.0 * np.pi * np.sqrt(l * c)) if l * c > 0 else 0.0
        return {
            "u0_v": u0,
            "g0_s": g0,
            "damping": damping,
            "threshold_g": threshold,
            "in_neg_branch": in_neg_branch,
            "can_oscillate_linear": can_oscillate,
            "f_lc_hz": f0,
        }

    @staticmethod
    def _rhs(_t: float, y: np.ndarray, params: dict) -> np.ndarray:
        u, i = float(y[0]), float(y[1])
        c, l = params["c_f"], params["l_h"]
        id_ = PhysicsModels.diode_current(u, params)
        du = (i - id_) / c
        di = (params["epsilon_v"] - params["r_ohm"] * i - u) / l
        return np.array([du, di], dtype=float)

    def _tunnel_diode_generator(self, params: dict) -> dict:
        t_eval = np.linspace(0.0, params["t_end_s"], params["n_points"])
        y0 = [params["u0_v"], params["i0_a"]]

        sol = solve_ivp(
            self._rhs,
            (0.0, params["t_end_s"]),
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            args=(params,),
        )
        if not sol.success:
            raise RuntimeError(f"Интегрирование не сошлось: {sol.message}")

        u = sol.y[0]
        i = sol.y[1]
        id_arr = np.array([self.diode_current(float(uv), params) for uv in u])

        return {
            "time": t_eval,
            "u_diode_v": u,
            "i_inductor_a": i,
            "i_diode_a": id_arr,
        }

    @staticmethod
    def survey_parameter_ranges(epsilon_v: float = 1.2, thd_pure: float = 0.12) -> dict:
        """Грубый перебор R, L, C для ответов на вопросы задания."""
        osc_cases: list[tuple[float, float, float, float]] = []
        pure_cases: list[tuple[float, float, float, float, float]] = []

        base = {
            "epsilon_v": epsilon_v,
            "u0_v": 1.15,
            "i0_a": 1e-4,
            "t_end_s": 0.025,
            "n_points": 8000,
            "a_a_per_v3": A_DIODE,
            "b_a_per_v2": B_DIODE,
            "c_a_per_v": C_DIODE,
        }

        pm = PhysicsModels()
        for r in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
            for l in (5e-4, 1e-3, 5e-3, 1e-2):
                for c in (5e-7, 1e-6, 2e-6, 5e-6):
                    p = {**base, "r_ohm": r, "l_h": l, "c_f": c}
                    try:
                        lin = PhysicsModels.linear_oscillation_criterion(p)
                    except ValueError:
                        continue
                    if not lin["can_oscillate_linear"]:
                        continue

                    sol = pm._tunnel_diode_generator(p)
                    tail = sol["u_diode_v"][int(0.65 * sol["u_diode_v"].size) :]
                    amp = 0.5 * (float(np.max(tail)) - float(np.min(tail)))
                    centered = tail - np.mean(tail)
                    zc = int(np.sum(np.diff(np.signbit(centered)) != 0))
                    if amp < 0.02 or zc < 6:
                        continue

                    osc_cases.append((r, l, c, amp))
                    dt = float(sol["time"][1] - sol["time"][0])
                    spec = np.fft.rfft(centered)
                    freqs = np.fft.rfftfreq(centered.size, d=dt)
                    mag = np.abs(spec)
                    mag[0] = 0.0
                    i1 = int(np.argmax(mag))
                    a1 = mag[i1]
                    if a1 < 1e-12:
                        continue
                    harm = 0.0
                    f1 = freqs[i1]
                    for n in range(2, 9):
                        j = int(np.argmin(np.abs(freqs - n * f1)))
                        harm += mag[j] ** 2
                    thd = float(np.sqrt(harm) / a1)
                    if thd <= thd_pure:
                        pure_cases.append((r, l, c, thd, amp))

        return {"oscillating": osc_cases, "sinusoidal": pure_cases}

    def get_model(self, params):
        _ = params
        return self._tunnel_diode_generator, "RLC-генератор с туннельным диодом"
