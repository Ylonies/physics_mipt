import numpy as np

from .models import PhysicsModels, U_NEG_HI, U_NEG_LO


class ResultAnalyzer:
    @staticmethod
    def _estimate_fundamental_and_thd(signal: np.ndarray, dt: float) -> tuple[float, float]:
        centered = signal - np.mean(signal)
        spectrum = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(centered.size, d=dt)

        if freqs.size < 3:
            return 0.0, 1.0

        mag = np.abs(spectrum)
        mag[0] = 0.0
        idx1 = int(np.argmax(mag))
        f1 = float(freqs[idx1])
        a1 = float(mag[idx1])

        if a1 <= 1e-12:
            return 0.0, 1.0

        harmonic_power = 0.0
        for n in range(2, 9):
            fn = n * f1
            j = int(np.argmin(np.abs(freqs - fn)))
            harmonic_power += mag[j] ** 2

        thd = float(np.sqrt(harmonic_power) / a1)
        return f1, thd

    @staticmethod
    def analyze(solution, params, model_name):
        t = solution["time"]
        u = solution["u_diode_v"]
        i = solution["i_inductor_a"]

        n = u.size
        dt = float(t[1] - t[0]) if t.size > 1 else 1.0
        t_total = float(t[-1] - t[0]) if t.size > 1 else 0.0
        # При коротком моделировании берём более ранний «хвост», иначе мало периодов колебаний
        start_frac = 0.35 if t_total < 0.02 else 0.65
        start = int(start_frac * n)
        tail_u = u[start:]
        tail_i = i[start:]

        amp_steady = 0.5 * (float(np.max(tail_u)) - float(np.min(tail_u)))
        centered = tail_u - np.mean(tail_u)
        zero_crossings = int(np.sum(np.diff(np.signbit(centered)) != 0))

        f1, thd = ResultAnalyzer._estimate_fundamental_and_thd(tail_u, dt)
        sine_purity = float(1.0 / (1.0 + thd))

        # Автогенерация: устойчивые колебания с ненулевой амплитудой
        self_oscillation = bool(amp_steady > 0.02 and zero_crossings >= 6)

        lin = PhysicsModels.linear_oscillation_criterion(params)
        u0, i0 = lin["u0_v"], PhysicsModels.diode_current(lin["u0_v"], params)

        # Оценка предела амплитуды по ширине области отриц. сопротивления
        u_limit = max(0.0, min(lin["u0_v"] - U_NEG_LO, U_NEG_HI - lin["u0_v"]))

        answers = {
            "q1_theory": (
                f"g_D(U₀)<0 и U₀∈({U_NEG_LO:.2f}; {U_NEG_HI:.2f}) В; "
                f"дополнительно RC+L·g_D(U₀)<0 (эквив. g_D<-RC/L)."
            ),
            "q1_linear_ok": lin["can_oscillate_linear"],
            "q1_sim_ok": self_oscillation,
            "q2_theory": (
                "Почти синусоида при RC+L·g_D≈0 (граница возбуждения) и малой нелинейности; "
                f"частота близка f₀≈1/(2π√(LC))≈{lin['f_lc_hz']:.0f} Гц."
            ),
            "q2_pure_sim": bool(self_oscillation and thd < 0.12),
            "q3_theory": (
                "Предел задаётся нелинейной ВАХ: амплитуда ограничена выходом из области "
                f"отриц. проводимости (оценка ≤{u_limit:.2f} В от рабочей точки)."
            ),
            "q3_amp_sim": amp_steady,
        }

        return {
            "model_name": model_name,
            "self_oscillation": self_oscillation,
            "u_max_abs": float(np.max(np.abs(u))),
            "u_amplitude_steady": amp_steady,
            "fundamental_freq_hz": f1,
            "thd": thd,
            "sine_purity": sine_purity,
            "time": t,
            "u_diode_v": u,
            "i_inductor_a": i,
            "i_diode_a": solution["i_diode_a"],
            "tail_u": tail_u,
            "tail_i": tail_i,
            "dc_u0_v": u0,
            "dc_i0_a": i0,
            "linear": lin,
            "answers": answers,
            "params": params,
        }

    @staticmethod
    def _describe_sine_quality(metrics: dict) -> str:
        if not metrics["self_oscillation"]:
            return (
                f"колебаний нет (A_U≈{metrics['u_amplitude_steady']:.4f} В) — "
                "THD здесь не характеризует форму автогенерации"
            )
        a = metrics["answers"]
        return (
            f"THD={metrics['thd']:.3f}, синусоидальность={metrics['sine_purity']:.3f} → "
            f"{'близко к синусоиде' if a['q2_pure_sim'] else 'заметные нелинейные искажения'}"
        )

    @staticmethod
    def format_assignment_report(metrics: dict, survey: dict | None = None) -> str:
        p = metrics["params"]
        lin = metrics["linear"]
        a = metrics["answers"]
        lines = [
            "",
            "=== ОТВЕТЫ НА ВОПРОСЫ ЗАДАНИЯ ===",
            "",
            "1) При каких параметрах происходит автогенерация?",
            f"   Теория: {a['q1_theory']}",
            f"   Ваша точка: U₀={lin['u0_v']:.3f} В, g_D={lin['g0_s']*1e3:.3f} мСм, RC+L·g_D={lin['damping']:.2e}",
            f"   Порог: g_D < {lin['threshold_g']*1e3:.3f} мСм → линейный критерий: "
            f"{'выполнен' if a['q1_linear_ok'] else 'НЕ выполнен'}.",
            f"   Моделирование (R={p['r_ohm']}, L={p['l_h']}, C={p['c_f']}): "
            f"{'автогенерация есть' if a['q1_sim_ok'] else 'автогенерации нет'}.",
            "",
            "2) При каких параметрах — практически чистая синусоида?",
            f"   Теория: {a['q2_theory']}",
            f"   Ваш режим: {ResultAnalyzer._describe_sine_quality(metrics)}",
            "",
            "3) Какова максимальная амплитуда колебаний?",
            f"   Теория: {a['q3_theory']}",
            f"   В моделировании (установившийся режим): A_U≈{a['q3_amp_sim']:.4f} В "
            f"(полуразмах), размах U_D≈{2*a['q3_amp_sim']:.4f} В.",
        ]

        if survey and survey["oscillating"]:
            r_vals = sorted({x[0] for x in survey["oscillating"]})
            lines += [
                "",
                "   Примеры R (Ом) с автогенерацией при ε=1.2 В (перебор L,C): "
                + ", ".join(f"{v:g}" for v in r_vals[:8]),
            ]
        if survey and survey["sinusoidal"]:
            best = min(survey["sinusoidal"], key=lambda x: x[3])
            lines += [
                f"   Пример почти синусоиды: R={best[0]} Ом, L={best[1]} Гн, C={best[2]} Ф "
                f"(THD={best[3]:.3f}, A_U≈{best[4]:.3f} В).",
            ]

        return "\n".join(lines)
