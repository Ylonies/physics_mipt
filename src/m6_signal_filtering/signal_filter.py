from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from src.m6_signal_filtering import config
from src.m6_signal_filtering.models import result_visualizer, simulation, spectrum, transfer


class SignalFilterLab:
    def run(self, *, save_plots: Path | None = None) -> None:
        # Проверка параметров перед расчётами
        self._validate_parameters()

        R, C = config.RC_R, config.RC_C
        wc, tau = transfer.rc_omega_c(R, C), R * C
        dt_rc = spectrum.suggest_dt(tau, 100 * wc)

        rc_sweep = simulation.sweep_rc_harmonic(
            R, C, config.RC_U0, config.RC_OMEGA_RATIOS, config.RC_T_MAX_HARM, dt_rc
        )
        rc_square = simulation.run_rc_square(
            R, C, config.RC_U0, config.RC_SQUARE_PERIOD, config.RC_SQUARE_T_MAX, dt_rc, config.RC_FOURIER_K_MAX
        )

        self._print_rc_header(R, C, wc, tau)
        self._print_table("RC", rc_sweep, wc, config.RC_U0)
        self._print_square("RC", rc_square)

        rlc_data = None
        if config.RUN_RLC:
            L, Cr = config.RLC_L, config.RLC_C
            w0 = transfer.rlc_omega_0(L, Cr)
            q = transfer.rlc_Q(config.RLC_R, L, Cr)
            dt_rlc = spectrum.suggest_dt(2 * q / w0, 1.5 * w0)
            rlc_sweep = simulation.sweep_rlc_harmonic(
                config.RLC_R, L, Cr, config.RLC_U0, config.RLC_OMEGA_RATIOS, config.RLC_T_MAX_HARM, dt_rlc
            )
            period_rlc = 2 * np.pi / w0
            rlc_square = simulation.run_rlc_square(
                config.RLC_R, L, Cr, config.RLC_U0, period_rlc, config.RLC_SQUARE_T_MAX, dt_rlc, config.RLC_FOURIER_K_MAX
            )
            self._print_rlc_header(config.RLC_R, L, Cr, w0, q)
            self._print_table("RLC", rlc_sweep, w0, config.RLC_U0)
            self._print_square("RLC", rlc_square)
            rlc_data = (config.RLC_R, L, Cr, rlc_sweep, rlc_square)

        if save_plots is not None:
            rc_fig = result_visualizer.build_rc_figures(
                R, C, rc_sweep, rc_square, config.RC_U0, config.RC_SQUARE_PERIOD
            )
            result_visualizer.save_figures(save_plots, "m6_rc", rc_fig, None)
            print(f"RC: {save_plots.resolve()}/m6_rc_main.png")
            if rlc_data is not None:
                Rl, L, Cr, sw, sq = rlc_data
                rlc_fig = result_visualizer.build_rlc_figures(Rl, L, Cr, sw, sq, config.RLC_U0)
                result_visualizer.save_figures(save_plots, "m6_rlc", rlc_fig, None)
                print(f"RLC: {save_plots.resolve()}/m6_rlc_main.png")
            return

        choice = self._ask_plot_choice()
        if choice == "n":
            return
        self._render_choice(choice, R, C, rc_sweep, rc_square, rlc_data)

    def _validate_parameters(self) -> None:
        """Проверяет корректность параметров из config.py. При ошибке выводит сообщение и завершает программу."""
        errors = []

        # RC параметры
        if config.RC_R <= 0:
            errors.append(f"RC_R = {config.RC_R} → сопротивление должно быть > 0")
        if config.RC_C <= 0:
            errors.append(f"RC_C = {config.RC_C} → ёмкость должна быть > 0")
        if config.RC_U0 <= 0:
            errors.append(f"RC_U0 = {config.RC_U0} → амплитуда должна быть > 0")
        if config.RC_T_MAX_HARM <= 0:
            errors.append(f"RC_T_MAX_HARM = {config.RC_T_MAX_HARM} → время моделирования должно быть > 0")
        if config.RC_SQUARE_PERIOD <= 0:
            errors.append(f"RC_SQUARE_PERIOD = {config.RC_SQUARE_PERIOD} → период меандра должен быть > 0")
        if config.RC_SQUARE_T_MAX <= 0:
            errors.append(f"RC_SQUARE_T_MAX = {config.RC_SQUARE_T_MAX} → время моделирования меандра должно быть > 0")
        if config.RC_FOURIER_K_MAX < 1:
            errors.append(f"RC_FOURIER_K_MAX = {config.RC_FOURIER_K_MAX} → должно быть >= 1")
        if len(config.RC_OMEGA_RATIOS) == 0:
            errors.append("RC_OMEGA_RATIOS не должен быть пустым")
        if np.any(config.RC_OMEGA_RATIOS <= 0):
            errors.append("RC_OMEGA_RATIOS: все значения должны быть положительными")

        # RLC параметры (проверяются, только если включены)
        if config.RUN_RLC:
            if config.RLC_R <= 0:
                errors.append(f"RLC_R = {config.RLC_R} → сопротивление должно быть > 0")
            if config.RLC_L <= 0:
                errors.append(f"RLC_L = {config.RLC_L} → индуктивность должна быть > 0")
            if config.RLC_C <= 0:
                errors.append(f"RLC_C = {config.RLC_C} → ёмкость должна быть > 0")
            if config.RLC_U0 <= 0:
                errors.append(f"RLC_U0 = {config.RLC_U0} → амплитуда должна быть > 0")
            if config.RLC_T_MAX_HARM <= 0:
                errors.append(f"RLC_T_MAX_HARM = {config.RLC_T_MAX_HARM} → время моделирования должно быть > 0")
            if config.RLC_SQUARE_T_MAX <= 0:
                errors.append(f"RLC_SQUARE_T_MAX = {config.RLC_SQUARE_T_MAX} → время моделирования меандра должно быть > 0")
            if config.RLC_FOURIER_K_MAX < 1:
                errors.append(f"RLC_FOURIER_K_MAX = {config.RLC_FOURIER_K_MAX} → должно быть >= 1")
            if len(config.RLC_OMEGA_RATIOS) == 0:
                errors.append("RLC_OMEGA_RATIOS не должен быть пустым")
            if np.any(config.RLC_OMEGA_RATIOS <= 0):
                errors.append("RLC_OMEGA_RATIOS: все значения должны быть положительными")

        if errors:
            print("\nОшибка в config.py: некорректные параметры", file=sys.stderr)
            for e in errors:
                print(f"  • {e}", file=sys.stderr)
            print("\nИсправьте параметры и запустите программу снова.", file=sys.stderr)
            sys.exit(1)

    def _ask_plot_choice(self) -> str:
        print("\n--- Графики ---")
        print("  a — RC (фильтр нижних частот)")
        print("  b — RLC (полосовой, выход на R)")
        print("  n — без графиков")
        while True:
            s = input("Выбор [a/b/n]: ").strip().lower().replace(" ", "")
            if s in ("a", "b", "n", "а", "б"):
                return {"а": "a", "б": "b"}.get(s, s)
            print("Введите a, b или n.")

    def _render_choice(self, choice, R, C, rc_sweep, rc_square, rlc_data):
        if choice == "a":
            fig = result_visualizer.build_rc_figures(
                R, C, rc_sweep, rc_square, config.RC_U0, config.RC_SQUARE_PERIOD
            )
            result_visualizer.show_figures(fig)
        elif choice == "b" and rlc_data is not None:
            Rl, L, Cr, sw, sq = rlc_data
            fig = result_visualizer.build_rlc_figures(Rl, L, Cr, sw, sq, config.RLC_U0)
            result_visualizer.show_figures(fig)

    def _print_rc_header(self, R, C, wc, tau):
        print("\n=== RC: фильтр нижних частот ===")
        print(f"R = {R:.0f} Ом, C = {C:.2e} Ф, τ = {tau*1e3:.3f} мс")
        print(f"ω_c = {wc:.1f} рад/с, f_c = {wc/(2*np.pi):.1f} Гц")

    def _print_rlc_header(self, R, L, C, w0, q):
        print("\n=== RLC: полосовой (U_out на R) ===")
        print(f"R = {R:.0f} Ом, L = {L:.2e} Гн, C = {C:.2e} Ф")
        print(f"ω_0 = {w0:.1f} рад/с, Q = {q:.2f}")

    def _print_table(self, tag, sweep, w_ref, u0):
        print(f"\n--- {tag}: гармоника vs H(ω) ---")
        print(f"{'ω/ω_ref':>8} {'|H|':>8} {'φ°,т':>8} {'A_num':>8} {'A_т':>8} {'δA%':>7} {'Δφ°':>7}")
        for h in sweep:
            r = h["omega"] / w_ref
            print(
                f"{r:8.2f} {h['amp_theor']/u0:8.4f} {np.degrees(h['phase_theor']):8.2f} "
                f"{h['amp_num']:8.4f} {h['amp_theor']:8.4f} "
                f"{abs(h['amp_num']-h['amp_theor'])/(h['amp_theor']+1e-30)*100:7.3f} "
                f"{abs(np.degrees(h['phase_num']-h['phase_theor'])):7.3f}"
            )

    def _print_square(self, tag, sq):
        print(f"\n--- {tag}: меандр, гармоники выхода ---")
        print(f"{'k':>4} {'A_num':>10} {'A_an':>10} {'δ%':>8}")
        for k, an, aa in zip(sq["ks_fft"][:6], sq["amps_fft"][:6], sq["amps_ana"][:6]):
            print(f"{int(k):4d} {an:10.5f} {aa:10.5f} {abs(an-aa)/(aa+1e-12)*100:8.2f}")