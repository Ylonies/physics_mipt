from __future__ import annotations

from pathlib import Path

import numpy as np

from src.m6_signal_filtering import config
from src.m6_signal_filtering.models import result_visualizer, simulation, spectrum, transfer


class SignalFilterLab:
    def run(self, *, save_plots: Path | None = None) -> None:
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
            self._save_choice("ab", save_plots, R, C, rc_sweep, rc_square, rlc_data)
            return

        choice = self._ask_plot_choice()
        if choice == "n":
            return
        self._render_choice(choice, R, C, rc_sweep, rc_square, rlc_data)

    def _ask_plot_choice(self) -> str:
        print("\n--- Графики ---")
        print("  a — RC (фильтр нижних частот)")
        print("  b — RLC (полосовой, выход на R)")
        print("  ab — оба набора")
        print("  n — без графиков")
        while True:
            s = input("Выбор [a/b/ab/n]: ").strip().lower().replace(" ", "")
            if s in ("a", "b", "ab", "n", "а", "б", "аб"):
                return {"а": "a", "б": "b", "аб": "ab"}.get(s, s)
            print("Введите a, b, ab или n.")

    def _render_choice(self, choice, R, C, rc_sweep, rc_square, rlc_data):
        figs: list = []
        if choice in ("a", "ab"):
            figs.extend(result_visualizer.build_rc_figures(
                R, C, rc_sweep, rc_square, config.RC_U0, config.RC_SQUARE_PERIOD
            ))
        if choice in ("b", "ab") and rlc_data is not None:
            Rl, L, Cr, sw, sq = rlc_data
            figs.extend(result_visualizer.build_rlc_figures(Rl, L, Cr, sw, sq, config.RLC_U0))
        result_visualizer.show_figures(*figs)

    def _save_choice(self, choice, path, R, C, rc_sweep, rc_square, rlc_data):
        if choice in ("a", "ab"):
            f1, f2 = result_visualizer.build_rc_figures(R, C, rc_sweep, rc_square, config.RC_U0, config.RC_SQUARE_PERIOD)
            result_visualizer.save_figures(path, "m6_rc", f1, f2)
            print(f"RC: {path.resolve()}/m6_rc_main.png")
        if choice in ("b", "ab") and rlc_data is not None:
            Rl, L, Cr, sw, sq = rlc_data
            f1, f2 = result_visualizer.build_rlc_figures(Rl, L, Cr, sw, sq, config.RLC_U0)
            result_visualizer.save_figures(path, "m6_rlc", f1, f2)
            print(f"RLC: {path.resolve()}/m6_rlc_main.png")

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
