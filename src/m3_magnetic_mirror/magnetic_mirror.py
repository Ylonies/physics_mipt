"""Оркестратор симуляции (аналог ElectrodeSystem для M1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.m3_magnetic_mirror import config
from src.m3_magnetic_mirror.models import result_visualizer
from src.m3_magnetic_mirror.models import simulation


class MagneticMirrorTrap:
    def run(self, *, save_plots: Path | None = None) -> None:
        R, d, I = config.RING_RADIUS, config.RING_SEPARATION, config.RING_CURRENT
        q, m = config.PARTICLE_CHARGE, config.PARTICLE_MASS

        hist = simulation.run_single(
            R,
            d,
            I,
            q,
            m,
            config.R0,
            config.V0,
            n_seg=config.N_SEGMENTS_PER_RING,
            grid_nr=config.FIELD_GRID_NR,
            grid_nz=config.FIELD_GRID_NZ,
            dt_frac_tcyc=config.DT_FRACTION_OF_TCYC,
            b_ref=config.B_REF_FOR_DT,
            t_max_tcyc=config.T_MAX_TCYC,
            escape_z=config.ESCAPE_Z,
            r_max_factor=config.FIELD_R_MAX_FACTOR,
            z_margin_factor=config.FIELD_Z_MARGIN_FACTOR,
        )

        esc = hist["escaped_at"]
        print(f"B_ref ≈ {hist['b_ref']:.4e} T, dt = {hist['dt']:.4e} s")
        print(f"steps = {len(hist['t']) - 1}, escaped_at = {esc}")

        if config.RUN_BATCH:
            rng = np.random.default_rng(config.BATCH_SEED)
            batch = simulation.run_batch(
                R,
                d,
                I,
                q,
                m,
                config.BATCH_N,
                rng,
                n_seg=config.N_SEGMENTS_PER_RING,
                grid_nr=config.FIELD_GRID_NR,
                grid_nz=config.FIELD_GRID_NZ,
                dt_frac_tcyc=config.DT_FRACTION_OF_TCYC,
                b_ref=config.B_REF_FOR_DT,
                t_max_tcyc=config.T_MAX_TCYC,
                escape_z=config.ESCAPE_Z,
                r0_max=config.BATCH_R0_MAX,
                z0_half=config.BATCH_Z0_RANGE,
                v_mag=config.BATCH_V_MAG,
                r_max_factor=config.FIELD_R_MAX_FACTOR,
                z_margin_factor=config.FIELD_Z_MARGIN_FACTOR,
            )
            trapped = sum(1 for h in batch if h["escaped_at"] is None)
            print(f"Batch: retained until t_max: {trapped}/{config.BATCH_N}")

        self._show_plots(hist, m, save_plots)

    def _show_plots(self, hist: dict, mass: float, save_plots: Path | None) -> None:
        try:
            import matplotlib.pyplot as plt

            field = hist["field"]
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            zs = np.linspace(field._z_lo, field._z_hi, 200)
            result_visualizer.plot_bz_axis_vs_numeric(ax1, field, zs)
            ax1.set_title(r"$B_z$ на оси: аналитика и Biot–Savart")
            fig1.tight_layout()

            fig2 = plt.figure(figsize=(10, 8))
            result_visualizer.plot_trajectory_overview(fig2, hist, mass)
            fig2.suptitle("Траектория частицы")
            fig2.tight_layout()

            fig3, ax3 = plt.subplots(figsize=(6, 5))
            result_visualizer.plot_field_magnitude_rz(ax3, field)
            ax3.set_title(r"$|B|(r,z)$")
            fig3.tight_layout()

            if save_plots is not None:
                save_plots = Path(save_plots)
                save_plots.mkdir(parents=True, exist_ok=True)
                fig1.savefig(save_plots / "m3_bz_axis.png", dpi=140, bbox_inches="tight")
                fig2.savefig(save_plots / "m3_trajectory.png", dpi=140, bbox_inches="tight")
                fig3.savefig(save_plots / "m3_field_rz.png", dpi=140, bbox_inches="tight")
                print(f"Графики сохранены в {save_plots.resolve()}")
                plt.close("all")
            else:
                plt.show()
        except Exception as e:
            print(f"(графики пропущены: {e})")
