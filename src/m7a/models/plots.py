# result_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from .config import SimulationConfig

class ResultVisualizer:

    @staticmethod
    def plot_basic(results):
        cfg = SimulationConfig

        times = np.array(results["times"])
        temps = np.array(results["temps"])
        pressures = np.array(results["pressures"])
        piston_xs = np.array(results["piston_xs"])
        pos_history = results["pos_history"]
        piston_x_history = results["piston_x_history"]

        # объём газа
        volumes = piston_xs * cfg.Ly

        # snapshot в начале и конце
        snapshots_idx = [0, -1]

        # --- усреднение давления для графика ---
        # например, берем скользящее среднее за 1 секунду
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        window_steps = max(1, int(1.0 / dt))  # шагов на 1 секунду
        pressures_avg = np.convolve(pressures, np.ones(window_steps)/window_steps, mode='same')

        # --- создаем единый холст 2x2 ---
        fig, axes = plt.subplots(2,2, figsize=(14,10))

        # 1. Температура T(t)
        axes[0,0].plot(times, temps, label="T(t)")
        axes[0,0].scatter(times[snapshots_idx], temps[snapshots_idx], color='red', zorder=5,
                          label='Snapshots')
        axes[0,0].set_xlabel("t [s]")
        axes[0,0].set_ylabel("Temperature T [K]")
        axes[0,0].set_title("Temperature vs Time")
        axes[0,0].grid(True)
        axes[0,0].legend()

        axes[0,1].plot(times, pressures_avg, color="orange", label="P_avg(t)")
        axes[0,1].scatter(times[snapshots_idx], pressures_avg[snapshots_idx], color='red', zorder=5,
                        label='Snapshots')

        axes[0,1].set_xlabel("t [s]")
        axes[0,1].set_ylabel("Pressure P [Pa]")
        axes[0,1].set_title("Average Pressure vs Time (1 s window)")
        axes[0,1].grid(True)
        axes[0,1].legend()

        # 3. Объём V(t)
        axes[1,0].plot(times, volumes, color="green", label="V(t)")
        axes[1,0].scatter(times[snapshots_idx], volumes[snapshots_idx], color='red', zorder=5,
                          label='Snapshots')
        axes[1,0].set_xlabel("t [s]")
        axes[1,0].set_ylabel("Volume V [m²]")
        axes[1,0].set_title("Volume vs Time")
        axes[1,0].grid(True)
        axes[1,0].legend()

        for idx, snap_idx in enumerate(snapshots_idx):
            pos = pos_history[snap_idx]
            piston_x = piston_x_history[snap_idx]
            axes[1,1].scatter(pos[:,0], pos[:,1], s=(cfg.radius*1000)**2, alpha=0.5,
                              label=f"Snapshot {idx+1}, t ≈ {times[snap_idx]:.2f} s")
            axes[1,1].axvline(piston_x, color="red", linestyle="--", label="Piston" if idx==0 else "")
        axes[1,1].set_xlim(0, cfg.Lx*1.05)
        axes[1,1].set_ylim(0, cfg.Ly)
        axes[1,1].set_xlabel("x [m]")
        axes[1,1].set_ylabel("y [m]")
        axes[1,1].set_title("Snapshots: Beginning & End")
        axes[1,1].legend()

        plt.tight_layout()
        plt.show()
