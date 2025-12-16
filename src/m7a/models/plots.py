import matplotlib.pyplot as plt
import numpy as np
from .config import SimulationConfig

class ResultVisualizer:

    @staticmethod
    def plot_basic(results, mode=0):
        cfg = SimulationConfig

        times = np.array(results["times"])
        temps = np.array(results["temps"])
        pressures = np.array(results["pressures"])
        piston_xs = np.array(results["piston_xs"])
        pos_history = results["pos_history"]
        piston_x_history = results["piston_x_history"]
        snapshot_indices = results["snapshot_indices"]

        volumes = piston_xs * cfg.Ly

        # нормализация объёма для пропорциональной шкалы
        V_min = volumes.min()
        V_max = volumes.max()
        if V_max > V_min:
            volumes_norm = (volumes - V_min) / (V_max - V_min)
        else:
            volumes_norm = np.zeros_like(volumes)

        dt_step = times[1]-times[0] if len(times)>1 else 1.0
        window_steps = max(1, int(1.0/dt_step))
        n = len(pressures)
        pressures_avg = []
        times_avg = []
        for start in range(0, n, window_steps):
            end = min(start+window_steps, n)
            pressures_avg.append(pressures[start:end].mean())
            times_avg.append(times[start:end].mean())
        pressures_avg = np.array(pressures_avg)
        times_avg = np.array(times_avg)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0,0].plot(times, temps, label="T(t)")
        for i, idx in enumerate(snapshot_indices):
            idx = min(idx, len(times)-1)
            axes[0,0].scatter(times[idx], temps[idx], s=80, color='red')
        axes[0,0].set_xlabel("t [s]")
        axes[0,0].set_ylabel("Temperature [K]")
        axes[0,0].set_title("Temperature vs Time")
        axes[0,0].grid(True)

        axes[0,1].plot(times, pressures, alpha=0.5, label="P_raw(t)")
        axes[0,1].plot(times_avg, pressures_avg, label="P_avg(t)")
        for i, idx in enumerate(snapshot_indices):
            idx = min(idx, len(times)-1)
            axes[0,1].scatter(times[idx], pressures[idx], s=80, color='red')
        axes[0,1].set_xlabel("t [s]")
        axes[0,1].set_ylabel("Pressure [Pa]")
        axes[0,1].set_title("Pressure vs Time")
        axes[0,1].grid(True)

        axes[1,0].plot(times, volumes, label="V(t)", color="blue")
        axes[1,0].set_xlabel("t [s]")
        axes[1,0].set_ylabel("Volume [m²]")
        axes[1,0].set_title("Volume vs Time")
        axes[1,0].grid(True)

        colors = ["blue", "green"]
        for i, si in enumerate(snapshot_indices):
            if i < pos_history.shape[0]:
                pos = pos_history[i]
                piston_x = piston_x_history[i]  # поршень именно на этом snapshot
                label_time = times[min(si, len(times)-1)] if len(times) > 0 else 0.0
                axes[1,1].scatter(pos[:,0], pos[:,1], s=(cfg.radius*1000)**2, alpha=0.5,
                                color=colors[i%len(colors)],
                                label=f"Snapshot {i} (t≈{label_time:.2f}s)")
                if mode != 0:
                    axes[1,1].axvline(piston_x, color="red", linestyle="--",
                                    linewidth=2, alpha=0.7,
                                    label="Piston" if i==0 else "")

        if len(piston_x_history) > 1:
            axes[1,1].axvline(piston_x_history[1], color="red", linestyle="--",
                            linewidth=2, alpha=0.7, label="Final Piston")


        handles, labels = axes[1,1].get_legend_handles_labels()
        unique_labels = {}
        for h, l in zip(handles, labels):
            unique_labels[l] = h
        axes[1,1].legend(unique_labels.values(), unique_labels.keys())

        axes[1,1].set_xlim(0, cfg.Lx*1.05)
        axes[1,1].set_ylim(0, cfg.Ly)
        axes[1,1].set_xlabel("x [m]")
        axes[1,1].set_ylabel("y [m]")
        axes[1,1].set_title("Particle Snapshots")
        axes[1,1].grid(True)

        plt.tight_layout()
        plt.show()
