import matplotlib.pyplot as plt
import numpy as np


class ResultVisualizer:
    @staticmethod
    def plot(results, params, model_name: str) -> None:
        _ = model_name
        t = results["time"]
        u = results["u_diode_v"]
        i = results["i_inductor_a"]
        id_d = results["i_diode_a"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

        axes[0, 0].plot(t, u, lw=1.2, label="U_D(t)")
        axes[0, 0].set_title("Напряжение на туннельном диоде")
        axes[0, 0].set_xlabel("t, c")
        axes[0, 0].set_ylabel("U, В")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc="upper right")

        axes[0, 1].plot(t, i, lw=1.2, label="I_L(t)")
        axes[0, 1].plot(t, id_d, lw=1.0, label="I_D(t)")
        axes[0, 1].set_title("Токи в контуре")
        axes[0, 1].set_xlabel("t, c")
        axes[0, 1].set_ylabel("I, А")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc="upper right")

        axes[1, 0].plot(u, i, lw=1.0)
        axes[1, 0].set_title("Фазовый портрет I_L(U_D)")
        axes[1, 0].set_xlabel("U_D, В")
        axes[1, 0].set_ylabel("I_L, А")
        axes[1, 0].grid(True, alpha=0.3)

        tail_u = results["tail_u"] - np.mean(results["tail_u"])
        dt = float(t[1] - t[0])
        spectrum = np.abs(np.fft.rfft(tail_u))
        freqs = np.fft.rfftfreq(tail_u.size, d=dt)
        axes[1, 1].plot(freqs, spectrum, lw=1.0)
        axes[1, 1].set_xlim(0.0, min(10_000.0, freqs[-1]))
        axes[1, 1].set_title("Спектр установившихся колебаний")
        axes[1, 1].set_xlabel("f, Гц")
        axes[1, 1].set_ylabel("|U(f)|")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(
            "Автогенератор на туннельном диоде | "
            f"R={params['r_ohm']}, L={params['l_h']}, C={params['c_f']}"
        )
        plt.show()
