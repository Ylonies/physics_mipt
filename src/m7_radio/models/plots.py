import matplotlib.pyplot as plt
import numpy as np

def _slice_for_visibility(t: np.ndarray, max_time: float = 0.002) -> slice:
    idx = int(np.searchsorted(t, max_time))
    return slice(0, max(100, idx))


def plot_simulation(results, params) -> None:
    t = results["time"]
    view = _slice_for_visibility(t)
    t_ms = t[view] * 1_000.0

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), constrained_layout=True)

    axes[0].plot(t_ms, results["source_signal"][view], label="Исходный сигнал (НЧ)", lw=1.2)
    axes[0].set_title("Исходный информационный сигнал")
    axes[0].set_ylabel("Амплитуда")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t_ms, results["transmitted_signal"][view], label="Переданный АМ-сигнал", lw=1.0)
    axes[1].set_title(f"Амплитудная модуляция, несущая = {params['carrier_freq_hz'] / 1_000:.0f} кГц")
    axes[1].set_ylabel("Амплитуда")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t_ms, results["received_signal"][view], label="После канала + ВЧ-помех", lw=1.0)
    axes[2].set_title("Принятый зашумлённый ВЧ-сигнал")
    axes[2].set_ylabel("Амплитуда")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    axes[3].plot(t_ms, results["rlc_output"][view], label="Выход RLC-контура", lw=1.0)
    axes[3].plot(t_ms, results["envelope"][view], label="Выпрямлённая огибающая", lw=1.0)
    axes[3].set_title("Резонансный отбор и идеальный диодный детектор")
    axes[3].set_ylabel("Амплитуда")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    axes[4].plot(t * 1_000.0, results["source_signal"], label="Исходный", lw=1.2)
    axes[4].plot(t * 1_000.0, results["recovered_signal"], label="Восстановленный после RC-ФНЧ", lw=1.2)
    axes[4].set_title("Качество восстановления информации")
    axes[4].set_xlabel("Время, мс")
    axes[4].set_ylabel("Амплитуда")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="upper right")

    plt.show()


class ResultVisualizer:
    @staticmethod
    def plot(results, params, model_name: str) -> None:
        _ = model_name
        plot_simulation(results, params)
