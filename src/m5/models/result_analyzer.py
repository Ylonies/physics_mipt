import numpy as np


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
        max_harm = 8
        for n in range(2, max_harm + 1):
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
        start = int(0.6 * n)
        tail = u[start:]
        tail_i = i[start:]

        amp_tail = 0.5 * (np.max(tail) - np.min(tail))
        amp_prev = 0.5 * (np.max(u[int(0.2 * n):int(0.5 * n)]) - np.min(u[int(0.2 * n):int(0.5 * n)]))
        self_oscillation = bool(amp_tail > 1e-3 and amp_tail >= 0.7 * amp_prev)

        dt = float(t[1] - t[0]) if t.size > 1 else 1.0
        f1, thd = ResultAnalyzer._estimate_fundamental_and_thd(tail, dt)
        sine_purity = float(1.0 / (1.0 + thd))

        results = {
            "model_name": model_name,
            "self_oscillation": self_oscillation,
            "u_max_abs": float(np.max(np.abs(u))),
            "fundamental_freq_hz": f1,
            "thd": thd,
            "sine_purity": sine_purity,
            "time": t,
            "u_diode_v": u,
            "i_inductor_a": i,
            "i_diode_a": solution["i_diode_a"],
            "tail_u": tail,
            "tail_i": tail_i,
        }
        results["params"] = params
        return results
