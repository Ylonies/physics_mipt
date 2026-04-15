import numpy as np


class PhysicsModels:
    @staticmethod
    def _normalize_signal(signal: np.ndarray) -> np.ndarray:
        max_abs = float(np.max(np.abs(signal)))
        if max_abs == 0.0:
            return signal.copy()
        return signal / max_abs

    @staticmethod
    def _generate_source_signal(t: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        low_component = (
            0.55 * np.sin(2.0 * np.pi * 1_200.0 * t)
            + 0.35 * np.sin(2.0 * np.pi * 2_100.0 * t + 0.2)
            + 0.25 * np.sin(2.0 * np.pi * 3_000.0 * t + 1.1)
        )
        burst = 0.15 * np.sign(np.sin(2.0 * np.pi * 450.0 * t))
        tremolo = 1.0 + 0.1 * np.sin(2.0 * np.pi * 70.0 * t)
        noisy_audio_like = tremolo * low_component + burst + 0.03 * rng.standard_normal(t.size)
        return PhysicsModels._normalize_signal(noisy_audio_like)

    @staticmethod
    def _generate_high_freq_noise(t: np.ndarray, params: dict) -> np.ndarray:
        rng = np.random.default_rng(params["seed"] + 1)
        frequencies = rng.uniform(
            low=params["noise_min_freq_hz"],
            high=params["noise_max_freq_hz"],
            size=params["noise_components"],
        )
        phases = rng.uniform(0.0, 2.0 * np.pi, size=params["noise_components"])
        amplitudes = rng.uniform(0.2, 1.0, size=params["noise_components"])

        noise = np.zeros_like(t)
        for f_hz, phase, amp in zip(frequencies, phases, amplitudes, strict=True):
            noise += amp * np.sin(2.0 * np.pi * f_hz * t + phase)
        noise = PhysicsModels._normalize_signal(noise)
        noise += 0.1 * rng.standard_normal(t.size)
        return PhysicsModels._normalize_signal(noise)

    @staticmethod
    def _simulate_rlc_response(input_signal: np.ndarray, dt: float, params: dict) -> np.ndarray:
        w0 = 2.0 * np.pi * params["carrier_freq_hz"]
        q_factor = w0 * params["rlc_inductance_h"] / params["rlc_resistance_ohm"]

        freqs_hz = np.fft.rfftfreq(input_signal.size, d=dt)
        omega = 2.0 * np.pi * freqs_hz
        ratio = omega / w0

        numerator = 1j * ratio / q_factor
        denominator = 1.0 - ratio**2 + 1j * ratio / q_factor
        transfer = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )

        spectrum = np.fft.rfft(input_signal)
        output_spectrum = spectrum * transfer
        return np.fft.irfft(output_spectrum, n=input_signal.size)

    @staticmethod
    def _rectifier_envelope_detector(signal: np.ndarray, scale: float) -> np.ndarray:
        return scale * np.abs(signal)

    @staticmethod
    def _single_pole_lowpass(signal: np.ndarray, dt: float, tau: float) -> np.ndarray:
        alpha = dt / (tau + dt)
        filtered = np.zeros_like(signal)
        for idx in range(1, signal.size):
            filtered[idx] = filtered[idx - 1] + alpha * (signal[idx] - filtered[idx - 1])
        return filtered

    @staticmethod
    def _align_by_max_correlation(reference: np.ndarray, candidate: np.ndarray, max_shift: int) -> np.ndarray:
        best_shift = 0
        best_corr = -np.inf
        n = reference.size
        for shift in range(-max_shift, max_shift + 1):
            if shift >= 0:
                ref_slice = reference[shift:]
                cand_slice = candidate[: n - shift]
            else:
                ref_slice = reference[: n + shift]
                cand_slice = candidate[-shift:]
            if ref_slice.size < 32:
                continue
            corr = float(np.corrcoef(ref_slice, cand_slice)[0, 1])
            if np.isnan(corr):
                continue
            if corr > best_corr:
                best_corr = corr
                best_shift = shift

        aligned = np.zeros_like(candidate)
        if best_shift >= 0:
            aligned[best_shift:] = candidate[: n - best_shift]
        else:
            aligned[: n + best_shift] = candidate[-best_shift:]
        return aligned

    @staticmethod
    def _radio_link_model(params: dict) -> dict:
        dt = 1.0 / params["sample_rate_hz"]
        t = np.arange(0.0, params["duration_s"], dt)

        source = PhysicsModels._generate_source_signal(t, params["seed"])
        transmitted = (1.0 + params["modulation_index"] * source) * np.cos(
            2.0 * np.pi * params["carrier_freq_hz"] * t
        )

        channel_noise = params["noise_relative_amplitude"] * PhysicsModels._generate_high_freq_noise(t, params)
        received = transmitted + channel_noise

        rlc_output = PhysicsModels._simulate_rlc_response(received, dt, params)
        envelope = PhysicsModels._rectifier_envelope_detector(rlc_output, params["envelope_scale"])
        lowpassed = PhysicsModels._single_pole_lowpass(envelope, dt, params["lowpass_tau_s"])

        recovered = PhysicsModels._normalize_signal(lowpassed - np.mean(lowpassed))
        max_shift = int(0.002 * params["sample_rate_hz"])
        recovered = PhysicsModels._align_by_max_correlation(source, recovered, max_shift=max_shift)
        if np.corrcoef(source, recovered)[0, 1] < 0:
            recovered = -recovered

        return {
            "time": t,
            "source_signal": source,
            "transmitted_signal": transmitted,
            "channel_noise": channel_noise,
            "received_signal": received,
            "rlc_output": rlc_output,
            "envelope": envelope,
            "recovered_signal": recovered,
        }

    def get_model(self, params):
        return self._radio_link_model, "АМ + ВЧ-помехи + RLC-контур + детектор + RC-фильтр"
