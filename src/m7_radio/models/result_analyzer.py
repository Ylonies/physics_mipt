import numpy as np


class ResultAnalyzer:
    @staticmethod
    def analyze(solution, params, model_name):
        source = solution["source_signal"]
        recovered = solution["recovered_signal"]
        warmup_idx = max(1, int(0.1 * recovered.size))

        mse = float(((recovered[warmup_idx:] - source[warmup_idx:]) ** 2).mean())
        corrcoef = float(np.corrcoef(source[warmup_idx:], recovered[warmup_idx:])[0, 1])

        results = {
            "model_name": model_name,
            "mse": mse,
            "corrcoef": corrcoef,
        }
        results.update(solution)
        results["params"] = params
        return results
