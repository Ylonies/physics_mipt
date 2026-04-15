"""Metrics for the recovered message quality."""

from m7.models.models import SimulationResult


class ResultAnalyzer:
    @staticmethod
    def analyze(result: SimulationResult, model_name: str) -> dict[str, float | str]:
        return {
            "model_name": model_name,
            "mse": result.mse,
            "corrcoef": result.corrcoef,
        }
