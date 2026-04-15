"""Metrics for the recovered message quality."""

from m7.models.models import SimulationResult


def print_metrics(result: SimulationResult) -> None:
    print("=== M7 radio-link simulation metrics ===")
    print(f"MSE(source, recovered): {result.mse:.6f}")
    print(f"Correlation(source, recovered): {result.corrcoef:.4f}")
