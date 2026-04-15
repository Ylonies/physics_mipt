"""Entry point for module m7."""

from m7.models.input import SimulationConfig
from m7.models.plots import plot_simulation
from m7.models.result_analyzer import print_metrics
from m7.models.solver import run_simulation


def main() -> None:
    config = SimulationConfig()
    result = run_simulation(config)
    print_metrics(result)
    plot_simulation(result, config)


if __name__ == "__main__":
    main()
