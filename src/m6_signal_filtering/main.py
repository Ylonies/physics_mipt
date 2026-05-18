from __future__ import annotations

import argparse
from pathlib import Path

from src.m6_signal_filtering.signal_filter import SignalFilterLab


def main() -> None:
    p = argparse.ArgumentParser(description="M6: RC и RLC фильтры")
    p.add_argument("--save-plots", type=Path, default=None, metavar="DIR")
    args = p.parse_args()
    SignalFilterLab().run(save_plots=args.save_plots)


if __name__ == "__main__":
    main()
