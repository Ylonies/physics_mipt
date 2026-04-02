"""Точка входа: `uv run python -m src.m3_magnetic_mirror.main`"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.m3_magnetic_mirror.magnetic_mirror import MagneticMirrorTrap


def main() -> None:
    p = argparse.ArgumentParser(description="M3: заряд в магнитной пробке двух колец")
    p.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        metavar="DIR",
        help="сохранить PNG в каталог (без GUI, удобно для MPLBACKEND=Agg)",
    )
    args = p.parse_args()
    MagneticMirrorTrap().run(save_plots=args.save_plots)


if __name__ == "__main__":
    main()
