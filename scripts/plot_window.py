from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.visualization import plot_window


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one saved raw window.")
    parser.add_argument("--artifact", required=True, help="Path to saved artifact.")
    parser.add_argument("--index", type=int, default=0, help="Window index to plot.")
    parser.add_argument("--output", default=None, help="Optional output image path.")
    args = parser.parse_args()

    output = None if args.output is None else Path(args.output)
    plot_window(Path(args.artifact), args.index, output)


if __name__ == "__main__":
    main()
