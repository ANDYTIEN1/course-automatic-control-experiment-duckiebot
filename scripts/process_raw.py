#!/usr/bin/env python3
# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Process a motor folder into angle, speed, acceleration, and input.

Step 3 of README.md, From video to figures: reads
``data/raw/db17-<robot>-<motor>-<id>/`` and writes ``<run>.parquet`` per run
and ``trials.parquet`` (data/schema.md §11). Runs without a frame table are
skipped with a message. A floor run whose frame table has the spoke columns
of scripts/track_video.py takes its wheel angle from the spokes (§9.5); one
without them keeps the marker angle, with a note.

Example:
    python scripts/process_raw.py data/raw/db17-daisy-left --run L_Sweep_20261015_012223
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from duckiebot_analysis.analysis.process import ProcessParams, process_folder
from duckiebot_analysis.data.layout import processed_folder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    defaults = ProcessParams()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "folder", type=Path, help="motor folder data/raw/db17-<robot>-<motor>-<id>"
    )
    p.add_argument(
        "--out",
        type=Path,
        help="output folder (default: data/processed/<same name>)",
    )
    p.add_argument("--run", nargs="+", action="extend", help="process only these runs")
    p.add_argument(
        "--n-min", type=int, default=defaults.n_min, help="minimum marker pixels (§9.2)"
    )
    p.add_argument(
        "--n-hub-min",
        type=int,
        default=defaults.n_hub_min,
        help="minimum hub pixels (§9.2)",
    )
    p.add_argument(
        "--w-omega",
        type=float,
        default=defaults.w_omega_s,
        help="half-width of the ω fit in s (§9.7)",
    )
    p.add_argument(
        "--w-alpha",
        type=float,
        default=defaults.w_alpha_s,
        help="half-width of the α fit in s (§9.7)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the processing and return the exit status."""
    args = parse_args(argv)
    params = ProcessParams(
        n_min=args.n_min,
        n_hub_min=args.n_hub_min,
        w_omega_s=args.w_omega,
        w_alpha_s=args.w_alpha,
    )
    try:
        out = args.out if args.out is not None else processed_folder(args.folder)
        process_folder(args.folder, out, params, runs=args.run)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
