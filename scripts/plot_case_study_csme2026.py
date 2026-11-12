# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures and numbers of the CSME 2026 case study for one motor.

Reads data/processed/<folder> (made by scripts/process_raw.py) and the Pi logs
in data/raw/<folder>, where <folder> is a motor folder such as db17-r5-left-m9
and <label> starts its run names, such as R5-L-m9. The defaults reproduce the
article (motor m5, folder db17-cherry-left-m5, label R3-L). It writes
fig-*.pdf into the figures folder and the LaTeX macros of the reported
numbers into results.tex. fig-methods.pdf, the
schematic of S3-S5 of the supplement, uses synthetic data only;
fig-rest-noise.pdf shows the speed noise at rest that sets omega_min;
fig-wheel-overlay.pdf shows the tracker's output on the frames that
scripts/select_wheel_frames.py picked from a video (skipped with a message
when that script has not been run).

Usage: python scripts/plot_case_study_csme2026.py --data data \\
    --folder db17-r5-left-m9 --label R5-L-m9 \\
    --figures build/figures/case_study
"""

import argparse
import sys
from pathlib import Path

from duckiebot_analysis.analysis.case_study import (
    LABEL,
    MOTOR,
    REST_EXCEEDANCE,
    case_study,
    configure,
)
from duckiebot_analysis.visualization.case_study_figures import make_all
from duckiebot_analysis.visualization.methods_figure import fig_methods
from duckiebot_analysis.visualization.rest_noise_figure import fig_rest_noise
from duckiebot_analysis.visualization.wheel_overlay_figure import (
    WHEEL_FRAMES_DIR,
    fig_wheel_overlay,
)

COMMAND = "python scripts/plot_case_study_csme2026.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help="folder holding raw/ and processed/ (default: data)",
    )
    p.add_argument(
        "--folder", default=MOTOR, help=f"motor folder name (default: {MOTOR})"
    )
    p.add_argument(
        "--label",
        default=LABEL,
        help=f"start of the run names, such as R5-L-m9 (default: {LABEL})",
    )
    p.add_argument(
        "--figures",
        type=Path,
        default=Path("build/figures/case_study_csme2026"),
        help="output folder of the figures",
    )
    p.add_argument(
        "--macros",
        type=Path,
        help="LaTeX macro file to write (default: <figures>/results.tex)",
    )
    p.add_argument(
        "--wheel-frames",
        type=Path,
        default=WHEEL_FRAMES_DIR,
        help=f"output folder of select_wheel_frames.py (default: {WHEEL_FRAMES_DIR})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compute the results and write the figures and the macro file."""
    args = parse_args(argv)
    figures = args.figures
    macros = args.macros if args.macros is not None else figures / "results.tex"
    configure(args.folder, args.label)
    command = f"{COMMAND} --folder {args.folder} --label {args.label}"
    cs = case_study(args.data)
    for path in make_all(cs, figures, macros, command):
        print(path)
    rest = figures / "fig-rest-noise.pdf"
    fig_rest_noise(cs.rest_noise, cs.omega_min, REST_EXCEEDANCE, rest)
    print(rest)
    methods = figures / "fig-methods.pdf"
    fig_methods(methods)
    print(methods)
    overlay = figures / "fig-wheel-overlay.pdf"
    try:
        fig_wheel_overlay(overlay, args.wheel_frames)
    except FileNotFoundError as exc:
        print(
            f"skipped {overlay}: run select_wheel_frames.py first ({exc})",
            file=sys.stderr,
        )
        return 0
    print(overlay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
