#!/usr/bin/env python3
# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Write the data quality report of every motor folder.

Reads ``data/raw/<folder>/`` and ``data/processed/<folder>/`` (run
``scripts/process_raw.py`` first), writes ``data/raw/<folder>/quality.md``, and
with ``--readme`` replaces the summary table between the quality-summary
markers of data/README.md.

Example:
    python scripts/quality_report.py --readme data/README.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from duckiebot_analysis.analysis import quality
from duckiebot_analysis.analysis import quality_markdown as md
from duckiebot_analysis.data import layout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "folders",
        nargs="*",
        help="motor folders to write reports for (default: all); all are assessed",
    )
    p.add_argument("--raw", type=Path, default=Path("data/raw"), help="raw data root")
    p.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed"),
        help="processed data root",
    )
    p.add_argument("--readme", type=Path, help="data/README.md to update")
    p.add_argument(
        "--date", default=date.today().isoformat(), help="report date, YYYY-MM-DD"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the reports and return the exit status."""
    args = parse_args(argv)
    try:
        every = sorted(
            p.name for p in args.raw.iterdir() if layout.is_motor_folder_name(p.name)
        )
        unknown = sorted(set(args.folders) - set(every))
        if unknown:
            raise ValueError(f"not motor folders in {args.raw}: {', '.join(unknown)}")
        bias = quality.motion_bias_by_experiment(args.raw, args.processed)
        # Every folder is assessed, so that the recommendations name the other
        # motors and the README summary keeps every folder; only the named
        # folders get a new report.
        folders = [
            quality.assess_folder(
                args.raw / name, args.processed / name, motion_bias=bias
            )
            for name in every
        ]
        quality.finish(folders)
        for fq in folders:
            if args.folders and fq.folder not in args.folders:
                continue
            path = args.raw / fq.folder / md.REPORT_NAME
            path.write_text(
                md.render_report(fq, date=args.date, motion_bias=bias),
                encoding="utf-8",
            )
            warned = sum(rq.warned for rq in fq.runs)
            print(f"{path}: {warned} of {len(fq.runs)} runs with warnings")
        if args.readme is not None:
            table = md.readme_header() + [md.readme_row(f) for f in folders]
            text = args.readme.read_text(encoding="utf-8")
            args.readme.write_text(
                md.replace_marked(
                    text,
                    md.README_START,
                    md.README_END,
                    "\n" + "\n".join(table) + "\n\n",
                ),
                encoding="utf-8",
            )
            print(f"{args.readme}: summary of {len(folders)} folders")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
