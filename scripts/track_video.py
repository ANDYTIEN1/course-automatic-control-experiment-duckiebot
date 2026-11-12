#!/usr/bin/env python3
# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""Track a video into its frame table (data/schema.md §3).

Writes <folder>/frames/<run>.parquet and prints the run's row for runs.csv
(§4) to stdout, so it can be appended with >>. The run identifier is the video
file name without extension. An existing frame table is never overwritten
unless --force is given.

Example:
    python scripts/track_video.py VIDEO.mp4 \\
        --folder data/raw/db17-daisy-left --mode stand \\
        --marker-colour blue --led-colour green \\
        --camera "phone, 1920x1080 H.264 30 fps"

The defaults (blue marker tape, green sync LED) fit the course kit; the
article's recordings used a white marker and a red LED.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from duckiebot_analysis.data.frames import write_frames
from duckiebot_analysis.data.layout import (
    SYNC_LEVELS,
    RunRecord,
    format_runs_row,
    frames_path,
    parse_motor_folder,
)
from duckiebot_analysis.tracking.blobs import PRESETS
from duckiebot_analysis.tracking.track import TrackConfig, track_video
from duckiebot_analysis.tracking.video import probe_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    presets = ", ".join(f"{k} ({v})" for k, v in PRESETS.items())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path, help="video file")
    ap.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="motor folder data/raw/db17-<robot>-<motor>-<id>",
    )
    ap.add_argument("--mode", choices=("stand", "floor"), required=True)
    ap.add_argument(
        "--camera",
        required=True,
        help='phone model and recording mode, such as "Samsung S7 Edge, '
        'slow motion 1280x720 240 fps"',
    )
    ap.add_argument(
        "--marker-colour",
        default="blue",
        help=f"preset or rule such as 'r>170 & g>170 & b>160'; presets: {presets}",
    )
    ap.add_argument(
        "--hub-colour", default="yellow", help="preset or rule, as --marker-colour"
    )
    ap.add_argument(
        "--marker-region",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "R"),
        help="stand: marker search disk in px (located from key frames if omitted)",
    )
    ap.add_argument(
        "--hub-region",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "R"),
        help="stand: hub search disk in px (located from key frames if omitted)",
    )
    ap.add_argument(
        "--led-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="stand: LED grid centre in px (located from key frames if omitted)",
    )
    ap.add_argument("--no-led", action="store_true", help="the run has no sync LED")
    ap.add_argument(
        "--led-colour",
        choices=("red", "green"),
        default="green",
        help="colour of the sync LED (default: green)",
    )
    ap.add_argument(
        "--floor-hub-radius", type=float, help="floor: hub region radius in px"
    )
    ap.add_argument(
        "--floor-marker-radius", type=float, help="floor: marker region radius in px"
    )
    ap.add_argument(
        "--floor-led-radius", type=float, help="floor: LED search radius in px"
    )
    ap.add_argument("--experiment", help="runs.csv experiment")
    ap.add_argument(
        "--sync", choices=SYNC_LEVELS, help="runs.csv synchronisation level"
    )
    ap.add_argument("--pi-log", help="runs.csv Pi log path relative to the folder")
    ap.add_argument(
        "--trials", default="", help="runs.csv trial identifiers, ';'-separated"
    )
    ap.add_argument("--notes", help="runs.csv notes")
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing frame table"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Track the video, write the frame table, and print the runs.csv row."""
    args = parse_args(argv)
    parse_motor_folder(args.folder)
    out = frames_path(args.folder, args.video.stem)
    if out.exists() and not args.force:
        print(
            f"{out} exists; data/raw is never overwritten without --force",
            file=sys.stderr,
        )
        return 1
    config = TrackConfig(
        mode=args.mode,
        camera=args.camera,
        marker_colour=args.marker_colour,
        hub_colour=args.hub_colour,
        marker_region=tuple(args.marker_region) if args.marker_region else None,
        hub_region=tuple(args.hub_region) if args.hub_region else None,
        led=not args.no_led,
        led_xy=tuple(args.led_xy) if args.led_xy else None,
        floor_hub_radius=args.floor_hub_radius,
        floor_marker_radius=args.floor_marker_radius,
        floor_led_radius=args.floor_led_radius,
        tracker_script="scripts/track_video.py",
        led_colour=args.led_colour,
    )
    cols, meta = track_video(args.video, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_frames(out, cols, meta)
    info = probe_video(args.video)
    row = RunRecord(
        run=meta["run"],
        video_file=meta["video_file"],
        video_sha256=meta["video_sha256"],
        video_bytes=meta["video_bytes"],
        duration_s=info.duration_s,
        resolution=f"{meta['width']}x{meta['height']}",
        fps_nominal=meta["fps_nominal"],
        mode=meta["mode"],
        experiment=args.experiment,
        sync=args.sync,
        pi_log=args.pi_log,
        trials=tuple(t.strip() for t in args.trials.split(";") if t.strip()),
        notes=args.notes,
    )
    n = len(cols["frame"])
    missing = int((cols["marker_n"] == 0).sum())
    print(f"{out}: {n} frames, marker absent in {missing}", file=sys.stderr)
    sys.stdout.write(format_runs_row(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
