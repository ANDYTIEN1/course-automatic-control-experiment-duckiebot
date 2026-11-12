#!/usr/bin/env python3
# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Pick the frames of a stand video by angle uncertainty, for Fig. S2 of the CSME
2026 supplement.

Runs the tracker and the angle computation on the video, then stores the crops
of the frames of smallest, median, mean and largest uncertainty, and the numbers
the figure draws. The videos are not stored in this repository
(README.md, Data), so the figure is redrawn from these files alone;
build/wheel_frames/wheel_frames_example.json names the video by its checksum.

Usage:
    python scripts/select_wheel_frames.py VIDEO.mp4 \\
        --turns ccw --marker-colour white --led-colour red

The article's Fig. S2 video has a white marker and a red LED; the defaults
(blue marker, green LED) fit the course kit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from duckiebot_analysis.analysis.angle import compute_angle
from duckiebot_analysis.analysis.frame_uncertainty import (
    angle_sd_parts,
    pick_frames,
)
from duckiebot_analysis.tracking.track import (
    TrackConfig,
    stand_crop,
    track_video,
)
from duckiebot_analysis.tracking.video import decode_frames, probe_video
from duckiebot_analysis.visualization.wheel_overlay_figure import (
    WHEEL_FRAMES_DIR,
    check_rotation,
    wheel_frame_files,
)

CAMERA = "phone, 1280x720 H.264 240 fps"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument(
        "--turns",
        required=True,
        choices=("cw", "ccw"),
        help="positive_duty_turns of the motor's setup.toml",
    )
    ap.add_argument("--marker-colour", default="blue", help="marker preset or rule")
    ap.add_argument("--led-colour", choices=("red", "green"), default="green")
    ap.add_argument(
        "--out",
        type=Path,
        default=WHEEL_FRAMES_DIR,
        help=f"folder for the frame crops and tables (default: {WHEEL_FRAMES_DIR})",
    )
    args = ap.parse_args(argv)
    frames_dir, picks_table, trace_table, regions_file = wheel_frame_files(args.out)
    info = probe_video(args.video)
    check_rotation(info.rotation)
    cols, meta = track_video(
        args.video,
        TrackConfig(
            mode="stand",
            camera=CAMERA,
            marker_colour=args.marker_colour,
            led_colour=args.led_colour,
        ),
    )
    result = compute_angle(
        cols,
        float(Fraction(meta["time_base"])),
        mode="stand",
        positive_duty_turns=args.turns,
    )
    marker_sd, hub_sd = angle_sd_parts(cols, result)
    picks = pick_frames(result.theta_sd, result.usable)
    led = (float(cols["led_x"][0]), float(cols["led_y"][0]))
    regions = [
        (r["cx"], r["cy"], r["r"]) for r in (meta["marker_region"], meta["hub_region"])
    ]
    size = stand_crop(*regions, led, info.width, info.height)
    width, height, x0, y0 = size
    crop = {"x0": x0, "y0": y0, "x1": x0 + width, "y1": y0 + height}
    frames_dir.mkdir(parents=True, exist_ok=True)
    last = max(picks.values())
    for index, (_, frame) in enumerate(decode_frames(args.video, info, crop=size)):
        for kind, wanted in picks.items():
            if index == wanted:
                mpimg.imsave(frames_dir / f"video-frame-wheel-{kind}.png", frame)
        if index >= last:
            break
    ellipse = result.ellipse
    assert ellipse is not None  # pick_frames found usable frames
    regions_file.write_text(
        json.dumps(
            {
                "video_file": meta["video_file"],
                "video_sha256": meta["video_sha256"],
                "rotation": meta["rotation"],
                "positive_duty_turns": args.turns,
                "height": meta["height"],
                "marker_colour": meta["marker_colour"],
                "hub_colour": meta["hub_colour"],
                "marker_region": meta["marker_region"],
                "hub_region": meta["hub_region"],
                "led": {"x": led[0], "y": led[1]},
                "ellipse_centre": list(map(float, ellipse.centre)),
                "ellipse_transform": np.asarray(ellipse.transform).tolist(),
                "semi_major_px": float(ellipse.semi_major_px),
                "semi_minor_px": float(ellipse.semi_minor_px),
                "k": result.k,
                "crop": crop,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with picks_table.open("w", newline="", encoding="utf-8") as fh:
        names = ["kind", "frame", "t_s", "theta_sd_rad", "marker_sd_rad", "hub_sd_rad"]
        moments = ("n", "x", "y", "cxx", "cyy", "cxy")
        names += [f"{p}_{m}" for p in ("marker", "hub") for m in moments]
        writer = csv.DictWriter(fh, names, lineterminator="\n")
        writer.writeheader()
        for kind, i in picks.items():
            row = {
                "kind": kind,
                "frame": i,
                "t_s": f"{result.t[i]:.6f}",
                "theta_sd_rad": f"{result.theta_sd[i]:.6e}",
                "marker_sd_rad": f"{marker_sd[i]:.6e}",
                "hub_sd_rad": f"{hub_sd[i]:.6e}",
            }
            for p in ("marker", "hub"):
                for m in moments:
                    row[f"{p}_{m}"] = cols[f"{p}_{m}"][i]
            writer.writerow(row)
    with trace_table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            [
                "frame",
                "t_s",
                "theta_mod_rad",
                "theta_sd_rad",
                "marker_sd_rad",
                "hub_sd_rad",
            ]
        )
        wrapped = np.mod(result.theta, 2 * np.pi)
        for i in np.nonzero(result.usable)[0]:
            writer.writerow(
                [
                    i,
                    f"{result.t[i]:.6f}",
                    f"{wrapped[i]:.5f}",
                    f"{result.theta_sd[i]:.6e}",
                    f"{marker_sd[i]:.6e}",
                    f"{hub_sd[i]:.6e}",
                ]
            )
    print(picks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
