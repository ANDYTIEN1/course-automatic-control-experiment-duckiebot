# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Overlay of the tracker's output on video frames picked by angle uncertainty.

Fig. S2 of the CSME 2026 supplement. scripts/select_wheel_frames.py runs the
tracker and the angle computation on a video and stores the frames and numbers
drawn here, because the videos are not stored in this repository. The pixels are
reclassified with the tracker's own rule and disk functions, which a test checks
against the tracker's output for these frames.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("pdf")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle, Wedge  # noqa: E402

from duckiebot_analysis.tracking.blobs import (  # noqa: E402
    LED_BLOCK_PX,
    LED_SPACING_PX,
    BlobMoments,
    ColourRule,
    blob_moments,
    disk_mask,
    resolve_colour_rule,
)
from duckiebot_analysis.visualization.case_study_figures import (  # noqa: E402
    BLUE,
    INK2,
    ORANGE,
)

# Default folder, relative to the directory the scripts run from, holding the
# frame crops and tables that select_wheel_frames.py writes.
WHEEL_FRAMES_DIR = Path("build") / "wheel_frames"


class WheelFrameFiles(NamedTuple):
    """Files that select_wheel_frames.py writes and fig_wheel_overlay reads."""

    frames_dir: Path
    picks_table: Path
    trace_table: Path
    regions_file: Path


def wheel_frame_files(folder: Path = WHEEL_FRAMES_DIR) -> WheelFrameFiles:
    """Return the paths of the wheel-frame files in ``folder``."""
    return WheelFrameFiles(
        folder,
        folder / "wheel_frames_example.csv",
        folder / "wheel_frames_example_trace.csv",
        folder / "wheel_frames_example.json",
    )


TEXT_WIDTH_IN = 17.0 / 2.54  # CSME text block: 21 - 2 - 2 cm
PANELS = (
    ("best", "Smallest"),
    ("median", "Median"),
    ("mean", "Mean"),
    ("worst", "Largest"),
)
# The covariances and the wedge are too small to see at their true size.
ENLARGE = 30.0
BACKGROUND_GAIN = 0.55  # Grey level of the unused pixels, so the used ones show.
MARK = "white"


class FrameRegions(NamedTuple):
    """Pixels the tracker uses in one frame, and their moments."""

    marker_mask: np.ndarray
    hub_mask: np.ndarray
    marker: BlobMoments
    hub: BlobMoments


def load_frame(path: Path) -> np.ndarray:
    """Read an image as the (rows, columns, 3) uint8 RGB array the tracker takes."""
    image = np.asarray(mpimg.imread(path))
    if image.dtype != np.uint8:
        image = np.rint(image * 255).astype(np.uint8)
    return image[..., :3]


def frame_regions(
    frame: np.ndarray,
    centre: tuple[float, float],
    marker_radius: float,
    hub_radius: float,
    marker_rule: ColourRule,
    hub_rule: ColourRule,
    x0: int = 0,
    y0: int = 0,
    hub_centre: tuple[float, float] | None = None,
) -> FrameRegions:
    """Classify the pixels of one frame as ``_track_stand`` does.

    Both regions are full disks, so a marker pixel inside the hub disk still
    counts, and a hub pixel the marker covers is lost.

    Args:
        frame: The frame or a crop of it.
        centre: Centre of the marker disk, full-frame px.
        marker_radius: Radius of the marker disk in px.
        hub_radius: Radius of the hub disk in px.
        marker_rule: Colour rule of the marker pixels.
        hub_rule: Colour rule of the hub pixels.
        x0: Full-frame column of the crop's first column.
        y0: Full-frame row of the crop's first row.
        hub_centre: Centre of the hub disk; ``centre`` when None.

    Raises:
        ValueError: If the frame is not uint8 RGB or a radius is not positive.
    """
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB, got {frame.dtype} {frame.shape}")
    if not (marker_radius > 0 and hub_radius > 0):
        raise ValueError(f"radii {marker_radius} and {hub_radius} must be positive")
    shape = frame.shape[:2]
    hub_at = centre if hub_centre is None else hub_centre
    marker_mask = marker_rule.mask(frame) & disk_mask(
        shape, *centre, marker_radius, x0=x0, y0=y0
    )
    hub_mask = hub_rule.mask(frame) & disk_mask(
        shape, *hub_at, hub_radius, x0=x0, y0=y0
    )
    return FrameRegions(
        marker_mask,
        hub_mask,
        blob_moments(marker_mask, x0, y0),
        blob_moments(hub_mask, x0, y0),
    )


def led_block_corners(
    led_x: float, led_y: float, block: int, spacing: int
) -> list[tuple[int, int]]:
    """Return (left, top) of the 25 blocks ``led_grid`` samples, in its order."""
    half = (block - 1) / 2
    return [
        (
            int(round(led_x + spacing * (j - 2) - half)),
            int(round(led_y + spacing * (i - 2) - half)),
        )
        for i in range(5)
        for j in range(5)
    ]


def check_rotation(rotation: int) -> int:
    """Return the video's rotation if the upright panels can handle it.

    Raises:
        ValueError: If it is not the -90 degrees of the phone, the only rotation
            that the panels' coordinates assume.
    """
    if rotation != -90:
        raise ValueError(f"rotation {rotation} is not the -90 the panels assume")
    return rotation


def _shaded(frame: np.ndarray, regions: FrameRegions) -> np.ndarray:
    grey = BACKGROUND_GAIN * frame.mean(axis=-1, keepdims=True)
    shaded = np.repeat(grey, 3, axis=-1).astype(np.uint8)
    for mask, colour in ((regions.hub_mask, ORANGE), (regions.marker_mask, BLUE)):
        shaded[mask] = np.rint(np.array(mcolors.to_rgb(colour)) * 255)
    return shaded


def _covariance_patch(m: BlobMoments) -> Ellipse:
    """Return the enlarged one-sd ellipse of a centroid, Sigma = C / n (k = 1).

    The axes are those of the upright picture, x_up = -y and y_up = x, up to the
    shift that imshow's coordinates add.
    """
    cov = np.array([[m.cyy, -m.cxy], [-m.cxy, m.cxx]]) / m.n
    values, vectors = np.linalg.eigh(cov)
    width, height = 2 * ENLARGE * np.sqrt(values[::-1])
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    return Ellipse((0, 0), width, height, angle=angle, fill=False, ec=MARK, lw=0.7)


def _panel(
    ax: plt.Axes,
    frame: np.ndarray,
    row: dict[str, str],
    meta: dict,
    title: str,
    labelled: bool,
) -> None:
    crop = meta["crop"]
    x0, y0 = crop["x0"], crop["y0"]
    marker_region, hub_region = meta["marker_region"], meta["hub_region"]
    regions = frame_regions(
        frame,
        (marker_region["cx"], marker_region["cy"]),
        marker_region["r"],
        hub_region["r"],
        resolve_colour_rule(meta["marker_colour"]),
        resolve_colour_rule(meta["hub_colour"]),
        x0=x0,
        y0=y0,
        hub_centre=(hub_region["cx"], hub_region["cy"]),
    )
    if regions.marker.n == 0 or regions.hub.n == 0:
        raise ValueError(f"frame {row['frame']} has no marker or no hub pixels")

    # The phone stores the frames on their side: rotate a stored (x, y) to the
    # upright picture, in the coordinates of the crop.
    height = frame.shape[0]

    def upright(x: float, y: float) -> tuple[float, float]:
        return height - 1 - (y - y0), x - x0

    ax.imshow(np.rot90(_shaded(frame, regions), k=-1), interpolation="none")
    ax.set_axis_off()
    centre = upright(marker_region["cx"], marker_region["cy"])
    ax.add_patch(Circle(centre, marker_region["r"], fill=False, ec=MARK, lw=0.8))
    ax.add_patch(
        Circle(centre, hub_region["r"], fill=False, ec=MARK, lw=0.8, ls=(0, (4, 2)))
    )
    ax.plot(*centre, "+", color=MARK, ms=6, mew=0.8)
    for left, top in led_block_corners(
        meta["led"]["x"], meta["led"]["y"], LED_BLOCK_PX, LED_SPACING_PX
    ):
        # A block's pixels run from left to left + block; upright, y runs across.
        corner = upright(left - 0.5, top - 0.5 + LED_BLOCK_PX)
        ax.add_patch(
            Rectangle(corner, LED_BLOCK_PX, LED_BLOCK_PX, fill=False, ec=MARK, lw=0.3)
        )
    m = upright(regions.marker.x, regions.marker.y)
    h = upright(regions.hub.x, regions.hub.y)
    ax.annotate(
        "", xy=m, xytext=h, arrowprops={"arrowstyle": "-|>", "color": MARK, "lw": 0.8}
    )
    # Angle uncertainty, enlarged, as a wedge about r_i from h_i.
    sd = np.degrees(float(row["theta_sd_rad"])) * ENLARGE
    direction = np.degrees(np.arctan2(m[1] - h[1], m[0] - h[0]))
    length = float(np.hypot(m[0] - h[0], m[1] - h[1]))
    ax.add_patch(
        Wedge(h, length, direction - sd, direction + sd, fc=MARK, ec="none", alpha=0.45)
    )
    for moments, point in ((regions.marker, m), (regions.hub, h)):
        patch = _covariance_patch(moments)
        patch.set_center(point)
        ax.add_patch(patch)
        ax.plot(*point, "o", ms=2.5, mfc="black", mec=MARK, mew=0.5)
    if labelled:
        diagonal_m = marker_region["r"] / np.sqrt(2)
        diagonal_h = hub_region["r"] / np.sqrt(2)
        for text, xy, offset in (
            (r"$\mathbf{m}_i$", m, (16, -4)),
            (r"$\mathbf{h}_i$", h, (-16, 8)),
            (
                "Marker disk",
                (centre[0] + diagonal_m, centre[1] - diagonal_m),
                (-26, 12),
            ),
            ("Hub disk", (centre[0] - diagonal_h, centre[1] - diagonal_h), (-34, 16)),
            ("LED grid", upright(meta["led"]["x"], meta["led"]["y"]), (-8, 20)),
        ):
            ax.annotate(
                text,
                xy,
                xytext=offset,
                textcoords="offset points",
                color=MARK,
                fontsize=7,
                arrowprops={"arrowstyle": "-", "color": MARK, "lw": 0.5},
            )
    ax.set_title(title, loc="left", fontsize=8, pad=2)


def _trace(ax: plt.Axes, picks: list[dict[str, str]], trace_table: Path) -> None:
    with trace_table.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    wrapped = np.degrees([float(r["theta_mod_rad"]) for r in rows])
    for name, label, colour in (
        ("theta_sd_rad", r"$\sigma_{\theta,i}$", "black"),
        ("hub_sd_rad", "hub part", ORANGE),
        ("marker_sd_rad", "marker part", BLUE),
    ):
        values = 1e3 * np.array([float(r[name]) for r in rows])
        ax.plot(
            wrapped, values, ".", ms=0.7, color=colour, label=label, rasterized=True
        )
    by_frame = {int(r["frame"]): r for r in rows}
    for letter, pick in zip("abcd", picks, strict=True):
        hit = by_frame[int(pick["frame"])]
        x = np.degrees(float(hit["theta_mod_rad"]))
        y = 1e3 * float(hit["theta_sd_rad"])
        ax.plot(x, y, "o", ms=4, mfc="none", mec=INK2, mew=0.8)
        ax.annotate(
            letter,
            (x, y),
            xytext=(0, -11 if letter == "c" else 5),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_ylim(0, 6)
    ax.set_xlabel(r"Wheel angle $\theta_i$ modulo a turn (degrees)")
    ax.set_ylabel(r"Uncertainty (mrad)")
    ax.legend(loc="center right", markerscale=8, frameon=False, fontsize=7)
    ax.set_title("(e) All frames of the video", loc="left", fontsize=8, pad=2)


def fig_wheel_overlay(path: Path, folder: Path = WHEEL_FRAMES_DIR) -> None:
    """Draw the four picked frames and the uncertainty of every frame.

    Args:
        path: The PDF to write.
        folder: The folder select_wheel_frames.py wrote its files into.

    Raises:
        ValueError: If a stored frame has no marker or no hub pixels.
        FileNotFoundError: If select_wheel_frames.py has not been run.
    """
    files = wheel_frame_files(folder)
    meta = json.loads(files.regions_file.read_text(encoding="utf-8"))
    check_rotation(meta["rotation"])
    with files.picks_table.open(encoding="utf-8") as fh:
        picks = list(csv.DictReader(fh))
    by_kind = {row["kind"]: row for row in picks}
    crop = meta["crop"]
    ratio = (crop["x1"] - crop["x0"]) / (crop["y1"] - crop["y0"])
    panel_height = TEXT_WIDTH_IN / 2 * ratio
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2 * panel_height + 2.0))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(panel_height, panel_height, 1.5),
        hspace=0.1,
        wspace=0.02,
        left=0.01,
        right=0.99,
        top=0.98,
        bottom=0.075,
    )
    for k, (kind, name) in enumerate(PANELS):
        row = by_kind[kind]
        frame = load_frame(files.frames_dir / f"video-frame-wheel-{kind}.png")
        sd = float(row["theta_sd_rad"])
        title = (
            f"({'abcd'[k]}) {name} uncertainty: "
            rf"$\sigma_{{\theta,i}}$ = {1e3 * sd:.2f} mrad"
            rf" = {np.degrees(sd):.3f}$\degree$"
        )
        _panel(fig.add_subplot(grid[k // 2, k % 2]), frame, row, meta, title, k == 0)
    # The trace has tick labels, which need room the image panels do not.
    trace = fig.add_subplot(grid[2, :])
    box = trace.get_position()
    trace.set_position((0.07, box.y0, 0.91, box.height))
    _trace(trace, [by_kind[kind] for kind, _ in PANELS], files.trace_table)
    fig.savefig(path, dpi=300)
    plt.close(fig)
