# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""Track a video into the frame table of data/schema.md §3.

Two modes (§3.3):

- ``stand``: the wheel is fixed in the picture. The wheel disk and the sync
  LED are located from the key frames, as the white marker sweeps a disk and
  the LED's red-minus-green value changes, then every frame is measured inside
  fixed regions.
- ``floor``: the robot drives through the picture. The hub region and the
  marker region are disks around the previous frame's hub, and the LED grid is
  centred on the reddest spot near the hub.

Adapted from the six-motor campaign tracker (``track_all.py``), with the
schema's rules: no quality threshold, image moments with covariances, and
full-frame coordinates.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]

from duckiebot_analysis.data.frames import (
    LED_COLUMNS,
    SCHEMA_VERSION,
    SPOKE_COLUMNS,
)
from duckiebot_analysis.tracking.blobs import (
    LED_BLOCK_PX,
    LED_SIGNAL_TEXT,
    LED_SPACING_PX,
    ColourRule,
    blob_moments,
    disk_mask,
    led_channels,
    led_grid,
    resolve_colour_rule,
)
from duckiebot_analysis.tracking.spokes import (
    N_SPOKES,
    RING_FRACTION,
    SpokeMeasurement,
    measure_spokes,
)
from duckiebot_analysis.tracking.video import (
    VideoInfo,
    decode_frames,
    ffmpeg_version,
    probe_video,
    sha256_file,
)

# Stand regions as multiples of the marker sweep radius, the 99th percentile
# of the distance of the sweep pixels from their mean. The reference tracker
# took 0.96 of it for the marker, which cut the outer edge of the marker off
# and pulled the centroid inward; 1.1 keeps the whole marker. Its hub region,
# 0.62 of it, is kept.
MARKER_REGION_FACTOR = 1.1
HUB_REGION_FACTOR = 0.62
# Key-frame location at half size (reference tracker values). A pixel belongs
# to the sweep when the marker covers it in a fraction of the key frames
# between the bounds; the lower bound drops to 1.5 frames for short clips.
LOCATE_SCALE = 2
SWEEP_MIN_FRACTION = 0.03
SWEEP_MAX_FRACTION = 0.7
# The LED is searched outside this multiple of the sweep radius.
LED_EXCLUSION_FACTOR = 1.12 * 1.15
# Half the extent of the LED grid from its centre to the outer pixel edge.
LED_GRID_HALF_PX = 2 * LED_SPACING_PX + LED_BLOCK_PX / 2
# Floor defaults as multiples of the hub blob's equivalent radius in the
# first frame, when no radius is given.
FLOOR_HUB_FACTOR = 2.0
FLOOR_MARKER_FACTOR = 2.5
FLOOR_LED_FACTOR = 3.0  # LED search radius as a multiple of the marker radius


@dataclass(frozen=True)
class TrackConfig:
    """Settings of one tracking run.

    Attributes:
        mode: ``"stand"`` or ``"floor"``.
        camera: Phone model and recording mode, for the metadata.
        marker_colour: Preset name or rule string of the marker pixels.
        hub_colour: Preset name or rule string of the hub pixels.
        marker_region: Stand only: ``(cx, cy, r)`` in px; located when None.
        hub_region: Stand only: ``(cx, cy, r)`` in px; located when None.
        led: Whether the run has a sync LED; without it the LED columns are null.
        led_xy: Stand only: LED grid centre in px; located when None.
        floor_hub_radius: Floor only: hub region radius in px.
        floor_marker_radius: Floor only: marker region radius in px.
        floor_led_radius: Floor only: LED search radius around the hub in px.
        tracker_script: Script path recorded in the metadata.
        led_colour: Sync LED colour, ``red`` or ``green`` (blobs.LED_SIGNALS).
    """

    mode: str
    camera: str
    marker_colour: str = "white"
    hub_colour: str = "yellow"
    marker_region: tuple[float, float, float] | None = None
    hub_region: tuple[float, float, float] | None = None
    led: bool = True
    led_xy: tuple[float, float] | None = None
    floor_hub_radius: float | None = None
    floor_marker_radius: float | None = None
    floor_led_radius: float | None = None
    tracker_script: str = "scripts/track_video.py"
    led_colour: str = "red"


# ---------------------------------------------------------------- helpers


def _snap_led(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    """Put the grid centre between pixels, with the whole grid in the frame."""
    lo = LED_GRID_HALF_PX - 0.5
    x = min(max(math.floor(x) + 0.5, lo), w - 1 - lo)
    y = min(max(math.floor(y) + 0.5, lo), h - 1 - lo)
    return x, y


def _tracker_commit() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],  # noqa: S603,S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--", "."],  # noqa: S603,S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def _empty_columns(n: int) -> dict[str, np.ndarray]:
    cols: dict[str, np.ndarray] = {
        "marker_n": np.zeros(n, np.uint32),
        "hub_n": np.zeros(n, np.uint32),
    }
    for name in ("marker_x", "marker_y", "hub_x", "hub_y", "led_x", "led_y"):
        cols[name] = np.full(n, np.nan, np.float32)
    for name in (
        "marker_cxx",
        "marker_cyy",
        "marker_cxy",
        "hub_cxx",
        "hub_cyy",
        "hub_cxy",
    ):
        cols[name] = np.full(n, np.nan, np.float16)
    return cols


def _store(cols: dict[str, np.ndarray], prefix: str, i: int, m: Any) -> None:
    cols[f"{prefix}_n"][i] = m.n
    cols[f"{prefix}_x"][i] = m.x
    cols[f"{prefix}_y"][i] = m.y
    cols[f"{prefix}_cxx"][i] = m.cxx
    cols[f"{prefix}_cyy"][i] = m.cyy
    cols[f"{prefix}_cxy"][i] = m.cxy


def _finish(
    rows: list[dict[str, Any]], pts: list[int], leds: list[np.ndarray | None]
) -> dict[str, np.ndarray]:
    n = len(rows)
    cols = _empty_columns(n)
    for i, row in enumerate(rows):
        for prefix in ("marker", "hub"):
            _store(cols, prefix, i, row[prefix])
        cols["led_x"][i], cols["led_y"][i] = row.get("led_xy", (np.nan, np.nan))
    cols["frame"] = np.arange(n, dtype=np.uint32)
    cols["pts"] = np.asarray(pts, dtype=np.int64)
    if all(v is not None for v in leds):
        grid = np.array(leds, np.uint8).reshape(n, 25)
    else:
        grid = np.full((n, 25), np.nan, np.float32)
        for i, v in enumerate(leds):
            if v is not None:
                grid[i] = v
    for k, name in enumerate(LED_COLUMNS):
        cols[name] = grid[:, k].copy()
    return cols


# ---------------------------------------------------------------- stand


def locate_stand(
    path: Path,
    info: VideoInfo,
    marker: ColourRule,
    find_led: bool,
    led_colour: str = "red",
) -> dict[str, Any]:
    """Locate the marker sweep and the LED from the key frames at half size.

    Returns:
        ``{"centre": (cx, cy), "radius": r99, "led": (x, y) or None,
        "n_key": n}`` in full-frame px, where ``r99`` is the 99th percentile
        of the sweep pixels' distance from their mean.

    Raises:
        RuntimeError: With fewer than five key frames, or no moving marker.
    """
    s = LOCATE_SCALE
    nk = 0
    h, w = info.height // s, info.width // s
    on, off = led_channels(led_colour)
    wsum, rsum, rsq = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))
    for _, f in decode_frames(path, info, scale=s, key_only=True):
        a = f.astype(np.int16)
        rg = (a[..., on] - a[..., off]).astype(np.float64)
        wsum += marker.mask(f)
        rsum += rg
        rsq += rg * rg
        nk += 1
    if nk < 5:
        raise RuntimeError(f"only {nk} key frames; give the regions explicitly")
    freq = wsum / nk
    lo = min(SWEEP_MIN_FRACTION, 1.5 / nk)
    sweep = (freq >= lo) & (freq < SWEEP_MAX_FRACTION)
    sweep = ndimage.binary_closing(
        ndimage.binary_opening(sweep, iterations=1), iterations=4
    )
    lab, n = ndimage.label(sweep)
    if n == 0:
        raise RuntimeError("no moving marker found; give the regions explicitly")
    sizes = ndimage.sum(sweep, lab, range(1, n + 1))
    yy, xx = np.nonzero(lab == 1 + int(np.argmax(sizes)))
    cx, cy = xx.mean(), yy.mean()
    r99 = float(np.percentile(np.hypot(xx - cx, yy - cy), 99))
    led = None
    if find_led:
        rgsd = np.sqrt(np.maximum(rsq / nk - (rsum / nk) ** 2, 0))
        sd = ndimage.gaussian_filter(rgsd, 2)
        loc = sd - ndimage.uniform_filter(sd, 41)
        hh, ww = loc.shape
        gy, gx = np.mgrid[0:hh, 0:ww]
        loc[np.hypot(gx - cx, gy - cy) < r99 * LED_EXCLUSION_FACTOR] = -np.inf
        edge = int(math.ceil(LED_GRID_HALF_PX / s))
        loc[:edge, :] = loc[-edge:, :] = -np.inf
        loc[:, :edge] = loc[:, -edge:] = -np.inf
        j = np.unravel_index(np.argmax(loc), loc.shape)
        led = (j[1] * s + (s - 1) / 2, j[0] * s + (s - 1) / 2)
    # A half-size pixel i covers full pixels s*i ... s*i + s - 1.
    centre = (float(cx) * s + (s - 1) / 2, float(cy) * s + (s - 1) / 2)
    return {"centre": centre, "radius": r99 * s, "led": led, "n_key": nk}


def stand_crop(
    mreg: tuple[float, float, float],
    hreg: tuple[float, float, float],
    led_xy: tuple[float, float] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Return the crop (w, h, x0, y0) holding both regions and the LED grid.

    The offsets are even for the chroma. The decoded pixels depend on the crop,
    so anything that reproduces the tracker's pixel counts must decode with it.

    Raises:
        ValueError: If the regions lie outside the frame.
    """
    xs = [mreg[0] - mreg[2], hreg[0] - hreg[2], mreg[0] + mreg[2], hreg[0] + hreg[2]]
    ys = [mreg[1] - mreg[2], hreg[1] - hreg[2], mreg[1] + mreg[2], hreg[1] + hreg[2]]
    if led_xy is not None:
        xs += [led_xy[0] - LED_GRID_HALF_PX, led_xy[0] + LED_GRID_HALF_PX]
        ys += [led_xy[1] - LED_GRID_HALF_PX, led_xy[1] + LED_GRID_HALF_PX]
    x0 = max(0, int(math.floor(min(xs))) - 2) // 2 * 2
    y0 = max(0, int(math.floor(min(ys))) - 2) // 2 * 2
    x1 = min(width, int(math.ceil(max(xs))) + 2)
    y1 = min(height, int(math.ceil(max(ys))) + 2)
    cw, ch = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    if cw <= 0 or ch <= 0:
        raise ValueError("the search regions lie outside the frame")
    return cw, ch, x0, y0


def _track_stand(
    path: Path, info: VideoInfo, cfg: TrackConfig, marker: ColourRule, hub: ColourRule
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    need_led = cfg.led and cfg.led_xy is None
    located: dict[str, Any] | None = None
    mreg: tuple[float, float, float]
    hreg: tuple[float, float, float]
    if cfg.marker_region is None or cfg.hub_region is None or need_led:
        located = locate_stand(path, info, marker, need_led, cfg.led_colour)
    if cfg.marker_region is not None:
        mreg = (
            float(cfg.marker_region[0]),
            float(cfg.marker_region[1]),
            float(cfg.marker_region[2]),
        )
    else:
        assert located is not None
        mreg = (*located["centre"], float(MARKER_REGION_FACTOR * located["radius"]))
    if cfg.hub_region is not None:
        hreg = (
            float(cfg.hub_region[0]),
            float(cfg.hub_region[1]),
            float(cfg.hub_region[2]),
        )
    else:
        assert located is not None
        hreg = (*located["centre"], float(HUB_REGION_FACTOR * located["radius"]))
    led_xy = None
    if cfg.led:
        raw = cfg.led_xy if cfg.led_xy is not None else located["led"]  # type: ignore[index]
        led_xy = _snap_led(raw[0], raw[1], info.width, info.height)
    cw, ch, x0, y0 = stand_crop(mreg, hreg, led_xy, info.width, info.height)
    mdisk = disk_mask((ch, cw), *mreg, x0=x0, y0=y0)
    hdisk = disk_mask((ch, cw), *hreg, x0=x0, y0=y0)
    rows: list[dict[str, Any]] = []
    pts: list[int] = []
    leds: list[np.ndarray | None] = []
    for p, f in decode_frames(path, info, crop=(cw, ch, x0, y0)):
        row: dict[str, Any] = {
            "marker": blob_moments(marker.mask(f) & mdisk, x0, y0),
            "hub": blob_moments(hub.mask(f) & hdisk, x0, y0),
        }
        if led_xy is not None:
            row["led_xy"] = led_xy
            leds.append(
                led_grid(f, led_xy[0], led_xy[1], x0, y0, colour=cfg.led_colour)
            )
        else:
            leds.append(None)
        rows.append(row)
        pts.append(p)
    if led_xy is None:
        method = "no sync LED; LED columns null"
    elif cfg.led_xy is not None:
        method = "given on the command line, fixed for the run"
    else:
        method = (
            "maximum of the local temporal spread of "
            f"{LED_SIGNAL_TEXT[cfg.led_colour]} over the key "
            "frames, outside the wheel; fixed for the run"
        )
    meta = {
        "marker_region": dict(zip(("cx", "cy", "r"), mreg, strict=True)),
        "hub_region": dict(zip(("cx", "cy", "r"), hreg, strict=True)),
        "led_grid": {
            "block_px": LED_BLOCK_PX,
            "spacing_px": LED_SPACING_PX,
            "method": method,
            "signal": LED_SIGNAL_TEXT[cfg.led_colour],
        },
    }
    return _finish(rows, pts, leds), meta


# ---------------------------------------------------------------- floor


def _sub(f: np.ndarray, cx: float, cy: float, r: float) -> tuple[np.ndarray, int, int]:
    h, w = f.shape[:2]
    x0 = max(0, int(math.floor(cx - r)))
    y0 = max(0, int(math.floor(cy - r)))
    x1 = min(w, int(math.ceil(cx + r)) + 1)
    y1 = min(h, int(math.ceil(cy + r)) + 1)
    return f[y0 : max(y0, y1), x0 : max(x0, x1)], x0, y0


def _region_mask(
    f: np.ndarray, rule: ColourRule, cx: float, cy: float, r: float
) -> tuple[np.ndarray, int, int]:
    sub, x0, y0 = _sub(f, cx, cy, r)
    return rule.mask(sub) & disk_mask(sub.shape[:2], cx, cy, r, x0, y0), x0, y0


def _region_moments(
    f: np.ndarray, rule: ColourRule, cx: float, cy: float, r: float
) -> Any:
    mask, x0, y0 = _region_mask(f, rule, cx, cy, r)
    return blob_moments(mask, x0, y0)


def _largest_blob(f: np.ndarray, rule: ColourRule) -> tuple[float, float, float] | None:
    mask = rule.mask(f)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes))
    ys, xs = np.nonzero(lab == k + 1)
    return float(xs.mean()), float(ys.mean()), float(sizes[k])


def _track_floor(
    path: Path, info: VideoInfo, cfg: TrackConfig, marker: ColourRule, hub: ColourRule
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    r_hub, r_marker, r_led = (
        cfg.floor_hub_radius,
        cfg.floor_marker_radius,
        cfg.floor_led_radius,
    )
    rows: list[dict[str, Any]] = []
    pts: list[int] = []
    leds: list[np.ndarray | None] = []
    prev: tuple[float, float] | None = None
    spokes: list[SpokeMeasurement] = []
    nan_row = blob_moments(np.zeros((0, 0), bool))
    no_spokes = measure_spokes(np.zeros((0, 0), bool), np.nan, np.nan)
    for p, f in decode_frames(path, info):
        pts.append(p)
        centre = prev
        if centre is None:
            blob = _largest_blob(f, hub)
            if blob is not None:
                centre = blob[:2]
                rho = math.sqrt(blob[2] / math.pi)
                r_hub = r_hub or FLOOR_HUB_FACTOR * rho
                r_marker = r_marker or FLOOR_MARKER_FACTOR * rho
        if centre is None or r_hub is None or r_marker is None:
            rows.append({"marker": nan_row, "hub": nan_row})
            leds.append(None)
            spokes.append(no_spokes)
            continue
        hmask, hx0, hy0 = _region_mask(f, hub, *centre, r_hub)
        hm = blob_moments(hmask, hx0, hy0)
        # The hub pixels are the yellow wheel, whose spokes give its angle
        # where the tape is hidden; the centre is this frame's hub centroid.
        spokes.append(
            measure_spokes(hmask, hm.x, hm.y, hx0, hy0) if hm.n else no_spokes
        )
        row: dict[str, Any] = {
            "hub": hm,
            "marker": _region_moments(f, marker, *centre, r_marker),
        }
        leds.append(None)
        if cfg.led and hm.n:
            rl = r_led or FLOOR_LED_FACTOR * r_marker
            sub, x0, y0 = _sub(f, hm.x, hm.y, rl)
            a = sub.astype(np.float32)
            # The LED's own channel minus the mean of the other two.
            on = led_channels(cfg.led_colour)[0]
            rest = [c for c in range(3) if c != on]
            red = ndimage.uniform_filter(
                a[..., on] - (a[..., rest[0]] + a[..., rest[1]]) / 2, 5
            )
            inside = disk_mask(sub.shape[:2], hm.x, hm.y, rl, x0, y0)
            red[~inside] = -np.inf
            # A saturated LED is a plateau, whose first maximum lies at its
            # corner; take the centroid of the plateau around the maximum.
            j = np.unravel_index(np.argmax(red), red.shape)
            top = float(red[j])
            near_top = red >= top - 0.1 * (top - float(np.median(red[inside])))
            lab, _ = ndimage.label(near_top)
            ys, xs = np.nonzero(lab == lab[j])
            led_xy = _snap_led(xs.mean() + x0, ys.mean() + y0, info.width, info.height)
            row["led_xy"] = led_xy
            leds[-1] = led_grid(f, *led_xy, colour=cfg.led_colour)
        rows.append(row)
        prev = (hm.x, hm.y) if hm.n else None
    centre_rule = (
        "previous frame's hub; when the hub was not found in the previous frame, "
        "the centroid of the largest hub-coloured blob in the frame"
    )
    method = (
        "no sync LED; LED columns null"
        if not cfg.led
        else "per frame: centroid of the connected pixels within 10 % of the maximum "
        f"of {cfg.led_colour} minus the mean of the other two colours "
        "(5 px box mean), searched within "
        "search_radius_px of the hub"
    )
    meta: dict[str, Any] = {
        "marker_region": {"r": r_marker, "centre": centre_rule},
        "hub_region": {"r": r_hub, "centre": centre_rule},
        "led_grid": {
            "block_px": LED_BLOCK_PX,
            "spacing_px": LED_SPACING_PX,
            "method": method,
            "signal": LED_SIGNAL_TEXT[cfg.led_colour],
        },
    }
    if cfg.led:
        led_r = r_led or (FLOOR_LED_FACTOR * r_marker if r_marker else None)
        meta["led_grid"]["search_radius_px"] = led_r
    meta["spokes"] = {
        "n_spokes": N_SPOKES,
        "ring_fraction": RING_FRACTION,
        "method": "phase of the n-th angular harmonic of the hub pixels on a ring "
        "at ring_fraction of the wheel radius (98th percentile of the hub pixels' "
        "distance from the hub centroid), around the hub centroid",
    }
    cols = _finish(rows, pts, leds)
    fields = dict(zip(SPOKE_COLUMNS, ("angle", "angle_sd", "contrast"), strict=True))
    for name, field in fields.items():
        cols[name] = np.array([getattr(m, field) for m in spokes], np.float32)
    return cols, meta


# ---------------------------------------------------------------- entry point


def track_video(
    video_path: str | Path, config: TrackConfig
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Track a video into the frame-table columns and metadata of §3.

    Args:
        video_path: The video file; its stem becomes the run identifier.
        config: Mode, colour rules, regions, and camera description.

    Returns:
        Columns for :func:`duckiebot_analysis.data.frames.write_frames`,
        and the §3.3 metadata.

    Raises:
        ValueError: If the mode is unknown or a colour rule is invalid.
        RuntimeError: If decoding fails or the wheel cannot be located.
    """
    path = Path(video_path)
    if config.mode not in ("stand", "floor"):
        raise ValueError(f"mode {config.mode!r} must be 'stand' or 'floor'")
    marker = resolve_colour_rule(config.marker_colour)
    hub = resolve_colour_rule(config.hub_colour)
    info = probe_video(path)
    track = _track_stand if config.mode == "stand" else _track_floor
    cols, region_meta = track(path, info, config, marker, hub)
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": path.stem,
        "video_file": path.name,
        "video_sha256": sha256_file(path),
        "video_bytes": path.stat().st_size,
        "width": info.width,
        "height": info.height,
        "rotation": info.rotation,
        "time_base": info.time_base,
        "fps_nominal": info.fps_nominal,
        "camera": config.camera,
        "mode": config.mode,
        "marker_colour": str(marker),
        "hub_colour": str(hub),
        **region_meta,
        "decoder": ffmpeg_version(),
        "tracker": {"script": config.tracker_script, **_tracker_commit()},
        "tracked": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return cols, meta
