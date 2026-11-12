# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Frame time, usable frames, and the marker relative to the hub.

Implements data/schema.md §9.1-§9.4: frame time and dropped frames (§9.1),
usable frames (§9.2), centroid covariance (§9.3), and the marker position
relative to the hub (§9.4). The input is the frame table of §3.1 as a mapping
from column name to a numpy array, with NaN where the column is null.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from duckiebot_analysis.analysis._types import (
    BoolArray,
    FloatArray,
    Frames,
    Mode,
)

N_MIN = 25
N_HUB_MIN = 50
GAP_FACTOR = 1.5
MAD_TO_SD = 1.482602218505602


@dataclass(frozen=True)
class TimeAxis:
    """Frame times of §9.1.

    Attributes:
        t: Time from the first frame in s.
        dropped: True for a frame preceded by a gap above 1.5 median intervals.
        n_dropped: Estimated number of frames missing in those gaps.
        median_dt: Median frame interval in s, NaN with fewer than two frames.
    """

    t: FloatArray
    dropped: BoolArray
    n_dropped: int
    median_dt: float


@dataclass(frozen=True)
class HubRelative:
    """Marker position relative to the hub and its covariance (§9.4).

    Attributes:
        r: Marker minus hub position in px, shape (N, 2), NaN where unusable.
        cov: Covariance Σᵣ of r in px², shape (N, 2, 2).
        usable: Frames where r is valid.
        hub_moved: Stand frames where the hub left its median position.
        hub_imputed: Stand frames where the median hub position was used.
        hub_cov: The hub's part of cov, with the spread of the found hubs on an
            imputed frame.
    """

    r: FloatArray
    cov: FloatArray
    usable: BoolArray
    hub_moved: BoolArray
    hub_imputed: BoolArray
    hub_cov: FloatArray


def _col(frames: Frames, key: str) -> FloatArray:
    return np.asarray(frames[key], dtype=np.float64)


def median_interval(t: npt.ArrayLike) -> float:
    """Return the median of the finite, positive frame intervals, NaN if none."""
    dt = np.diff(np.asarray(t, dtype=np.float64))
    finite = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(finite)) if finite.size else np.nan


def time_axis(pts: npt.ArrayLike, time_base: float) -> TimeAxis:
    """Frame times and dropped frames (§9.1).

    Args:
        pts: Presentation timestamps in units of the time base.
        time_base: Seconds per pts unit.

    Returns:
        The frame times, the dropped-frame flags, and the number dropped.
    """
    p = np.asarray(pts)
    if p.size == 0:
        return TimeAxis(np.zeros(0), np.zeros(0, bool), 0, np.nan)
    # Subtract in integers, so a large pts loses no precision before scaling.
    if np.issubdtype(p.dtype, np.integer):
        t = (p - p[0]).astype(np.float64) * time_base
    else:
        p = p.astype(np.float64)
        t = (p - p[0]) * time_base
    dt = np.diff(t)
    dropped = np.zeros(t.size, bool)
    median_dt = median_interval(t)
    if not np.isfinite(median_dt):
        return TimeAxis(t, dropped, 0, np.nan)
    with np.errstate(invalid="ignore"):
        gap = dt > GAP_FACTOR * median_dt
    dropped[1:] = gap
    n_dropped = int(np.sum(np.round(dt[gap] / median_dt) - 1))
    return TimeAxis(t, dropped, n_dropped, median_dt)


def _found(n: FloatArray, x: FloatArray, y: FloatArray, threshold: float) -> BoolArray:
    with np.errstate(invalid="ignore"):
        return (n >= threshold) & np.isfinite(x) & np.isfinite(y)


def usable_mask(
    frames: Frames, *, n_min: float = N_MIN, n_hub_min: float = N_HUB_MIN
) -> BoolArray:
    """Frames where both the marker and the hub were found (§9.2).

    Args:
        frames: Frame table columns; uses marker_n, marker_x, marker_y, hub_n,
            hub_x, hub_y.
        n_min: Minimum marker pixel count.
        n_hub_min: Minimum hub pixel count.

    Returns:
        Boolean mask of usable frames.
    """
    marker = _found(
        _col(frames, "marker_n"),
        _col(frames, "marker_x"),
        _col(frames, "marker_y"),
        n_min,
    )
    hub = _found(
        _col(frames, "hub_n"), _col(frames, "hub_x"), _col(frames, "hub_y"), n_hub_min
    )
    return marker & hub


def centroid_covariance(
    n: npt.ArrayLike,
    cxx: npt.ArrayLike,
    cyy: npt.ArrayLike,
    cxy: npt.ArrayLike,
    k: float = 1.0,
) -> FloatArray:
    """Covariance of a centroid, Σ = k² C / n (§9.3).

    The stored covariance is float16, whose rounding can make C slightly
    indefinite; cxy is clipped to |cxy| ≤ (cxx cyy)^½ so Σ is positive
    semidefinite.

    Args:
        n: Pixel counts.
        cxx: Pixel position variance along x in px².
        cyy: Pixel position variance along y in px².
        cxy: Pixel position covariance in px².
        k: Calibration factor k ≥ 1.

    Returns:
        Array of shape (N, 2, 2) in px², NaN where an input is missing.
    """
    n_ = np.asarray(n, dtype=np.float64)
    xx = np.asarray(cxx, dtype=np.float64)
    yy = np.asarray(cyy, dtype=np.float64)
    xy = np.asarray(cxy, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        bound = np.sqrt(np.clip(xx, 0, None) * np.clip(yy, 0, None))
        xy = np.clip(xy, -bound, bound)
        scale = k**2 / n_
    cov = np.empty((n_.size, 2, 2))
    cov[:, 0, 0] = scale * xx
    cov[:, 1, 1] = scale * yy
    cov[:, 0, 1] = cov[:, 1, 0] = scale * xy
    return cov


def hub_relative(
    frames: Frames,
    *,
    mode: Mode,
    k: float = 1.0,
    n_min: float = N_MIN,
    n_hub_min: float = N_HUB_MIN,
) -> HubRelative:
    """Marker position relative to the hub and its covariance (§9.4).

    On the stand a frame is flagged hub_moved when either coordinate of the
    hub is more than three robust standard deviations (1.4826 MAD) from its
    median. The robust standard deviation is floored at the median standard
    error of the hub centroid, so a hub that is still to within its own noise
    is not flagged. A stand frame without the hub uses the median hub position
    with the sample covariance of the found hub positions as Σₕ. On the floor
    such a frame is unusable, and hub_moved is not computed, as the hub moves
    by design.

    Args:
        frames: Frame table columns of §3.1.
        mode: "stand" or "floor".
        k: Calibration factor of §9.3, applied to the measured centroids.
        n_min: Minimum marker pixel count (§9.2).
        n_hub_min: Minimum hub pixel count (§9.2).

    Returns:
        The hub-relative positions, their covariance, and the flags.
    """
    if mode not in ("stand", "floor"):
        raise ValueError(f"mode must be 'stand' or 'floor', not {mode!r}")
    mx, my = _col(frames, "marker_x"), _col(frames, "marker_y")
    hx, hy = _col(frames, "hub_x"), _col(frames, "hub_y")
    mn, hn = _col(frames, "marker_n"), _col(frames, "hub_n")
    cov_m = centroid_covariance(
        mn,
        _col(frames, "marker_cxx"),
        _col(frames, "marker_cyy"),
        _col(frames, "marker_cxy"),
        k,
    )
    cov_h = centroid_covariance(
        hn, _col(frames, "hub_cxx"), _col(frames, "hub_cyy"), _col(frames, "hub_cxy"), k
    )
    n = mx.size
    marker_ok = _found(mn, mx, my, n_min) & np.isfinite(cov_m).all(axis=(1, 2))
    hub_ok = _found(hn, hx, hy, n_hub_min) & np.isfinite(cov_h).all(axis=(1, 2))
    hub = np.column_stack([hx, hy])
    hub_moved = np.zeros(n, bool)
    hub_imputed = np.zeros(n, bool)
    if mode == "stand" and hub_ok.any():
        found = hub[hub_ok]
        median = np.median(found, axis=0)
        mad_sd = MAD_TO_SD * np.median(np.abs(found - median), axis=0)
        unit = centroid_covariance(
            hn[hub_ok],
            _col(frames, "hub_cxx")[hub_ok],
            _col(frames, "hub_cyy")[hub_ok],
            _col(frames, "hub_cxy")[hub_ok],
        )
        noise_sd = np.sqrt(np.median(unit[:, [0, 1], [0, 1]], axis=0))
        robust_sd = np.maximum(mad_sd, noise_sd)
        hub_moved[hub_ok] = np.any(np.abs(found - median) > 3 * robust_sd, axis=1)
        missing = marker_ok & ~hub_ok
        if missing.any():
            spread = (
                np.cov(found.T) if found.shape[0] > 1 else np.zeros((2, 2))
            ).reshape(2, 2)
            hub[missing] = median
            cov_h[missing] = spread
            hub_imputed = missing
        usable = marker_ok
    else:
        usable = marker_ok & hub_ok
    r = np.column_stack([mx, my]) - hub
    cov = cov_m + cov_h
    r[~usable] = np.nan
    cov[~usable] = np.nan
    cov_h[~usable] = np.nan
    return HubRelative(r, cov, usable, hub_moved, hub_imputed, cov_h)
