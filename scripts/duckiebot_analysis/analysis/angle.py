# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Wheel angle and its uncertainty from the tracked marker and hub.

Implements data/schema.md §9.5 and combines it with §9.1-§9.4
(duckiebot_analysis.analysis.hub) and the §9.6 correction
(duckiebot_analysis.analysis.harmonics) into the corrected angle with
the calibrated factor k. The public names of those modules are re-exported
here, so callers can import everything of §9.1-§9.6 from this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from duckiebot_analysis.analysis._types import (
    BoolArray,
    Direction,
    FloatArray,
    Frames,
    IntArray,
    Mode,
    Window,
)
from duckiebot_analysis.analysis.ellipse import Ellipse, ellipse_phase, fit_ellipse
from duckiebot_analysis.analysis.harmonics import (
    N_HARMONICS,
    HarmonicCorrection,
    TrialHarmonics,
    TrialWindows,
    harmonic_correct,
)
from duckiebot_analysis.analysis.hub import (
    N_HUB_MIN,
    N_MIN,
    HubRelative,
    TimeAxis,
    centroid_covariance,
    hub_relative,
    time_axis,
    usable_mask,
)

__all__ = [
    "N_HARMONICS",
    "N_HUB_MIN",
    "N_MIN",
    "AngleResult",
    "Direction",
    "Ellipse",
    "HarmonicCorrection",
    "HubRelative",
    "Mode",
    "TimeAxis",
    "TrialHarmonics",
    "TrialWindows",
    "Window",
    "centroid_covariance",
    "compute_angle",
    "ellipse_phase",
    "fit_ellipse",
    "harmonic_correct",
    "hub_relative",
    "process_angle",
    "time_axis",
    "unwrap_segments",
    "usable_mask",
]

# Usable intervals whose local speed is estimated from a running median.
RATE_WINDOW = 11
# A per-turn ellipse (§9.5, floor) needs this many points spread over this
# many of 12 sectors of the turn; otherwise the run's ellipse is used.
MIN_WINDOW_POINTS = 20
MIN_WINDOW_SECTORS = 10


@dataclass(frozen=True)
class AngleResult:
    """Wheel angle of every frame (§9.1-§9.6).

    Attributes:
        t: Time from the first frame in s (§9.1).
        dropped: Frames preceded by dropped frames (§9.1).
        usable: Frames with a valid angle (§9.2, §9.4).
        hub_moved: Hub flag of §9.4.
        hub_imputed: Hub flag of §9.4.
        theta: Angle in rad, corrected by §9.6 when trials were given.
        theta_sd: Standard deviation of theta in rad.
        theta_raw: Angle of §9.5 before the correction.
        theta_raw_sd: Its standard deviation, with the factor k.
        segment: Unwrapping segment of each frame, -1 where unusable (§9.5).
        ellipse: Ellipse of the whole run, None when it could not be fitted.
        window: Index into window_ellipses of the ellipse used per frame, -1
            where the run's ellipse was used (always on the stand).
        window_ellipses: Per-turn ellipses of a floor run (§9.5).
        semi_major_px: Semi-major axis of the ellipse used for each frame.
        semi_major_sd_px: Its standard deviation.
        k: Factor k of §9.3 used for theta_sd.
        n_min: Minimum marker pixel count used (§9.2).
        n_hub_min: Minimum hub pixel count used (§9.2).
        mode: "stand" or "floor", as computed.
        chi2: χ² of the §9.6 fit with k = 1, from which k was calibrated.
        nu: Degrees of freedom of that fit.
        harmonics: Per-trial §9.6 fits with the calibrated k.
    """

    t: FloatArray
    dropped: BoolArray
    usable: BoolArray
    hub_moved: BoolArray
    hub_imputed: BoolArray
    theta: FloatArray
    theta_sd: FloatArray
    theta_raw: FloatArray
    theta_raw_sd: FloatArray
    segment: IntArray
    ellipse: Ellipse | None
    window: IntArray
    window_ellipses: tuple[Ellipse, ...]
    semi_major_px: FloatArray
    semi_major_sd_px: FloatArray
    k: float
    n_min: float = N_MIN
    n_hub_min: float = N_HUB_MIN
    mode: Mode = "stand"
    chi2: float = np.nan
    nu: int = 0
    harmonics: tuple[TrialHarmonics, ...] = ()


def _wrap(x: FloatArray) -> FloatArray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def unwrap_segments(
    t: FloatArray, phi: FloatArray, valid: BoolArray
) -> tuple[FloatArray, IntArray]:
    """Unwrap the phase over valid frames and split it into segments (§9.5).

    Each step between consecutive valid frames is taken as the wrapped
    difference, which assumes less than half a turn between them. A jump of
    half a turn or more cannot be seen in the wrapped phase, so it is
    predicted: the local speed is the running median of the signed wrapped
    rates over 11 intervals, and a gap over which this speed turns the wheel
    by π or more starts a new segment. The unwrapped value stays congruent to
    the phase modulo 2π across a break, so the once-per-turn terms of §9.6
    remain functions of the wheel position.

    Args:
        t: Frame times in s.
        phi: Wrapped phase in rad.
        valid: Frames to use.

    Returns:
        The unwrapped phase (NaN where not valid) and the segment index of
        each frame (-1 where not valid).
    """
    out = np.full(t.size, np.nan)
    seg = np.full(t.size, -1, dtype=np.int64)
    idx = np.flatnonzero(valid & np.isfinite(phi) & np.isfinite(t))
    if idx.size == 0:
        return out, seg
    p = phi[idx]
    step = _wrap(np.diff(p))
    dt = np.diff(t[idx])
    breaks = np.zeros(step.size, bool)
    if step.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(dt > 0, step / dt, 0.0)
        half = RATE_WINDOW // 2
        padded = np.pad(rate, half, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, RATE_WINDOW)
        local = np.median(windows, axis=1)
        breaks = np.abs(local) * dt >= np.pi
    out[idx] = p[0] + np.concatenate([[0.0], np.cumsum(step)])
    seg[idx] = np.concatenate([[0], np.cumsum(breaks)])
    return out, seg


def _circular_mean(x: FloatArray) -> float:
    return float(np.angle(np.mean(np.exp(1j * x))))


def _sectors(phi: FloatArray) -> int:
    return int(np.unique(np.floor((phi + np.pi) / (np.pi / 6)) % 12).size)


def _floor_phase(
    hr: HubRelative,
    phi_run: FloatArray,
    t: FloatArray,
    run_ellipse: Ellipse,
) -> tuple[FloatArray, FloatArray, IntArray, tuple[Ellipse, ...]]:
    """Phase with the ellipse refitted per wheel turn (§9.5, floor).

    Windows are consecutive usable frames over which the run-ellipse phase
    accumulates one turn of absolute rotation. Each window's map A is fixed
    only up to a rotation, so its phase is aligned with the previous window
    by the mean difference of the two maps over both windows' frames.
    """
    n = t.size
    idx = np.flatnonzero(hr.usable)
    unwrapped, _ = unwrap_segments(t, phi_run, hr.usable)
    turned = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(unwrapped[idx])))])
    turn = np.floor(turned / (2 * np.pi)).astype(np.int64)
    phi = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    window = np.full(n, -1, dtype=np.int64)
    ellipses: list[Ellipse] = []
    prev: tuple[Ellipse, IntArray, float] | None = None
    bounds = np.flatnonzero(np.diff(turn)) + 1
    for part in np.split(idx, bounds):
        ell = run_ellipse
        if part.size >= MIN_WINDOW_POINTS and _sectors(phi_run[part]) >= (
            MIN_WINDOW_SECTORS
        ):
            try:
                ell = fit_ellipse(hr.r[part, 0], hr.r[part, 1])
            except ValueError:
                ell = run_ellipse
        p, s = ellipse_phase(hr.r[part], hr.cov[part], ell)
        if prev is None:
            offset = _circular_mean(phi_run[part] - p)
        else:
            prev_ell, prev_part, prev_offset = prev
            both = np.concatenate([prev_part, part])
            p_prev, _ = ellipse_phase(hr.r[both], hr.cov[both], prev_ell)
            p_this, _ = ellipse_phase(hr.r[both], hr.cov[both], ell)
            offset = prev_offset + _circular_mean(p_prev - p_this)
        phi[part] = _wrap(p + offset)
        sd[part] = s
        if ell is not run_ellipse:
            window[part] = len(ellipses)
            ellipses.append(ell)
        prev = (ell, part, offset)
    return phi, sd, window, tuple(ellipses)


def _direction_sign(positive_duty_turns: Direction) -> float:
    # A det(A) > 0 keeps the image orientation; with y down, atan2 grows
    # clockwise on screen, so a counter-clockwise positive needs s = -1.
    if positive_duty_turns == "ccw":
        return -1.0
    if positive_duty_turns == "cw":
        return 1.0
    raise ValueError(
        f"positive_duty_turns must be 'ccw' or 'cw', not {positive_duty_turns!r}"
    )


def compute_angle(
    frames: Frames,
    time_base: float,
    *,
    mode: Mode,
    positive_duty_turns: Direction,
    k: float = 1.0,
    n_min: float = N_MIN,
    n_hub_min: float = N_HUB_MIN,
) -> AngleResult:
    """Wheel angle of §9.5 with the covariance of §9.3 scaled by k.

    The ellipse is fitted without weights, so θ does not depend on k; only
    its standard deviation scales with k (the hub spread of an imputed frame
    does not).

    Args:
        frames: Frame table columns of §3.1.
        time_base: Seconds per pts unit.
        mode: "stand" or "floor"; the floor refits the ellipse per turn.
        positive_duty_turns: Direction of a positive duty as seen on screen
            (§6), which sets the sign s.
        k: Calibration factor of §9.3.
        n_min: Minimum marker pixel count (§9.2).
        n_hub_min: Minimum hub pixel count (§9.2).

    Returns:
        The angle without the §9.6 correction (theta equals theta_raw).
    """
    sign = _direction_sign(positive_duty_turns)
    axis = time_axis(frames["pts"], time_base)
    t = axis.t
    n = t.size
    hr = hub_relative(frames, mode=mode, k=k, n_min=n_min, n_hub_min=n_hub_min)
    usable = hr.usable & np.isfinite(t)
    nan = np.full(n, np.nan)
    empty = AngleResult(
        t=t,
        dropped=axis.dropped,
        usable=np.zeros(n, bool),
        hub_moved=hr.hub_moved,
        hub_imputed=np.zeros(n, bool),
        theta=nan,
        theta_sd=nan.copy(),
        theta_raw=nan.copy(),
        theta_raw_sd=nan.copy(),
        segment=np.full(n, -1, dtype=np.int64),
        ellipse=None,
        window=np.full(n, -1, dtype=np.int64),
        window_ellipses=(),
        semi_major_px=nan.copy(),
        semi_major_sd_px=nan.copy(),
        k=k,
        n_min=n_min,
        n_hub_min=n_hub_min,
        mode=mode,
    )
    if usable.sum() < 5:
        return empty
    try:
        ellipse = fit_ellipse(hr.r[usable, 0], hr.r[usable, 1])
    except ValueError:
        return empty
    phi, phi_sd = ellipse_phase(hr.r, hr.cov, ellipse)
    window = np.full(n, -1, dtype=np.int64)
    window_ellipses: tuple[Ellipse, ...] = ()
    if mode == "floor":
        phi, phi_sd, window, window_ellipses = _floor_phase(hr, phi, t, ellipse)
    usable = usable & np.isfinite(phi) & np.isfinite(phi_sd) & (phi_sd > 0)
    unwrapped, segment = unwrap_segments(t, phi, usable)
    theta = sign * unwrapped
    theta_sd = np.where(usable, phi_sd, np.nan)
    a = np.full(n, ellipse.semi_major_px)
    a_sd = np.full(n, ellipse.semi_major_sd_px)
    for j, ell in enumerate(window_ellipses):
        a[window == j] = ell.semi_major_px
        a_sd[window == j] = ell.semi_major_sd_px
    return replace(
        empty,
        usable=usable,
        hub_imputed=hr.hub_imputed & usable,
        theta=theta,
        theta_sd=theta_sd,
        theta_raw=theta.copy(),
        theta_raw_sd=theta_sd.copy(),
        segment=segment,
        ellipse=ellipse,
        window=window,
        window_ellipses=window_ellipses,
        semi_major_px=a,
        semi_major_sd_px=a_sd,
    )


def process_angle(
    frames: Frames,
    time_base: float,
    *,
    mode: Mode,
    positive_duty_turns: Direction,
    trials: Sequence[TrialWindows],
    n_min: float = N_MIN,
    n_hub_min: float = N_HUB_MIN,
    n_harmonics: int = N_HARMONICS,
) -> AngleResult:
    """Corrected wheel angle with the calibrated k (§9.3-§9.6).

    The angle is computed with k = 1, corrected per trial, and k² is taken
    as max(1, χ²/ν) pooled over the corrected trials of the run; the
    uncertainty is then recomputed with that k and the correction refitted
    (§9.3). Trials without a steady part are not corrected and use the run's
    k.

    Args:
        frames: Frame table columns of §3.1.
        time_base: Seconds per pts unit.
        mode: "stand" or "floor".
        positive_duty_turns: Direction of a positive duty as seen on screen.
        trials: Trial spans and steady parts of the run.
        n_min: Minimum marker pixel count (§9.2).
        n_hub_min: Minimum hub pixel count (§9.2).
        n_harmonics: Number of harmonics of §9.6.

    Returns:
        The angle result with theta corrected and k calibrated.
    """
    kw = {
        "mode": mode,
        "positive_duty_turns": positive_duty_turns,
        "n_min": n_min,
        "n_hub_min": n_hub_min,
    }
    first = compute_angle(frames, time_base, k=1.0, **kw)  # type: ignore[arg-type]
    if first.ellipse is None:
        return first
    calib = harmonic_correct(
        first.t,
        first.theta_raw,
        first.theta_raw_sd,
        first.segment,
        trials,
        n_harmonics=n_harmonics,
    )
    k = calib.k
    second = (
        first if k == 1.0 else compute_angle(frames, time_base, k=k, **kw)  # type: ignore[arg-type]
    )
    final = harmonic_correct(
        second.t,
        second.theta_raw,
        second.theta_raw_sd,
        second.segment,
        trials,
        n_harmonics=n_harmonics,
    )
    return replace(
        second,
        theta=final.theta,
        theta_sd=final.theta_sd,
        k=k,
        chi2=calib.chi2,
        nu=calib.nu,
        harmonics=final.trials,
    )
