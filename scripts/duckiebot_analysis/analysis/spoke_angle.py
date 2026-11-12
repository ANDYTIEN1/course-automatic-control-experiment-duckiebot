# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Wheel angle of a floor run from the tracked spoke angle (data/schema.md §9.5).

The spoke tracker (tracking.spokes) gives the angle of the five spokes modulo
one spoke period, 2π/5, per frame. The wheel angle is the sum of the
frame-to-frame increments, each wrapped to half a period, times the sign s
that makes a positive duty give a positive angle. This assumes less than
half a spoke period, a tenth of a turn, between usable frames; a new segment
starts where more than MAX_GAP_INTERVALS median frame intervals separate two
usable frames, and θ stays continuous across it, as in §9.5.

The spokes are symmetric, so the angle carries no once-per-turn error from a
marker and needs no correction of §9.6.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from duckiebot_analysis.analysis._types import Direction
from duckiebot_analysis.analysis.angle import _direction_sign
from duckiebot_analysis.analysis.hub import median_interval
from duckiebot_analysis.tracking.spokes import N_SPOKES, SPOKE_MIN_CONTRAST

FloatArray = npt.NDArray[np.float64]
PERIOD = 2 * np.pi / N_SPOKES
# Over 4 frame intervals at 240 frames/s, an increment stays below half a spoke
# period, a tenth of a turn, up to 37 rad/s.
MAX_GAP_INTERVALS = 4.0


class SpokeAngle(NamedTuple):
    """Wheel angle from the spokes.

    Attributes:
        theta: Angle in rad, NaN where not usable.
        theta_sd: Its standard deviation in rad.
        usable: Frames with a spoke angle of sufficient contrast.
        segment: Segment index, -1 where not usable.
    """

    theta: FloatArray
    theta_sd: FloatArray
    usable: npt.NDArray[np.bool_]
    segment: npt.NDArray[np.int64]


def spoke_theta(
    t: npt.ArrayLike,
    angle: npt.ArrayLike,
    angle_sd: npt.ArrayLike,
    contrast: npt.ArrayLike,
    *,
    positive_duty_turns: Direction,
    min_contrast: float = SPOKE_MIN_CONTRAST,
) -> SpokeAngle:
    """Continuous wheel angle from the per-frame spoke angle.

    Args:
        t: Frame times in s.
        angle: Spoke angle in rad, clockwise on screen, modulo 2π/5.
        angle_sd: Its standard deviation in rad.
        contrast: Spoke contrast of each frame (tracking.spokes).
        positive_duty_turns: Direction of a positive duty on screen (§6).
        min_contrast: Smallest contrast of a usable frame.

    Returns:
        θ, its standard deviation, the usable frames, and the segments.

    Raises:
        ValueError: If positive_duty_turns is not "cw" or "ccw".
    """
    sign = _direction_sign(positive_duty_turns)
    t = np.asarray(t, dtype=np.float64)
    psi = np.asarray(angle, dtype=np.float64)
    sd = np.asarray(angle_sd, dtype=np.float64)
    con = np.asarray(contrast, dtype=np.float64)
    n = t.size
    usable = (
        np.isfinite(t)
        & np.isfinite(psi)
        & np.isfinite(sd)
        & (sd > 0)
        & np.isfinite(con)
        & (con >= min_contrast)
    )
    theta = np.full(n, np.nan)
    segment = np.full(n, -1, dtype=np.int64)
    idx = np.flatnonzero(usable)
    if idx.size:
        # Over the finite intervals only, so that a frame without a time
        # does not turn the median, and every comparison with it, into NaN.
        median_dt = median_interval(t)
        gaps = np.diff(t[idx])
        breaks = gaps > MAX_GAP_INTERVALS * median_dt
        step = (np.diff(psi[idx]) + PERIOD / 2) % PERIOD - PERIOD / 2
        # Across a break the turn is unknown; θ continues from the last value.
        step[breaks] = 0.0
        theta[idx] = sign * np.r_[psi[idx[0]], psi[idx[0]] + np.cumsum(step)]
        segment[idx] = np.r_[0, np.cumsum(breaks)]
    return SpokeAngle(theta, np.where(usable, sd, np.nan), usable, segment)


__all__ = ["MAX_GAP_INTERVALS", "SpokeAngle", "spoke_theta"]
