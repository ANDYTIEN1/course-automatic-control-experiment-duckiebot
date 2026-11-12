# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""Spoke angle of the yellow wheel, for the floor runs (data/schema.md §3.1).

On the floor the white tape is hidden in many frames, but the five yellow
spokes stay in view. The hub mask (the yellow pixels of the wheel) is sampled
on a ring at RING_FRACTION of the wheel radius, which crosses every spoke and
not the rim, and the phase of the ring's N_SPOKES-th angular harmonic gives
the spoke angle modulo one spoke period, 2π/5. This follows the tracker of
the six-motor campaign, which took the angle from the 5th harmonic of the
spokes on a ring at 0.55 of the rim radius.

Angles are measured in image coordinates (x right, y down), so they grow
clockwise on screen, as the atan2 of §9.5 does.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

N_SPOKES = 5
RING_FRACTION = 0.55
N_RING = 360
# Fraction of the pixel distances from the centre that lie inside the radius;
# the 98th percentile keeps the rim and drops stray pixels beyond it.
RADIUS_PERCENTILE = 98.0
# Below this contrast |c_n| / sum of the ring samples, the spokes are not seen;
# ideal spokes of 1/5 of the ring give 0.94, and a uniform ring 0.
SPOKE_MIN_CONTRAST = 0.3
# Standard deviation of one binary ring sample at its largest (p = 0.5),
# which bounds the noise of the harmonic.
SAMPLE_SD = 0.5

_ANGLES = np.linspace(0.0, 2 * np.pi, N_RING, endpoint=False)


class SpokeMeasurement(NamedTuple):
    """Spoke angle of one frame.

    Attributes:
        angle: Spoke angle in rad in (-π/5, π/5], clockwise on screen; NaN
            when the wheel is not found.
        angle_sd: Its standard deviation in rad, NaN with the angle.
        contrast: |c_5| / sum of the ring samples, 0 when no sample is set.
        radius: Wheel radius in px, NaN when the mask is empty.
    """

    angle: float
    angle_sd: float
    contrast: float
    radius: float


_NONE = SpokeMeasurement(np.nan, np.nan, 0.0, np.nan)


def measure_spokes(
    mask: np.ndarray, cx: float, cy: float, x0: int = 0, y0: int = 0
) -> SpokeMeasurement:
    """Measure the spoke angle in a wheel mask around (cx, cy).

    Args:
        mask: Boolean mask of the wheel pixels, a crop whose top-left pixel
            is (x0, y0) of the frame.
        cx: Wheel centre x in frame px, such as the hub centroid.
        cy: Wheel centre y in frame px.
        x0: Frame x of the crop's first column.
        y0: Frame y of the crop's first row.

    Returns:
        The spoke angle, its standard deviation, the contrast, and the radius.
    """
    if mask.size == 0 or not (np.isfinite(cx) and np.isfinite(cy)):
        return _NONE
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return _NONE
    radius = float(
        np.percentile(np.hypot(xs + x0 - cx, ys + y0 - cy), RADIUS_PERCENTILE)
    )
    rr = RING_FRACTION * radius
    px = np.rint(cx + rr * np.cos(_ANGLES) - x0).astype(int)
    py = np.rint(cy + rr * np.sin(_ANGLES) - y0).astype(int)
    inside = (px >= 0) & (px < mask.shape[1]) & (py >= 0) & (py < mask.shape[0])
    ring = np.zeros(N_RING)
    ring[inside] = mask[py[inside], px[inside]]
    total = float(ring.sum())
    if total == 0:
        return SpokeMeasurement(np.nan, np.nan, 0.0, radius)
    c = complex(np.sum(ring * np.exp(-1j * N_SPOKES * _ANGLES)))
    if abs(c) == 0:
        return SpokeMeasurement(np.nan, np.nan, 0.0, radius)
    # A spoke at angle β puts the ring's mass at β + 2πk/5, so arg c = -5β.
    angle = -np.angle(c) / N_SPOKES
    period = 2 * np.pi / N_SPOKES
    if angle <= -period / 2:
        angle += period
    sd = SAMPLE_SD * np.sqrt(N_RING / 2) / abs(c) / N_SPOKES
    return SpokeMeasurement(float(angle), float(sd), abs(c) / total, radius)


__all__ = [
    "N_SPOKES",
    "RING_FRACTION",
    "SPOKE_MIN_CONTRAST",
    "SpokeMeasurement",
    "measure_spokes",
]
