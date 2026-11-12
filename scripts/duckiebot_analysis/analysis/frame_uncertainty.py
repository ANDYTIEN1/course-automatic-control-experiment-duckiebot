# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Frames picked by their angle uncertainty, and what that uncertainty is made of."""

from __future__ import annotations

import numpy as np

from duckiebot_analysis.analysis.angle import AngleResult
from duckiebot_analysis.analysis.ellipse import ellipse_phase
from duckiebot_analysis.analysis.hub import hub_relative


def pick_frames(sd: np.ndarray, usable: np.ndarray) -> dict[str, int]:
    """Return the usable frames of smallest, median, mean and largest sd.

    The median and mean frames are those nearest to these values, the first of
    equally near ones.

    Raises:
        ValueError: If the lengths differ, no frame is usable, or a usable frame
            has no finite sd.
    """
    if sd.shape != usable.shape:
        raise ValueError(f"sd {sd.shape} and usable {usable.shape} differ in shape")
    index = np.nonzero(usable)[0]
    if index.size == 0:
        raise ValueError("no usable frame to pick from")
    used = sd[index]
    if not np.isfinite(used).all():
        raise ValueError("a usable frame has no finite uncertainty")
    return {
        "best": int(index[np.argmin(used)]),
        "median": int(index[np.argmin(np.abs(used - np.median(used)))]),
        "mean": int(index[np.argmin(np.abs(used - used.mean()))]),
        "worst": int(index[np.argmax(used)]),
    }


def angle_sd_parts(
    frames: dict[str, np.ndarray], result: AngleResult
) -> tuple[np.ndarray, np.ndarray]:
    """Return the parts of theta_sd of a stand video due to marker and hub.

    The variance is g Sigma_r g^T with Sigma_r the sum of the two covariances,
    so the squares of the two parts add up to the square of theta_sd.

    Args:
        frames: The frame table columns that the result was computed from.
        result: The angle of a stand video, with the run's ellipse.

    Returns:
        The marker part and the hub part in rad, NaN where no angle is usable.

    Raises:
        ValueError: If the result has no ellipse or is not of a stand video.
    """
    if result.ellipse is None:
        raise ValueError("the angle result has no ellipse")
    if result.mode != "stand":
        raise ValueError(f"the parts need a stand result, not {result.mode}")
    hub = hub_relative(
        frames,
        mode="stand",
        k=result.k,
        n_min=result.n_min,
        n_hub_min=result.n_hub_min,
    )
    _, marker_sd = ellipse_phase(hub.r, hub.cov - hub.hub_cov, result.ellipse)
    _, hub_sd = ellipse_phase(hub.r, hub.hub_cov, result.ellipse)
    return (
        np.where(result.usable, marker_sd, np.nan),
        np.where(result.usable, hub_sd, np.nan),
    )
