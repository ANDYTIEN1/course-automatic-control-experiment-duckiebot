# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Ellipse fitted to the marker positions, and the phase on it (§9.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from duckiebot_analysis.analysis._types import (
    FloatArray,
)


@dataclass(frozen=True)
class Ellipse:
    """Ellipse fitted to hub-relative marker positions (§9.5).

    Attributes:
        centre: Ellipse centre c in px, relative to the hub.
        transform: The matrix A = D^½ Vᵀ that maps the ellipse onto the unit
            circle, with det(A) > 0 so the orientation of the image is kept.
        semi_major_px: Semi-major axis a in px.
        semi_minor_px: Semi-minor axis in px.
        semi_major_sd_px: Approximate standard deviation of a, from the
            radial scatter of the points about the fitted ellipse.
        n_points: Number of points fitted.
    """

    centre: FloatArray
    transform: FloatArray
    semi_major_px: float
    semi_minor_px: float
    semi_major_sd_px: float
    n_points: int


def fit_ellipse(x: npt.ArrayLike, y: npt.ArrayLike) -> Ellipse:
    """Least-squares ellipse through points (§9.5).

    Uses the direct ellipse-specific least-squares fit of Fitzgibbon et al.
    (1999) in the numerically stable form of Halíř and Flusser (1998), on
    centred and scaled coordinates, which always returns an ellipse.

    Args:
        x: Point x coordinates in px.
        y: Point y coordinates in px.

    Returns:
        The fitted ellipse.

    Raises:
        ValueError: With fewer than five finite points, or when the points do
            not determine an ellipse.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 5:
        raise ValueError("an ellipse needs at least five points")
    mean = np.array([xa.mean(), ya.mean()])
    scale = float(np.sqrt(np.mean((xa - mean[0]) ** 2 + (ya - mean[1]) ** 2)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("the points do not determine an ellipse")
    u, v = (xa - mean[0]) / scale, (ya - mean[1]) / scale
    d1 = np.column_stack([u * u, u * v, v * v])
    d2 = np.column_stack([u, v, np.ones_like(u)])
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    try:
        tmat = -np.linalg.solve(s3, s2.T)
        m = s1 + s2 @ tmat
        m = np.vstack([m[2] / 2, -m[1], m[0] / 2])
        _, vecs = np.linalg.eig(m)
    except np.linalg.LinAlgError as err:
        raise ValueError("the points do not determine an ellipse") from err
    vecs = np.real(vecs)
    cond = 4 * vecs[0] * vecs[2] - vecs[1] ** 2
    if not np.any(cond > 0):
        raise ValueError("the points do not determine an ellipse")
    a1 = vecs[:, int(np.argmax(cond))]
    a, b, c = a1
    d, e, f = tmat @ a1
    quad = np.array([[a, b / 2], [b / 2, c]])
    try:
        centre_n = np.linalg.solve(2 * quad, [-d, -e])
    except np.linalg.LinAlgError as err:
        raise ValueError("the points do not determine an ellipse") from err
    level = centre_n @ quad @ centre_n - f
    quad_n = quad / level
    vals, vecs2 = np.linalg.eigh(quad_n)
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        raise ValueError("the points do not determine an ellipse")
    # Back to px: u = (p - mean) / scale, so the quadratic form scales by 1/scale².
    vals = vals / scale**2
    transform = np.diag(np.sqrt(vals)) @ vecs2.T
    if np.linalg.det(transform) < 0:
        transform[1] *= -1
    centre = mean + scale * centre_n
    axes = 1 / np.sqrt(vals)
    q = (np.column_stack([xa, ya]) - centre) @ transform.T
    radial = np.hypot(q[:, 0], q[:, 1]) - 1
    semi_major = float(np.max(axes))
    # a rests on roughly half the points, those near the ends of the major axis.
    sd = semi_major * float(np.std(radial)) * np.sqrt(2 / xa.size)
    return Ellipse(
        centre=centre,
        transform=transform,
        semi_major_px=semi_major,
        semi_minor_px=float(np.min(axes)),
        semi_major_sd_px=sd,
        n_points=int(xa.size),
    )


def ellipse_phase(
    r: FloatArray, cov: FloatArray, ellipse: Ellipse
) -> tuple[FloatArray, FloatArray]:
    """Phase on the ellipse and its standard deviation (§9.5).

    q = A(r - c), φ = atan2(q_y, q_x), g = (-q_y, q_x) A / |q|², σ² = g Σᵣ gᵀ.

    Args:
        r: Hub-relative positions in px, shape (N, 2).
        cov: Their covariance in px², shape (N, 2, 2).
        ellipse: Fitted ellipse.

    Returns:
        φ in (-π, π] and σ_φ in rad, NaN where r is NaN.
    """
    q = (r - ellipse.centre) @ ellipse.transform.T
    phi = np.arctan2(q[:, 1], q[:, 0])
    rho2 = q[:, 0] ** 2 + q[:, 1] ** 2
    g = np.column_stack([-q[:, 1], q[:, 0]]) @ ellipse.transform / rho2[:, None]
    var = np.einsum("ni,nij,nj->n", g, cov, g)
    return phi, np.sqrt(np.clip(var, 0, None))
