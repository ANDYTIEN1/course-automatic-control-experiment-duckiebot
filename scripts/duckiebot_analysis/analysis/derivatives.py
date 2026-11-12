# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Angular velocity, acceleration, and ground motion from the wheel angle.

Implements data/schema.md §9.7, the weighted local quadratic fit on the
non-uniform frame times, and §9.8, the ground position, speed, and slip of a
floor run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from duckiebot_analysis.analysis._types import BoolArray, FloatArray, IntArray

W_OMEGA = 0.023
W_ALPHA = 0.050
MIN_POINTS = 5
MIN_SIDE = 2
# Window entries gathered per chunk, which bounds the memory of the fit.
CHUNK_ENTRIES = 2_000_000


@dataclass(frozen=True)
class LocalQuadratic:
    """Local fits y(t) ≈ β₀ + β₁(t − tᵢ) + β₂(t − tᵢ)² at every frame (§9.7).

    Attributes:
        beta: β₀, β₁, β₂ per frame, shape (N, 3), NaN where no fit.
        beta_sd: Standard deviations from (XᵀWX)⁻¹ · max(1, χ²/ν).
        n: Number of points in each fit, 0 where no fit.
    """

    beta: FloatArray
    beta_sd: FloatArray
    n: IntArray


@dataclass(frozen=True)
class Rates:
    """Angular velocity and acceleration per frame (§9.7), NaN where missing.

    Attributes:
        omega: ω in rad/s.
        omega_sd: Its standard deviation.
        alpha: α in rad/s².
        alpha_sd: Its standard deviation.
    """

    omega: FloatArray
    omega_sd: FloatArray
    alpha: FloatArray
    alpha_sd: FloatArray


@dataclass(frozen=True)
class GroundPosition:
    """Ground position of a floor run (§9.8).

    Attributes:
        x_m: Position along the direction of travel from the first usable
            frame, in m, NaN where unusable.
        x_sd_m: Total standard deviation, including the scale and the first
            frame's hub position.
        x_frame_sd_m: Independent part per frame, from the hub centroid; the
            weight for a local slope.
        rel_scale_sd: Relative standard deviation of the pixel scale per
            frame, common to all frames that share an ellipse.
        direction: Unit vector of travel in image px coordinates.
    """

    x_m: FloatArray
    x_sd_m: FloatArray
    x_frame_sd_m: FloatArray
    rel_scale_sd: FloatArray
    direction: FloatArray


def _runs(valid: BoolArray, segment: IntArray) -> tuple[IntArray, IntArray]:
    """Run index of each valid frame and the run to evaluate each frame in.

    A run is a maximal sequence of valid frames of one segment. A frame that
    is not valid is evaluated in the run of its neighbours when the valid
    frames before and after it belong to the same run.
    """
    idx = np.flatnonzero(valid)
    seg = segment[idx]
    run = np.concatenate([[0], np.cumsum(seg[1:] != seg[:-1])]).astype(np.int64)
    n = valid.size
    pos = np.searchsorted(idx, np.arange(n), side="left")
    eval_run = np.full(n, -1, dtype=np.int64)
    eval_run[idx] = run
    other = ~valid
    before = pos - 1
    after = pos
    inside = other & (before >= 0) & (after < idx.size)
    b = before[inside]
    a = after[inside]
    same = run[b] == run[a]
    target = np.flatnonzero(inside)[same]
    eval_run[target] = run[b[same]]
    return run, eval_run


def local_quadratic(
    t: npt.ArrayLike,
    y: npt.ArrayLike,
    y_sd: npt.ArrayLike,
    half_width: float,
    *,
    usable: npt.ArrayLike | None = None,
    segment: npt.ArrayLike | None = None,
) -> LocalQuadratic:
    """Weighted local quadratic fit at every frame (§9.7).

    Fits β₀ + β₁(t − tᵢ) + β₂(t − tᵢ)² with weights 1/σ² to the usable frames
    with |t − tᵢ| ≤ half_width in the same segment as frame i. A fit needs at
    least five frames with at least two on each side of tᵢ. The covariance of
    β is (XᵀWX)⁻¹ · max(1, χ²/ν) with ν = n − 3. A frame that is not usable
    but lies inside a segment also gets a fit, from its neighbours.

    Args:
        t: Frame times in s, non-decreasing.
        y: Values to fit.
        y_sd: Their standard deviations; a frame with σ ≤ 0 or NaN is unused.
        half_width: Window half-width w in s.
        usable: Frames that may enter a fit; default all.
        segment: Segment index per frame (§9.5); frames of different segments
            never share a fit, and -1 marks an unusable frame.

    Returns:
        The coefficients, their standard deviations, and the point counts.
    """
    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ss = np.asarray(y_sd, dtype=np.float64)
    n = tt.size
    beta = np.full((n, 3), np.nan)
    beta_sd = np.full((n, 3), np.nan)
    count = np.zeros(n, dtype=np.int64)
    valid = np.isfinite(tt) & np.isfinite(yy) & np.isfinite(ss) & (ss > 0)
    if usable is not None:
        valid &= np.asarray(usable, dtype=bool)
    seg = (
        np.zeros(n, dtype=np.int64)
        if segment is None
        else np.asarray(segment, dtype=np.int64)
    )
    valid &= seg >= 0
    if valid.sum() < MIN_POINTS:
        return LocalQuadratic(beta, beta_sd, count)
    run, eval_run = _runs(valid, seg)
    idx = np.flatnonzero(valid)
    tv, yv, wv = tt[idx], yy[idx], 1 / ss[idx] ** 2
    starts = np.searchsorted(run, np.arange(run[-1] + 1), side="left")
    ends = np.searchsorted(run, np.arange(run[-1] + 1), side="right")
    evals = np.flatnonzero((eval_run >= 0) & np.isfinite(tt))
    er = eval_run[evals]
    lo = np.maximum(np.searchsorted(tv, tt[evals] - half_width, "left"), starts[er])
    hi = np.minimum(np.searchsorted(tv, tt[evals] + half_width, "right"), ends[er])
    width = hi - lo
    keep = width >= MIN_POINTS
    evals, lo, hi = evals[keep], lo[keep], hi[keep]
    if evals.size == 0:
        return LocalQuadratic(beta, beta_sd, count)
    span = int((hi - lo).max())
    step = max(1, CHUNK_ENTRIES // span)
    offsets = np.arange(span)
    for c in range(0, evals.size, step):
        e, lo_c, hi_c = evals[c : c + step], lo[c : c + step], hi[c : c + step]
        j = lo_c[:, None] + offsets
        inside = j < hi_c[:, None]
        j = np.where(inside, j, lo_c[:, None])
        # Scale the time by w so the normal equations are well conditioned.
        tau = (tv[j] - tt[e][:, None]) / half_width
        w = np.where(inside, wv[j], 0.0)
        yc = yv[j] - yv[lo_c][:, None]
        left = np.sum(inside & (tau < 0), axis=1)
        right = np.sum(inside & (tau > 0), axis=1)
        npts = inside.sum(axis=1)
        ok = (npts >= MIN_POINTS) & (left >= MIN_SIDE) & (right >= MIN_SIDE)
        if not ok.any():
            continue
        tau, w, yc, e = tau[ok], w[ok], yc[ok], e[ok]
        npts, lo_c = npts[ok], lo_c[ok]
        t2 = tau * tau
        s = [np.sum(w * tau**p, axis=1) for p in range(5)]
        m = np.empty((e.size, 3, 3))
        for r in range(3):
            for q in range(3):
                m[:, r, q] = s[r + q]
        rhs = np.stack(
            [
                np.sum(w * yc, axis=1),
                np.sum(w * tau * yc, axis=1),
                np.sum(w * t2 * yc, axis=1),
            ],
            axis=1,
        )
        inv = np.linalg.inv(m)
        b = np.einsum("nij,nj->ni", inv, rhs)
        res = yc - (b[:, :1] + b[:, 1:2] * tau + b[:, 2:3] * t2)
        chi2 = np.sum(w * res * res, axis=1)
        factor = np.maximum(1.0, chi2 / (npts - 3))
        var = np.diagonal(inv, axis1=1, axis2=2) * factor[:, None]
        unscale = np.array([1.0, 1 / half_width, 1 / half_width**2])
        b = b * unscale
        b[:, 0] += yv[lo_c]
        beta[e] = b
        beta_sd[e] = np.sqrt(var) * unscale
        count[e] = npts
    return LocalQuadratic(beta, beta_sd, count)


def angular_rates(
    t: npt.ArrayLike,
    theta: npt.ArrayLike,
    theta_sd: npt.ArrayLike,
    *,
    usable: npt.ArrayLike | None = None,
    segment: npt.ArrayLike | None = None,
    w_omega: float = W_OMEGA,
    w_alpha: float = W_ALPHA,
) -> Rates:
    """Angular velocity and acceleration from the corrected angle (§9.7).

    ω = β₁ of the fit with half-width w_ω, and α = 2β₂ of the fit with
    half-width w_α. Values from overlapping windows are correlated, so a fit
    across frames must use the angle, not these values.

    Args:
        t: Frame times in s.
        theta: Corrected angle θᶜ in rad.
        theta_sd: Its standard deviation in rad.
        usable: Usable frames (§9.2); default all.
        segment: Segment index of §9.5; default one segment.
        w_omega: Half-width for ω in s, 23 ms in §9.7.
        w_alpha: Half-width for α in s, 50 ms in §9.7.

    Returns:
        ω, α, and their standard deviations per frame.
    """
    fw = local_quadratic(t, theta, theta_sd, w_omega, usable=usable, segment=segment)
    fa = local_quadratic(t, theta, theta_sd, w_alpha, usable=usable, segment=segment)
    return Rates(
        omega=fw.beta[:, 1],
        omega_sd=fw.beta_sd[:, 1],
        alpha=2 * fa.beta[:, 2],
        alpha_sd=2 * fa.beta_sd[:, 2],
    )


# §9.8 ---------------------------------------------------------------------------


def pixel_scale(
    semi_major_px: npt.ArrayLike,
    semi_major_sd_px: npt.ArrayLike,
    marker_radius_m: float,
    marker_radius_sd_m: float,
) -> tuple[FloatArray, FloatArray]:
    """Pixel scale at the marker, ρ = marker_radius_m / a (§9.8).

    Args:
        semi_major_px: Semi-major axis a of the ellipse in px, per frame or
            scalar (§9.5).
        semi_major_sd_px: Its standard deviation.
        marker_radius_m: Distance of the marker from the axle (§6).
        marker_radius_sd_m: Its standard deviation.

    Returns:
        ρ in m/px and its standard deviation, from both relative errors.
    """
    a = np.asarray(semi_major_px, dtype=np.float64)
    a_sd = np.asarray(semi_major_sd_px, dtype=np.float64)
    rho = marker_radius_m / a
    rel = np.hypot(marker_radius_sd_m / marker_radius_m, a_sd / a)
    return rho, rho * rel


def ground_position(
    hub_x: npt.ArrayLike,
    hub_y: npt.ArrayLike,
    hub_cov: npt.ArrayLike,
    rho: npt.ArrayLike,
    rho_sd: npt.ArrayLike,
    *,
    usable: npt.ArrayLike | None = None,
    direction: npt.ArrayLike | None = None,
) -> GroundPosition:
    """Ground position along the direction of travel (§9.8).

    The direction is the principal axis of the usable hub positions, signed so
    that the net displacement is positive, unless given. §9.8 writes
    x = ρ (h_x − h₀,x) rotated to the direction of travel; this projects the
    hub displacement on that direction and sums ρ over the frame-to-frame
    steps, so a scale that changes along the run (the perspective of §9.5) is
    applied to the step where it holds. With constant ρ this equals
    ρ · (hᵢ − h₀)·u.

    Args:
        hub_x: Hub x in px.
        hub_y: Hub y in px.
        hub_cov: Hub centroid covariance Σₕ in px², shape (N, 2, 2) (§9.3).
        rho: Pixel scale in m/px, per frame or scalar.
        rho_sd: Its standard deviation.
        usable: Frames to use; default those with a finite hub.
        direction: Unit vector of travel in px coordinates, to override.

    Returns:
        The ground position and its uncertainty.
    """
    hx = np.asarray(hub_x, dtype=np.float64)
    hy = np.asarray(hub_y, dtype=np.float64)
    cov = np.asarray(hub_cov, dtype=np.float64).reshape(-1, 2, 2)
    n = hx.size
    rho_a = np.broadcast_to(np.asarray(rho, dtype=np.float64), (n,))
    rho_sd_a = np.broadcast_to(np.asarray(rho_sd, dtype=np.float64), (n,))
    ok = np.isfinite(hx) & np.isfinite(hy) & np.isfinite(rho_a)
    if usable is not None:
        ok &= np.asarray(usable, dtype=bool)
    x = np.full(n, np.nan)
    x_sd = np.full(n, np.nan)
    x_frame_sd = np.full(n, np.nan)
    rel = np.full(n, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return GroundPosition(x, x_sd, x_frame_sd, rel, np.array([np.nan, np.nan]))
    h = np.column_stack([hx[idx], hy[idx]])
    if direction is None:
        centred = h - h.mean(axis=0)
        if idx.size > 1 and np.any(centred):
            u = np.linalg.svd(centred, full_matrices=False)[2][0]
        else:
            u = np.array([1.0, 0.0])
        if np.dot(h[-1] - h[0], u) < 0:
            u = -u
    else:
        u = np.asarray(direction, dtype=np.float64)
        u = u / np.linalg.norm(u)
    along = h @ u
    r = rho_a[idx]
    step_rho = (r[1:] + r[:-1]) / 2
    x[idx] = np.concatenate([[0.0], np.cumsum(step_rho * np.diff(along))])
    var_px = np.einsum("i,nij,j->n", u, cov[idx], u)
    x_frame_sd[idx] = r * np.sqrt(var_px)
    rel[idx] = rho_sd_a[idx] / r
    noise = r**2 * (var_px + var_px[0])
    noise[0] = 0.0
    x_sd[idx] = np.sqrt(noise + (x[idx] * rel[idx]) ** 2)
    return GroundPosition(x, x_sd, x_frame_sd, rel, u)


def ground_speed(
    t: npt.ArrayLike,
    ground: GroundPosition,
    *,
    usable: npt.ArrayLike | None = None,
    half_width: float = W_OMEGA,
) -> tuple[FloatArray, FloatArray]:
    """Ground speed as the local slope of the ground position (§9.8, §9.7).

    The slope is fitted with the independent per-frame uncertainty; the scale
    uncertainty, common to the frames, is added afterwards as |v| σ_ρ/ρ.

    Args:
        t: Frame times in s.
        ground: Ground position of :func:`ground_position`.
        usable: Frames to use; default those with a position.
        half_width: Window half-width in s; default that of ω.

    Returns:
        v in m/s and its standard deviation.
    """
    sd = ground.x_frame_sd_m
    # A hub without spread would get infinite weight; give such frames a
    # tiny uncertainty instead, so they still count.
    finite = sd[np.isfinite(sd) & (sd > 0)]
    floor = 1e-12 if finite.size == 0 else 1e-6 * float(np.median(finite))
    sd = np.where(np.isfinite(sd), np.maximum(sd, floor), np.nan)
    fit = local_quadratic(t, ground.x_m, sd, half_width, usable=usable)
    v = fit.beta[:, 1]
    v_sd = np.hypot(fit.beta_sd[:, 1], v * ground.rel_scale_sd)
    return v, v_sd


def slip(
    v: npt.ArrayLike,
    v_sd: npt.ArrayLike,
    omega: npt.ArrayLike,
    omega_sd: npt.ArrayLike,
    wheel_radius_m: float,
    wheel_radius_sd_m: float,
) -> tuple[FloatArray, FloatArray]:
    """Slip 1 − v / (R ω) and its first-order standard deviation (§9.8).

    v and ω must share a sign convention: a wheel turning in the direction
    that drives the robot forward gives the same sign to both.

    Args:
        v: Ground speed in m/s.
        v_sd: Its standard deviation.
        omega: Wheel angular velocity in rad/s.
        omega_sd: Its standard deviation.
        wheel_radius_m: Wheel radius R (§6).
        wheel_radius_sd_m: Its standard deviation.

    Returns:
        The slip and its standard deviation.
    """
    va = np.asarray(v, dtype=np.float64)
    wa = np.asarray(omega, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = va / (wheel_radius_m * wa)
        rel = np.sqrt(
            (np.asarray(v_sd, dtype=np.float64) / va) ** 2
            + (np.asarray(omega_sd, dtype=np.float64) / wa) ** 2
            + (wheel_radius_sd_m / wheel_radius_m) ** 2
        )
        sd = np.abs(ratio) * rel
        # v = 0 makes the relative form 0/0; the slip's error is then σ_v/(Rω).
        sd = np.where(va == 0, np.asarray(v_sd) / np.abs(wheel_radius_m * wa), sd)
    return 1 - ratio, sd
