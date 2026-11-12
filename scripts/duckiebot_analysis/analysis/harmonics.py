# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Per-trial once-per-turn correction of the wheel angle (data/schema.md §9.6).

The fit also calibrates the factor k of the centroid covariance (§9.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from duckiebot_analysis.analysis._types import (
    BoolArray,
    FloatArray,
    IntArray,
    Window,
)

N_HARMONICS = 3
MIN_BLOCK_FRAMES = 3
# A steady part that sweeps this many turns also gets t² and t³ terms, for
# a speed that drifts at constant duty. Over 4 turns a cubic in time can mimic
# at most 11 % of the variance of a once-per-turn term (3 turns: 18 %), so the
# harmonics stay determined; shorter parts keep a constant speed.
DRIFT_MIN_TURNS = 4.0
DRIFT_DEGREE = 3


@dataclass(frozen=True)
class TrialWindows:
    """Frames of one trial and its steady parts, for §9.6.

    Attributes:
        span: Frames of the trial; these receive the trial's correction.
        steady: Steady parts of the trial (constant duty for at least 1.6 s,
            from 1.2 s after the change), as masks or (t_start, t_end).
    """

    span: Window
    steady: tuple[Window, ...] = ()


@dataclass(frozen=True)
class TrialHarmonics:
    """Once-per-turn fit of one trial (§9.6).

    Attributes:
        corrected: Whether the correction was fitted and applied.
        coef: a₁, b₁, a₂, b₂, a₃, b₃ in rad, NaN when not corrected.
        coef_cov: Covariance of coef, unscaled (XᵀWX)⁻¹.
        chi2: Weighted residual sum of squares of the fit.
        nu: Degrees of freedom of the fit.
        k: max(1, χ²/ν)^½ of this trial, NaN when not corrected.
        n_frames: Usable frames in the steady parts.
    """

    corrected: bool
    coef: FloatArray
    coef_cov: FloatArray
    chi2: float
    nu: int
    k: float
    n_frames: int


@dataclass(frozen=True)
class HarmonicCorrection:
    """Result of §9.6 over all trials of a run.

    Attributes:
        theta: Corrected angle θᶜ in rad.
        theta_sd: Its standard deviation in rad.
        trials: Per-trial fits, in the order given.
        chi2: χ² summed over corrected trials.
        nu: ν summed over corrected trials.
        k: Run factor max(1, Σχ²/Σν)^½; 1 when no trial was corrected.
    """

    theta: FloatArray
    theta_sd: FloatArray
    trials: tuple[TrialHarmonics, ...]
    chi2: float
    nu: int
    k: float


def _window_mask(window: Window, t: FloatArray) -> BoolArray:
    if isinstance(window, tuple):
        start, end = window
        with np.errstate(invalid="ignore"):
            return (t >= start) & (t <= end)
    mask = np.asarray(window, dtype=bool)
    if mask.shape != t.shape:
        raise ValueError(f"window mask has shape {mask.shape}, frames {t.shape}")
    return mask


def _harmonic_basis(theta: FloatArray, n_harmonics: int) -> FloatArray:
    k = np.arange(1, n_harmonics + 1)
    kt = theta[:, None] * k
    basis = np.empty((theta.size, 2 * n_harmonics))
    basis[:, 0::2] = np.cos(kt)
    basis[:, 1::2] = np.sin(kt)
    return basis


def _harmonic_slope(theta: FloatArray, n_harmonics: int) -> FloatArray:
    k = np.arange(1, n_harmonics + 1)
    kt = theta[:, None] * k
    slope = np.empty((theta.size, 2 * n_harmonics))
    slope[:, 0::2] = -k * np.sin(kt)
    slope[:, 1::2] = k * np.cos(kt)
    return slope


def _not_corrected(n_harmonics: int, n_frames: int) -> TrialHarmonics:
    p = 2 * n_harmonics
    return TrialHarmonics(
        corrected=False,
        coef=np.full(p, np.nan),
        coef_cov=np.full((p, p), np.nan),
        chi2=np.nan,
        nu=0,
        k=np.nan,
        n_frames=n_frames,
    )


def _fit_trial(
    t: FloatArray,
    theta: FloatArray,
    weight: FloatArray,
    blocks: list[BoolArray],
    n_harmonics: int,
) -> TrialHarmonics:
    """Weighted least squares of §9.6 for one trial.

    Each steady block (a steady part within one segment) gets its own θ₀ and
    ω, since a trial may hold steady parts at different duties and θ is not
    compared across segments; the harmonic terms are shared by the trial.
    A block that sweeps at least DRIFT_MIN_TURNS turns also gets t² and t³
    terms, because the speed of some motors drifts at constant duty, by up
    to 40 % over 7 s; a constant ω would pass that motion to χ² and k and
    bend the harmonics.
    """
    sel = np.any(blocks, axis=0)
    n_frames = int(sel.sum())
    p = 2 * n_harmonics
    motion: list[FloatArray] = []
    for b in blocks:
        in_b = b[sel]
        tb = t[b]
        # Centre time per block, so the intercept and slope are uncorrelated.
        x = tb - tb.mean()
        powers = [x**0, x]
        turns = float(np.ptp(theta[b])) / (2 * np.pi)
        if turns >= DRIFT_MIN_TURNS:
            # Scaled to [-1, 1], so the higher powers stay well conditioned.
            xs = x / max(float(np.max(np.abs(x))), np.finfo(float).tiny)
            powers += [xs**d for d in range(2, DRIFT_DEGREE + 1)]
        for col in powers:
            full = np.zeros(n_frames)
            full[in_b] = col
            motion.append(full)
    nm = len(motion)
    cols = np.empty((n_frames, nm + p))
    cols[:, :nm] = np.column_stack(motion)
    cols[:, nm:] = _harmonic_basis(theta[sel], n_harmonics)
    nu = n_frames - cols.shape[1]
    if nu < 1:
        return _not_corrected(n_harmonics, n_frames)
    sw = np.sqrt(weight[sel])
    xw = cols * sw[:, None]
    yw = theta[sel] * sw
    beta, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
    if rank < cols.shape[1]:
        return _not_corrected(n_harmonics, n_frames)
    cov = np.linalg.inv(xw.T @ xw)
    chi2 = float(np.sum((yw - xw @ beta) ** 2))
    return TrialHarmonics(
        corrected=True,
        coef=beta[nm:],
        coef_cov=cov[nm:, nm:],
        chi2=chi2,
        nu=int(nu),
        k=float(np.sqrt(max(1.0, chi2 / nu))),
        n_frames=n_frames,
    )


def harmonic_correct(
    t: FloatArray,
    theta: FloatArray,
    theta_sd: FloatArray,
    segment: IntArray,
    trials: Sequence[TrialWindows],
    *,
    n_harmonics: int = N_HARMONICS,
) -> HarmonicCorrection:
    """Remove the once-per-turn tracking error per trial (§9.6).

    The model θᵢ = θ₀ + ω tᵢ + Σₖ [aₖ cos kθᵢ + bₖ sin kθᵢ] is fitted with
    weights 1/σ²_θ over the steady parts of each trial, with θ₀ and ω per
    steady part and segment, plus t² and t³ terms for a part that sweeps at
    least DRIFT_MIN_TURNS turns. A trial is corrected only when its steady parts
    sweep at least one turn, so the harmonics are determined. The corrected
    angle θᶜ = θ − h(θ) has, to first order,

      σ²_c = (1 − h′(θ))² σ²_θ + H Cov(a, b) Hᵀ,

    where H = ∂h/∂(a, b) and Cov(a, b) = (XᵀWX)⁻¹ unscaled; the factor
    (1 − h′) is the effect of the correction on the frame's own noise, which
    §9.6 leaves implicit. Frames outside every trial span are not corrected.
    Errors of θᶜ are correlated across frames through the common coefficients.

    Args:
        t: Frame times in s.
        theta: Angle of §9.5 in rad.
        theta_sd: Its standard deviation in rad.
        segment: Segment index of §9.5, -1 where unusable.
        trials: Trial spans and steady parts.
        n_harmonics: Number of harmonics, 3 in §9.6.

    Returns:
        The corrected angle, its standard deviation, the per-trial fits, and
        the run factor k = max(1, Σχ²/Σν)^½ over the corrected trials.

    Raises:
        ValueError: When two trial spans share a frame.
    """
    t = np.asarray(t, dtype=np.float64)
    theta_c = np.array(theta, dtype=np.float64)
    sd = np.asarray(theta_sd, dtype=np.float64)
    var_c = sd**2
    valid = np.isfinite(theta_c) & np.isfinite(sd) & (sd > 0) & (segment >= 0)
    with np.errstate(divide="ignore"):
        weight = np.where(valid, 1 / np.where(valid, sd, 1.0) ** 2, 0.0)
    claimed = np.zeros(t.size, bool)
    fits: list[TrialHarmonics] = []
    for trial in trials:
        span = _window_mask(trial.span, t)
        if np.any(claimed & span):
            raise ValueError("trial spans overlap")
        claimed |= span
        steady = np.zeros(t.size, bool)
        for w in trial.steady:
            steady |= _window_mask(w, t)
        steady &= span & valid
        blocks: list[BoolArray] = []
        for w in trial.steady:
            in_w = _window_mask(w, t) & span & valid
            for sg in np.unique(segment[in_w]):
                b = in_w & (segment == sg)
                if b.sum() >= MIN_BLOCK_FRAMES:
                    blocks.append(b)
        sweep = sum(float(np.ptp(theta_c[b])) for b in blocks)
        if not blocks or sweep < 2 * np.pi:
            fits.append(_not_corrected(n_harmonics, int(steady.sum())))
            continue
        fit = _fit_trial(t, theta_c, weight, blocks, n_harmonics)
        fits.append(fit)
        if not fit.corrected:
            continue
        apply = span & np.isfinite(theta_c)
        th = theta_c[apply]
        basis = _harmonic_basis(th, n_harmonics)
        slope = _harmonic_slope(th, n_harmonics) @ fit.coef
        var_c[apply] = (1 - slope) ** 2 * var_c[apply] + np.einsum(
            "ni,ij,nj->n", basis, fit.coef_cov, basis
        )
        theta_c[apply] = th - basis @ fit.coef
    done = [f for f in fits if f.corrected]
    chi2 = float(sum(f.chi2 for f in done))
    nu = int(sum(f.nu for f in done))
    k = float(np.sqrt(max(1.0, chi2 / nu))) if nu > 0 else 1.0
    return HarmonicCorrection(
        theta=theta_c,
        theta_sd=np.sqrt(var_c),
        trials=tuple(fits),
        chi2=chi2 if done else np.nan,
        nu=nu,
        k=k,
    )
