# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Textbook quantities of a measured step response, and first-order fits.

The quantities follow Franklin, Powell and Emami-Naeini (2019), Feedback
Control of Dynamic Systems, 8th ed., §3.4: the rise time from 10 % to 90 %
of the change, the overshoot Mp as the largest excursion beyond the final
value divided by the change, and the settling time as the time after which
the response stays within 1 % of the change about its final value. For a step
from an operating point, every fraction is of the change, not of the final
value, so a step from 0.6 to 1.0 is judged like a unit step.

A measured response is noisy, so a quantity is reported only where the noise
lets it be resolved: the overshoot when it exceeds both the settling band and
k_noise standard deviations of the response about its final value, and the
settling time when the band is wider than k_noise standard deviations.

The fits estimate the first-order model, optionally with a dead time, by
weighted least squares on the angular velocity of all repetitions together,
with one gain, time constant and delay and one initial speed per repetition.
The speeds of neighbouring frames are correlated, because each comes from a
window of frames (data/schema.md §9.7), so the covariance of that fit
understates the uncertainty; the spread of the same fit made per repetition
is returned beside it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import least_squares

from duckiebot_analysis.analysis.six_motor import hold_segments, hold_window

FloatArray = npt.NDArray[np.float64]
RISE_LIMITS = (0.1, 0.9)  # Franklin et al. (2019) §3.4.1
SETTLING_BAND = 0.01  # Franklin et al. (2019) §3.4.3
K_NOISE = 2.0


@dataclass(frozen=True)
class StepInfo:
    """Rise, overshoot and settling of one step response; NaN where absent.

    Times are from the command, in s. ``overshoot`` is a fraction of the change.
    """

    initial: float
    final: float
    t_rise_start_s: float
    t_rise_end_s: float
    peak_time_s: float
    overshoot: float
    settling_time_s: float

    @property
    def rise_time_s(self) -> float:
        """Time from 10 % to 90 % of the change."""
        return self.t_rise_end_s - self.t_rise_start_s

    @property
    def moved(self) -> bool:
        """Whether the response reached 90 % of a resolvable change."""
        return bool(np.isfinite(self.t_rise_end_s))


def _clean(t: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return t and y sorted by t, without samples where t or y is NaN.

    y is 1-D or holds one column per series; a NaN in any column drops the row.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.shape[:1] != t.shape or y.ndim > 2 or t.size == 0:
        raise ValueError("need equally long, non-empty 1-D t and y")
    ok = np.isfinite(t) & np.all(np.isfinite(y.reshape(t.size, -1)), axis=1)
    order = np.argsort(t[ok], kind="stable")
    return t[ok][order], y[ok][order]


def _first_crossing(t: FloatArray, z: FloatArray, level: float) -> float:
    above = np.flatnonzero(z >= level)
    if above.size == 0:
        return np.nan
    i = int(above[0])
    if i == 0:
        return float(t[0])
    frac = (level - z[i - 1]) / (z[i] - z[i - 1])
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def step_info(
    t: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    initial: float,
    final: float,
    noise_sd: float = 0.0,
    min_change: float = 0.0,
    y_settle: npt.ArrayLike | None = None,
    t_final_s: float = np.inf,
    band: float = SETTLING_BAND,
    limits: tuple[float, float] = RISE_LIMITS,
    k_noise: float = K_NOISE,
) -> StepInfo:
    """Return the rise time, overshoot and settling time of a step response.

    Args:
        t: Time from the command in s; samples before 0 are ignored.
        y: Response; NaN samples are ignored.
        initial: Value before the step.
        final: Value the response settles to.
        noise_sd: Standard deviation of y about its final value, from noise and
            ripple.
        min_change: Smallest change counted as a response, in the unit of y.
        y_settle: Response, on the same t, from which the overshoot and the
            settling time are taken, such as y smoothed over the period of a
            ripple; y if None.
        t_final_s: Start of the window whose mean is the final value. An
            excursion inside it is noise about that mean, so the peak and the
            exits from the band are searched before it, and the response must
            be inside the band when the window starts.
        band: Settling band as a fraction of the change.
        limits: Fractions of the change between which the rise time is taken.
        k_noise: Number of noise_sd an overshoot must exceed, and the band must
            exceed, to be resolved.

    Returns:
        The quantities; all NaN when the change is not larger than min_change
        and k_noise noise_sd.

    Raises:
        ValueError: For empty or mismatched t and y.
    """
    ys = y if y_settle is None else y_settle
    t, both = _clean(t, np.c_[np.asarray(y, float), np.asarray(ys, float)])
    nan = StepInfo(initial, final, np.nan, np.nan, np.nan, np.nan, np.nan)
    change = final - initial
    after = t >= 0
    if abs(change) <= max(min_change, k_noise * noise_sd) or after.sum() < 2:
        return nan
    t, z, zs = t[after], *((both[after].T - initial) / change)
    lo = _first_crossing(t, z, limits[0])
    hi = _first_crossing(t, z, limits[1])
    if not np.isfinite(hi):
        return nan
    peak_t, mp = np.nan, np.nan
    post = (t >= hi) & (t < t_final_s)
    if post.any():
        k = int(np.argmax(zs[post]))
        excess = zs[post][k] - 1
        # An excursion inside the settling band counts as settled (§3.4.3).
        if excess > band and excess * abs(change) > k_noise * noise_sd:
            peak_t, mp = float(t[post][k]), float(excess)
    ts = np.nan
    before = t < t_final_s
    if band * abs(change) > k_noise * noise_sd and before.sum() >= 2:
        tb, e = t[before], np.abs(zs[before] - 1)
        out = np.flatnonzero(e > band)
        if out.size == 0:
            ts = float(tb[0])
        elif out[-1] < tb.size - 1:
            i = int(out[-1])
            frac = (e[i] - band) / (e[i] - e[i + 1])
            ts = float(tb[i] + frac * (tb[i + 1] - tb[i]))
    return StepInfo(initial, final, lo, hi, peak_t, mp, ts)


def moving_mean(t: npt.ArrayLike, y: npt.ArrayLike, width_s: float) -> FloatArray:
    """Return the centred moving mean of y over width_s, skipping NaN.

    t must be increasing. A point with no finite sample in its window is NaN.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    csum = np.r_[0.0, np.cumsum(np.where(ok, y, 0.0))]
    ccnt = np.r_[0, np.cumsum(ok)]
    # A tolerance keeps the window symmetric where its edge falls on a sample.
    half = width_s / 2 + 1e-9
    a = np.searchsorted(t, t - half, side="left")
    b = np.searchsorted(t, t + half, side="right")
    n = ccnt[b] - ccnt[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (csum[b] - csum[a]) / n, np.nan)


@dataclass(frozen=True)
class StepData:
    """One repetition of a step: time from the command, speed and its sd."""

    t: FloatArray
    omega: FloatArray
    omega_sd: FloatArray


def mean_response(
    reps: Sequence[StepData], grid: npt.ArrayLike, max_gap_s: float = 0.02
) -> FloatArray:
    """Return the mean over repetitions of the speed interpolated onto grid.

    A repetition contributes nothing where its finite samples are more than
    max_gap_s apart or outside its record.
    """
    grid = np.asarray(grid, dtype=float)
    rows = []
    for r in reps:
        ok = np.isfinite(r.t) & np.isfinite(r.omega)
        ts, ws = r.t[ok], r.omega[ok]
        row = np.full(grid.size, np.nan)
        if ts.size >= 2:
            j = np.clip(np.searchsorted(ts, grid), 1, ts.size - 1)
            inside = (
                (grid >= ts[0]) & (grid <= ts[-1]) & (ts[j] - ts[j - 1] <= max_gap_s)
            )
            row[inside] = np.interp(grid[inside], ts, ws)
        rows.append(row)
    if not rows:
        return np.full(grid.size, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.array(rows), axis=0)


def first_order_step(
    t: npt.ArrayLike,
    omega0: float,
    gain: float,
    delta_u: float,
    tau: float,
    delay: float = 0.0,
) -> FloatArray:
    """Return the first-order response, with dead time, to a duty step at t = 0."""
    t = np.asarray(t, dtype=float)
    s = np.maximum(t - delay, 0.0)
    return omega0 + gain * delta_u * (1.0 - np.exp(-s / tau))


@dataclass(frozen=True)
class FirstOrderFit:
    """First-order fit of all repetitions of a step.

    Attributes:
        gain: G, in rad/s per unit duty.
        tau_s: Time constant.
        delay_s: Dead time; 0 when not fitted.
        omega0: Initial speed of each repetition.
        delta_u: Duty step.
        gain_se, tau_se, delay_se: Standard errors from the fit covariance,
            scaled by max(1, chi2/nu); they assume independent residuals.
        gain_sd, tau_sd, delay_sd: Standard deviation of the per-repetition
            fits; NaN for one repetition.
        per_rep: (gain, tau_s, delay_s) of each repetition fitted alone.
        rms_rel: RMS of the residuals after the command, relative to the
            fitted change G * delta_u.
        n: Number of frames fitted.
    """

    gain: float
    tau_s: float
    delay_s: float
    omega0: list[float]
    delta_u: float
    gain_se: float
    tau_se: float
    delay_se: float
    gain_sd: float
    tau_sd: float
    delay_sd: float
    per_rep: list[tuple[float, float, float]]
    rms_rel: float
    n: int

    def predict(self, t: npt.ArrayLike, omega0: float) -> FloatArray:
        """Return the fitted response from the initial speed omega0."""
        return first_order_step(
            t, omega0, self.gain, self.delta_u, self.tau_s, self.delay_s
        )


def _arrays(
    reps: Sequence[StepData],
) -> list[tuple[FloatArray, FloatArray, FloatArray]]:
    out = []
    for r in reps:
        t = np.asarray(r.t, dtype=float)
        w = np.asarray(r.omega, dtype=float)
        sd = np.asarray(r.omega_sd, dtype=float)
        ok = np.isfinite(t) & np.isfinite(w) & np.isfinite(sd) & (sd > 0)
        if ok.sum() < 5:
            raise ValueError("every repetition needs at least five finite frames")
        out.append((t[ok], w[ok], sd[ok]))
    return out


def _solve(
    data: list[tuple[FloatArray, FloatArray, FloatArray]], du: float, delay: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the parameters, their covariance and the residuals per frame."""
    n_rep = len(data)
    n_shape = 3 if delay else 2

    def unpack(z: FloatArray) -> tuple[float, float, float, FloatArray]:
        theta = float(z[2]) if delay else 0.0
        return float(z[0]), float(np.exp(z[1])), theta, z[n_shape:]

    def resid(z: FloatArray) -> FloatArray:
        g, tau, theta, w0 = unpack(z)
        return np.concatenate(
            [
                (first_order_step(t, w0[i], g, du, tau, theta) - w) / sd
                for i, (t, w, sd) in enumerate(data)
            ]
        )

    pre = [
        float(np.mean(w[t < 0])) if np.any(t < 0) else float(w[0]) for t, w, _ in data
    ]
    late = [float(np.mean(w[t >= 0.8 * t.max()])) for t, w, _ in data]
    g0 = (np.mean(late) - np.mean(pre)) / du
    lower = [-np.inf, np.log(1e-4)] + ([0.0] if delay else []) + [-np.inf] * n_rep
    upper = [np.inf, np.log(1e2)] + ([0.3] if delay else []) + [np.inf] * n_rep
    best = None
    for tau0 in (0.02, 0.1, 0.4):
        for th0 in (0.0, 0.01, 0.03) if delay else (None,):
            z0 = [g0, np.log(tau0)] + ([th0] if delay else []) + pre
            r = least_squares(resid, z0, bounds=(lower, upper), x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
    assert best is not None
    nu = max(best.fun.size - best.x.size, 1)
    scale = max(1.0, float(np.sum(best.fun**2)) / nu)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * scale
    except np.linalg.LinAlgError:
        cov = np.full((best.x.size, best.x.size), np.nan)
    sds = np.concatenate([sd for _, _, sd in data])
    return best.x, cov, best.fun * sds


def fit_first_order(
    reps: Sequence[StepData],
    delta_u: float,
    *,
    delay: bool = False,
    rms_window_s: float | None = None,
) -> FirstOrderFit:
    """Fit the first-order model, with dead time if asked, to all repetitions.

    Args:
        reps: Repetitions of the step, time from the logged command; frames
            before the command fix the initial speed.
        delta_u: Duty step.
        delay: Fit a dead time (Eq. fopdt) instead of none (Eq. step).
        rms_window_s: Length after the command over which the RMS error is
            taken; the whole record after the command if None.

    Raises:
        ValueError: Without repetitions, with a repetition of fewer than five
            finite frames, or for a zero step.
    """
    if not reps:
        raise ValueError("no repetitions to fit")
    if delta_u == 0:
        raise ValueError("the duty step must not be zero")
    data = _arrays(reps)
    x, cov, res = _solve(data, delta_u, delay)
    gain, tau = float(x[0]), float(np.exp(x[1]))
    theta = float(x[2]) if delay else 0.0
    se = np.sqrt(np.abs(np.diag(cov)))
    t_all = np.concatenate([t for t, _, _ in data])
    sel = t_all >= 0
    if rms_window_s is not None:
        sel &= t_all <= rms_window_s
    rms = float(np.sqrt(np.mean(res[sel] ** 2))) / abs(gain * delta_u)
    per_rep: list[tuple[float, float, float]] = []
    if len(data) > 1:
        for d in data:
            xi, _, _ = _solve([d], delta_u, delay)
            per_rep.append(
                (float(xi[0]), float(np.exp(xi[1])), float(xi[2]) if delay else 0.0)
            )
    spread = (
        np.std(np.array(per_rep), axis=0, ddof=1) if per_rep else np.full(3, np.nan)
    )
    n_shape = 3 if delay else 2
    return FirstOrderFit(
        gain=gain,
        tau_s=tau,
        delay_s=theta,
        omega0=[float(v) for v in x[n_shape:]],
        delta_u=delta_u,
        gain_se=float(se[0]),
        tau_se=float(tau * se[1]),
        delay_se=float(se[2]) if delay else 0.0,
        gain_sd=float(spread[0]),
        tau_sd=float(spread[1]),
        delay_sd=float(spread[2]) if delay else (np.nan if not per_rep else 0.0),
        per_rep=per_rep,
        rms_rel=rms,
        n=int(t_all.size),
    )


@dataclass(frozen=True)
class LevelSpeed:
    """Speed at the end of one held duty level of a sweep."""

    duty: float
    t_start_s: float
    t_end_s: float
    rising: bool
    mean: float
    sd: float
    n: int


def mean_ci95(level: LevelSpeed) -> float:
    """Return the half-width of the 95 % t confidence interval of a level's mean.

    The frames are taken as independent, so with correlated speed noise the
    interval is a lower bound.
    """
    if level.n < 2 or not np.isfinite(level.sd):
        return float("nan")
    return float(stats.t.ppf(0.975, level.n - 1) * level.sd / np.sqrt(level.n))


def level_speeds(
    t: npt.ArrayLike,
    omega: npt.ArrayLike,
    u: npt.ArrayLike,
    last_s: float,
    min_hold_s: float,
) -> list[LevelSpeed]:
    """Return the mean and sd of the speed over the last last_s of each level.

    Levels of zero duty are left out; a level is rising when its |duty|
    exceeds that of the level before it.
    """
    t = np.asarray(t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    u = np.asarray(u, dtype=float)
    out: list[LevelSpeed] = []
    prev = 0.0
    for duty, start, end in hold_segments(t, u, min_s=min_hold_s):
        if duty != 0:
            w = hold_window(t, omega, start, end, last_s)
            out.append(
                LevelSpeed(
                    duty=duty,
                    t_start_s=start,
                    t_end_s=end,
                    rising=abs(duty) > abs(prev),
                    mean=float(np.mean(w)) if w.size else np.nan,
                    sd=float(np.std(w, ddof=1)) if w.size > 1 else np.nan,
                    n=int(w.size),
                )
            )
        prev = duty
    return out


def start_stop_duties(
    levels: Sequence[LevelSpeed], moving: float
) -> tuple[float, float]:
    """Return the |duty| that started the wheel and the |duty| it stopped at.

    The start duty is the first rising level whose mean |speed| exceeds
    moving; the stop duty is the first falling level after it at which the
    wheel is at rest. Each is NaN if it did not occur.
    """
    start, stop = np.nan, np.nan
    for lv in levels:
        turning = abs(lv.mean) > moving
        if np.isnan(start):
            if lv.rising and turning:
                start = abs(lv.duty)
        elif not lv.rising and not turning:
            stop = abs(lv.duty)
            break
    return start, stop
