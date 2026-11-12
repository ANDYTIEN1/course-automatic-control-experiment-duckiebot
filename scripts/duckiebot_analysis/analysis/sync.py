# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""The applied input on the video clock, with its timing uncertainty (schema.md §10).

- §10.1 `applied_duty`: the duty after the driver's quantisation.
- §10.2 `detect_led_edges`, §10.3 `fit_clock`, and `sync_by_led`: Pi clock to video
  clock by the
  sync LED.
- §10.5 `align_by_motion` and `align_by_correlation`: the backups without a readable
  LED.
- `command_on_video`: the per-frame input and the per-trial sync record (§11.1, §11.2).

Times are seconds; `tau` is the Pi time of an event from the trial start (§7.1), `t`
video time.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from duckiebot_analysis.data.pilogs import MOTOR_CHANNELS, PiEvents

QUANTISATIONS = ("pca9685", "count255")
DRIFT_PRIOR_SPAN_S = 10.0  # §10.3: below this span the drift is not fitted
# Reweighting steps of the clock fit; it converges in a few.
CLOCK_ITERATIONS = 100
DRIFT_PRIOR_SD = 1e-4  # §10.3: two quartz clocks


# --- §10.1 --------------------------------------------------------------------------
# -------------


def applied_duty(u: np.ndarray | float, quantisation: str) -> np.ndarray:
    """Duty that reaches the motor after the driver's quantisation (§10.1).

    adafruit_motor's DCMotor.throttle sets the 16-bit duty `int(0xFFFF * abs(u))`, and
    adafruit_pca9685's PWMChannel.duty_cycle keeps 12 bits, `(d16 + 1) >> 4`, with
    4096 meaning
    fully on; the applied duty is that count over 4096 with the sign of u.
    With "count255" the script first rounded to `round(255 u) / 255` (Python rounds
    half to even,
    as does numpy).

    Args:
        u: Commanded duty in [-1, 1]; NaN stays NaN.
        quantisation: "pca9685" or "count255" (setup.toml, §6).

    Returns:
        Applied duty, a multiple of 1/4096, same shape as u.

    Raises:
        ValueError: For an unknown quantisation or a duty outside [-1, 1].
    """
    if quantisation not in QUANTISATIONS:
        raise ValueError(
            f"unknown quantisation {quantisation!r}; expected one of {QUANTISATIONS}"
        )
    arr = np.asarray(u, dtype=np.float64)
    if np.any(np.abs(arr[~np.isnan(arr)]) > 1):
        raise ValueError("duty out of range [-1, 1]")
    mag = np.abs(arr)
    if quantisation == "count255":
        mag = np.round(255.0 * mag) / 255.0
    d16 = np.floor(
        65535.0 * mag
    )  # same float64 product as the library's int(0xFFFF * abs(u))
    return np.sign(arr) * np.floor((d16 + 1) / 16) / 4096


# --- §10.2 --------------------------------------------------------------------------
# -------------


def step_signal(x: np.ndarray, m: int = 4) -> np.ndarray:
    """Mean of frames i..i+m-1 minus mean of frames i-m..i-1 (§10.2), else NaN.

    A peak at i marks an edge between frames i-1 and i.
    Null frames are skipped; a side needs at least m/2 valid frames.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    s = np.full(n, np.nan)
    if n < 2 * m:
        return s
    valid = np.isfinite(x)
    cs = np.r_[0.0, np.cumsum(np.where(valid, x, 0.0))]
    cn = np.r_[0, np.cumsum(valid)]
    i = np.arange(m, n - m + 1)
    na, nb = cn[i + m] - cn[i], cn[i] - cn[i - m]
    ok = (na >= max(1, m // 2)) & (nb >= max(1, m // 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        s[i] = np.where(ok, (cs[i + m] - cs[i]) / na - (cs[i] - cs[i - m]) / nb, np.nan)
    return s


@dataclass(frozen=True)
class LedEdges:
    """LED edges found in the video (§10.2).

    Attributes:
        t_s: Edge time, the midpoint of the interval.
        sd_s: Standard deviation, interval / sqrt(12).
        sign: +1 for LED on (R - G rises), -1 for off.
        strength: Step size over the robust noise of the step signal.
        t_before_s: Time of the last frame at the old level.
        t_after_s: Time of the first frame at the new level.
    """

    t_s: np.ndarray
    sd_s: np.ndarray
    sign: np.ndarray
    strength: np.ndarray
    t_before_s: np.ndarray
    t_after_s: np.ndarray


def detect_led_edges(
    t: np.ndarray,
    x: np.ndarray,
    *,
    m: int = 4,
    k_sigma: float = 8.0,
    min_step: float = 4.0,
) -> LedEdges:
    """Edges of one LED block (§10.2).

    An edge is a local extreme of the step signal above `k_sigma` robust standard
    deviations of
    the step signal and above `min_step` counts.
    Its interval runs from the last frame below a quarter of the step to the first
    above three
    quarters, so a frame at an intermediate level (exposed while the LED switched)
    widens it; the
    exposure time is not known, so it cannot narrow it.

    Args:
        t: Frame times (§9.1).
        x: The block's values (`led_NN`, §3.1), NaN for null.
        m: Frames averaged on each side.
        k_sigma: Detection threshold in robust standard deviations.
        min_step: Smallest step in counts.

    Raises:
        ValueError: If t and x differ in length.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.shape != x.shape:
        raise ValueError("t and x must have the same length")
    s = step_signal(x, m)
    fin = np.isfinite(s)
    empty = np.zeros(0)
    if not fin.any():
        return LedEdges(empty, empty, empty.astype(int), empty, empty, empty)
    mad = 1.4826 * np.median(np.abs(s[fin] - np.median(s[fin])))
    noise = max(mad, 0.25)  # a mean of 4 uint8 values has a resolution of 0.25
    a = np.where(fin, np.abs(s), 0.0)
    peaks: list[int] = []
    for i in np.argsort(-a, kind="stable"):
        if a[i] <= max(k_sigma * mad, min_step):
            break
        if all(abs(i - p) > m for p in peaks):
            peaks.append(int(i))
    peaks.sort()
    rows = []
    for p in peaks:
        lo0, hi0 = np.nanmedian(x[p - m : p]), np.nanmedian(x[p : p + m])
        with np.errstate(invalid="ignore"):
            f = (x - lo0) / (hi0 - lo0)
        lo = p - 1
        while lo > p - m and not f[lo] < 0.25:
            lo -= 1
        hi = p
        while hi < p + m - 1 and not f[hi] > 0.75:
            hi += 1
        rows.append((t[lo], t[hi], np.sign(s[p]), a[p] / noise))
    if not rows:
        return LedEdges(empty, empty, empty.astype(int), empty, empty, empty)
    tb, ta, sg, st = (np.array(c) for c in zip(*rows, strict=True))
    return LedEdges((tb + ta) / 2, (ta - tb) / np.sqrt(12), sg.astype(int), st, tb, ta)


# --- §10.3 --------------------------------------------------------------------------
# -------------


@dataclass(frozen=True)
class ClockFit:
    """Map t = a + (1 + delta) tau from a trial's Pi clock to the video clock (§10.3).

    Attributes:
        a: Video time of tau = 0, the trial start.
        delta: Relative clock drift.
        cov: Covariance of (a, delta).
        timing_sd_s: Residual timing uncertainty sigma_tau of one event (§10.3).
        n_edges: Number of events the fit used.
        span_s: Span of their Pi times.
        drift_prior: True if delta was set to 0 with sd 1e-4 (span below 10 s).
        residuals_s: Residuals of the fit.
        method: "led", "motion", or "correlation".
        led_block: LED block used, or None.
        tau_s: Pi times of the events the fit used, the middle of each call;
            residuals_s[i] belongs to tau_s[i].
    """

    a: float
    delta: float
    cov: np.ndarray
    timing_sd_s: float
    n_edges: int
    span_s: float
    drift_prior: bool
    residuals_s: np.ndarray
    method: str = "led"
    led_block: int | None = None
    tau_s: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_video(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Video time of Pi times tau, and its sd (fit plus sigma_tau)."""
        tau = np.asarray(tau, dtype=np.float64)
        var = self.cov[0, 0] + tau**2 * self.cov[1, 1] + 2 * tau * self.cov[0, 1]
        return self.a + (1 + self.delta) * tau, np.sqrt(var + self.timing_sd_s**2)


def fit_clock(
    tau_s: np.ndarray,
    t_video_s: np.ndarray,
    t_video_sd_s: np.ndarray,
    *,
    tau_after_s: np.ndarray | None = None,
    tau_sd_s: np.ndarray | None = None,
    frame_interval_s: float | None = None,
    camera_readout_s: float = 0.0,
    method: str = "led",
    led_block: int | None = None,
) -> ClockFit:
    """Weighted least-squares fit of t = a + (1 + delta) tau (§10.3).

    Each Pi time is the middle of its call, (t_s + t_after_s) / 2, with the uniform sd
    (t_after_s - t_s) / sqrt(12) (§7.1); its variance adds to the video edge
    variance and the
    nominal timing variance `tau_sd_s` (§10.4).
    With a span below 10 s, delta = 0 with sd 1e-4 and only a is fitted.
    Over a longer span, delta is fitted with the prior delta ~ N(0, 1e-4^2), the
    tolerance of two quartz clocks, and the weights are divided by
    max(1, chi2 / nu) of that fit, repeated until it settles, so events that
    scatter more than their stated uncertainties, such as motion departures
    whose bias varies with the duty (§10.5), cannot outweigh the prior.
    The covariance is scaled by max(1, chi2 / nu).
    sigma_tau is the larger of the residual rms (nu degrees of freedom) and the rms
    of the
    stated uncertainties, plus, when `camera_readout_s` is 0 (unknown, §6), half a frame
    interval for the rolling shutter (§9.1).

    Raises:
        ValueError: For no events, unequal lengths, or non-finite values.
    """
    tau = np.asarray(tau_s, dtype=np.float64)
    tv = np.asarray(t_video_s, dtype=np.float64)
    sd = np.asarray(t_video_sd_s, dtype=np.float64)
    after = tau if tau_after_s is None else np.asarray(tau_after_s, dtype=np.float64)
    tsd = (
        np.zeros_like(tau)
        if tau_sd_s is None
        else np.asarray(tau_sd_s, dtype=np.float64)
    )
    if len(tau) == 0:
        raise ValueError("no events to fit the clock")
    if not (tau.shape == tv.shape == sd.shape == after.shape == tsd.shape):
        raise ValueError(
            "tau, t_video, and their uncertainties must have the same length"
        )
    if not all(np.all(np.isfinite(v)) for v in (tau, tv, sd, after, tsd)):
        raise ValueError("non-finite value in the clock fit")
    s2 = np.maximum(sd**2 + (after - tau) ** 2 / 12 + tsd**2, 1e-14)
    tau = (tau + after) / 2
    w = 1 / s2
    y = tv - tau
    span = float(np.ptp(tau))
    prior = span < DRIFT_PRIOR_SPAN_S
    if prior:
        a, delta, p = float(np.sum(w * y) / np.sum(w)), 0.0, 1
    else:
        # The quartz tolerance enters as a prior, delta ~ N(0, 1e-4^2), so that
        # a few events or a bias that varies between them (§10.5) cannot give
        # a drift that no pair of ordinary clocks has; many LED edges over a
        # long span outweigh it. The prior counts as one observation.
        x = np.column_stack([np.ones_like(tau), tau])
        scale = 1.0
        for _ in range(CLOCK_ITERATIONS):
            info = x.T @ ((w / scale)[:, None] * x) + np.diag(
                [0.0, 1 / DRIFT_PRIOR_SD**2]
            )
            a, delta = np.linalg.solve(info, x.T @ (w / scale * y))
            r = y - a - delta * tau
            dof = len(tau) - 1
            new = max(1.0, float(np.sum(w * r**2)) / dof) if dof > 0 else 1.0
            done = abs(new - scale) <= 1e-9 * new
            scale = new
            if done:
                break
        p = 2
    r = y - a - delta * tau
    nu = len(tau) - p
    if prior:
        scale = max(1.0, float(np.sum(w * r**2)) / nu) if nu > 0 else 1.0
        tbar, v = float(np.sum(w * tau) / np.sum(w)), DRIFT_PRIOR_SD**2
        cov = np.array([[scale / np.sum(w) + tbar**2 * v, -tbar * v], [-tbar * v, v]])
    else:
        # The information at the final scale, also when the reweighting
        # stopped before it converged.
        info = x.T @ ((w / scale)[:, None] * x) + np.diag([0.0, 1 / DRIFT_PRIOR_SD**2])
        cov = np.linalg.inv(info)
    resid_var = float(np.sum(r**2)) / nu if nu > 0 else 0.0
    rolling = (
        frame_interval_s / 2 if (camera_readout_s == 0 and frame_interval_s) else 0.0
    )
    timing = np.sqrt(max(resid_var, float(np.mean(s2))) + rolling**2)
    return ClockFit(
        float(a),
        float(delta),
        cov,
        float(timing),
        len(tau),
        span,
        prior,
        r,
        method,
        led_block,
        tau,
    )


def _pi_led_edges(
    ev: PiEvents,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LED switchings of a trial: (t_s, t_after_s, t_sd_s, sign); the LED starts off."""
    m = np.flatnonzero(ev.channel == "led")
    keep, prev = [], 0.0
    for i in m:
        if ev.value[i] != prev:
            keep.append(i)
            prev = ev.value[i]
    k = np.array(keep, dtype=int)
    return ev.t_s[k], ev.t_after_s[k], ev.t_sd_s[k], np.where(ev.value[k] > 0, 1, -1)


def _match(
    edges: LedEdges,
    tau: np.ndarray,
    sign: np.ndarray,
    offset: float,
    tol: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pair shifted Pi edges with the nearest video edge of the same sign within tol.

    Returns video indices, Pi indices, and a score: matches plus up to 0.5 for
        closeness.
    """
    tol = np.broadcast_to(np.asarray(tol, dtype=np.float64), tau.shape)
    iv, ip, err = [], [], []
    for sg in (1, -1):
        vi = np.flatnonzero(edges.sign == sg)
        pi = np.flatnonzero(sign == sg)
        if not len(vi) or not len(pi):
            continue
        tvs = edges.t_s[vi]
        pred = offset + tau[pi]
        j = np.clip(np.searchsorted(tvs, pred), 1, len(tvs)) - 1
        j = np.where(
            (j + 1 < len(tvs))
            & (
                np.abs(tvs[np.minimum(j + 1, len(tvs) - 1)] - pred)
                < np.abs(tvs[j] - pred)
            ),
            j + 1,
            j,
        )
        d = np.abs(tvs[j] - pred)
        best: dict[int, tuple[float, int]] = {}
        for qi in np.flatnonzero(d <= tol[pi]):
            q = int(qi)
            jq = int(j[q])
            if jq not in best or d[q] < best[jq][0]:
                best[jq] = (float(d[q]), q)
        for jj, (dd, qq) in best.items():
            iv.append(vi[jj])
            ip.append(pi[qq])
            err.append(dd / tol[pi[qq]])
    score = len(iv) + 0.5 * (1 - float(np.mean(err))) if iv else 0.0
    return np.array(iv, dtype=int), np.array(ip, dtype=int), score


def _candidates(
    edges: LedEdges, tau: np.ndarray, sign: np.ndarray, tol: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Offsets that put one of up to 10 Pi edges on a same-sign video edge, scored."""
    ks = np.unique(np.linspace(0, len(tau) - 1, min(10, len(tau))).round().astype(int))
    offs = np.unique(
        np.concatenate([edges.t_s[edges.sign == sign[k]] - tau[k] for k in ks])
    )
    return offs, np.array([_match(edges, tau, sign, o, tol)[2] for o in offs])


def sync_by_led(
    t: np.ndarray,
    led: np.ndarray,
    trials: Sequence[PiEvents],
    *,
    start_tolerance_s: float = 1.5,
    camera_readout_s: float = 0.0,
) -> tuple[dict[str, ClockFit], int]:
    """Clock fit of every trial of a run by the sync LED (§10.2, §10.3).

    The LED block whose edges best match the Pi's LED schedule is used for the run:
    for each
    block, each trial's offset is searched among the offsets that put a Pi edge on a
    video edge;
    with several trials their offsets must agree with the differences of their
    wall-clock starts
    within `start_tolerance_s` (the six-motor IDs give the start to 1 s), which
    removes the
    ambiguity of identical, periodic trials.
    A trial is then fitted on its matched edges, re-matched with the fitted clock,
    and refitted.

    Args:
        t: Frame times (§9.1).
        led: LED blocks, shape (frames, 25), columns `led_00` ... `led_24` (§3.1).
        trials: The trials of the run, each with LED events.
        start_tolerance_s: Allowed error of the wall-clock start differences.
        camera_readout_s: From setup.toml; 0 if unknown (§6).

    Returns:
        ClockFit per trial identifier (trials without a matched edge are left
        out), and the block.

    Raises:
        ValueError: If no block matches any edge, or several trials lack a
            wall-clock start.
    """
    t = np.asarray(t, dtype=np.float64)
    led = np.asarray(led, dtype=np.float64)
    if led.ndim != 2 or led.shape[0] != len(t):
        raise ValueError("led must have one row per frame")
    if len(trials) > 1 and any(tr.start is None for tr in trials):
        raise ValueError(
            "several trials in a run need their wall-clock start to be told apart"
        )
    frame = float(np.median(np.diff(t)))
    pis = [_pi_led_edges(tr) for tr in trials]
    start0 = trials[0].start if trials else None
    rel = [
        (tr.start - start0).total_seconds()
        if tr.start is not None and start0 is not None
        else 0.0
        for tr in trials
    ]
    best: tuple[float, int, list[float], LedEdges] | None = None
    for b in range(led.shape[1]):
        edges = detect_led_edges(t, led[:, b])
        if not len(edges.t_s):
            continue
        cands = []
        for ts, ta, _, sg in pis:
            tau = (ts + ta) / 2
            tol = (
                max(3 * frame, 0.015) + 2e-4 * (tau - tau.min())
                if len(tau)
                else np.zeros(0)
            )
            cands.append(
                _candidates(edges, tau, sg, tol)
                if len(tau)
                else (np.zeros(0), np.zeros(0))
            )
        globals_ = np.unique(
            np.concatenate([o - r for (o, _), r in zip(cands, rel, strict=True)])
        )
        for g in globals_:
            total, offs = 0.0, []
            for (o, sc), r in zip(cands, rel, strict=True):
                ok = np.abs(o - g - r) <= (
                    start_tolerance_s if len(trials) > 1 else np.inf
                )
                k = int(np.argmax(np.where(ok, sc, -1.0))) if len(o) else -1
                total += sc[k] if k >= 0 and ok[k] else 0.0
                offs.append(o[k] if k >= 0 and ok[k] else np.nan)
            if best is None or total > best[0]:
                best = (total, b, offs, edges)
    if best is None or best[0] < 1:
        raise ValueError("no LED edges in the video match the Pi's LED schedule")
    _, block, offs, edges = best
    clocks: dict[str, ClockFit] = {}
    for tr, (ts, ta, tsd, sg), off in zip(trials, pis, offs, strict=True):
        if not np.isfinite(off):
            continue
        tau = (ts + ta) / 2
        iv, ip, _ = _match(
            edges, tau, sg, off, max(3 * frame, 0.015) + 2e-4 * (tau - tau.min())
        )
        for _ in range(2):
            fit = fit_clock(
                ts[ip],
                edges.t_s[iv],
                edges.sd_s[iv],
                tau_after_s=ta[ip],
                tau_sd_s=tsd[ip],
                frame_interval_s=frame,
                camera_readout_s=camera_readout_s,
                led_block=block,
            )
            pred, psd = fit.to_video(tau)
            iv, ip, _ = _match(edges, pred, sg, 0.0, np.maximum(4 * psd, 2 * frame))
        clocks[tr.trial_id] = fit_clock(
            ts[ip],
            edges.t_s[iv],
            edges.sd_s[iv],
            tau_after_s=ta[ip],
            tau_sd_s=tsd[ip],
            frame_interval_s=frame,
            camera_readout_s=camera_readout_s,
            led_block=block,
        )
    return clocks, block


# --- §10.5 --------------------------------------------------------------------------
# -------------


def _first_command(ev: PiEvents, motor: str) -> int:
    ch = MOTOR_CHANNELS[motor][0]
    idx = np.flatnonzero((ev.channel == ch) & (ev.value != 0))
    if not len(idx):
        raise ValueError(f"trial {ev.trial_id} has no non-zero {ch} command")
    return int(idx[0])


def departure_time(
    t: np.ndarray,
    omega: np.ndarray,
    omega_sd: np.ndarray,
    *,
    t_from: float,
    rest_s: float = 0.5,
    k_sigma: float = 5.0,
    n_consecutive: int = 3,
) -> tuple[float, float]:
    """First departure of omega from its rest level by more than k_sigma sigma (§10.5).

    The rest level is the median of omega over `rest_s` from `t_from`; sigma per
    frame is the
    larger of `omega_sd` and the robust spread at rest.
    The departure must last `n_consecutive` usable frames; its time is the midpoint
    between the
    last usable frame before it and its first frame, with sd interval / sqrt(12).

    Raises:
        ValueError: Without enough rest frames or without a departure.
    """
    t, w, s = (np.asarray(v, dtype=np.float64) for v in (t, omega, omega_sd))
    idx = np.flatnonzero(np.isfinite(w) & np.isfinite(s) & (t >= t_from))
    rest = idx[t[idx] < t_from + rest_s]
    if len(rest) < 5:
        raise ValueError("fewer than 5 usable rest frames before the departure")
    level = np.median(w[rest])
    spread = 1.4826 * np.median(np.abs(w[rest] - level))
    out = np.abs(w[idx] - level) > k_sigma * np.maximum(s[idx], spread)
    out[: len(rest)] = False
    run = (
        np.convolve(out.astype(int), np.ones(n_consecutive, dtype=int), "valid")
        == n_consecutive
    )
    hits = np.flatnonzero(run)
    if not len(hits):
        raise ValueError("no departure of omega from rest found")
    j = hits[0]
    t0, t1 = t[idx[j - 1]], t[idx[j]]
    return (t0 + t1) / 2, (t1 - t0) / np.sqrt(12)


def align_by_motion(
    t: np.ndarray,
    omega: np.ndarray,
    omega_sd: np.ndarray,
    events: PiEvents,
    motor: str,
    *,
    t_from: float | None = None,
    rest_s: float = 0.5,
    k_sigma: float = 5.0,
    n_consecutive: int = 3,
    camera_readout_s: float = 0.0,
) -> ClockFit:
    """Clock of a trial from the first wheel motion after its first command (§10.5).

    The first non-zero command is the anchor pulse or the first step.
    Bias: the clock places the command at the departure, so it absorbs the motor
        delay plus the
    time the speed takes to rise by k_sigma sigma, about tau_m * k_sigma sigma /
    omega_ss for a
    first-order motor; ω from a windowed fit (§9.7) can depart up to the half window
    early.
    Delay and zero-crossing times are therefore not computed from this alignment
    (§10.5).

    Args:
        t, omega, omega_sd: Frame times, angular velocity, and its sd (§9.7).
        events: The trial.
        motor: "left", "right", or "both".
        t_from: Start of the rest window; default the first frame.

    Raises:
        ValueError: Without a non-zero command or a departure.
    """
    k = _first_command(events, motor)
    t = np.asarray(t, dtype=np.float64)
    t_dep, sd = departure_time(
        t,
        omega,
        omega_sd,
        t_from=float(t[0]) if t_from is None else t_from,
        rest_s=rest_s,
        k_sigma=k_sigma,
        n_consecutive=n_consecutive,
    )
    return fit_clock(
        events.t_s[[k]],
        np.array([t_dep]),
        np.array([sd]),
        tau_after_s=events.t_after_s[[k]],
        tau_sd_s=events.t_sd_s[[k]],
        frame_interval_s=float(np.median(np.diff(t))),
        camera_readout_s=camera_readout_s,
        method="motion",
    )


def _staircase(ev: PiEvents, motor: str) -> tuple[np.ndarray, np.ndarray]:
    ch = MOTOR_CHANNELS[motor][0]
    m = ev.channel == ch
    if not m.any():
        raise ValueError(f"trial {ev.trial_id} has no {ch} events")
    return (ev.t_s[m] + ev.t_after_s[m]) / 2, ev.value[m]


def align_by_correlation(
    t: np.ndarray,
    omega: np.ndarray,
    events: PiEvents,
    motor: str,
    *,
    lag_min_s: float,
    lag_max_s: float,
    camera_readout_s: float = 0.0,
    sign_free: bool = False,
    coarse_step_s: float = 0.02,
) -> ClockFit:
    """Clock of a trial without anchor pulse, by corr(omega, duty) (§10.5).

    The duty staircase of the log (0 before the first command) is shifted by each
    lag on a grid,
    first at max(frame, `coarse_step_s`) on every frame closer than a quarter of
    that step thinned out, then at the frame interval around the best on every
    frame, and the peak is
    refined by a parabola.
    Bias: the peak lags the command by about the motor delay plus its time constant;
        the clock
    absorbs it.
    The sd is the frame interval / sqrt(12), a lower bound.

    Args:
        sign_free: Maximise |corr| instead of corr, for an omega whose sign is
            not known, such as before `positive_duty_turns` is set (§6).
        coarse_step_s: Step of the coarse grid; the fine grid spans one step on
            each side of the best coarse lag.

    Raises:
        ValueError: If the motor has no events or the lag range is empty.
    """
    if not lag_max_s > lag_min_s:
        raise ValueError("lag_max_s must exceed lag_min_s")
    tau, val = _staircase(events, motor)
    t = np.asarray(t, dtype=np.float64)
    ok = np.isfinite(np.asarray(omega, dtype=np.float64))
    tt, w = t[ok], np.asarray(omega, dtype=np.float64)[ok]
    wc = (w - w.mean()) / (w.std() or 1.0)
    frame = float(np.median(np.diff(t)))
    step = max(frame, coarse_step_s)

    def corr(lag: float, stride: int = 1) -> float:
        ts, ws = tt[::stride], wc[::stride]
        j = np.searchsorted(tau, ts - lag, side="right") - 1
        u = np.where(j >= 0, val[np.maximum(j, 0)], 0.0)
        sd = u.std()
        if not sd > 0:
            return -np.inf
        c = float(np.mean(ws * (u - u.mean())) / sd)
        return abs(c) if sign_free else c

    # A grid of step s cannot resolve a peak narrower than s, so frames closer
    # than s/4 add no information to the coarse search.
    stride = max(1, int(step / (4 * frame)))
    coarse = np.arange(lag_min_s, lag_max_s + 1e-12, step)
    c0 = coarse[int(np.argmax([corr(g, stride) for g in coarse]))]
    half = max(0.05, step)
    fine = np.arange(c0 - half, c0 + half + 1e-12, frame)
    r = np.array([corr(g) for g in fine])
    k = int(np.argmax(r))
    lag = fine[k]
    if 0 < k < len(fine) - 1 and np.all(np.isfinite(r[k - 1 : k + 2])):
        den = r[k - 1] - 2 * r[k] + r[k + 1]
        if den < 0:
            lag += frame * 0.5 * (r[k - 1] - r[k + 1]) / den
    fit = fit_clock(
        np.zeros(1),
        np.array([lag]),
        np.array([frame / np.sqrt(12)]),
        frame_interval_s=frame,
        camera_readout_s=camera_readout_s,
        method="correlation",
    )
    # The lag places the whole schedule, not an event, so the clock counts no
    # event of its own (§11.2 n_sync_events).
    return dataclasses.replace(
        fit, n_edges=0, residuals_s=np.zeros(0), tau_s=np.zeros(0)
    )


# --- per-frame command (§11)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CommandOnVideo:
    """The input per frame (§11.1) and the sync record per trial (§11.2).

    Attributes:
        u_cmd: Commanded duty at each frame; NaN before the first command.
        u: Applied duty (§10.1).
        trial: Trial running at each frame, or None.
        t_trial_s: Time from that trial's start on the video clock, NaN outside trials.
        trials: Per trial: trial, run, sync, align, offset_s, offset_sd_s, drift,
            drift_sd, timing_sd_s, led_block.
    """

    u_cmd: np.ndarray
    u: np.ndarray
    trial: np.ndarray
    t_trial_s: np.ndarray
    trials: list[dict[str, object]]


def command_on_video(
    t: np.ndarray,
    events: Sequence[PiEvents],
    clocks: Mapping[str, ClockFit],
    quantisation: str,
    motor: str,
    *,
    run: str = "",
) -> CommandOnVideo:
    """Place the logged or reconstructed commands on the video clock (§10.3, §11).

    A command at Pi time tau, the middle of its call, holds from a + (1 + delta) tau
    until the next.
    A trial runs from its start (or its first event, if earlier) until the next
    trial starts; the
    last runs to the end of the video.

    Args:
        t: Frame times (§9.1).
        events: The trials of the run.
        clocks: ClockFit per trial identifier.
        quantisation: "pca9685" or "count255" (§6).
        motor: "left", "right", or "both" (both channels must then carry the same duty).
        run: Run identifier for the records.

    Raises:
        ValueError: For a trial without clock or without events of the motor.
    """
    t = np.asarray(t, dtype=np.float64)
    tv_all, val_all, starts, offsets, ids, records = [], [], [], [], [], []
    for ev in events:
        clock = clocks.get(ev.trial_id)
        if clock is None:
            raise ValueError(f"no clock for trial {ev.trial_id}")
        tau, val = _staircase(ev, motor)
        if motor == "both":
            tau_r, val_r = _staircase(ev, "right")
            if not (np.array_equal(val, val_r) and np.allclose(tau, tau_r, atol=0.05)):
                raise ValueError(
                    f"trial {ev.trial_id}: the two motors differ; choose one motor"
                )
        tv_all.append(clock.to_video(tau)[0])
        val_all.append(val)
        starts.append(float(clock.to_video(np.array([min(0.0, ev.t_s.min())]))[0][0]))
        offsets.append(clock.a)
        ids.append(ev.trial_id)
        sync = ("script" if ev.from_script else "log") + (
            "+led" if clock.method == "led" else "+motion"
        )
        records.append(
            {
                "trial": ev.trial_id,
                "run": run,
                "sync": sync,
                # §10.5 groups motion and correlation under "+motion"; the
                # method tells their different biases apart (§11.2).
                "align": clock.method,
                "offset_s": clock.a,
                "offset_sd_s": float(np.sqrt(clock.cov[0, 0])),
                "drift": clock.delta,
                "drift_sd": float(np.sqrt(clock.cov[1, 1])),
                "timing_sd_s": clock.timing_sd_s,
                "led_block": clock.led_block,
            }
        )
    tv = np.concatenate(tv_all) if tv_all else np.zeros(0)
    vals = np.concatenate(val_all) if val_all else np.zeros(0)
    order = np.argsort(tv, kind="stable")
    tv, vals = tv[order], vals[order]
    j = np.searchsorted(tv, t, side="right") - 1
    u_cmd = np.where(j >= 0, vals[np.maximum(j, 0)] if len(vals) else np.nan, np.nan)
    so = np.argsort(starts, kind="stable")
    st = np.array(starts)[so]
    k = np.searchsorted(st, t, side="right") - 1
    trial = np.array([None if kk < 0 else ids[so[kk]] for kk in k], dtype=object)
    off = np.array(offsets)[so] if len(so) else np.zeros(0)
    t_trial = np.where(k >= 0, t - (off[np.maximum(k, 0)] if len(off) else 0.0), np.nan)
    return CommandOnVideo(
        u_cmd,
        applied_duty(u_cmd, quantisation),
        trial,
        t_trial,
        [records[i] for i in so],
    )
