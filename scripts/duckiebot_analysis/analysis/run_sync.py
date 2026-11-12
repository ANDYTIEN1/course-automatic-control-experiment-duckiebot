# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""One clock per run, and the comparison of synchronisations (schema.md §10.6-§10.7).

- §10.6 `run_schedule`, `trial_clock`, `sync_run_by_led`, and `align_run_by_motion`:
  one clock for all trials of a run, through the Pi clock they share.
- §10.7 `synchronise`, `compare_syncs`: every synchronisation the sources of a
  run allow, and how far each puts the trial starts from the one used.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import numpy as np
import numpy.typing as npt

from duckiebot_analysis.analysis import sync
from duckiebot_analysis.analysis.sync import (
    ClockFit,
    _first_command,
    _pi_led_edges,
    align_by_correlation,
    departure_time,
    fit_clock,
    sync_by_led,
)
from duckiebot_analysis.data import pilogs
from duckiebot_analysis.data.pilogs import PiEvents

FloatArray = npt.NDArray[np.float64]
# §10.6, §10.7: the ways of synchronising a run's trials, in the order they are
# preferred within one alignment method; a run clock uses every trial at once.
SYNC_NAMES = ("run_led", "trial_led", "run_motion", "trial_motion")


# --- §10.6 one clock per run ---------------------------------------------------


START_TOLERANCE_S = 1.5  # six-motor trial IDs give the start to 1 s (§5)


def shared_clock_problem(
    trials: Sequence[PiEvents], tolerance_s: float = START_TOLERANCE_S
) -> str | None:
    """Why the trials cannot share one Pi clock, or None if they can (§10.6).

    Every trial needs `clock_s`, and where the wall-clock starts are known,
    the differences of `clock_s` must equal theirs within `tolerance_s`, as a
    reboot resets the monotonic clock of a motor-trial log. A §7.1 log takes
    `clock_s` from its own start, so the check cannot see a step of its wall
    clock.
    """
    missing = [
        tr.trial_id + (f" ({tr.clock_note})" if tr.clock_note else "")
        for tr in trials
        if tr.clock_s is None
    ]
    if missing:
        return f"no Pi run clock in the logs of {', '.join(missing)}"
    timed = [tr for tr in trials if tr.start is not None]
    if len(timed) < 2:
        return None
    if len({tr.start.tzinfo is None for tr in timed}) > 1:  # type: ignore[union-attr]
        # A naive start has no known zone, so it cannot be set against an
        # aware one.
        return "some wall-clock starts have a time zone and some have none"
    ref = timed[0]
    bad = []
    for tr in timed[1:]:
        start = tr.start - ref.start  # type: ignore[operator]
        clock = float(tr.clock_s) - float(ref.clock_s)  # type: ignore[arg-type]
        if abs(clock - start.total_seconds()) > tolerance_s:
            bad.append(tr.trial_id)
    if bad:
        return (
            f"the Pi run clock of {', '.join(bad)} disagrees with the wall-clock "
            f"starts by more than {tolerance_s:g} s"
        )
    return None


def run_schedule(
    trials: Sequence[PiEvents], run: str
) -> tuple[PiEvents, dict[str, float]]:
    """All trials of a run as one schedule on the Pi clock they share (§10.6).

    Each trial's events are shifted by its `clock_s` minus the earliest one, so
    the schedule's time 0 is the start of the first trial.

    Args:
        trials: The trials of the run, each with `clock_s`.
        run: Identifier given to the merged schedule.

    Returns:
        The merged events, and each trial's start on the merged time axis.

    Raises:
        ValueError: Without trials, or if a trial has no `clock_s`.
    """
    if not trials:
        raise ValueError("no trials to place on the run clock")
    missing = [tr.trial_id for tr in trials if tr.clock_s is None]
    if missing:
        raise ValueError(
            f"trial(s) {', '.join(missing)} have no Pi run clock (clock_s, §10.6)"
        )
    c0 = min(float(tr.clock_s) for tr in trials)  # type: ignore[arg-type]
    rel = {tr.trial_id: float(tr.clock_s) - c0 for tr in trials}  # type: ignore[arg-type]

    def cat(name: str, shift: bool = False) -> np.ndarray:
        return np.concatenate(
            [
                getattr(tr, name) + rel[tr.trial_id] if shift else getattr(tr, name)
                for tr in trials
            ]
        )

    t_s = cat("t_s", shift=True)
    order = np.argsort(t_s, kind="stable")
    first = min(trials, key=lambda tr: float(tr.clock_s))  # type: ignore[arg-type]
    merged = PiEvents(
        trial=np.full(len(t_s), run),
        t_s=t_s[order],
        t_after_s=cat("t_after_s", shift=True)[order],
        channel=cat("channel")[order],
        value=cat("value")[order],
        event=cat("event")[order],
        t_sd_s=cat("t_sd_s")[order],
        start=first.start,
        source="run",
        clock_s=c0,
    )
    return merged, rel


def trial_clock(
    run: ClockFit,
    rel_s: float,
    start_s: float | None = None,
    end_s: float | None = None,
) -> ClockFit:
    """The run clock seen from a trial that starts `rel_s` into the run (§10.6).

    t = a + (1 + delta)(rel + tau) = [a + (1 + delta) rel] + (1 + delta) tau, so
    the trial's offset is a + (1 + delta) rel, with the covariance carried by
    the Jacobian [[1, rel], [0, 1]]; the drift and timing are the run's.

    Args:
        run: The run clock.
        rel_s: The trial's start on the run's Pi time axis.
        start_s, end_s: The trial's span in its own time; when given, only the
            events the run fit used within it are kept, so `n_edges` and
            `residuals_s` describe this trial.
    """
    jac = np.array([[1.0, rel_s], [0.0, 1.0]])
    tau = run.tau_s - rel_s
    keep = np.ones(len(tau), dtype=bool)
    if start_s is not None:
        keep &= tau >= start_s - 1e-9
    if end_s is not None:
        keep &= tau <= end_s + 1e-9
    return dataclasses.replace(
        run,
        a=run.a + (1 + run.delta) * rel_s,
        cov=jac @ run.cov @ jac.T,
        n_edges=int(keep.sum()),
        residuals_s=run.residuals_s[keep],
        tau_s=tau[keep],
    )


def _split(
    fit: ClockFit, trials: Sequence[PiEvents], rel: Mapping[str, float]
) -> dict[str, ClockFit]:
    return {
        tr.trial_id: trial_clock(
            fit,
            rel[tr.trial_id],
            float(np.min(tr.t_s)),
            float(np.max(tr.t_after_s)),
        )
        for tr in trials
    }


def sync_run_by_led(
    t: np.ndarray,
    led: np.ndarray,
    trials: Sequence[PiEvents],
    *,
    run: str = "run",
    camera_readout_s: float = 0.0,
    min_support: float = 0.5,
) -> tuple[ClockFit, dict[str, ClockFit]]:
    """One LED clock for every trial of a run (§10.2, §10.3, §10.6).

    The trials are merged on their shared Pi clock and matched to the video's
    LED edges as one schedule, so every edge of the run constrains the offset
    and, over a run longer than 10 s, the drift; a trial whose LED was not seen
    is placed by the others.
    The clock must match at least `min_support` of the LED edges of all the
    run's trials, which runs.csv lists as filmed (§4): a schedule of identical
    trials repeats, so a few edges fit a shift by whole trials as well as the
    truth, and such a shift also moves the other trials out of the video.

    Returns:
        The run clock, and the clock of each trial, whose `n_edges` counts the
        trial's own matched edges.

    Raises:
        ValueError: If a trial has no `clock_s`, no LED edge matches, or the
            match covers less than `min_support` of the run's LED edges.
    """
    merged, rel = run_schedule(trials, run)
    clocks, _ = sync_by_led(t, led, [merged], camera_readout_s=camera_readout_s)
    fit = clocks.get(merged.trial_id)
    if fit is None:
        raise ValueError("no LED edges in the video match the run's LED schedule")
    total = len(_pi_led_edges(merged)[0])
    if fit.n_edges < min_support * total:
        raise ValueError(
            f"only {fit.n_edges} of the {total} LED edges of the run's trials "
            f"match, fewer than {min_support:.0%}"
        )
    return fit, _split(fit, trials, rel)


def align_run_by_motion(
    t: np.ndarray,
    omega: np.ndarray,
    omega_sd: np.ndarray,
    trials: Sequence[PiEvents],
    motor: str,
    *,
    run: str = "run",
    window_s: float = 0.4,
    outside_s: float = 1.0,
    rest_s: float = 0.5,
    k_sigma: float = 5.0,
    n_consecutive: int = 3,
    camera_readout_s: float = 0.0,
) -> tuple[ClockFit, dict[str, ClockFit]]:
    """One motion clock for every trial of a run (§10.5, §10.6).

    The run's whole duty schedule, on the shared Pi clock, is first aligned to
    omega by correlation, sign-free, so the wheel must follow every trial at
    once.
    Each trial's first command is then placed by its departure from rest
    (`departure_time`) searched within `window_s` of where that alignment puts
    it, and the clock is fitted to the departures found.
    A trial without a departure there is placed by the others; with none at all
    the correlation is the clock.
    runs.csv lists the trials a video shows (§4), so the correlation searches
    only lags that keep the whole schedule in the video, give or take
    `outside_s` at each end; a schedule of similar trials otherwise fits a
    shift by whole trials, which moves some of them out of the recording.
    A schedule longer than the video is searched over every lag.
    Bias: as for `align_by_motion`, the clock absorbs the motor's delay.

    Returns:
        The run clock, and each trial's clock, whose `n_edges` is 1 when its
        own departure was used and 0 otherwise.

    Raises:
        ValueError: If a trial has no `clock_s` or the motor has no events.
    """
    t = np.asarray(t, dtype=np.float64)
    merged, rel = run_schedule(trials, run)
    first = float(np.min(merged.t_s))
    last = float(np.max(merged.t_after_s))
    frame = float(np.median(np.diff(t)))
    lo, hi = float(t[0]) - first - outside_s, float(t[-1]) - last + outside_s
    if not hi > lo:
        lo, hi = float(t[0]) - first, float(t[-1]) - first
    coarse = align_by_correlation(
        t,
        omega,
        merged,
        motor,
        lag_min_s=lo,
        lag_max_s=hi,
        camera_readout_s=camera_readout_s,
        sign_free=True,
        coarse_step_s=0.05,
    )
    tau, tau_after, tau_sd, dep, dep_sd = [], [], [], [], []
    for ev in trials:
        try:
            k = _first_command(ev, motor)
        except ValueError:
            continue
        r = rel[ev.trial_id]
        mid = r + (ev.t_s[k] + ev.t_after_s[k]) / 2
        pred = float(coarse.to_video(np.array([mid]))[0][0])
        try:
            d, sd = departure_time(
                t,
                omega,
                omega_sd,
                t_from=pred - window_s - rest_s,
                rest_s=rest_s,
                k_sigma=k_sigma,
                n_consecutive=n_consecutive,
            )
        except ValueError:
            continue
        if abs(d - pred) > window_s:
            continue
        tau.append(r + ev.t_s[k])
        tau_after.append(r + ev.t_after_s[k])
        tau_sd.append(ev.t_sd_s[k])
        dep.append(d)
        dep_sd.append(sd)
    if not dep:
        return coarse, _split(coarse, trials, rel)
    fit = fit_clock(
        np.array(tau),
        np.array(dep),
        np.array(dep_sd),
        tau_after_s=np.array(tau_after),
        tau_sd_s=np.array(tau_sd),
        frame_interval_s=frame,
        camera_readout_s=camera_readout_s,
        method="motion",
    )
    return fit, _split(fit, trials, rel)


# --- §10.7 every synchronisation, and their comparison -------------------------


def by_start(events: Iterable[pilogs.PiEvents]) -> list[pilogs.PiEvents]:
    """Return the trials in the order of their wall-clock starts.

    Starts that all carry a zone are ordered in real time. A naive start,
    such as one from a trial ID, has no known zone, so where one occurs the
    starts are ordered by their wall time as it reads, which sorts a naive
    start among aware ones of its own zone. A trial without a start sorts
    as time 0.
    """
    evs = list(events)
    wall = any(e.start is not None and e.start.tzinfo is None for e in evs)

    def key(e: pilogs.PiEvents) -> float:
        if e.start is None:
            return 0.0
        start = e.start.replace(tzinfo=UTC) if wall else e.start
        return start.timestamp()

    return sorted(evs, key=key)


@dataclass(frozen=True)
class Syncs:
    """Every synchronisation of a run's trials from one schedule source.

    Attributes:
        clocks: Clock per trial, for each name of SYNC_NAMES that succeeded.
        run_fits: The run clock behind each ``run_*`` entry (§10.6).
        notes: Why a synchronisation could not be made.
        shared: True when every trial is on the shared Pi clock.
    """

    clocks: dict[str, dict[str, sync.ClockFit]]
    run_fits: dict[str, sync.ClockFit]
    notes: list[str]
    shared: bool = False


def synchronise(
    events: dict[str, pilogs.PiEvents],
    t: FloatArray,
    grid: FloatArray | None,
    motion: Callable[[], tuple[FloatArray, FloatArray]],
    motor: str,
    readout: float | None,
    *,
    run: str,
    led: bool,
) -> Syncs:
    """Synchronise the trials in every way the sources allow (§10.2-§10.7).

    The LED ways are tried only when ``led`` is set, since a run whose
    experimenter found the LED unreadable could match noise.
    A run clock needs every trial on the shared Pi clock (§10.6); the per-trial
    ways are kept to compare with it.
    """
    ordered = by_start(events.values())
    # A readout time of 0 tells sync that it is unknown (§9.1, §10.3).
    readout_s = 0.0 if readout is None else readout
    problem = shared_clock_problem(ordered)
    shared = problem is None
    out = Syncs({}, {}, [], shared)
    if not shared:
        out.notes.append(f"{problem}; each trial is aligned alone")
    if led and grid is None:
        out.notes.append("LED: the LED columns are null")
    elif led and grid is not None:
        if shared:
            try:
                fit, clocks = sync_run_by_led(
                    t, grid, ordered, run=run, camera_readout_s=readout_s
                )
                out.run_fits["run_led"], out.clocks["run_led"] = fit, clocks
            except ValueError as exc:
                out.notes.append(f"run LED clock: {exc}")
        try:
            clocks, _ = sync.sync_by_led(t, grid, ordered, camera_readout_s=readout_s)
            if clocks:
                out.clocks["trial_led"] = clocks
        except ValueError as exc:
            out.notes.append(f"per-trial LED: {exc}")
    try:
        omega, omega_sd = motion()
    except ValueError as exc:
        # The motion ways need the first-pass speed; the LED ways do not.
        out.notes.append(f"motion: no first-pass speed: {exc}")
        return out
    if shared:
        try:
            fit, clocks = align_run_by_motion(
                t, omega, omega_sd, ordered, motor, run=run, camera_readout_s=readout_s
            )
            out.run_fits["run_motion"], out.clocks["run_motion"] = fit, clocks
        except ValueError as exc:
            out.notes.append(f"run motion clock: {exc}")
    try:
        out.clocks["trial_motion"] = motion_per_trial(
            ordered, t, omega, omega_sd, motor, readout_s
        )
    except ValueError as exc:
        out.notes.append(f"per-trial motion: {exc}")
    return out


def motion_per_trial(
    ordered: Sequence[pilogs.PiEvents],
    t: FloatArray,
    omega: FloatArray,
    omega_sd: FloatArray,
    motor: str,
    readout_s: float,
) -> dict[str, sync.ClockFit]:
    """Clock of each trial alone from its first motion, in time order (§10.5).

    Raises:
        ValueError: When a trial can be aligned neither by motion nor by
            correlation.
    """
    clocks = {}
    t_from = float(t[0])
    for ev in ordered:
        try:
            clock = sync.align_by_motion(
                t, omega, omega_sd, ev, motor, t_from=t_from, camera_readout_s=readout_s
            )
        except ValueError:
            # §10.5: a log without an anchor pulse is aligned by correlation.
            first = float(np.min(ev.t_s))
            clock = sync.align_by_correlation(
                t,
                omega,
                ev,
                motor,
                lag_min_s=t_from - first,
                lag_max_s=float(t[-1]) - first,
                camera_readout_s=readout_s,
            )
        clocks[ev.trial_id] = clock
        # The next trial's rest window starts after this trial's last command.
        last, _ = clock.to_video(np.array([float(np.max(ev.t_after_s))]))
        t_from = max(t_from, float(last[0]))
    return clocks


def compare_syncs(
    options: dict[str, dict[str, sync.ClockFit]], chosen: str | None
) -> dict[str, dict[str, Any]]:
    """How far each other synchronisation puts the trial starts from the one used.

    For each other way, the differences of the trial offsets, other minus
    used, over the trials both placed (§10.7).
    A motion way absorbs the motor's delay (§10.5), so against an LED way it
    differs by the delay plus the rise, not by an error.
    """
    if chosen is None:
        return {}
    used = options[chosen]
    out: dict[str, dict[str, Any]] = {}
    for name in SYNC_NAMES:
        if name == chosen or name not in options:
            continue
        diff = {
            trial: options[name][trial].a - used[trial].a
            for trial in used
            if trial in options[name]
        }
        if not diff:
            continue
        worst = max(diff, key=lambda k: abs(diff[k]))
        out[f"{name} - {chosen}"] = {
            "n": len(diff),
            "median_s": float(np.median(list(diff.values()))),
            "max_abs_s": abs(diff[worst]),
            "trial_max_abs": worst,
        }
    return out


def comparison_note(pair: str, c: dict[str, Any]) -> str:
    return (
        f"sync comparison {pair}: n = {c['n']}, median {c['median_s']:+.4f} s, "
        f"max |diff| {c['max_abs_s']:.4f} s ({c['trial_max_abs']})"
    )


def clock_record(
    chosen: str | None, fit: sync.ClockFit | None
) -> dict[str, Any] | None:
    """The clock used, for the file metadata; the run clock's fit when there is one."""
    if chosen is None:
        return None
    scope, method = chosen.split("_")
    rec: dict[str, Any] = {"scope": scope, "method": method}
    if fit is not None:
        rec.update(
            align=fit.method,
            a_s=fit.a,
            a_sd_s=float(np.sqrt(fit.cov[0, 0])),
            drift=fit.delta,
            drift_sd=float(np.sqrt(fit.cov[1, 1])),
            drift_prior=bool(fit.drift_prior),
            timing_sd_s=fit.timing_sd_s,
            n_events=int(fit.n_edges),
            span_s=fit.span_s,
            led_block=fit.led_block,
        )
    return rec


__all__ = [
    "SYNC_NAMES",
    "Syncs",
    "align_run_by_motion",
    "clock_record",
    "compare_syncs",
    "comparison_note",
    "motion_per_trial",
    "run_schedule",
    "shared_clock_problem",
    "by_start",
    "sync_run_by_led",
    "synchronise",
    "trial_clock",
]
