# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Results of the CSME 2026 case study of one DB17 wheel motor.

the CSME 2026 article reports what a
student meets when step-testing motor m5 (campaign label R3-L) on the stand.
This module reads the processed runs of that motor (data/schema.md §11) and
the Pi logs (§7) and computes every quantity the article reports:

* E1, the dense sweep: the steady speed of each level, raised and lowered,
  and the duties at which the wheel started and stopped;
* E2, the C trials stepping 0 -> 0.6 -> 1.0 -> 0.6 -> 0: per step the
  response averaged over the repetitions, its rise time, overshoot and
  settling time (analysis.step_response), and the first-order fits.

A wheel counts as at rest at a level when the magnitude of its mean speed is
at most omega_min, the magnitude exceeded by REST_EXCEEDANCE of the speeds of
a wheel at rest before the first command of each trial.

Time is measured from the logged command: the middle of the Pi's call to the
motor driver, placed on the video clock by the trial's clock (§10.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from duckiebot_analysis.analysis.derivatives import W_OMEGA
from duckiebot_analysis.analysis.six_motor import hold_segments, hold_speed
from duckiebot_analysis.analysis.step_response import (
    FirstOrderFit,
    LevelSpeed,
    StepData,
    StepInfo,
    fit_first_order,
    level_speeds,
    mean_response,
    moving_mean,
    start_stop_duties,
    step_info,
)
from duckiebot_analysis.data.pilogs import read_pi_log

FloatArray = npt.NDArray[np.float64]
# The motor folder and the start of its run names (label), as in the article.
# configure() points these at another motor; runs and trials are found by
# pattern or prefix, because each name ends in its recording time, and
# _one_run/_one_trial stop unless exactly one matches.
MOTOR = "db17-cherry-left-m5"
LABEL = "R3-L"
MOTOR_CHANNEL = "motor_left"  # the wheel of MOTOR in the Pi log
E1_RUN = f"{LABEL}_A_sweep_dense_*.parquet"
E1_TRIAL = f"{LABEL}_A_sweep_forward_010to036by020_rep01_"
E1_TRIAL_NEG = f"{LABEL}_A_sweep_backward_010to036by020_rep01_"
E2_RUN = f"{LABEL}_C_b060p040*.parquet"
E2_PREFIX = f"{LABEL}_C_b060p040_rep"
DT = 1 / 240  # frame interval of the slow-motion video
# Share of the speeds of a wheel at rest, both signs together, that exceed the
# rest threshold omega_min: the risk that noise alone makes a frame at rest
# count as turning.
REST_EXCEEDANCE = 0.01
LEVEL_LAST_S = 0.8  # end of a 2 s level averaged, as in six_motor.a_levels
# Moving mean of the averaged response for the rise time; less than half the
# 46 ms window of the speed (derivatives, ±23 ms).
SMOOTH_S = 0.02
# The steady speed ripples by a few tenths of a rad/s on time scales below
# 0.1 s; the overshoot and the settling time are taken after a moving mean
# over 0.1 s removes it.
SETTLE_SMOOTH_S = 0.1
FINAL_S = 1.0  # end of a hold that gives the final value and the noise
BEFORE_S = 0.5  # time before the command that gives the initial value
REST_AFTER_S = 2.0  # the C schedule holds duty 0 for 2 s before its end
# The steps of E2, in the order of the schedule: (name, from, to).
E2_STEPS = (
    ("up06", 0.0, 0.6),
    ("up10", 0.6, 1.0),
    ("down06", 1.0, 0.6),
    ("down0", 0.6, 0.0),
)


@dataclass(frozen=True)
class Trace:
    """Per-frame data of one trial (§11.1), time from the trial start."""

    trial: str
    t: FloatArray
    omega: FloatArray
    omega_sd: FloatArray
    u_cmd: FloatArray


@dataclass(frozen=True)
class StepResult:
    """One step of E2: its repetitions, averaged response, quantities and fits."""

    name: str
    u_from: float
    u_to: float
    hold_s: float
    reps: list[StepData]
    grid: FloatArray
    mean: FloatArray
    smooth: FloatArray
    smooth_settle: FloatArray
    noise_sd: float
    info: StepInfo
    fit: FirstOrderFit
    fit_delay: FirstOrderFit


@dataclass(frozen=True)
class CaseStudy:
    """Everything the article reports."""

    e1: Trace
    e1_levels: list[LevelSpeed]
    e1_neg_levels: list[LevelSpeed]
    rest_noise: FloatArray
    omega_min: float
    e1_start: float
    e1_stop: float
    e1_neg_start: float
    e1_neg_stop: float
    e2_trials: list[str]
    steps: dict[str, StepResult]
    trials: dict[str, dict]
    meta_e1: dict
    meta_e2: dict
    call_s: float
    speeds_06: dict[str, float]
    # Name of the backward sweep trial, found by its prefix E1_TRIAL_NEG.
    e1_neg_trial: str = ""


def load_traces(path: Path) -> dict[str, Trace]:
    """Return the trace of every trial of a processed run."""
    d = pq.read_table(
        path, columns=["trial", "t_trial_s", "omega_rad_s", "omega_sd_rad_s", "u_cmd"]
    ).to_pydict()
    names = np.array([x or "" for x in d["trial"]])
    out = {}
    for name in sorted(set(names) - {""}):
        s = names == name
        out[name] = Trace(
            trial=name,
            t=np.asarray(d["t_trial_s"], float)[s],
            omega=np.asarray(d["omega_rad_s"], float)[s],
            omega_sd=np.asarray(d["omega_sd_rad_s"], float)[s],
            u_cmd=np.asarray(d["u_cmd"], float)[s],
        )
    return out


def run_metadata(path: Path) -> dict:
    """Return the `duckiebot` metadata of a processed run (§11.1)."""
    md = pq.read_schema(path).metadata or {}
    return dict(json.loads(md.get(b"duckiebot", b"{}")))


def trial_table(path: Path) -> dict[str, dict]:
    """Return the rows of trials.parquet (§11.2) keyed by trial."""
    return {r["trial"]: r for r in pq.read_table(path).to_pylist()}


def command_changes(log: Path, drift: float) -> list[tuple[float, float]]:
    """Return (time from trial start, duty) of every change of the motor duty.

    The command time is the middle of the Pi's call to the driver, scaled by
    the clock drift (§10.3), as the processing places it.
    """
    ev = read_pi_log(log)
    m = ev.channel == MOTOR_CHANNEL
    tau = (ev.t_s[m] + ev.t_after_s[m]) / 2 * (1 + drift)
    out, prev = [], 0.0
    for t, v in zip(tau, ev.value[m], strict=True):
        if v != prev:
            out.append((float(t), float(v)))
        prev = v
    return out


def call_duration(log: Path) -> float:
    """Return the median time the Pi's calls to the motor driver took, in s."""
    ev = read_pi_log(log)
    m = ev.channel == MOTOR_CHANNEL
    return float(np.median(ev.t_after_s[m] - ev.t_s[m]))


def speeds_at_06(processed: Path) -> dict[str, float]:
    """Return per motor the mean |speed| at duty 0.6 of its C_b060p040 trials.

    The speed is the mean over the last FINAL_S of the first hold at 0.6, and
    the motor is named by the identifier ending its folder, such as m5.
    """
    out: dict[str, float] = {}
    if not processed.is_dir():
        return out
    # One wheel per folder: db17-cherry-both-m1-m2 also ends in a motor name.
    folders = [
        f
        for side in ("left", "right")
        # Any robot name, so that a student's own robot is included too.
        for f in processed.glob(f"db17-*-{side}-m*")
    ]
    for folder in sorted(folders):
        speeds = []
        for path in sorted(folder.glob("*_C_b060p040*.parquet")):
            d = pq.read_table(
                path, columns=["trial", "t_trial_s", "omega_rad_s", "u_cmd"]
            ).to_pydict()
            names = np.array([x or "" for x in d["trial"]])
            for name in sorted(set(names)):
                if "_C_b060p040_rep" not in name:
                    continue
                s = names == name
                t = np.asarray(d["t_trial_s"], float)[s]
                w = np.asarray(d["omega_rad_s"], float)[s]
                u = np.asarray(d["u_cmd"], float)[s]
                # A trial without a 1 s hold at 0.6 gives no speed.
                hold = next((h for h in hold_segments(t, u, 1.0) if h[0] == 0.6), None)
                if hold is None:
                    continue
                speeds.append(abs(hold_speed(t, w, hold[1], hold[2], FINAL_S)))
        if speeds:
            out[folder.name.rsplit("-", 1)[1]] = float(np.mean(speeds))
    return out


def rest_speeds(trace: Trace, t_cmd: float) -> FloatArray:
    """Return the finite speeds of a trial before its first command at t_cmd.

    Only frames whose speed window (S4, +-W_OMEGA) lies wholly between the
    trial start and the command are kept, so no frame sees the wheel move.
    """
    s = (trace.t >= W_OMEGA) & (trace.t <= t_cmd - W_OMEGA) & np.isfinite(trace.omega)
    return trace.omega[s]


def rest_threshold(speeds: npt.ArrayLike, exceedance: float = REST_EXCEEDANCE) -> float:
    """Return the smallest |speed| that at most `exceedance` of speeds exceed.

    Both signs count together (two-tailed): with n speeds, the threshold is
    the sorted magnitude that floor(n * exceedance) magnitudes lie above.

    Raises:
        ValueError: If speeds is empty or non-finite, or exceedance is not
            in (0, 1).
    """
    a = np.sort(np.abs(np.asarray(speeds, float)))
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("rest speeds must be finite and not empty")
    if not 0.0 < exceedance < 1.0:
        raise ValueError(f"exceedance {exceedance} not in (0, 1)")
    return float(a[a.size - 1 - int(np.floor(a.size * exceedance))])


def check_command(trace: Trace, t_cmd: float, duty: float) -> None:
    """Raise unless u_cmd changes to duty between the frames around t_cmd."""
    before = trace.t < t_cmd
    after = ~before
    if not (before.any() and after.any()):
        raise ValueError(f"{trace.trial}: command at {t_cmd:.4f} s outside the trial")
    i = int(np.flatnonzero(after)[0])
    if not np.isclose(trace.u_cmd[i], duty) or np.isclose(trace.u_cmd[i - 1], duty):
        raise ValueError(f"{trace.trial}: u_cmd does not change to {duty} at {t_cmd}")


def step_data(trace: Trace, t_cmd: float, before_s: float, after_s: float) -> StepData:
    """Return the frames from before_s before to after_s after the command."""
    rel = trace.t - t_cmd
    s = (rel >= -before_s) & (rel <= after_s)
    return StepData(t=rel[s], omega=trace.omega[s], omega_sd=trace.omega_sd[s])


def analyse_step(
    name: str,
    reps: list[StepData],
    u_from: float,
    u_to: float,
    hold_s: float,
    omega_min: float,
) -> StepResult:
    """Return the averaged response, its quantities and the fits of one step."""
    grid = np.arange(-BEFORE_S, hold_s + DT / 2, DT)
    mean = mean_response(reps, grid)
    smooth = moving_mean(grid, mean, SMOOTH_S)
    slow = moving_mean(grid, mean, SETTLE_SMOOTH_S)
    pre = grid < 0
    t_final = hold_s - FINAL_S
    end = grid >= t_final
    initial = float(np.nanmean(mean[pre]))
    final = float(np.nanmean(mean[end]))
    noise = float(np.nanstd(slow[end], ddof=1))
    info = step_info(
        grid,
        smooth,
        initial=initial,
        final=final,
        noise_sd=noise,
        min_change=omega_min,
        y_settle=slow,
        t_final_s=t_final,
    )
    du = u_to - u_from
    return StepResult(
        name=name,
        u_from=u_from,
        u_to=u_to,
        hold_s=hold_s,
        reps=reps,
        grid=grid,
        mean=mean,
        smooth=smooth,
        smooth_settle=slow,
        noise_sd=noise,
        info=info,
        fit=fit_first_order(reps, du),
        fit_delay=fit_first_order(reps, du, delay=True),
    )


def _e2_changes(
    raw: Path, traces: dict[str, Trace], trials: dict[str, dict]
) -> dict[str, list[float]]:
    changes = {}
    for name, tr in traces.items():
        ch = command_changes(raw / "pi" / f"{name}_events.csv", trials[name]["drift"])
        if [v for _, v in ch] != [s[2] for s in E2_STEPS]:
            raise ValueError(f"{name}: duty schedule {ch} is not that of E2")
        for (t_cmd, duty), _ in zip(ch, E2_STEPS, strict=True):
            check_command(tr, t_cmd, duty)
        changes[name] = [t for t, _ in ch]
    return changes


def _e2_steps(
    traces: dict[str, Trace], changes: dict[str, list[float]], omega_min: float
) -> dict[str, StepResult]:
    out = {}
    for k, (step, u_from, u_to) in enumerate(E2_STEPS):
        holds = [
            (c[k + 1] - c[k]) if k + 1 < len(c) else np.inf for c in changes.values()
        ]
        hold = min(holds) if np.isfinite(min(holds)) else REST_AFTER_S
        # Stop one frame before the next command so that no frame of the next
        # step enters the fit.
        hold = float(np.floor((hold - DT) / DT) * DT)
        reps = [
            step_data(traces[n], changes[n][k], BEFORE_S, hold) for n in sorted(traces)
        ]
        out[step] = analyse_step(step, reps, u_from, u_to, hold, omega_min)
    return out


def configure(folder: str, label: str) -> None:
    """Point the case study at the runs of one motor.

    Args:
        folder: Motor folder, such as db17-r5-left-m9; its side (left or right)
            gives the wheel's channel in the Pi log.
        label: Start of the run and trial names, such as R5-L-m9.

    Raises:
        ValueError: If the folder is not a left or right motor folder.
    """
    global MOTOR, LABEL, MOTOR_CHANNEL, E1_RUN, E1_TRIAL, E1_TRIAL_NEG
    global E2_RUN, E2_PREFIX
    parts = folder.split("-")
    side = parts[2] if len(parts) > 2 else ""
    if side not in ("left", "right"):
        raise ValueError(f"{folder}: need a left or right motor folder")
    MOTOR, LABEL, MOTOR_CHANNEL = folder, label, f"motor_{side}"
    E1_RUN = f"{label}_A_sweep_dense_*.parquet"
    E1_TRIAL = f"{label}_A_sweep_forward_010to036by020_rep01_"
    E1_TRIAL_NEG = f"{label}_A_sweep_backward_010to036by020_rep01_"
    E2_RUN = f"{label}_C_b060p040*.parquet"
    E2_PREFIX = f"{label}_C_b060p040_rep"


def check_led_sync(trials: dict[str, dict], names: list[str]) -> None:
    """Raise unless every named trial has the clock drift of an LED sync.

    A trial synchronised by motion (log+motion, script+motion) has no drift
    or timing uncertainty (§10.3), and the case study needs both.

    Raises:
        ValueError: Naming the first trial without them.
    """
    for name in names:
        row = trials.get(name, {})
        if row.get("drift") is None or row.get("timing_sd_s") is None:
            raise ValueError(
                f"{name}: synchronised by {row.get('sync') or 'no clock'}, but the "
                "case study needs LED synchronisation (log+led or script+led); "
                "re-record the clip with the LED in view."
            )


def _one_run(folder: Path, pattern: str) -> Path:
    """Return the only run of ``folder`` matching ``pattern``."""
    found = sorted(folder.glob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(
            f"{folder}: expected one run matching {pattern}, found {len(found)}"
        )
    return found[0]


def _one_trial(traces: dict[str, Trace], prefix: str) -> Trace:
    """Return the only trace whose name starts with ``prefix``."""
    found = [k for k in traces if k.startswith(prefix)]
    if len(found) != 1:
        raise ValueError(f"expected one trial {prefix}*, found {len(found)}")
    return traces[found[0]]


def case_study(data_root: Path, motor: str | None = None) -> CaseStudy:
    """Compute every result of the case study from data/processed and data/raw.

    Args:
        data_root: The data/ folder holding raw/ and processed/.
        motor: The motor folder name, the same in raw/ and processed/
            (default: MOTOR, set by configure()).

    Raises:
        ValueError: If motor differs from the folder set by configure(), whose
            run and trial names and wheel channel would not fit it, or a trial
            used is not synchronised by the LED.
    """
    if motor is not None and motor != MOTOR:
        raise ValueError(
            f"{motor}: call configure({motor!r}, label) first; the case study "
            f"is configured for {MOTOR}"
        )
    motor = MOTOR
    proc = data_root / "processed" / motor
    raw = data_root / "raw" / motor
    e1_path = _one_run(proc, E1_RUN)
    e2_path = _one_run(proc, E2_RUN)
    trials = trial_table(proc / "trials.parquet")
    e1_all = load_traces(e1_path)
    e1 = _one_trial(e1_all, E1_TRIAL)
    neg = _one_trial(e1_all, E1_TRIAL_NEG)
    e2 = {k: v for k, v in load_traces(e2_path).items() if k.startswith(E2_PREFIX)}
    check_led_sync(trials, [e1.trial, neg.trial, *sorted(e2)])
    e2_changes = _e2_changes(raw, e2, trials)
    first_cmd = {n: c[0] for n, c in e2_changes.items()}
    for tr in (e1, neg):
        log = raw / "pi" / f"{tr.trial}_events.csv"
        first_cmd[tr.trial] = command_changes(log, trials[tr.trial]["drift"])[0][0]
    traces = {e1.trial: e1, neg.trial: neg} | e2
    noise = np.concatenate(
        [rest_speeds(traces[n], first_cmd[n]) for n in sorted(first_cmd)]
    )
    omega_min = rest_threshold(noise)
    lv = level_speeds(e1.t, e1.omega, e1.u_cmd, LEVEL_LAST_S, min_hold_s=1.5)
    start, stop = start_stop_duties(lv, omega_min)
    lv_neg = level_speeds(neg.t, neg.omega, neg.u_cmd, LEVEL_LAST_S, min_hold_s=1.5)
    neg_start, neg_stop = start_stop_duties(lv_neg, omega_min)
    return CaseStudy(
        e1=e1,
        e1_levels=lv,
        e1_neg_levels=lv_neg,
        rest_noise=noise,
        omega_min=omega_min,
        e1_start=start,
        e1_stop=stop,
        e1_neg_start=neg_start,
        e1_neg_stop=neg_stop,
        e2_trials=sorted(e2),
        steps=_e2_steps(e2, e2_changes, omega_min),
        trials=trials,
        meta_e1=run_metadata(e1_path),
        meta_e2=run_metadata(e2_path),
        call_s=float(
            np.median([call_duration(raw / "pi" / f"{n}_events.csv") for n in e2])
        ),
        speeds_06=speeds_at_06(data_root / "processed"),
        e1_neg_trial=neg.trial,
    )
