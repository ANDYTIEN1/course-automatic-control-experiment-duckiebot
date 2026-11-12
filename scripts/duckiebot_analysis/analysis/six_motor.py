# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Steady speeds, thresholds and responses of the six-motor campaign.

Reads the processed runs of data/processed/db17-* (data/schema.md
§11), made from data/raw by scripts/process_raw.py, and returns what the
model fits of scripts/experiment_design/ need:

* load_trials: the B, C and D responses about each command change;
* static_speeds: the steady speed of every turning A sweep level and the
  mean B steady speed per duty;
* start_duties_per_motor: the lowest duty that started the wheel;
* air_speeds: the speed in the air with both wheels driven (E);
* floor_speeds: the wheel speed on the floor with both wheels driven (E),
  from the runs whose wheel angle comes from the spokes.

Trial and run names start with the robot and position, such as R1-L, which
is the motor label used throughout. A trial is used only if it is
synchronised and its once-per-turn correction succeeded (harm_k <= 100).
Trials with tracking failures (spikes, lost speed, a speed or level timing
that disagrees with the other trials) and B starts more than 0.5 s late are
also left out; every exclusion is printed and, where the caller passes a
``left_out`` list, appended to it as (trial, reason).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from duckiebot_analysis.analysis.model_fits import Trial

FloatArray = npt.NDArray[np.float64]
LeftOut = list[tuple[str, str]]
MAX_HARM_K = 100.0
MOVING = 1.0  # rad/s; a slower level counts as standing still
B_TOLERANCE = 0.15
SWEEP_TOLERANCE = 1.0  # rad/s RMS from the B steady speeds
SPIKE = 5.0  # rad/s; smallest deviation that counts as a tracking spike
# Windows relative to the command change, and the time of that change in the
# trial: B steps from rest at 2 s, C steps at 5 s (with the return at 8 s), D
# reversals at 6 s.
WINDOW = {"B": (2.0, -0.3, 3.0), "C": (5.0, -1.0, 5.9), "D": (6.0, -1.0, 2.95)}
EVERY = 2  # keep every second frame (120 per s) to halve the fitting time
# E on the floor: the speed is the mean over the end of the 1.5 s hold, when
# the robot has accelerated, and needs ω in most of those frames, since the
# wheel can leave the picture.
FLOOR_LAST_S = 0.5
FLOOR_MIN_COVERAGE = 0.5


def hold_segments(
    t: FloatArray, u: FloatArray, min_s: float
) -> list[tuple[float, float, float]]:
    """Return (duty, start, end) of every constant duty held min_s or longer."""
    ok = np.isfinite(u)
    t, u = t[ok], u[ok]
    if t.size < 2:
        return []
    edges = np.r_[0, np.flatnonzero(np.abs(np.diff(u)) > 1e-6) + 1, u.size]
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if t[b - 1] - t[a] >= min_s:
            out.append((round(float(u[a]), 4), float(t[a]), float(t[b - 1])))
    return out


def hold_window(
    t: FloatArray, w: FloatArray, start: float, end: float, last_s: float
) -> FloatArray:
    """Return the known speeds over the last last_s of the hold start to end."""
    sel = (t >= max(start, end - last_s)) & (t <= end) & np.isfinite(w)
    return w[sel]


def hold_speed(
    t: FloatArray, w: FloatArray, start: float, end: float, last_s: float
) -> float:
    """Return the mean speed over the last last_s of a hold, NaN if none."""
    sel = hold_window(t, w, start, end, last_s)
    return float(np.mean(sel)) if sel.size else np.nan


def start_duties(
    levels: list[tuple[float, float]], moving: float
) -> tuple[float, float]:
    """Return the lowest |duty| that started the wheel, forward and backward.

    levels are (duty, speed) in time order; a level starts the wheel when it
    turns and the level before it in the same direction did not.
    """
    out = {1: np.nan, -1: np.nan}
    prev: dict[int, bool] = {}
    for u, w in levels:
        s = 1 if u > 0 else -1
        turning = abs(w) > moving
        if turning and prev.get(s) is False:
            out[s] = np.nanmin([out[s], abs(u)])
        prev[s] = turning
    return out[1], out[-1]


def realign_levels(
    levels: list[tuple[float, float]], ref: dict[float, float], max_shift: int
) -> tuple[list[tuple[float, float]], float]:
    """Shift a sweep's speeds by whole levels to match reference speeds.

    A sweep placed on the video by the wheel motion can be off by whole
    levels, so that each level reads the speed of a neighbour. The speeds
    are paired with the duty k levels later, for |k| <= max_shift, and the k
    with the smallest RMS difference from ref (steady speed per duty) is kept.

    Returns:
        The realigned (duty, speed) levels and their RMS difference from ref,
        NaN if no level has a reference duty.
    """
    duties = [d for d, _ in levels]
    best: tuple[float, int, list[tuple[float, float]]] = (np.inf, 0, levels)
    for k in sorted(range(-max_shift, max_shift + 1), key=abs):
        pairs = [
            (duties[i + k], w)
            for i, (_, w) in enumerate(levels)
            if 0 <= i + k < len(levels)
        ]
        diff = [w - ref[d] for d, w in pairs if d in ref]
        if diff:
            err = float(np.sqrt(np.mean(np.square(diff))))
            if err < best[0] - 1e-9:
                best = (err, k, pairs)
    if not np.isfinite(best[0]):
        return levels, np.nan
    return best[2], best[0]


def has_spikes(w: FloatArray) -> bool:
    """Return whether a settled speed has tracking spikes.

    A spike is a frame far from the median; more than 5 % of the frames
    deviating by more than half the median speed and more than SPIKE rad/s
    marks a failed tracking. Fewer, isolated spikes barely bias the mean.
    The absolute floor keeps the noise and the once-per-turn ripple
    of a slow wheel, about +-3 rad/s at 6 rad/s, from counting as spikes.
    """
    w = w[np.isfinite(w)]
    if w.size == 0:
        return False
    med = float(np.median(w))
    if abs(med) < MOVING:
        return False
    return bool(np.mean(np.abs(w - med) > max(0.5 * abs(med), SPIKE)) > 0.05)


def creep_share(t: FloatArray, w: FloatArray, start: float, end: float) -> float:
    """Return the share of the steady speed gained after the main rise.

    The speed 1.5-2.5 s after the step, when the main rise of 0.1-0.4 s has
    settled, is compared with the mean over the last 2 s of the hold.
    """
    mid = hold_speed(t, w, start + 1.5, start + 2.5, last_s=1.0)
    late = hold_speed(t, w, start, end, last_s=2.0)
    return (late - mid) / late


def level_steps(
    sweep: list[tuple[float, float]], duties: tuple[float, ...]
) -> list[float]:
    """Return the speed steps between consecutive duties on the way up.

    Takes the first level at each duty; empty if a duty is missing.
    """
    first: dict[float, float] = {}
    for d, w in sweep:
        first.setdefault(round(d, 2), w)
    if any(round(d, 2) not in first for d in duties):
        return []
    w = [first[round(d, 2)] for d in duties]
    return [b - a for a, b in zip(w[:-1], w[1:], strict=True)]


def b_creep(root: Path) -> list[dict]:
    """Return the creep share of every clean B trial that turned for 8 s."""
    out = []
    for name, _, _, t, w, u in iter_trials(root, "*_B_*.parquet"):
        if name.split("_")[1] != "B":
            continue
        duty, speed, spikes, late = _b_summary(t, w, u)
        segs = [s for s in hold_segments(t, u, min_s=5.0) if s[0] != 0]
        if spikes or late or not abs(speed) > MOVING or not segs:
            continue
        _, start, end = segs[0]
        out.append(
            {
                "motor": motor_of(name),
                "trial": name,
                "duty": duty,
                "speed": speed,
                "creep": creep_share(t, w, start, end),
            }
        )
    return out


def leave_out(left_out: LeftOut | None, name: str, reason: str) -> None:
    """Print the exclusion of a trial and record it in left_out if given."""
    print(f"left out {name}: {reason}")
    if left_out is not None:
        left_out.append((name, reason))


def _good(folder: Path) -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Return the usable trials, the sync of every trial, and the reason
    each unusable trial is left out."""
    tr = pq.read_table(
        folder / "trials.parquet", columns=["trial", "sync", "harm_k"]
    ).to_pydict()
    ok, bad = set(), {}
    for t, s, k in zip(tr["trial"], tr["sync"], tr["harm_k"], strict=True):
        if s is None:
            bad[t] = "no timing"
        elif k is None or not k <= MAX_HARM_K:
            bad[t] = "once-per-turn correction failed" + (
                f" (k = {k:.0f})" if k is not None else ""
            )
        else:
            ok.add(t)
    return ok, dict(zip(tr["trial"], tr["sync"], strict=True)), bad


def iter_trials(
    root: Path, pattern: str, left_out: LeftOut | None = None
) -> Iterator[tuple[str, str, Path, FloatArray, FloatArray, FloatArray]]:
    """Yield (trial, sync, file, t, omega, u) of every usable trial.

    The unusable trials in the matched files go to left_out, if given.
    """
    for folder in sorted(root.glob("db17-*")):
        ok, sync, bad = _good(folder)
        for path in sorted(folder.glob(pattern)):
            d = pq.read_table(
                path, columns=["trial", "t_trial_s", "omega_rad_s", "u"]
            ).to_pydict()
            names = np.array([x or "" for x in d["trial"]])
            if left_out is not None:
                for name in sorted(set(names) & set(bad)):
                    leave_out(left_out, name, bad[name])
            for name in sorted(set(names) & ok):
                s = names == name
                yield (
                    name,
                    sync[name],
                    path,
                    np.asarray(d["t_trial_s"], float)[s],
                    np.asarray(d["omega_rad_s"], float)[s],
                    np.asarray(d["u"], float)[s],
                )


def motor_of(name: str) -> str:
    """Return R1-L from a trial or run name."""
    return name.split("_")[0]


def _b_summary(
    t: FloatArray, w: FloatArray, u: FloatArray
) -> tuple[float, float, bool, bool]:
    """Return duty, steady speed, spikes and late start of a B trial.

    The steady speed is the mean over the last 2 s of the hold; the start is
    late when the wheel still stands 0.5 s after the step.
    """
    segs = [s for s in hold_segments(t, u, min_s=5.0) if s[0] != 0]
    if not segs:
        return np.nan, np.nan, False, False
    duty, start, end = segs[0]
    speed = hold_speed(t, w, start, end, last_s=2.0)
    spikes = has_spikes(w[(t >= end - 2.0) & (t <= end)])
    early = hold_speed(t, w, start, start + 0.6, last_s=0.2)
    late = abs(speed) > MOVING and not abs(early) > MOVING
    return duty, speed, spikes, late


def b_steady(root: Path) -> dict[str, list[tuple[float, float]]]:
    """Return per motor the (duty, steady speed) of every B trial.

    The steady speed is the mean over the last 2 s of the 8 s hold; trials
    with tracking spikes are left out.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for name, _, _, t, w, u in iter_trials(root, "*_B_*.parquet"):
        if name.split("_")[1] != "B":
            continue
        duty, speed, spikes, _ = _b_summary(t, w, u)
        if np.isfinite(speed) and not spikes:
            out.setdefault(motor_of(name), []).append((duty, speed))
    return out


def b_starts(root: Path) -> list[tuple[str, float, bool]]:
    """Return (trial, duty, started) of every B step from rest.

    A step started when the steady speed of _b_summary turns faster than
    MOVING; trials without a steady speed or with tracking spikes are left
    out, as in b_steady.
    """
    out = []
    for name, _, _, t, w, u in iter_trials(root, "*_B_*.parquet"):
        if name.split("_")[1] != "B":
            continue
        duty, speed, spikes, _ = _b_summary(t, w, u)
        if np.isfinite(speed) and not spikes:
            out.append((name, duty, bool(abs(speed) > MOVING)))
    return out


def _consistent(b: list[tuple[float, float]], duty: float, end: float) -> bool:
    # A trial whose speed differs from the median of its duty is a tracking
    # failure (flipped sign or lost marker), not a property of the motor. b
    # holds the trial itself; the median over three or more resists one bad
    # trial, and two that disagree both fail.
    if not np.isfinite(end):
        return False
    same = [s for d, s in b if d == duty]
    if len(same) <= 1:
        # Nothing to compare with: the wheel must at least turn with the duty.
        return bool(np.sign(end) == np.sign(duty) or not abs(end) > MOVING)
    ref = float(np.median(same))
    return bool(abs(end - ref) <= B_TOLERANCE * abs(ref))


def load_trials(
    root: Path, left_out: LeftOut | None = None
) -> dict[tuple[str, str], list[Trial]]:
    """Return the B, C and D responses of every motor, keyed by (motor, exp).

    Every trial left out goes to left_out, if given, as (trial, reason).
    """
    b = b_steady(root)
    out: dict[tuple[str, str], list[Trial]] = {}
    for name, sync, _, t, w, u in iter_trials(root, "*_[BCD]*.parquet", left_out):
        exp = name.split("_")[1]
        if exp not in WINDOW:
            continue
        u = np.nan_to_num(u)
        t_ev, a, end = WINDOW[exp]
        k = np.flatnonzero(np.abs(np.diff(u)) > 0.01)
        k = k[np.abs(t[k + 1] - t_ev) < 0.2]
        if not k.size:
            continue
        tc = 0.5 * (t[k[0]] + t[k[0] + 1])
        rel = t - tc
        win = (rel >= a) & (rel <= end) & np.isfinite(w)
        if win.sum() < 100:
            continue
        if exp == "B":
            duty, speed, spikes, late = _b_summary(t, w, u)
            if not _consistent(b.get(motor_of(name), []), duty, speed):
                leave_out(left_out, name, f"steady speed {speed:.2f} at {duty}")
                continue
            if spikes or late:
                leave_out(left_out, name, "spikes" if spikes else "late start")
                continue
        else:
            before = (rel >= a) & (rel < 0)
            u_before = float(np.median(u[(t - tc >= a) & (t - tc < 0)]))
            lost = abs(u_before) >= 0.4 and not abs(np.nanmean(w[before])) > MOVING
            if lost or has_spikes(w[before]):
                leave_out(left_out, name, "no speed" if lost else "spikes")
                continue
        init = (rel >= a) & (rel <= a + 0.2) & np.isfinite(w)
        sel = np.flatnonzero(win)[::EVERY]
        out.setdefault((motor_of(name), exp), []).append(
            Trial(
                t=rel[sel],
                omega=w[sel],
                t_u=rel,
                u=u,
                led=sync == "log+led",
                omega_init=0.0 if exp == "B" else float(np.mean(w[init])),
            )
        )
    return out


def a_sweeps(
    root: Path,
    b: dict[str, list[tuple[float, float]]],
    left_out: LeftOut | None = None,
) -> dict[str, tuple[str, list[tuple[float, float]]]]:
    """Return per A sweep kept its motor and (duty, steady speed) levels.

    The speed is the mean over the last 1 s of a 3 s level and the last
    0.8 s of a 2 s level. A sweep timed by the LED is used as it is. A sweep
    timed by the wheel motion can be off by whole levels: a coarse one is
    realigned to the B steady speeds of the same motor (b, from b_steady)
    and left out if it still differs from them by more than 1 rad/s RMS; a
    dense one has no B duties to check it against and is left out. Every
    sweep left out goes to left_out, if given, as (trial, reason).

    Returns:
        {trial: (motor, levels)} in the order the trials are read.
    """
    out: dict[str, tuple[str, list[tuple[float, float]]]] = {}
    for name, sync, _, t, w, u in iter_trials(root, "*_A_*.parquet", left_out):
        levels = []
        for duty, start, end in hold_segments(t, u, min_s=1.5):
            if duty != 0:
                last = min(1.0, 0.4 * (end - start))
                levels.append((duty, hold_speed(t, w, start, end, last)))
        m = motor_of(name)
        if sync != "log+led":
            if "010to036" in name:
                leave_out(left_out, name, "dense sweep timed by motion")
                continue
            ref = {
                d: float(np.mean([s for dd, s in b.get(m, []) if dd == d]))
                for d in {d for d, _ in b.get(m, [])}
            }
            levels, err = realign_levels(levels, ref, max_shift=2)
            if not err <= SWEEP_TOLERANCE:
                leave_out(left_out, name, f"{err:.2f} rad/s from the B speeds")
                continue
        out[name] = (m, levels)
    return out


def a_levels(
    root: Path,
    b: dict[str, list[tuple[float, float]]],
    left_out: LeftOut | None = None,
) -> dict[str, list[list[tuple[float, float]]]]:
    """Return per motor the (duty, steady speed) levels of each A sweep kept.

    The sweeps and their levels are those of a_sweeps.
    """
    out: dict[str, list[list[tuple[float, float]]]] = {}
    for m, levels in a_sweeps(root, b, left_out).values():
        out.setdefault(m, []).append(levels)
    return out


def static_speeds(
    levels: dict[str, list[list[tuple[float, float]]]],
    b: dict[str, list[tuple[float, float]]],
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Return per motor the duties and steady speeds of the turning wheel.

    These are the A levels at which the wheel turned and the mean B steady
    speed of each duty over the trials that started.
    """
    out = {}
    for m in sorted(set(levels) | set(b)):
        pts = [(d, s) for sweep in levels.get(m, []) for d, s in sweep]
        pts = [(d, s) for d, s in pts if abs(s) > MOVING]
        bm = [(d, s) for d, s in b.get(m, []) if abs(s) > MOVING]
        for d in sorted({d for d, _ in bm}):
            pts.append((d, float(np.mean([s for dd, s in bm if dd == d]))))
        out[m] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def start_duties_per_motor(
    levels: dict[str, list[list[tuple[float, float]]]],
    b: dict[str, list[tuple[float, float]]],
    motors: Iterable[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return per motor the lowest |duty| seen to start the wheel from rest.

    Taken over the A sweeps and the B steps from rest, forward and backward,
    for every motor in levels or b, or only those of them in motors.

    Raises:
        ValueError: Naming the motor and direction in which no sweep or step
            started the wheel; a NaN start duty would pass unnoticed.
    """
    out = {}
    known = set(levels) | set(b)
    for m in sorted(known if motors is None else known & set(motors)):
        fwd, bwd = [], []
        for sweep in levels.get(m, []):
            f, r = start_duties(sweep, MOVING)
            fwd.append(f)
            bwd.append(r)
        for d, s in b.get(m, []):
            if abs(s) > MOVING:
                (fwd if d > 0 else bwd).append(abs(d))
        for name, seen in (("forward", fwd), ("backward", bwd)):
            if not np.isfinite(seen).any():
                raise ValueError(f"{m}: no sweep or step started the wheel {name}")
        out[m] = (float(np.nanmin(fwd)), float(np.nanmin(bwd)))
    return out


def air_speeds(root: Path) -> dict[str, float]:
    """Return per motor the mean speed in the air with both wheels at 0.6.

    The filmed wheel is named in the run name (filmedLeft, filmedRight); the
    speed is the mean over the last 1.5 s of the hold.
    """
    speeds: dict[str, list[float]] = {}
    for name, _, path, t, w, u in iter_trials(root, "*_E_air_*.parquet"):
        side = "L" if "filmedLeft" in path.name else "R"
        robot = name.split("-")[0]
        for duty, start, end in hold_segments(t, u, min_s=2.0):
            if duty != 0:
                speeds.setdefault(f"{robot}-{side}", []).append(
                    hold_speed(t, w, start, end, last_s=1.5)
                )
    return {m: float(np.nanmean(v)) for m, v in speeds.items()}


def floor_hold_coverage(
    t: FloatArray, w: FloatArray, start: float, end: float
) -> float:
    """Return the share of frames with a known ω in the last part of a hold.

    The part is the last FLOOR_LAST_S of the hold from start to end; a part
    without frames has coverage 0.
    """
    sel = (t >= max(start, end - FLOOR_LAST_S)) & (t <= end)
    return float(np.isfinite(w[sel]).mean()) if sel.any() else 0.0


def floor_hold_covered(t: FloatArray, w: FloatArray, start: float, end: float) -> bool:
    """Return whether ω is known in FLOOR_MIN_COVERAGE of a hold's last part."""
    return floor_hold_coverage(t, w, start, end) >= FLOOR_MIN_COVERAGE


def floor_speeds(root: Path) -> dict[str, list[float]]:
    """Return per motor the wheel speed of every usable floor run of E.

    A floor run is used when its wheel angle comes from the spokes (metadata
    ``wheel_angle``, data/schema.md §9.5), its trial is synchronised, and ω is
    known in at least FLOOR_MIN_COVERAGE of the last FLOOR_LAST_S of the hold
    (floor_hold_covered); every other floor run, and one without the
    duckiebot metadata, is printed with the reason. The filmed wheel is
    named in the run name (filmedLeft, filmedRight).
    """
    out: dict[str, list[float]] = {}
    for folder in sorted(root.glob("db17-*")):
        paths = sorted(folder.glob("*_E_floor_*.parquet"))
        if not paths:
            continue
        tr = pq.read_table(folder / "trials.parquet", columns=["trial", "sync"])
        synced = {t for t, s in zip(*tr.to_pydict().values(), strict=True) if s}
        for path in paths:
            raw = (pq.read_schema(path).metadata or {}).get(b"duckiebot")
            if raw is None:
                print(f"{path.stem}: floor, no duckiebot metadata", file=sys.stderr)
                continue
            meta = json.loads(raw)
            angle = meta.get("wheel_angle", "marker")
            if angle != "spokes":
                print(
                    f"{path.stem}: floor, wheel angle from the {angle}", file=sys.stderr
                )
                continue
            d = pq.read_table(
                path, columns=["trial", "t_trial_s", "omega_rad_s", "u"]
            ).to_pydict()
            names = np.array([x or "" for x in d["trial"]])
            for name in sorted(set(names) - {""}):
                if name not in synced:
                    print(
                        f"{path.stem}: floor, {name} not synchronised", file=sys.stderr
                    )
                    continue
                s = names == name
                t = np.asarray(d["t_trial_s"], float)[s]
                w = np.asarray(
                    [np.nan if x is None else x for x in d["omega_rad_s"]], float
                )[s]
                u = np.asarray([np.nan if x is None else x for x in d["u"]], float)[s]
                for duty, start, end in hold_segments(t, u, min_s=1.0):
                    if duty == 0:
                        continue
                    known = floor_hold_coverage(t, w, start, end)
                    if known < FLOOR_MIN_COVERAGE:
                        print(
                            f"{path.stem}: floor, ω known in "
                            f"{known:.0%} of the hold's end",
                            file=sys.stderr,
                        )
                        continue
                    side = "L" if "filmedLeft" in path.name else "R"
                    motor = f"{path.name.split('-')[0]}-{side}"
                    out.setdefault(motor, []).append(
                        abs(hold_speed(t, w, start, end, last_s=FLOOR_LAST_S))
                    )
    return out
