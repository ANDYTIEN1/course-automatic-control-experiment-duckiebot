# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Pi logs and command schedules as one event table (data/schema.md §5, §7, §10.4).

Every source of the command schedule, a Pi log in any format (§7) or a schedule
reconstructed
from the trial list (§10.4), is turned into a `PiEvents`: one row per output change,
with the Pi
time just before (`t_s`) and just after (`t_after_s`) the output call, and the nominal
timing
standard deviation `t_sd_s` (0 for a logged time, 2 ms for a nominal one, §10.4).

Values are the duty the script commanded, with the script's sign (§1).
Saloni's cherry scripts multiply the right wheel's duty by -1 before the driver
(`RIGHT_SIGN = -1`); that sign is not applied here, because the six-motor script that is
reconstructed from trial IDs is not in the repository and its driver sign is unknown, so
only the
commanded sign is available for every source.
`setup.toml` (§6, `positive_duty_turns`) relates the sign to the rotation direction per
motor.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np

from duckiebot_analysis.data.layout import (
    parse_motor_folder,
)

CHANNELS = ("motor_left", "motor_right", "led")
MOTOR_CHANNELS = {
    "left": ("motor_left",),
    "right": ("motor_right",),
    "both": ("motor_left", "motor_right"),
}
NOMINAL_TIMING_SD_S = 0.002  # §10.4: sleep timing of the Pi, per change
NEW_COLUMNS = ("trial", "t_s", "t_after_s", "channel", "value", "event")
HEADER_FORMATS = {
    ("t_s", "event", "u", "count"): "daisy-task1",
    ("repeat", "duty", "on_s", "off_s", "wheel", "note"): "cherry-steps",
    ("repeat", "base", "duty", "on_s", "off_s", "wheel", "note"): "cherry-smallsteps",
    ("level", "duty", "start_s", "hold_s", "wheel", "note"): "cherry-staircase",
    (
        "repeat",
        "from_duty",
        "to_duty",
        "reverse_s",
        "stop_s",
        "wheel",
        "note",
    ): "cherry-reversal",
    ("time", "duty", "seconds", "wheel", "measured_on_s", "note"): "cherry-step",
    (
        "t_s",
        "t_monotonic_s",
        "event",
        "label",
        "requested_duty",
        "written_throttle",
        "led",
    ): "motor-trial",
}
_MOTOR_TRIAL_CHANNELS = {"motor_L": "motor_left", "motor_R": "motor_right"}
_MOTOR_TRIAL_SILENT = ("start", "end")  # rows that change no output
_START_RE = re.compile(r"#\s*start\s+(\S+)\s*\Z")

Row = tuple[
    float, float, str, float, str, float
]  # t_s, t_after_s, channel, value, event, t_sd_s


@dataclass(frozen=True)
class PiEvents:
    """Output changes of one trial on the Pi clock (§7.1 columns plus `t_sd_s`).

    Attributes:
        trial: Trial identifier per row; one trial per object.
        t_s: Pi time just before the output call, seconds from the trial start.
        t_after_s: Pi time just after the call returned; equals `t_s` where unknown.
        channel: `motor_left`, `motor_right`, or `led`.
        value: Commanded duty, or 0 and 1 for the LED.
        event: Name of the step in the schedule.
        t_sd_s: Nominal timing standard deviation of `t_s` (0 when logged, §10.4
            when not).
        start: Pi wall clock at `t_s = 0`, or None if unknown.
        source: Log format, or `script:<name>` for a reconstructed schedule.
        clock_s: Pi run clock at `t_s = 0` in seconds, a clock that runs on
            across the trials of a session, so that the trials of a run share
            one time axis (§10.6); None if the log holds none.
        clock_note: Why a log that records a run clock gives none, or "".
    """

    trial: np.ndarray
    t_s: np.ndarray
    t_after_s: np.ndarray
    channel: np.ndarray
    value: np.ndarray
    event: np.ndarray
    t_sd_s: np.ndarray
    start: datetime | None = None
    source: str = field(default="")
    clock_s: float | None = None
    clock_note: str = ""

    def __post_init__(self) -> None:
        cols = {
            "trial": str,
            "t_s": float,
            "t_after_s": float,
            "channel": str,
            "value": float,
            "event": str,
            "t_sd_s": float,
        }
        n = None
        for name, kind in cols.items():
            arr = np.array(
                getattr(self, name), dtype=np.float64 if kind is float else np.str_
            )
            if arr.ndim != 1 or (n is not None and len(arr) != n):
                raise ValueError(
                    f"{name}: expected a 1-D array of the same length as the others"
                )
            n = len(arr)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if n == 0:
            raise ValueError("no events")
        if len(set(self.trial.tolist())) != 1:
            raise ValueError(
                f"one trial per PiEvents, got {sorted(set(self.trial.tolist()))}"
            )
        bad = sorted(set(self.channel.tolist()) - set(CHANNELS))
        if bad:
            raise ValueError(f"unknown channel {bad}; expected one of {CHANNELS}")
        for name in ("t_s", "t_after_s", "value", "t_sd_s"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name}: non-finite value")
        if np.any(self.t_after_s < self.t_s - 1e-9):
            raise ValueError("t_after_s before t_s")
        if np.any(np.diff(self.t_s) < 0):
            raise ValueError("t_s decreasing: events must be in time order")
        led = self.value[self.channel == "led"]
        if not np.all((led == 0) | (led == 1)):
            raise ValueError("led values must be 0 or 1")
        mot = self.value[self.channel != "led"]
        if np.any(np.abs(mot) > 1):
            raise ValueError("motor duty outside [-1, 1]")
        if np.any(self.t_sd_s < 0):
            raise ValueError("t_sd_s negative")
        if self.clock_s is not None:
            if not np.isfinite(self.clock_s):
                raise ValueError(f"clock_s must be finite, got {self.clock_s!r}")
            object.__setattr__(self, "clock_s", float(self.clock_s))

    def __len__(self) -> int:
        return len(self.t_s)

    @property
    def trial_id(self) -> str:
        """The trial identifier."""
        return str(self.trial[0])

    @property
    def from_script(self) -> bool:
        """True for a schedule reconstructed from the script (§10.4)."""
        return self.source.startswith("script:")


def _build(
    trial: str,
    rows: Iterable[Row],
    start: datetime | None,
    source: str,
    clock_s: float | None = None,
    clock_note: str = "",
) -> PiEvents:
    """Sort rows stably by `t_s` and pack them into a PiEvents."""
    rows = sorted(rows, key=lambda r: r[0])
    if not rows:
        raise ValueError("no events")
    t, ta, ch, v, ev, sd = zip(*rows, strict=True)
    return PiEvents(
        trial=np.full(len(rows), trial),
        t_s=np.array(t),
        t_after_s=np.array(ta),
        channel=np.array(ch),
        value=np.array(v),
        event=np.array(ev),
        t_sd_s=np.array(sd),
        start=start,
        source=source,
        clock_s=clock_s,
        clock_note=clock_note,
    )


# --- Trial identifiers
# ---------------------------------------------------------------------------

# A motor identifier m1 to m16.
_MOTOR = r"m(?:1[0-6]|[1-9])"
_TRIAL_RE = re.compile(
    r"(?P<label>(?P<robot>R[1-8])-"
    rf"(?:(?P<side>[LR])-(?P<one>{_MOTOR})"
    rf"|both-(?P<left>{_MOTOR})-(?P<right>{_MOTOR})"
    rf"|(?P<older>{_MOTOR}|both|L|R)))"
    r"_(?P<exp>[A-E])_(?P<tag>[A-Za-z0-9][A-Za-z0-9_-]*)"
    r"_rep(?P<rep>[0-9]{2})_(?P<stamp>[0-9]{8}_[0-9]{6})",
    re.ASCII,
)
_SIDE_CHANNELS = {"L": "left", "R": "right", "both": "both"}


def motor_position(motor: str) -> str:
    """Return the position of `m1` to `m16` or `both`: odd left, even right."""
    if motor == "both":
        return "both"
    return "left" if int(motor[1:]) % 2 else "right"


@dataclass(frozen=True)
class TrialId:
    """Parts of a trial identifier such as `R5-L-m9_B_p050_rep03_20261024_113544` (§5).

    The label before the first underscore names the robot, the side, and the
    motor: `R5-L-m9`, or `R5-both-m9-m10` for both wheels. The older labels
    `R1-L`, `R3-both`, and `R2-m4` are also read; for `R2-m4` the side follows
    from the motor number, odd left and even right.
    """

    label: str
    robot: str
    side: str
    motors: tuple[str, ...]
    experiment: str
    tag: str
    rep: int
    start: datetime

    @property
    def motor(self) -> str:
        """The motor of a single-wheel trial, else `both` or the older side L or R."""
        if len(self.motors) == 1:
            return self.motors[0]
        return "both" if self.side == "both" else self.side

    def __str__(self) -> str:
        s = self.start
        date = f"{s.year:04d}{s.month:02d}{s.day:02d}"
        stamp = f"{date}_{s.hour:02d}{s.minute:02d}{s.second:02d}"
        return f"{self.label}_{self.experiment}_{self.tag}_rep{self.rep:02d}_{stamp}"

    @property
    def channels(self) -> tuple[str, ...]:
        """Motor channels the trial drives."""
        return MOTOR_CHANNELS[_SIDE_CHANNELS[self.side]]


def parse_trial_id(text: str) -> TrialId:
    """Split a trial identifier into label, parts, experiment, tag, repeat, and start.

    Args:
        text: Identifier `<label>_<A-E>_<tag>_rep<NN>_<YYYYmmdd_HHMMSS>`, where the
            label is `R<1-8>-<L|R>-m<1-16>`, `R<1-8>-both-m<1-16>-m<1-16>`,
            or the older `R<1-8>-<L|R|both>` or `R<1-8>-m<1-16>`.

    Returns:
        The parts; `start` is the Pi wall clock to the second, without time zone.

    Raises:
        ValueError: If the text does not match or the date is invalid.
    """
    m = _TRIAL_RE.fullmatch(text)
    if not m:
        raise ValueError(f"not a trial identifier: {text!r}")
    try:
        start = datetime.strptime(m["stamp"], "%Y%m%d_%H%M%S")
    except ValueError as exc:
        raise ValueError(f"invalid date in trial identifier {text!r}") from exc
    older = m["older"]
    if m["one"]:
        side, motors = m["side"], (m["one"],)
    elif m["left"]:
        side, motors = "both", (m["left"], m["right"])
    elif older in ("both", "L", "R"):
        side, motors = older, ()
    else:
        side = "L" if motor_position(older) == "left" else "R"
        motors = (older,)
    return TrialId(
        m["label"],
        m["robot"],
        side,
        motors,
        m["exp"],
        m["tag"],
        int(m["rep"]),
        start,
    )


# --- Readers ------------------------------------------------------------------------
# -------------


def _lines(text: str) -> list[str]:
    return [ln for ln in text.lstrip("﻿").splitlines() if ln.strip()]


def detect_format(text: str) -> str:
    """Name the log format from its first line(s): `new` (§7.1) or one of §7.2.

    Raises:
        ValueError: If the text is empty or the header is unknown.
    """
    lines = _lines(text)
    if not lines:
        raise ValueError("empty log")
    if lines[0].startswith("#"):
        return "new"
    header = tuple(c.strip() for c in next(csv.reader([lines[0]])))
    if header in HEADER_FORMATS:
        return HEADER_FORMATS[header]
    if header == NEW_COLUMNS:
        raise ValueError("new-format log without the '# start <ISO time>' line (§7.1)")
    raise ValueError(f"unknown log format, header {header}")


def _records(lines: list[str]) -> list[tuple[int, dict[str, str]]]:
    """Rows as dicts with their 1-based line number in the non-blank lines."""
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    out = []
    for k, rec in enumerate(reader, start=2):
        if None in rec or any(v is None for v in rec.values()):
            raise ValueError(f"line {k}: wrong number of fields")
        out.append((k, rec))
    if not out:
        raise ValueError("no events")
    return out


def _num(rec: dict[str, str], key: str, line: int) -> float:
    try:
        x = float(rec[key])
    except ValueError:
        raise ValueError(f"line {line}: {key}={rec[key]!r} is not a number") from None
    if not np.isfinite(x):
        raise ValueError(f"line {line}: {key} is not finite")
    return x


def _check_motor(motor: str | None, *, required: bool = False) -> None:
    """Raise ValueError unless `motor` is left, right, or both (or None if allowed)."""
    if motor is None and not required:
        return
    if motor not in MOTOR_CHANNELS:
        raise ValueError(
            f"motor must be 'left', 'right', or 'both', got {motor!r}"
            + ("; this log format does not name the wheel" if required else "")
        )


def _require_motor(motor: str | None) -> str:
    """Return `motor` if it is left, right, or both, for a format without a wheel."""
    _check_motor(motor, required=True)
    return str(motor)


def _motor_of(wheels: set[str], motor: str | None) -> str:
    if len(wheels) != 1 or not wheels <= set(MOTOR_CHANNELS):
        raise ValueError(
            f"wheel column must hold one of left/right/both, got {sorted(wheels)}"
        )
    wheel = wheels.pop()
    _check_motor(motor)
    if motor is not None and motor != wheel:
        raise ValueError(f"wheel {wheel!r} in the log contradicts motor={motor!r}")
    return wheel


def _period(times: Sequence[float]) -> float:
    """Schedule period of repeated steps, from the logged times rounded to 10 ms."""
    return float(np.round(np.median(np.diff(times)), 2)) if len(times) > 1 else 0.0


def _bound(nominal: float, after: float) -> float:
    """Scheduled time of a call as its start, or the logged time if implausible."""
    return nominal if 0 <= after - nominal < 0.05 else after


def _parse_new(lines: list[str]) -> PiEvents:
    m = _START_RE.fullmatch(lines[0].strip())
    try:
        start = datetime.fromisoformat(m.group(1)) if m else None
    except ValueError:
        start = None
    if start is None:
        raise ValueError(
            f"first line must be '# start <ISO 8601 time>' (§7.1), got {lines[0]!r}"
        )
    if (
        len(lines) < 2
        or tuple(c.strip() for c in next(csv.reader([lines[1]]))) != NEW_COLUMNS
    ):
        raise ValueError(f"second line must be the header {','.join(NEW_COLUMNS)}")
    recs = _records(lines[1:]) if len(lines) > 2 else []
    if not recs:
        raise ValueError("no events")
    rows = [
        (
            _num(r, "t_s", k + 1),
            _num(r, "t_after_s", k + 1),
            r["channel"],
            _num(r, "value", k + 1),
            r["event"],
            0.0,
        )
        for k, r in recs
    ]
    trials = {r["trial"] for _, r in recs}
    if len(trials) != 1:
        raise ValueError(f"one trial per file (§7.1), got {sorted(trials)}")
    t = [r[0] for r in rows]
    if any(b < a for a, b in zip(t, t[1:], strict=False)):
        raise ValueError("t_s decreasing")
    # Only a start given to a fraction of a second can place trials against
    # each other on one clock (§10.6); a start to the second cannot.
    fraction = "." in m.group(1).split("T")[-1] if m else False
    clock = start.timestamp() if fraction else None
    return _build(trials.pop(), rows, start, "new", clock)


def _parse_daisy(
    recs: list[tuple[int, dict[str, str]]], trial: str, motor: str | None
) -> list[Row]:
    motor = _require_motor(motor)
    rows: list[Row] = []
    last = -np.inf
    for k, r in recs:
        t, u, count = _num(r, "t_s", k), _num(r, "u", k), _num(r, "count", k)
        if t < last:
            raise ValueError(f"line {k}: t_s decreasing")
        last = t
        if count != round(abs(u) * 255):
            raise ValueError(
                f"line {k}: count {count:g} does not match u {u} "
                "(count = round(|u| 255))"
            )
        # task1_static.py takes the time after set_duty returned, so t_s = t_after_s.
        rows += [(t, t, ch, u, r["event"], 0.0) for ch in MOTOR_CHANNELS[motor]]
    return rows


def _parse_cherry(
    fmt: str,
    recs: list[tuple[int, dict[str, str]]],
    motor: str | None,
    settle_s: float,
    run_s: float,
) -> list[Row]:
    """Events of Saloni's cherry scripts; times follow each script's calls."""
    chs = MOTOR_CHANNELS[_motor_of({r["wheel"] for _, r in recs}, motor)]
    aborted = any("ABORTED" in r["note"] for _, r in recs)
    rows: list[Row] = []

    def put(
        t: float, ta: float, value: float, name: str, led: float | None, sd: float = 0.0
    ) -> None:
        rows.extend(
            (t, ta, ch, value, name, sd) for ch in chs
        )  # motor first, then the LED
        if led is not None:
            rows.append((t, ta, "led", led, name, sd))

    if fmt in ("cherry-steps", "cherry-smallsteps"):
        on = [_num(r, "on_s", k) for k, r in recs]
        off = [_num(r, "off_s", k) for k, r in recs]
        rep = [int(_num(r, "repeat", k)) for k, r in recs]
        p = _period(on)
        d = float(
            np.round(
                np.median([b - (i - 1) * p for b, i in zip(off, rep, strict=True)]), 2
            )
        )
        base = (
            _num(recs[0][1], "base", recs[0][0]) if fmt == "cherry-smallsteps" else 0.0
        )
        if fmt == "cherry-smallsteps":
            put(-settle_s, -settle_s, base, "base", None, NOMINAL_TIMING_SD_S)
        for (k, r), i, a, b in zip(recs, rep, on, off, strict=True):
            duty = _num(r, "duty", k)
            put(_bound((i - 1) * p, a), a, duty, f"rep{i:02d}_on", 1.0)
            put(_bound((i - 1) * p + d, b), b, base, f"rep{i:02d}_off", 0.0)
        if fmt == "cherry-smallsteps" and not aborted:
            put(len(recs) * p, len(recs) * p, 0.0, "stop", None, NOMINAL_TIMING_SD_S)
    elif fmt == "cherry-staircase":
        hold = _num(recs[0][1], "hold_s", recs[0][0])
        for k, r in recs:
            i, t = int(_num(r, "level", k)), _num(r, "start_s", k)
            # level 0: motor set just before the LED and the clock reset,
            # so both at t = 0
            put(
                0.0 if i == 0 else _bound(i * hold, t),
                0.0 if i == 0 else t,
                _num(r, "duty", k),
                f"level{i:02d}",
                1.0 if i == 0 else None,
            )
        if not aborted:
            put(
                len(recs) * hold, len(recs) * hold, 0.0, "end", 0.0, NOMINAL_TIMING_SD_S
            )
    elif fmt == "cherry-reversal":
        rev = [_num(r, "reverse_s", k) for k, r in recs]
        stop = [_num(r, "stop_s", k) for k, r in recs]
        p = _period(rev)
        h = float(np.round(np.median([b - i * p for i, b in enumerate(stop)]), 2))
        for i, ((k, r), a, b) in enumerate(zip(recs, rev, stop, strict=True)):
            t0 = i * p - run_s
            put(
                t0,
                t0,
                _num(r, "from_duty", k),
                f"rep{i + 1:02d}_run",
                None,
                NOMINAL_TIMING_SD_S,
            )
            put(min(i * p, a), a, _num(r, "to_duty", k), f"rep{i + 1:02d}_reverse", 1.0)
            put(_bound(i * p + h, b), b, 0.0, f"rep{i + 1:02d}_stop", 0.0)
    elif fmt == "cherry-step":
        k, r = recs[0]
        t_off, dur = _num(r, "measured_on_s", k), _num(r, "seconds", k)
        # step_test.py: LED on, clock read, then motors
        rows.append((0.0, 0.0, "led", 1.0, "on", 0.0))
        put(0.0, 0.0, _num(r, "duty", k), "on", None)
        put(min(dur, t_off), t_off, 0.0, "off", 0.0)
    return rows


def _motor_trial_channels(
    recs: list[tuple[int, dict[str, str]]], trial_id: TrialId | None, motor: str | None
) -> list[str]:
    """Motors of a motor-trial log, checked against its trial id and `motor`.

    The trial id and `motor` must name the same motors, and since the script
    commands every motor of the trial at each change, a log with any motor row
    must drive exactly those motors.
    """
    _check_motor(motor)
    logged = {
        _MOTOR_TRIAL_CHANNELS[r["event"]]
        for _, r in recs
        if r["event"] in _MOTOR_TRIAL_CHANNELS
    }
    named = [set(trial_id.channels)] if trial_id is not None else []
    if motor is not None:
        named.append(set(MOTOR_CHANNELS[motor]))
    if len(named) == 2 and named[0] != named[1]:
        raise ValueError(
            f"the trial id names {', '.join(sorted(named[0]))}, "
            f"but motor={motor!r} names {', '.join(sorted(named[1]))}"
        )
    if named and logged and logged != named[0]:
        raise ValueError(
            f"the log drives {', '.join(sorted(logged))}, but the "
            f"trial id or motor argument names {', '.join(sorted(named[0]))}"
        )
    return sorted(named[0] if named else logged)


def _parse_motor_trial(
    recs: list[tuple[int, dict[str, str]]], trial_id: TrialId | None, motor: str | None
) -> list[Row]:
    """Events of Saloni's six-motor `motor_trial.py` (`<trial>_events.csv`).

    The script logs `time.monotonic()` right after each output call, the LED first
    and then each motor. A motor call therefore starts when the row before it was
    logged (`t_s`) and has returned at its own row (`t_after_s`); an LED call
    (gpiozero) is taken as instantaneous. `requested_duty` is the commanded duty,
    before the wiring sign the script applies (`written_throttle`), as §1
    requires; the written throttle must equal it times a sign that stays the same
    for each motor. Which sign is right depends on the wiring
    (`pi/motor_mapping.json`), which the log does not hold, so it is not checked.
    The final `stop` row, logged after the script switched every motor of the
    trial and then the LED off, must record that. A paused and resumed trial
    restarts its clock and is not supported.
    """
    motors = _motor_trial_channels(recs, trial_id, motor)
    signs: dict[str, float] = {}
    rows: list[Row] = []
    prev: float | None = None
    for k, r in recs:
        t, event = _num(r, "t_s", k), r["event"]
        if event == "resume":
            raise ValueError(f"line {k}: pause/resume is not supported")
        if prev is not None and t < prev:
            raise ValueError(f"line {k}: t_s decreasing")
        led = _num(r, "led", k)
        if led not in (0.0, 1.0):
            raise ValueError(f"line {k}: led={r['led']!r} must be 0 or 1")
        before = t if prev is None else prev
        if event in _MOTOR_TRIAL_CHANNELS:
            ch = _MOTOR_TRIAL_CHANNELS[event]
            u = _num(r, "requested_duty", k)
            w = _num(r, "written_throttle", k)
            if abs(abs(w) - abs(u)) > 5e-5:
                raise ValueError(
                    f"line {k}: written_throttle {w:+.4f} differs in size from "
                    f"requested_duty {u:+.4f}"
                )
            if u != 0.0 and w != 0.0:
                sign = float(np.sign(w * u))
                if signs.setdefault(ch, sign) != sign:
                    raise ValueError(
                        f"line {k}: written_throttle changes the wiring sign of {ch}"
                    )
            rows.append((before, t, ch, u, r["label"], 0.0))
        elif event == "led":
            _num(r, "requested_duty", k)
            rows.append((t, t, "led", led, r["label"], 0.0))
        elif event == "stop":
            if (
                led != 0.0
                or _num(r, "written_throttle", k) != 0.0
                or _num(r, "requested_duty", k) != 0.0
            ):
                raise ValueError(
                    f"line {k}: the stop row must record LED and motors off"
                )
            rows += [(before, t, ch, 0.0, r["label"], 0.0) for ch in motors]
            rows.append((t, t, "led", 0.0, r["label"], 0.0))
        elif event not in _MOTOR_TRIAL_SILENT:
            raise ValueError(f"line {k}: unknown event {event!r}")
        prev = t
    if not rows:
        raise ValueError("no events")
    return rows


MONOTONIC_TOLERANCE_S = 1e-5  # both columns are logged to 1 µs


def _monotonic_origin(
    recs: list[tuple[int, dict[str, str]]],
) -> tuple[float | None, str]:
    """`time.monotonic()` at `t_s = 0` of a motor-trial log (§10.6).

    The script logs `t_s` as the monotonic time minus the trial's start, so
    `t_monotonic_s - t_s` is the same on every row; the monotonic clock runs
    on between trials, which places the trials of a run on one time axis.

    Returns:
        The origin and "", or None and why: when a `t_monotonic_s` is not a
        number, or `t_monotonic_s - t_s` varies by more than 10 µs. The
        shared clock is optional, so the log keeps its events, timed by
        `t_s`, and only loses its place on the run clock.
    """
    try:
        mono = [_num(r, "t_monotonic_s", k) for k, r in recs]
    except ValueError as err:
        return None, str(err)
    diff = [m - _num(r, "t_s", k) for m, (k, r) in zip(mono, recs, strict=True)]
    spread = max(diff) - min(diff)
    if spread > MONOTONIC_TOLERANCE_S:
        return None, (
            f"its two clocks disagree: t_monotonic_s - t_s varies by "
            f"{1e6 * spread:.0f} µs"
        )
    return float(np.median(diff)), ""


def parse_pi_log(
    text: str,
    *,
    fmt: str | None = None,
    trial: str | None = None,
    motor: str | None = None,
    settle_s: float = 3.0,
    run_s: float = 2.0,
) -> PiEvents:
    """Read a Pi log of any format of §7 into a PiEvents.

    Args:
        text: Content of the log file.
        fmt: Expected format; detected from the header when None.
        trial: Trial identifier; required for the §7.2 formats, which do not record it.
        motor: `left`, `right`, or `both`; required for `daisy-task1`, which
            does not record it, checked against the `wheel` column of the cherry
            formats, and against the trial id and the motors driven in a
            `motor-trial` log.
        settle_s: `--settle` of `small_steps.py` (not logged; its default is 3 s).
        run_s: `--run` of `reversal_test.py` (not logged; its default is 2 s).

    Returns:
        The events in time order.

    Raises:
        ValueError: If the log is empty, of unknown or unexpected format, or malformed.
    """
    detected = detect_format(text)
    if fmt is not None and fmt != detected:
        raise ValueError(f"expected format {fmt!r}, found {detected!r}")
    lines = _lines(text)
    if detected == "new":
        return _parse_new(lines)
    if not trial:
        raise ValueError(f"trial is required for format {detected!r}")
    recs = _records(lines)
    if detected == "daisy-task1":
        rows = _parse_daisy(recs, trial, motor)
    elif detected == "motor-trial":
        try:
            trial_id: TrialId | None = parse_trial_id(trial)
        except ValueError:
            trial_id = None
        rows = _parse_motor_trial(recs, trial_id, motor)
        start = trial_id.start if trial_id is not None else None
        return _build(trial, rows, start, detected, *_monotonic_origin(recs))
    else:
        rows = _parse_cherry(detected, recs, motor, settle_s, run_s)
    return _build(trial, rows, None, detected)


def _motor_from_path(path: Path) -> str | None:
    """Motor of a log in `data/raw/<motor folder>/`, from the folder's name.

    Data is stored only there, so a log elsewhere gives None.

    Raises:
        ValueError: If the motor folder is misnamed.
    """
    parts = path.parts
    for k in range(1, len(parts) - 2):
        if parts[k - 1 : k + 1] == ("data", "raw"):
            return parse_motor_folder(parts[k + 1])[1]
    return None


def read_pi_log(
    path: str | Path,
    *,
    fmt: str | None = None,
    trial: str | None = None,
    motor: str | None = None,
) -> PiEvents:
    """Read a Pi log file (§7).

    The trial defaults to the file name stem (without `_events` for a
    `motor-trial` log), and the motor of a `daisy-task1` log to that of its motor
    folder, `data/raw/db17-<robot>-<motor>[-<id>...]/`.

    Raises:
        ValueError: As `parse_pi_log`.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    detected = detect_format(text)
    if motor is None and detected == "daisy-task1":
        motor = _motor_from_path(path.absolute())
        if motor is None:
            raise ValueError(
                f"{path}: the log is not in a data/raw motor folder; "
                "pass motor='left', 'right', or 'both'"
            )
    stem = path.stem
    if detected == "motor-trial" and stem.endswith("_events"):
        stem = stem[: -len("_events")]  # motor_trial.py names it <trial>_events.csv
    return parse_pi_log(text, fmt=fmt, trial=trial or stem, motor=motor)


def write_pi_log(events: PiEvents, out: TextIO) -> None:
    """Write events in the format for new runs (§7.1).

    Raises:
        ValueError: If the start time is unknown.
    """
    if events.start is None:
        raise ValueError("the format of §7.1 needs the wall-clock start")
    out.write(f"# start {events.start.isoformat()}\n")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(NEW_COLUMNS)
    for i in range(len(events)):
        v = float(events.value[i])
        w.writerow(
            [
                events.trial[i],
                f"{events.t_s[i]:.6f}",
                f"{events.t_after_s[i]:.6f}",
                events.channel[i],
                f"{v:.0f}" if events.channel[i] == "led" else repr(v),
                events.event[i],
            ]
        )


# --- Schedule reconstruction (§10.4)
# -------------------------------------------------------------

_PRE_S = 2.0  # six-motor schedules: rest before the first change


def _duty_of(tag: str) -> float:
    m = re.fullmatch(r"([pm])(\d{3})", tag)
    if not m:
        raise ValueError(f"tag {tag!r} is not a duty such as p050")
    return (1 if m.group(1) == "p" else -1) * int(m.group(2)) / 100


def six_motor_schedule(
    tid: TrialId, duration_s: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Command changes and LED-on intervals of a six-motor trial (ID and duration).

    Port of `schedule()` in scripts/six_motor/schedule.py (branch
    feature/six-motor-characterisation), which reconstructs what the Pi script
    motor_trial.py did;
    that script is not in the repository, so this is the weakest source of a schedule.

    Returns:
        (changes as (t, duty), LED-on intervals as (t_on, t_off)), seconds from
        the trial start.
    """
    e, tag, pre = tid.experiment, tid.tag, _PRE_S
    if e == "A":
        m = re.fullmatch(r"sweep_(forward|backward)_(\d{3})to(\d{3})by(\d{3})", tag)
        if not m:
            raise ValueError(f"A tag {tag!r} is not sweep_<dir>_<start>to<top>by<step>")
        sgn = 1 if m.group(1) == "forward" else -1
        lo, top, step = (
            int(m.group(2)) / 100,
            int(m.group(3)) / 100,
            int(m.group(4)) / 1000,
        )
        hold = 3.0 if step >= 0.05 else 2.0
        up = [round(lo + k * step, 4) for k in range(round((top - lo) / step) + 1)]
        levels = up + up[-2::-1]
        ch = [(0.0, 0.0)] + [(pre + k * hold, sgn * v) for k, v in enumerate(levels)]
        led = [
            (pre + k * hold + 0.04, pre + (k + 1) * hold) for k in range(len(levels))
        ]
        return ch + [(pre + len(levels) * hold, 0.0)], led
    if e in ("B", "E"):
        holds = {13.0: 8.0, 12.0: 8.0, 8.0: 3.0, 6.5: 1.5}
        if duration_s not in holds:
            raise ValueError(
                f"no known hold for duration {duration_s} s; known {sorted(holds)}"
            )
        u, hold = _duty_of(tag), holds[duration_s]
        return [(0.0, 0.0), (pre, u), (pre + hold, 0.0)], [(pre, pre + hold)]
    if e == "C":
        m = re.fullmatch(r"b(\d{3})([pm])(\d{3})", tag)
        if not m:
            raise ValueError(f"C tag {tag!r} is not b<base><p|m><delta>")
        base = int(m.group(1)) / 100
        du = (1 if m.group(2) == "p" else -1) * int(m.group(3)) / 100
        t1, hold = pre + (3.0 if duration_s == 13.0 else 4.0), 3.0
        return (
            [
                (0.0, 0.0),
                (pre, base),
                (t1, round(base + du, 4)),
                (t1 + hold, base),
                (t1 + 2 * hold, 0.0),
            ],
            [(t1, t1 + hold)],
        )
    m = re.fullmatch(r"([pm])to([pm])(\d{3})", tag)
    if not m:
        raise ValueError(f"D tag {tag!r} is not <p|m>to<p|m><duty>")
    u2 = (1 if m.group(2) == "p" else -1) * int(m.group(3)) / 100
    return [(0.0, 0.0), (pre, -u2), (pre + 4, u2), (pre + 7, 0.0)], [(pre + 4, pre + 7)]


def _arg_value(row: Mapping[str, str], keys: Sequence[str]) -> str | None:
    """A value from a trial-row column, or from `args` (`--key value`, `key=value`)."""
    for k in keys:
        if row.get(k, "").strip():
            return row[k].strip()
    tokens = shlex.split(row.get("args", ""))
    for i, tok in enumerate(tokens):
        for k in keys:
            if tok == f"--{k}" and i + 1 < len(tokens):
                return tokens[i + 1]
            if tok.startswith((f"--{k}=", f"{k}=")):
                return tok.split("=", 1)[1]
    return None


def _reconstruct_six_motor(row: Mapping[str, str], start: datetime | None) -> PiEvents:
    tid = parse_trial_id(row["trial"])
    dur, n = (
        _arg_value(row, ("duration", "duration_s", "T")),
        _arg_value(row, ("changes",)),
    )
    if dur is None:
        raise ValueError(
            "six-motor reconstruction needs the trial duration "
            "(column T or duration_s, or --duration)"
        )
    changes, led = six_motor_schedule(tid, float(dur))
    expected = 2 * len(led) + 2 if tid.experiment == "A" else len(changes)
    if n is not None and int(n) != expected:
        raise ValueError(
            f"logged change count {n} does not match the reconstructed {expected}"
        )
    sd = NOMINAL_TIMING_SD_S
    rows: list[Row] = [
        (t, t, ch, u, f"change{k:02d}", sd)
        for k, (t, u) in enumerate(changes)
        for ch in tid.channels
    ]
    for a, b in led:
        rows += [(a, a, "led", 1.0, "led_on", sd), (b, b, "led", 0.0, "led_off", sd)]
    return _build(row["trial"], rows, start or tid.start, "script:motor_trial.py")


def _task1_parser() -> argparse.ArgumentParser:
    """Argument parser of scripts/task1_static.py (branch daisy_bot_DB17)."""
    p = argparse.ArgumentParser(
        prog="task1_static.py", exit_on_error=False, add_help=False
    )
    p.add_argument("--bot", default="daisy")
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sweep", exit_on_error=False, add_help=False)
    s.add_argument("--wheel", choices=["left", "right"], required=True)
    s.add_argument("--duties", default="paper")
    s.add_argument("--sides", choices=["both", "pos", "neg"], default="both")
    s.add_argument("--on", type=float, default=3.0)
    s.add_argument("--off", type=float, default=2.0)
    s.add_argument("--anchor", type=float, default=0.8)
    h = sub.add_parser("hyst", exit_on_error=False, add_help=False)
    h.add_argument("--wheel", choices=["left", "right"], required=True)
    h.add_argument("--start", type=int, default=20)
    h.add_argument("--stop", type=int, default=100)
    h.add_argument("--step", type=int, default=1)
    h.add_argument("--dwell", type=float, default=1.0)
    h.add_argument("--dir", type=int, choices=[1, -1], default=1)
    h.add_argument("--anchor", type=float, default=0.8)
    f = sub.add_parser("floor", exit_on_error=False, add_help=False)
    f.add_argument("--duty", type=float, required=True)
    f.add_argument("--time", type=float, default=2.0)
    return p


def _task1_duties(spec: str, sides: str) -> list[float]:
    paper = [round(0.05 * i, 2) for i in range(1, 21)]
    if spec == "paper":
        mags = paper
    elif spec == "dense":
        mags = sorted(set(paper + [round(c / 255, 4) for c in range(24, 66, 2)]))
    else:
        mags = sorted({abs(float(x)) for x in spec.split(",")})
    return (mags if sides in ("both", "pos") else []) + (
        [-m for m in mags] if sides in ("both", "neg") else []
    )


def _q255(u: float) -> float:
    """Duty after the 8-bit rounding of motors.set_duty, q(u) = round(255 u) / 255."""
    return round(255 * u) / 255


def _reconstruct_task1(row: Mapping[str, str], start: datetime | None) -> PiEvents:
    try:
        a = _task1_parser().parse_args(shlex.split(row.get("args", "")))
    except (argparse.ArgumentError, SystemExit, ValueError) as exc:
        raise ValueError(
            f"task1_static.py argument error in {row.get('args')!r}: {exc}"
        ) from None
    steps: list[
        tuple[float, str, float]
    ] = []  # (sleep before, event, duty) in script order
    if a.mode == "floor":
        chs = MOTOR_CHANNELS["both"]
        steps = [(0.0, "on", _q255(a.duty)), (a.time, "off", 0.0)]
    else:
        chs = MOTOR_CHANNELS[a.wheel]
        anchor = a.dir * a.anchor if a.mode == "hyst" else a.anchor
        steps = [(0.0, "anchor_on", _q255(anchor)), (1.0, "off", 0.0)]
        if a.mode == "hyst":
            up = list(range(a.start, a.stop + 1, a.step))
            down = list(range(a.stop, a.start - 1, -a.step))
            seq = [("up", c) for c in up] + [("down", c) for c in down]
            for j, (phase, c) in enumerate(seq):
                steps.append(
                    (3.0 if j == 0 else a.dwell, phase, _q255(a.dir * c / 255))
                )
            steps.append((a.dwell, "off", 0.0))
        else:
            for u in _task1_duties(a.duties, a.sides):
                steps += [(a.off, "on", _q255(u)), (a.on, "off", 0.0)]
    # Each time is the sum of the sleeps before it; the script's per-call overhead
    # (about 4 ms
    # per step in R_hyst1) adds a lag that grows linearly, which the clock drift
    # absorbs, and the
    # sleep jitter accumulates, hence 2 ms times the root of the number of sleeps.
    rows: list[Row] = []
    t = 0.0
    for k, (dt, name, u) in enumerate(steps):
        t += dt
        rows += [
            (t, t, ch, u, name, NOMINAL_TIMING_SD_S * np.sqrt(k + 1)) for ch in chs
        ]
    return _build(row["trial"], rows, start, "script:task1_static.py")


def reconstruct_schedule(trial_row: Mapping[str, str]) -> PiEvents:
    """Command schedule of a trial without Pi log, from its trials.csv row (§5, §10.4).

    Supported scripts:
        - `task1_static.py` (Sam): modes `hyst`, `sweep` (also `--duties
          dense`), and `floor`,
          from `args` exactly as typed.
        - `motor_trial.py` (Saloni, six-motor experiments A-E): from the trial
          identifier and
          the trial duration, plus the change count as a check, given as columns `T` (or
          `duration_s`) and `changes`, or in `args` as `--duration X --changes N`.
          The script is not in the repository; this path ports schedule.py's
          reconstruction
          and is the weakest source of a schedule.

    The start is `start_pi` when given, else the time in the trial identifier (if any).

    Raises:
        ValueError: For an unknown script, invalid arguments, or an inconsistent
            change count.
    """
    script = Path(trial_row.get("script", "")).name
    start = (
        datetime.fromisoformat(trial_row["start_pi"])
        if trial_row.get("start_pi", "").strip()
        else None
    )
    if script == "task1_static.py":
        return _reconstruct_task1(trial_row, start)
    if script == "motor_trial.py":
        return _reconstruct_six_motor(trial_row, start)
    raise ValueError(
        f"no schedule reconstruction for script {trial_row.get('script')!r}"
    )
