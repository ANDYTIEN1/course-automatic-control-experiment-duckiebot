# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""Motor folders and their text files (data/schema.md §2, §4, §5, §6, §8.2).

A motor folder is ``data/raw/db17-<robot>-<motor>-<id>/``, or
``db17-<robot>-both-<left id>-<right id>/`` for runs driving both motors; its
processed data goes to ``data/processed/`` under the same name
(data/README.md). Folders named before the motor identifiers were required,
such as ``db17-daisy-left``, carry no identifier.
"""

from __future__ import annotations

import csv
import io
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

MOTORS = ("left", "right", "both")
MODES = ("stand", "floor")
SYNC_LEVELS = ("log+led", "script+led", "log+motion", "script+motion")
TRIAL_STATUSES = ("completed", "aborted", "failed")

RUNS_COLUMNS = (
    "run",
    "video_file",
    "video_sha256",
    "video_bytes",
    "duration_s",
    "resolution",
    "fps_nominal",
    "mode",
    "experiment",
    "sync",
    "pi_log",
    "trials",
    "notes",
)
TRIALS_COLUMNS = (
    "trial",
    "run",
    "script",
    "script_commit",
    "args",
    "start_pi",
    "duration_s",
    "changes",
    "status",
    "notes",
)
MEASUREMENT_COLUMNS = (
    "quantity",
    "repeat",
    "value",
    "unit",
    "instrument",
    "resolution",
    "date",
    "by",
    "notes",
)

_ROBOT = re.compile(r"^[a-z0-9]+$")
# Any name that claims to be a motor folder, valid or not, so that a misnamed
# one is reported instead of being taken for an unrelated folder.
_FOLDER_CLAIM = re.compile(
    r"^db17[^a-z0-9]((.*[^a-z])?(left|right|both)([^a-z]|$)"
    r"|[a-z0-9]+[^a-z0-9](left|right|both))",
    re.IGNORECASE,
)
_FOLDER = re.compile(
    r"^db17-(?P<robot>[a-z0-9]+)-(?P<motor>left|right|both)(?P<ids>(?:-[a-z0-9]+)*)$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------- paths


def _id_count(motor: str) -> int:
    return 2 if motor == "both" else 1


def _check_ids(motor: str, ids: tuple[str, ...]) -> None:
    """Raise ValueError unless ``ids`` are valid motor identifiers for ``motor``.

    The single copy of the identifier rules, used to build and to parse names.
    """
    if ids and len(ids) != _id_count(motor):
        raise ValueError(
            f"motor {motor!r} needs {_id_count(motor)} motor identifier(s), "
            f"got {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"the left and right motor share the identifier {ids[0]!r}")
    for motor_id in ids:
        if not _ROBOT.match(motor_id):
            raise ValueError(
                f"motor identifier {motor_id!r} must be lower-case letters and digits"
            )


def motor_folder_name(robot: str, motor: str, motor_ids: tuple[str, ...]) -> str:
    """Return ``db17-<robot>-<motor>``, followed by ``-<id>`` for each motor.

    Args:
        robot: The robot the motors were mounted on during the test.
        motor: ``left`` or ``right``, the position the motor was mounted in, or
            ``both``.
        motor_ids: The identifier of the motor, or of the left and then the right
            motor for ``both``; ``()`` only for a folder named before identifiers
            were required. It has no default, so that leaving it out is an error.

    Raises:
        TypeError: If ``motor_ids`` is a string rather than a tuple of strings.
        ValueError: If ``robot`` or an identifier is not lower-case
            alphanumeric, ``motor`` is not left, right, or both, the number of
            identifiers does not match ``motor``, or both identifiers are equal.
    """
    if isinstance(motor_ids, str):
        raise TypeError(f"motor_ids must be a tuple of strings, not {motor_ids!r}")
    if not _ROBOT.match(robot or ""):
        raise ValueError(f"robot name {robot!r} must be lower-case letters and digits")
    if motor not in MOTORS:
        raise ValueError(f"motor {motor!r} must be one of {', '.join(MOTORS)}")
    _check_ids(motor, tuple(motor_ids))
    return "-".join(("db17", robot, motor, *motor_ids))


def raw_folder(
    robot: str,
    motor: str,
    motor_ids: tuple[str, ...],
    root: str | Path = "data",
) -> Path:
    """Return ``<root>/raw/`` joined with :func:`motor_folder_name`."""
    return Path(root) / "raw" / motor_folder_name(robot, motor, motor_ids)


def processed_folder(raw: str | Path) -> Path:
    """Map ``<root>/raw/<name>`` to ``<root>/processed/<name>``.

    Raises:
        ValueError: If the folder's parent is not named ``raw``.
    """
    raw = Path(raw)
    if raw.parent.name != "raw":
        raise ValueError(f"{raw} is not inside a raw/ folder")
    parse_motor_folder(raw)
    return raw.parent.parent / "processed" / raw.name


def _parse_folder(folder: str | Path) -> tuple[str, str, tuple[str, ...]]:
    name = Path(folder).name
    m = _FOLDER.match(name)
    ids = tuple(m.group("ids")[1:].split("-")) if m and m.group("ids") else ()
    try:
        if m is None:
            raise ValueError
        _check_ids(m.group("motor"), ids)
    except ValueError:
        raise ValueError(
            f"{name!r} is not of the form db17-<robot>-<left|right>-<id> or "
            "db17-<robot>-both-<left id>-<right id>, with the robot and distinct ids "
            "in lower-case letters and digits (or the older db17-<robot>-<motor>)"
        ) from None
    return m.group("robot"), m.group("motor"), ids


def is_motor_folder_name(name: str) -> bool:
    """Return whether ``name`` claims to be a motor folder, valid or not.

    True for a name that starts with ``db17`` and a separator and holds left,
    right, or both as a word of its own or at the start of its third field (a
    typo such as ``db17-daisy-lefty``), in any case, so that a misnamed motor
    folder, or a scratch copy such as ``db17-daisy-left-m1_old``, is reported by
    :func:`parse_motor_folder` instead of being taken for an unrelated folder.
    """
    return _FOLDER_CLAIM.match(name) is not None


def parse_motor_folder(folder: str | Path) -> tuple[str, str]:
    """Return ``(robot, motor)`` from a motor folder name.

    Raises:
        ValueError: If the name does not follow the pattern.
    """
    robot, motor, _ = _parse_folder(folder)
    return robot, motor


def motor_ids(folder: str | Path) -> tuple[str, ...]:
    """Return the motor identifiers of a motor folder name, left before right.

    Returns:
        One identifier for a left or right folder, two for a both folder, and
        none for a folder named before identifiers were required.

    Raises:
        ValueError: If the name does not follow the pattern.
    """
    return _parse_folder(folder)[2]


def frames_path(folder: str | Path, run: str) -> Path:
    """Return the frame table path ``<folder>/frames/<run>.parquet``."""
    return Path(folder) / "frames" / f"{run}.parquet"


# ---------------------------------------------------------------- CSV helpers


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str | None]]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in columns if c not in header]
        if missing:
            raise ValueError(f"{path} lacks column(s) {', '.join(missing)}")
        return [{c: (row.get(c) or None) for c in columns} for row in reader]


def _where(path: Path, line: int) -> str:
    return f"{path}, data row {line}"


def _required(row: dict[str, str | None], key: str, where: str) -> str:
    value = row[key]
    if value is None:
        raise ValueError(f"{where}: {key} is empty")
    return value


def _number(
    row: dict[str, str | None], key: str, where: str, kind: type = float
) -> Any:
    value = row[key]
    if value is None:
        return None
    try:
        return kind(value)
    except ValueError as exc:
        raise ValueError(f"{where}: {key} {value!r} is not a {kind.__name__}") from exc


def _choice(
    row: dict[str, str | None], key: str, allowed: tuple[str, ...], where: str
) -> Any:
    value = row[key]
    if value is not None and value not in allowed:
        raise ValueError(f"{where}: {key} {value!r} is not one of {', '.join(allowed)}")
    return value


# ---------------------------------------------------------------- runs.csv (§4)


@dataclass(frozen=True)
class RunRecord:
    """One row of ``runs.csv`` (§4); an empty cell is ``None``."""

    run: str
    video_file: str
    video_sha256: str
    video_bytes: int
    duration_s: float | None
    resolution: str | None
    fps_nominal: float | None
    mode: str
    experiment: str | None
    sync: str | None
    pi_log: str | None
    trials: tuple[str, ...]
    notes: str | None


def read_runs(folder: str | Path) -> list[RunRecord]:
    """Read ``<folder>/runs.csv``.

    Raises:
        ValueError: If a column is missing or a value is invalid.
    """
    path = Path(folder) / "runs.csv"
    out = []
    for k, row in enumerate(_read_csv(path, RUNS_COLUMNS), 1):
        where = _where(path, k)
        sha = _required(row, "video_sha256", where)
        if not _SHA256.match(sha):
            raise ValueError(f"{where}: video_sha256 must be 64 lower-case hex digits")
        video_bytes = _number(row, "video_bytes", where, int)
        if video_bytes is None:
            raise ValueError(f"{where}: video_bytes is empty")
        mode = _choice(row, "mode", MODES, where)
        if mode is None:
            raise ValueError(f"{where}: mode is empty")
        trials = row["trials"]
        out.append(
            RunRecord(
                run=_required(row, "run", where),
                video_file=_required(row, "video_file", where),
                video_sha256=sha,
                video_bytes=video_bytes,
                duration_s=_number(row, "duration_s", where),
                resolution=row["resolution"],
                fps_nominal=_number(row, "fps_nominal", where),
                mode=mode,
                experiment=row["experiment"],
                sync=_choice(row, "sync", SYNC_LEVELS, where),
                pi_log=row["pi_log"],
                trials=tuple(t.strip() for t in trials.split(";") if t.strip())
                if trials
                else (),
                notes=row["notes"],
            )
        )
    return out


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = repr(value)
        return text[:-2] if text.endswith(".0") else text
    if isinstance(value, tuple):
        return ";".join(value)
    return str(value)


def format_runs_row(record: RunRecord) -> str:
    """Return ``record`` as one CSV line of ``runs.csv``, ending in LF."""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(
        [_cell(getattr(record, name)) for name in RUNS_COLUMNS]
    )
    return buf.getvalue()


# ---------------------------------------------------------------- trials.csv (§5)


@dataclass(frozen=True)
class TrialRecord:
    """One row of ``trials.csv`` (§5); an empty cell is ``None``."""

    trial: str
    run: str
    script: str | None
    script_commit: str | None
    args: str | None
    start_pi: str | None
    duration_s: float | None
    changes: int | None
    status: str | None
    notes: str | None


def read_trials(folder: str | Path) -> list[TrialRecord]:
    """Read ``<folder>/trials.csv``.

    Raises:
        ValueError: If a column is missing or a value is invalid.
    """
    path = Path(folder) / "trials.csv"
    out = []
    for k, row in enumerate(_read_csv(path, TRIALS_COLUMNS), 1):
        where = _where(path, k)
        out.append(
            TrialRecord(
                trial=_required(row, "trial", where),
                run=_required(row, "run", where),
                script=row["script"],
                script_commit=row["script_commit"],
                args=row["args"],
                start_pi=row["start_pi"],
                duration_s=_number(row, "duration_s", where),
                changes=_number(row, "changes", where, int),
                status=_choice(row, "status", TRIAL_STATUSES, where),
                notes=row["notes"],
            )
        )
    return out


# ---------------------------------------------------------------- setup.toml (§6)


@dataclass(frozen=True)
class Setup:
    """Equipment of a motor folder (§6); a fact not known is ``None``.

    Keys that §6 does not list, such as ``computer``, ``camera``, or
    ``marker``, are kept unchecked in ``extra``.
    """

    robot: str
    motor: str
    motor_label: str | None = None
    hat: str | None = None
    hat_channel: str | None = None
    pwm_frequency_hz: float | None = None
    driver: str | None = None
    quantisation: str | None = None
    supply: str | None = None
    positive_duty_turns: str | None = None
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


_SETUP_CHOICES = {
    "motor": MOTORS,
    "quantisation": ("pca9685", "count255"),
    "positive_duty_turns": ("cw", "ccw"),
}
_SETUP_NUMBERS = ("pwm_frequency_hz",)


def read_setup(folder: str | Path) -> Setup:
    """Read ``<folder>/setup.toml``.

    When the folder name claims to be a motor folder
    (:func:`is_motor_folder_name`), it must follow the pattern of
    :func:`parse_motor_folder` and ``robot`` and ``motor`` must match it. When
    the name carries motor identifiers, ``motor_label`` must name the same
    motors, compared case-insensitively, the left before the right separated by
    a comma for ``both``.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If ``robot`` or ``motor`` is missing, a listed key has an
            invalid value, or the setup differs from the folder name.
    """
    folder = Path(folder)
    path = folder / "setup.toml"
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(Setup)} - {"extra"}
    values: dict[str, Any] = {}
    for key in known:
        if key not in data:
            continue
        value = data[key]
        if key in _SETUP_NUMBERS:
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or value <= 0
            ):
                raise ValueError(
                    f"{path}: {key} must be a positive number, got {value!r}"
                )
            value = float(value)
        elif not isinstance(value, str):
            raise ValueError(f"{path}: {key} must be a string, got {value!r}")
        if key in _SETUP_CHOICES and value not in _SETUP_CHOICES[key]:
            raise ValueError(
                f"{path}: {key} {value!r} is not one of "
                f"{', '.join(_SETUP_CHOICES[key])}"
            )
        values[key] = value
    for key in ("robot", "motor"):
        if key not in values:
            raise ValueError(f"{path}: {key} is missing")
    robot = motor = ""
    ids: tuple[str, ...] = ()
    if is_motor_folder_name(folder.name):  # other names are not checked
        robot, motor, ids = _parse_folder(folder)
    if robot and (values["robot"], values["motor"]) != (robot, motor):
        raise ValueError(
            f"{path}: robot {values['robot']!r} and motor {values['motor']!r} "
            f"differ from the folder name {folder.name!r}"
        )
    label = values.get("motor_label") or ""
    if ids and tuple(x.strip().lower() for x in label.split(",")) != ids:
        raise ValueError(
            f"{path}: motor_label {label!r} differs from {', '.join(ids)!r}, "
            f"the motor identifier(s) of the folder name {folder.name!r}"
        )
    extra = MappingProxyType({k: v for k, v in data.items() if k not in known})
    return Setup(**values, extra=extra)


# ---------------------------------------------------------------- measurements (§8.2)


@dataclass(frozen=True)
class Measurement:
    """One reading of ``manual/measurements.csv`` (§8.2)."""

    quantity: str
    repeat: int
    value: float
    unit: str
    instrument: str | None
    resolution: float
    date: str | None
    by: str | None
    notes: str | None


def read_measurements(folder: str | Path) -> list[Measurement]:
    """Read ``<folder>/manual/measurements.csv``; an absent file gives ``[]``.

    Raises:
        ValueError: If a column is missing or a value is invalid.
    """
    path = Path(folder) / "manual" / "measurements.csv"
    if not path.exists():
        return []
    out = []
    for k, row in enumerate(_read_csv(path, MEASUREMENT_COLUMNS), 1):
        where = _where(path, k)
        repeat = _number(row, "repeat", where, int)
        if repeat is None or repeat < 1:
            raise ValueError(f"{where}: repeat must be a positive integer")
        value = _number(row, "value", where)
        if value is None or not math.isfinite(value):
            raise ValueError(f"{where}: value must be a finite number")
        resolution = _number(row, "resolution", where)
        if resolution is None or not math.isfinite(resolution) or resolution < 0:
            raise ValueError(f"{where}: resolution must be a non-negative number")
        out.append(
            Measurement(
                quantity=_required(row, "quantity", where),
                repeat=repeat,
                value=value,
                unit=_required(row, "unit", where),
                instrument=row["instrument"],
                resolution=resolution,
                date=row["date"],
                by=row["by"],
                notes=row["notes"],
            )
        )
    return out


def measured_constant(folder: str | Path, quantity: str) -> tuple[float, float] | None:
    """Return the mean of a quantity's readings and its standard uncertainty.

    u = sqrt(s²/n + r²/12) (§8.2, GUM), with s the sample standard deviation
    and r the resolution. With one reading s is undefined and only the
    resolution term remains. When readings were taken at different
    resolutions, the largest is used, which is the conservative choice.

    Args:
        folder: The motor folder.
        quantity: Name in the ``quantity`` column, such as ``wheel_radius``.

    Returns:
        ``(mean, u)`` in the unit of the readings, or ``None`` when the
        quantity was not measured.

    Raises:
        ValueError: If the readings of the quantity use different units.
    """
    rows = [m for m in read_measurements(folder) if m.quantity == quantity]
    if not rows:
        return None
    units = {m.unit for m in rows}
    if len(units) > 1:
        raise ValueError(
            f"{quantity} is recorded in more than one unit: {sorted(units)}"
        )
    values = [m.value for m in rows]
    n = len(values)
    mean = math.fsum(values) / n
    scatter = math.fsum((v - mean) ** 2 for v in values) / (n - 1) / n if n > 1 else 0.0
    r = max(m.resolution for m in rows)
    return mean, math.sqrt(scatter + r * r / 12)
