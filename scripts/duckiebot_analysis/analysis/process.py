# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Process a motor folder into the per-frame and per-trial tables of §11.

Orchestrates data/schema.md §9 and §10 for each run of a motor folder:

1. read the frame table (§3), the run and trial lists (§4, §5), the setup
   (§6), and the measured constants (§8.2);
2. place each trial's commands on the video clock, at the run's
   synchronisation level or the next lower one whose source exists (§10),
   with one clock for the whole run when its trials share a Pi clock (§10.6),
   and compare every synchronisation the sources allow with the one used
   (§10.7);
3. compute the angle relative to the hub with k = 1, correct it once per turn
   in the steady parts of the schedule, and recompute it with the calibrated
   k (§9.3-§9.6); on the floor, take the wheel angle from the spoke columns
   of the frame table where it has them (§9.5);
4. compute ω and α (§9.7), the commanded and applied duty (§10.1), and on the
   floor the ground motion when the constants are measured (§9.8);
5. write ``<run>.parquet`` (§11.1) and ``trials.parquet`` (§11.2).

The synchronisation levels, from best to worst, are ``log+led``,
``script+led``, ``log+motion``, and ``script+motion`` (§10). A trial whose Pi
log is missing takes its schedule from ``trials.csv`` (§10.4), so a run may
mix log and script trials; the level used is recorded per trial.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from duckiebot_analysis.analysis import angle as angle_mod
from duckiebot_analysis.analysis import derivatives, sync
from duckiebot_analysis.analysis.run_sync import (
    SYNC_NAMES,
    Syncs,
    by_start,
    clock_record,
    compare_syncs,
    comparison_note,
    synchronise,
)
from duckiebot_analysis.analysis.spoke_angle import SpokeAngle, spoke_theta
from duckiebot_analysis.data import layout, pilogs
from duckiebot_analysis.data.frames import (
    LED_COLUMNS,
    SPOKE_COLUMNS,
    read_frames,
    time_base_seconds,
)

SCHEMA_VERSION = "2.1.0"
METADATA_KEY = "duckiebot"
LEVELS = ("log+led", "script+led", "log+motion", "script+motion")
# §9.6: a steady part has constant duty for at least 1.6 s and starts 1.2 s
# after the change. The hold is read as the time from the change to the next
# change, so the steady part is [change + 1.2 s, next change].
STEADY_HOLD_S = 1.6
STEADY_SETTLE_S = 1.2

FloatArray = npt.NDArray[np.float64]

PER_FRAME_SCHEMA = pa.schema(
    [
        pa.field("frame", pa.uint32(), nullable=False),
        pa.field("t_s", pa.float64()),
        pa.field("trial", pa.string()),
        pa.field("t_trial_s", pa.float64()),
        pa.field("usable", pa.bool_(), nullable=False),
        pa.field("hub_moved", pa.bool_(), nullable=False),
        pa.field("hub_imputed", pa.bool_(), nullable=False),
        pa.field("theta_rad", pa.float64()),
        pa.field("theta_sd_rad", pa.float64()),
        pa.field("omega_rad_s", pa.float64()),
        pa.field("omega_sd_rad_s", pa.float64()),
        pa.field("alpha_rad_s2", pa.float64()),
        pa.field("alpha_sd_rad_s2", pa.float64()),
        pa.field("u_cmd", pa.float64()),
        pa.field("u", pa.float64()),
        pa.field("x_ground_m", pa.float64()),
        pa.field("x_ground_sd_m", pa.float64()),
        pa.field("v_ground_m_s", pa.float64()),
        pa.field("v_ground_sd_m_s", pa.float64()),
    ]
)

HARM_COLUMNS = [f"harm_{c}{k}" for k in (1, 2, 3) for c in "ab"]

TRIALS_SCHEMA = pa.schema(
    [
        pa.field("trial", pa.string(), nullable=False),
        pa.field("run", pa.string(), nullable=False),
        pa.field("sync", pa.string()),
        pa.field("align", pa.string()),
        pa.field("offset_s", pa.float64()),
        pa.field("offset_sd_s", pa.float64()),
        pa.field("drift", pa.float64()),
        pa.field("drift_sd", pa.float64()),
        pa.field("timing_sd_s", pa.float64()),
        pa.field("led_block", pa.uint8()),
        pa.field("clock", pa.string()),
        pa.field("n_sync_events", pa.uint32()),
        *[pa.field(f"offset_{name}_s", pa.float64()) for name in SYNC_NAMES],
        *[pa.field(name, pa.float64()) for name in HARM_COLUMNS],
        pa.field("harm_k", pa.float64()),
        pa.field("n_frames", pa.uint32(), nullable=False),
        pa.field("n_usable", pa.uint32(), nullable=False),
        pa.field("n_dropped", pa.uint32(), nullable=False),
    ]
)

# Fields of §11.2 that need the LED: with motion alignment the clock absorbs
# the motor's delay (§10.5), so no drift, timing residual, or LED block exists.
LED_ONLY_FIELDS = ("drift", "drift_sd", "timing_sd_s", "led_block")


@dataclass(frozen=True)
class ProcessParams:
    """Processing parameters stored with the processed file (§9.2, §9.7).

    Attributes:
        n_min: Minimum marker pixel count of a usable frame.
        n_hub_min: Minimum hub pixel count of a usable frame.
        w_omega_s: Half-width of the ω fit in s.
        w_alpha_s: Half-width of the α fit in s.
    """

    n_min: int = angle_mod.N_MIN
    n_hub_min: int = angle_mod.N_HUB_MIN
    w_omega_s: float = derivatives.W_OMEGA
    w_alpha_s: float = derivatives.W_ALPHA


class RunSkippedError(Exception):
    """A run that cannot be processed yet, such as one without a frame table."""


@dataclass(frozen=True)
class _Schedule:
    """Command events of one trial and where they came from."""

    trial: str
    events: pilogs.PiEvents | None
    source: str | None  # "log" or "script"
    note: str


# ---------------------------------------------------------------- helpers


def code_commit() -> str | None:
    """Return the git commit of this code, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def _trial_row(record: layout.TrialRecord) -> dict[str, str]:
    """trials.csv row as reconstruct_schedule expects it, empty cells as ''."""
    return {k: ("" if v is None else str(v)) for k, v in asdict(record).items()}


def _motor_channel(motor: str) -> str:
    """Motor whose duty the per-frame u_cmd reports.

    A run of both motors drives them with the same duty in every script the
    data holds, so its u_cmd is that of the left motor.
    """
    return "left" if motor == "both" else motor


def _schedules(
    folder: Path,
    run: layout.RunRecord,
    trials: dict[str, layout.TrialRecord],
    motor: str,
) -> tuple[dict[str, pilogs.PiEvents], dict[str, pilogs.PiEvents], list[str]]:
    """Logged and reconstructed schedules of the run's trials.

    Returns:
        Logged events per trial, reconstructed events per trial, and notes on
        sources that are missing or unreadable.
    """
    notes: list[str] = []
    logged: dict[str, pilogs.PiEvents] = {}
    paths = [p.strip() for p in (run.pi_log or "").split(";") if p.strip()]
    for rel in paths:
        path = folder / rel
        if not path.is_file():
            notes.append(f"Pi log {rel} is missing")
            continue
        try:
            events = pilogs.read_pi_log(path, motor=motor)
        except ValueError as exc:
            notes.append(f"Pi log {rel} is unreadable: {exc}")
            continue
        trial = events.trial_id
        if trial not in run.trials and len(paths) == 1 and len(run.trials) == 1:
            # A single log of a single trial belongs to it whatever its name.
            trial = run.trials[0]
        logged[trial] = events
    scripted: dict[str, pilogs.PiEvents] = {}
    for trial in run.trials:
        record = trials.get(trial)
        if record is None:
            notes.append(f"trial {trial} is not in trials.csv")
            continue
        try:
            scripted[trial] = pilogs.reconstruct_schedule(_trial_row(record))
        except (ValueError, KeyError) as exc:
            notes.append(f"trial {trial} has no reconstructable schedule: {exc}")
    return logged, scripted, notes


def _led_grid(frames: dict[str, np.ndarray]) -> FloatArray | None:
    """LED blocks as an (N, 25) float array, or None when no LED was tracked."""
    grid = np.column_stack(
        [np.asarray(frames[c], dtype=np.float64) for c in LED_COLUMNS]
    )
    if grid.size == 0 or not np.isfinite(grid).any():
        return None
    return grid


def _level_chain(preferred: str | None) -> tuple[str, ...]:
    """The preferred level and every lower one, best first."""
    start = LEVELS.index(preferred) if preferred in LEVELS else 0
    return LEVELS[start:]


def _events_for(
    source: str,
    run_trials: Sequence[str],
    logged: dict[str, pilogs.PiEvents],
    scripted: dict[str, pilogs.PiEvents],
) -> dict[str, pilogs.PiEvents] | None:
    """Events of each trial for a level's source, or None if it is absent.

    The ``log`` source needs at least one logged trial; a trial without a
    log then falls back to its script, and its level is recorded as such.
    """
    if source == "log":
        if not any(t in logged for t in run_trials):
            return None
        out = {t: logged.get(t) or scripted.get(t) for t in run_trials}
    else:
        if not any(t in scripted for t in run_trials):
            return None
        out = {t: scripted.get(t) for t in run_trials}
    return {t: ev for t, ev in out.items() if ev is not None}


def _has_spokes(frames: dict[str, np.ndarray]) -> bool:
    return all(c in frames for c in SPOKE_COLUMNS)


def _spoke_rates(
    frames: dict[str, np.ndarray], t: FloatArray, turns: str, params: ProcessParams
) -> tuple[SpokeAngle, derivatives.Rates]:
    """Wheel angle from the spokes, and its ω and α (§9.5, §9.7)."""
    sp = spoke_theta(
        t,
        frames["spoke_angle"],
        frames["spoke_angle_sd"],
        frames["spoke_contrast"],
        positive_duty_turns=turns,  # type: ignore[arg-type]
    )
    return sp, derivatives.angular_rates(
        t,
        sp.theta,
        sp.theta_sd,
        usable=sp.usable,
        segment=sp.segment,
        w_omega=params.w_omega_s,
        w_alpha=params.w_alpha_s,
    )


def _first_pass_rates(
    frames: dict[str, np.ndarray],
    time_base: float,
    mode: str,
    direction: str,
    params: ProcessParams,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """ω from the uncorrected angle with k = 1, for the motion alignment.

    On the floor, the spoke angle is used where the frame table has it (§9.5);
    the alignment is sign-free, so the direction does not matter.
    """
    if mode == "floor" and _has_spokes(frames):
        t = angle_mod.time_axis(frames["pts"], time_base).t
        _, rates = _spoke_rates(frames, t, direction, params)
        return t, rates.omega, rates.omega_sd
    first = angle_mod.compute_angle(
        frames,
        time_base,
        mode=mode,  # type: ignore[arg-type]
        positive_duty_turns=direction,  # type: ignore[arg-type]
        k=1.0,
        n_min=params.n_min,
        n_hub_min=params.n_hub_min,
    )
    rates = derivatives.angular_rates(
        first.t,
        first.theta,
        first.theta_sd,
        usable=first.usable,
        segment=first.segment,
        w_omega=params.w_omega_s,
        w_alpha=params.w_alpha_s,
    )
    return first.t, rates.omega, rates.omega_sd


def _steady_windows(
    ev: pilogs.PiEvents,
    clock: sync.ClockFit,
    motor: str,
    span_end: float,
) -> tuple[tuple[float, float], ...]:
    """Steady parts of a trial on the video clock (§9.6)."""
    channel = f"motor_{motor}"
    m = ev.channel == channel
    if not m.any():
        return ()
    tau = (ev.t_s[m] + ev.t_after_s[m]) / 2
    value = ev.value[m]
    keep = np.r_[True, value[1:] != value[:-1]]
    times, _ = clock.to_video(tau[keep])
    ends = np.r_[times[1:], span_end]
    return tuple(
        (float(a + STEADY_SETTLE_S), float(b))
        for a, b in zip(times, ends, strict=True)
        if b - a >= STEADY_HOLD_S
    )


def _null_if_nan(values: npt.ArrayLike) -> pa.Array:
    a = np.asarray(values, dtype=np.float64)
    return pa.array(a, type=pa.float64(), mask=~np.isfinite(a))


def _paired(value: npt.ArrayLike, sd: npt.ArrayLike) -> tuple[pa.Array, pa.Array]:
    """A value and its standard deviation, both null where either is missing.

    A value without its uncertainty is not reported; derivatives.ground_speed,
    for one, fills unusable frames from their neighbours without an sd.
    """
    v = np.asarray(value, dtype=np.float64)
    s = np.asarray(sd, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(s)
    return _null_if_nan(np.where(ok, v, np.nan)), _null_if_nan(np.where(ok, s, np.nan))


def _ellipse_record(ellipse: angle_mod.Ellipse | None) -> dict[str, Any] | None:
    if ellipse is None:
        return None
    return {
        "centre_px": [float(x) for x in ellipse.centre],
        "transform": [[float(x) for x in row] for row in ellipse.transform],
        "semi_major_px": float(ellipse.semi_major_px),
        "semi_minor_px": float(ellipse.semi_minor_px),
        "semi_major_sd_px": float(ellipse.semi_major_sd_px),
        "n_points": int(ellipse.n_points),
    }


def _trial_k(fit: angle_mod.TrialHarmonics, run_k: float) -> float | None:
    """Per-trial k of §9.6, max(1, χ²/ν)^½ of the trial's fit with k = 1.

    The stored fit used σ scaled by the run's k, so its χ² is χ²(k = 1) / k²;
    this is exact except for hub-imputed frames, whose hub spread is not
    scaled by k (§9.4).
    """
    if fit.nu <= 0 or not np.isfinite(fit.chi2):
        return None
    return float(np.sqrt(max(1.0, fit.chi2 * run_k**2 / fit.nu)))


def _finite_or_none(x: Any) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------- one run


def process_run(
    folder: str | Path, run: str, params: ProcessParams | None = None
) -> tuple[pa.Table, list[dict[str, Any]], dict[str, Any]]:
    """Process one run of a motor folder (§9, §10).

    Args:
        folder: The motor folder ``data/raw/db17-<robot>-<motor>-<id>``.
        run: Run identifier, a row of ``runs.csv``.
        params: Processing parameters; defaults as in §9.2 and §9.7.

    Returns:
        The per-frame table of §11.1 (without file metadata), one record per
        trial of the run with the fields of §11.2, and the file metadata.

    Raises:
        RunSkippedError: When the run has no frame table.
        ValueError: When the run is not in ``runs.csv`` or an input is invalid.
    """
    params = params or ProcessParams()
    folder = Path(folder)
    setup = layout.read_setup(folder)
    runs = {r.run: r for r in layout.read_runs(folder)}
    if run not in runs:
        raise ValueError(f"run {run!r} is not in {folder / 'runs.csv'}")
    record = runs[run]
    path = layout.frames_path(folder, run)
    if not path.is_file():
        raise RunSkippedError(f"no frame table frames/{run}.parquet")
    trials = {t.trial: t for t in layout.read_trials(folder)}
    frames, frame_meta = read_frames(path)
    time_base = float(time_base_seconds(frame_meta))
    mode = frame_meta["mode"]
    notes: list[str] = []
    if mode != record.mode:
        notes.append(f"mode {mode!r} of the frame table overrides runs.csv")
    u_cmd_motor = _motor_channel(setup.motor)
    axis = angle_mod.time_axis(frames["pts"], time_base)
    t = axis.t
    n = t.size

    direction: str = setup.positive_duty_turns or "cw"
    sign_source = "setup.toml" if setup.positive_duty_turns else "assumed"

    # §10: clocks of the trials, at the first level whose sources exist.
    logged, scripted, sched_notes = _schedules(folder, record, trials, setup.motor)
    notes += sched_notes
    grid = _led_grid(frames)
    readout_c = layout.measured_constant(folder, "camera_readout")
    readout = None if readout_c is None else readout_c[0]
    cache: dict[str, tuple[FloatArray, FloatArray]] = {}

    def motion() -> tuple[FloatArray, FloatArray]:
        if "rates" not in cache:
            _, omega, omega_sd = _first_pass_rates(
                frames, time_base, mode, direction, params
            )
            cache["rates"] = (omega, omega_sd)
        return cache["rates"]

    clocks: dict[str, sync.ClockFit] = {}
    events: dict[str, pilogs.PiEvents] = {}
    found: dict[str, Syncs] = {}
    chosen: str | None = None
    led_wanted = _level_chain(record.sync)[0].endswith("+led")
    chain = _level_chain(record.sync)
    if all(t in logged for t in record.trials):
        # The script levels stand in for lost logs (§10.4); with every log
        # present each becomes the log level of the same alignment, which
        # keeps the timing the logs record.
        chain = tuple(dict.fromkeys(lv.replace("script+", "log+") for lv in chain))
    for level in chain:
        source, method = level.split("+")
        level_events = _events_for(source, record.trials, logged, scripted)
        if not level_events:
            notes.append(f"{level}: no {source} schedule")
            continue
        if source not in found:
            found[source] = synchronise(
                level_events,
                t,
                grid,
                motion,
                u_cmd_motor,
                readout,
                run=run,
                led=led_wanted,
            )
            notes += found[source].notes
        options = found[source].clocks
        # With a shared Pi clock, a per-trial way is a comparison only: when the
        # run clock fails, the trials alone are less constrained still (§10.6).
        ways = [f"run_{method}"]
        if not found[source].shared:
            ways.append(f"trial_{method}")
        name = next((n for n in ways if options.get(n)), None)
        if name is None:
            notes.append(f"{level}: no trial could be aligned")
            continue
        chosen, clocks = name, options[name]
        events = {k: v for k, v in level_events.items() if k in clocks}
        break
    if not clocks:
        notes.append("no trial could be placed on the video clock")
    options = found[source].clocks if chosen else {}
    comparison = compare_syncs(options, chosen)
    notes += [comparison_note(pair, c) for pair, c in comparison.items()]

    # Per-frame command and trial records (§10.1, §10.3).
    ordered = by_start(events.values())
    if ordered:
        cmd = sync.command_on_video(
            t, ordered, clocks, setup.quantisation or "pca9685", u_cmd_motor, run=run
        )
        u_cmd = np.asarray(cmd.u_cmd, dtype=np.float64)
        u_applied = np.asarray(cmd.u, dtype=np.float64)
        trial_col = list(cmd.trial)
        t_trial = np.asarray(cmd.t_trial_s, dtype=np.float64)
        sync_records = {r["trial"]: dict(r) for r in cmd.trials}
    else:
        u_cmd = np.full(n, np.nan)
        u_applied = np.full(n, np.nan)
        trial_col = [None] * n
        t_trial = np.full(n, np.nan)
        sync_records = {}
    if setup.quantisation is None and ordered:
        notes.append("quantisation unknown; applied duty computed as pca9685")
    trial_arr = np.array(trial_col, dtype=object)

    # Spans and steady parts of each trial (§9.6).
    windows: list[angle_mod.TrialWindows] = []
    window_trials: list[str] = []
    for ev in ordered:
        span = trial_arr == ev.trial_id
        if not span.any():
            continue
        end = float(t[np.flatnonzero(span)[-1]])
        steady = _steady_windows(ev, clocks[ev.trial_id], u_cmd_motor, end)
        windows.append(angle_mod.TrialWindows(span=span, steady=steady))
        window_trials.append(ev.trial_id)

    def angle_for(turns: str) -> angle_mod.AngleResult:
        return angle_mod.process_angle(
            frames,
            time_base,
            mode=mode,
            positive_duty_turns=turns,  # type: ignore[arg-type]
            trials=windows,
            n_min=params.n_min,
            n_hub_min=params.n_hub_min,
        )

    def rates_of(res: angle_mod.AngleResult) -> derivatives.Rates:
        return derivatives.angular_rates(
            res.t,
            res.theta,
            res.theta_sd,
            usable=res.usable,
            segment=res.segment,
            w_omega=params.w_omega_s,
            w_alpha=params.w_alpha_s,
        )

    result = angle_for(direction)
    rates = rates_of(result)
    if sign_source == "assumed":
        # §9.5: s makes a positive duty give a positive angle; without the
        # direction in setup.toml, take it from the sign of ω under the duty.
        both = np.isfinite(rates.omega) & np.isfinite(u_cmd) & (u_cmd != 0)
        if both.any() and np.sum(rates.omega[both] * u_cmd[both]) < 0:
            direction = "ccw"
            result = angle_for(direction)
            rates = rates_of(result)
        sign_source = "inferred from ω and u_cmd" if both.any() else "assumed"

    # §9.5: on the floor, the spokes give the wheel angle where the tape is
    # hidden; the marker angle stays in use for the ellipse of §9.8.
    wheel_angle: str | None = None
    out_usable, out_theta, out_theta_sd = result.usable, result.theta, result.theta_sd
    if mode == "floor":
        if _has_spokes(frames):
            wheel_angle = "spokes"
            if sign_source != "setup.toml":
                direction = "cw"
            spoke, rates = _spoke_rates(frames, t, direction, params)
            if sign_source != "setup.toml":
                both = np.isfinite(rates.omega) & np.isfinite(u_cmd) & (u_cmd != 0)
                if both.any() and np.sum(rates.omega[both] * u_cmd[both]) < 0:
                    direction = "ccw"
                    spoke, rates = _spoke_rates(frames, t, direction, params)
                sign_source = (
                    "inferred from the spoke ω and u_cmd" if both.any() else "assumed"
                )
            out_usable, out_theta, out_theta_sd = (
                spoke.usable,
                spoke.theta,
                spoke.theta_sd,
            )
        else:
            wheel_angle = "marker"
            notes.append(
                "no spoke columns in the frame table; wheel angle from the marker"
            )

    # §9.8: ground motion of a floor run, when the marker radius is measured.
    x_g = x_g_sd = v_g = v_g_sd = np.full(n, np.nan)
    marker_r = layout.measured_constant(folder, "marker_radius")
    if mode == "floor":
        if marker_r is None:
            notes.append("marker_radius not measured; no ground motion")
        elif result.ellipse is not None:
            rho, rho_sd = derivatives.pixel_scale(
                result.semi_major_px, result.semi_major_sd_px, *marker_r
            )
            hub_cov = angle_mod.centroid_covariance(
                frames["hub_n"],
                frames["hub_cxx"],
                frames["hub_cyy"],
                frames["hub_cxy"],
                result.k,
            )
            ground = derivatives.ground_position(
                frames["hub_x"],
                frames["hub_y"],
                hub_cov,
                rho,
                rho_sd,
                usable=result.usable,
            )
            x_g, x_g_sd = ground.x_m, ground.x_sd_m
            v_g, v_g_sd = derivatives.ground_speed(
                t, ground, usable=result.usable, half_width=params.w_omega_s
            )

    table = pa.Table.from_arrays(
        [
            pa.array(np.asarray(frames["frame"], dtype=np.uint32), pa.uint32()),
            pa.array(t, pa.float64()),
            pa.array(trial_col, pa.string()),
            _null_if_nan(t_trial),
            pa.array(out_usable, pa.bool_()),
            pa.array(result.hub_moved, pa.bool_()),
            pa.array(result.hub_imputed, pa.bool_()),
            *_paired(np.where(out_usable, out_theta, np.nan), out_theta_sd),
            *_paired(rates.omega, rates.omega_sd),
            *_paired(rates.alpha, rates.alpha_sd),
            _null_if_nan(u_cmd),
            _null_if_nan(u_applied),
            *_paired(x_g, x_g_sd),
            *_paired(v_g, v_g_sd),
        ],
        schema=PER_FRAME_SCHEMA,
    )

    # §11.2 records.
    harm = dict(zip(window_trials, result.harmonics, strict=False))
    gap = np.zeros(n)
    if axis.n_dropped and np.isfinite(axis.median_dt):
        dt = np.r_[0.0, np.diff(t)]
        gap = np.where(axis.dropped, np.round(dt / axis.median_dt) - 1, 0.0)
    records: list[dict[str, Any]] = []
    levels_used: dict[str, str | None] = {}
    for trial in record.trials:
        rec: dict[str, Any] = {name: None for name in TRIALS_SCHEMA.names}
        rec.update(trial=trial, run=run)
        span = trial_arr == trial
        rec.update(
            n_frames=int(span.sum()),
            n_usable=int((span & out_usable).sum()),
            n_dropped=int(gap[span].sum()),
        )
        for name, option in options.items():
            if trial in option:
                rec[f"offset_{name}_s"] = _finite_or_none(option[trial].a)
        if trial in sync_records:
            rec["clock"] = chosen.split("_")[0] if chosen else None
            rec["n_sync_events"] = int(clocks[trial].n_edges)
            ev = events[trial]
            clock = clocks[trial]
            method = "led" if clock.method == "led" else "motion"
            level = f"{'script' if ev.from_script else 'log'}+{method}"
            src = sync_records[trial]
            for key in (
                "offset_s",
                "offset_sd_s",
                "drift",
                "drift_sd",
                "timing_sd_s",
                "led_block",
            ):
                rec[key] = src.get(key)
            rec["sync"] = level
            rec["align"] = clock.method
            if method != "led":
                for key in LED_ONLY_FIELDS:
                    rec[key] = None
            for key in ("offset_s", "offset_sd_s", "drift", "drift_sd", "timing_sd_s"):
                rec[key] = _finite_or_none(rec[key])
            if rec["led_block"] is not None:
                rec["led_block"] = int(rec["led_block"])
            fit = harm.get(trial)
            if fit is not None and fit.corrected:
                for name, value in zip(HARM_COLUMNS, fit.coef, strict=True):
                    rec[name] = _finite_or_none(value)
                rec["harm_k"] = _trial_k(fit, result.k)
            levels_used[trial] = level
        else:
            levels_used[trial] = None
        records.append(rec)

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "robot": setup.robot,
        "motor": setup.motor,
        "mode": mode,
        "frames_file": f"frames/{run}.parquet",
        "video_sha256": frame_meta.get("video_sha256"),
        "parameters": asdict(params),
        "k": float(result.k),
        "chi2": _finite_or_none(result.chi2),
        "nu": int(result.nu),
        "ellipse": _ellipse_record(result.ellipse),
        "n_window_ellipses": len(result.window_ellipses),
        "sync_preferred": record.sync,
        "sync": _run_level(levels_used),
        "sync_per_trial": levels_used,
        "clock": clock_record(
            chosen, found[source].run_fits.get(chosen) if chosen else None
        ),
        "sync_comparison": comparison,
        "positive_duty_turns": direction,
        "positive_duty_turns_source": sign_source,
        "u_cmd_motor": u_cmd_motor,
        "quantisation": setup.quantisation,
        "steady_hold_s": STEADY_HOLD_S,
        "steady_settle_s": STEADY_SETTLE_S,
        "camera_readout_s": readout,
        "marker_radius_m": None if marker_r is None else list(marker_r),
        "code_commit": code_commit(),
        "processed": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
    }
    if wheel_angle is not None:
        metadata["wheel_angle"] = wheel_angle
    return table, records, metadata


def _run_level(levels: dict[str, str | None]) -> str | None:
    """One level for the run: the common one, or the worst used."""
    used = [lv for lv in levels.values() if lv is not None]
    if not used:
        return None
    return max(used, key=LEVELS.index)


# ---------------------------------------------------------------- folder


def write_run(path: Path, table: pa.Table, metadata: dict[str, Any]) -> None:
    """Write a per-frame table with its metadata (§11.1)."""
    table = table.replace_schema_metadata(
        {METADATA_KEY: json.dumps(metadata, ensure_ascii=False)}
    )
    pq.write_table(table, path, compression="zstd", compression_level=9)


def records_table(records: Sequence[dict[str, Any]]) -> pa.Table:
    """The §11.2 table of trial records."""
    return pa.Table.from_pylist(list(records), schema=TRIALS_SCHEMA)


def process_folder(
    folder: str | Path,
    out_folder: str | Path,
    params: ProcessParams | None = None,
    *,
    runs: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Process the runs of a motor folder and write §11.1 and §11.2.

    A run without a frame table is skipped with a message. When only some runs
    are processed, the rows of ``trials.parquet`` for the other runs are kept.

    Args:
        folder: The motor folder ``data/raw/db17-<robot>-<motor>-<id>``.
        out_folder: Output folder, ``data/processed/db17-<robot>-<motor>-<id>``.
        params: Processing parameters.
        runs: Runs to process; default every run in ``runs.csv``.
        log: Receives one message per run.

    Returns:
        The runs that were processed.

    Raises:
        ValueError: When a requested run is not in ``runs.csv`` or an input is
            invalid.
        FileNotFoundError: When ``setup.toml`` or ``runs.csv`` is absent.
    """
    params = params or ProcessParams()
    folder = Path(folder)
    out = Path(out_folder)
    layout.read_setup(folder)
    available = [r.run for r in layout.read_runs(folder)]
    if runs:
        unknown = [r for r in runs if r not in available]
        if unknown:
            raise ValueError(f"run(s) not in runs.csv: {', '.join(unknown)}")
        selected = list(dict.fromkeys(runs))
    else:
        selected = available
    if not selected:
        log(f"no runs in {folder / 'runs.csv'}; nothing written")
        return []
    done: list[str] = []
    records: list[dict[str, Any]] = []
    for run in selected:
        try:
            table, recs, meta = process_run(folder, run, params)
        except RunSkippedError as exc:
            log(f"skipped {run}: {exc}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        write_run(out / f"{run}.parquet", table, meta)
        records += recs
        done.append(run)
        levels = ", ".join(
            f"{t}: {lv or 'not synchronised'}"
            for t, lv in meta["sync_per_trial"].items()
        )
        log(f"processed {run}: {table.num_rows} frames, k = {meta['k']:.3g}; {levels}")
        for note in meta["notes"]:
            log(f"  {run}: {note}")
    if not done:
        log("no run had a frame table; nothing written")
        return done
    trials_path = out / "trials.parquet"
    if trials_path.exists():
        kept = [
            r
            for r in pq.read_table(trials_path).to_pylist()
            # A skipped run keeps its earlier rows, matching its <run>.parquet.
            if r["run"] not in done and (out / f"{r['run']}.parquet").exists()
        ]
        records = kept + records
    pq.write_table(records_table(records), trials_path, compression="zstd")
    return done


__all__ = [
    "PER_FRAME_SCHEMA",
    "TRIALS_SCHEMA",
    "ProcessParams",
    "RunSkippedError",
    "process_folder",
    "process_run",
    "records_table",
    "write_run",
]
