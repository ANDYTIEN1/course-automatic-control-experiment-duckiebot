# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Quality of the data of a motor folder, and its report (README.md, Data).

Every trial and run is checked against the thresholds of `Thresholds`.
A finding is a warning, when the data should not be used for the quantities
it affects, or a note, when it states a limitation.
A trial with a warning gets a recommendation: use a trial of the same
condition without warnings, or redo the experiment.

`quality_markdown` writes the report and a summary table for a README;
`scripts/quality_report.py` runs both on the processed data (data/schema.md
§11).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from duckiebot_analysis.data import layout
from duckiebot_analysis.data.pilogs import parse_trial_id

MOTION_ALIGNS = ("motion", "correlation")


@dataclass(frozen=True)
class Thresholds:
    """Limits of the checks; each is motivated where it is used.

    Attributes:
        k_note, k_warn: Angle scatter over its pixel-based uncertainty in the
            steady parts (k, schema.md §9.6); the campaign's median trial k is
            about 3, so 5 is unusual and 10 means tracking errors.
        usable_min: Smallest fraction of usable frames (§9.2).
        dropped_note: Fraction of frames the phone dropped; the bursts lower
            the effective frame rate.
        gap_warn: Longest gap between frames above which ω is undetermined
            across it: the speed fit needs frames within its half-width,
            23 ms, on both sides (§9.7).
        timing_warn: LED timing residual sigma_tau (§10.3), about 3 ms when
            the LED is read well.
        led_diff_warn: Two LED alignments of a correct run agree to a few
            milliseconds (§10.7); 10 ms is more than two frame intervals.
        motion_diff_warn: Two motion alignments of a trial agree to a few
            frames; 0.1 s is more than the delay and rise they absorb.
        motion_bias_warn: Median |motion - LED| clock in the LED runs of an
            experiment above which its motion alignment is ambiguous (§10.7);
            the delay and rise give a few tens of milliseconds.
        omega_max: Speed no DG01D reaches: the campaign's steady speeds at
            duty 1 are about 31 rad/s.
        omega_over_warn: Fraction of usable frames above omega_max.
        sign_duty_min: Duty above which every motor of the campaign turns.
        steady_after_s: Time after a change of the duty before a frame counts
            as steady (§9.6).
        sign_min: Smallest fraction of steady frames turning with the duty.
    """

    k_note: float = 5.0
    k_warn: float = 10.0
    usable_min: float = 0.9
    dropped_note: float = 0.05
    gap_warn: float = 0.046
    timing_warn: float = 0.010
    led_diff_warn: float = 0.010
    motion_diff_warn: float = 0.1
    motion_bias_warn: float = 0.25
    omega_max: float = 50.0
    omega_over_warn: float = 0.001
    sign_duty_min: float = 0.4
    steady_after_s: float = 1.2
    sign_min: float = 0.95


THRESHOLDS = Thresholds()

# Warnings that a new recording with the LED in view would remove.
LED_CODES = (
    "not-synchronised",
    "motion-ambiguous",
    "sync-disagree",
    "correlation-only",
)


@dataclass(frozen=True)
class Issue:
    """One finding: severity ``warning`` or ``note``, a code, one sentence."""

    severity: str
    code: str
    text: str


@dataclass
class TrialQuality:
    """Findings of one trial, with the values they rest on."""

    trial: str
    run: str
    condition: str | None = None
    experiment: str | None = None
    record: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    recommendation: str | None = None

    @property
    def warned(self) -> bool:
        """True if the trial has a warning."""
        return any(i.severity == "warning" for i in self.issues)


@dataclass
class RunQuality:
    """Findings of one run and its trials."""

    run: str
    experiment: str | None
    mode: str
    sync: str | None
    meta: dict[str, Any] = field(default_factory=dict)
    n_frames: int = 0
    dropped: float | None = None
    issues: list[Issue] = field(default_factory=list)
    trials: list[TrialQuality] = field(default_factory=list)

    @property
    def run_warned(self) -> bool:
        """True if the run itself has a warning, which holds for all its trials."""
        return any(i.severity == "warning" for i in self.issues)

    @property
    def warned(self) -> bool:
        """True if the run or one of its trials has a warning."""
        return self.run_warned or any(t.warned for t in self.trials)


@dataclass
class FolderQuality:
    """Findings of a motor folder."""

    folder: str
    robot: str
    motor: str
    motor_label: str | None
    issues: list[Issue] = field(default_factory=list)
    runs: list[RunQuality] = field(default_factory=list)


# ---------------------------------------------------------------- metrics


def _pct(x: float) -> str:
    return f"{100 * x:.3g} %"


def _secs(x: float) -> str:
    """A signed time in ms below 1 s and in s above, to three digits."""
    return f"{1e3 * x:+.3g} ms" if abs(x) < 1 else f"{x:+.3g} s"


def trial_metrics(
    table: pa.Table, trial: str, th: Thresholds = THRESHOLDS
) -> dict[str, float | None]:
    """Speed plausibility and direction of one trial's frames.

    Returns:
        ``omega_over``, the fraction of usable frames faster than
        ``th.omega_max``, and ``sign_agreement``, the fraction of steady frames
        (duty unchanged for ``th.steady_after_s``, |u| >= ``th.sign_duty_min``)
        whose ω has the sign of the duty, and ``longest_gap_s``, the longest
        interval between its frames; None without frames.
    """
    names = table.column("trial").to_pylist()
    sel = np.array([name == trial for name in names], dtype=bool)
    if not sel.any():
        return {"omega_over": None, "sign_agreement": None, "longest_gap_s": None}
    t = table.column("t_s").to_numpy()[sel]
    usable = np.asarray(table.column("usable").to_pylist(), dtype=bool)[sel]
    omega = np.asarray(table.column("omega_rad_s").to_pylist(), dtype=float)[sel]
    u = np.asarray(table.column("u_cmd").to_pylist(), dtype=float)[sel]
    ok = usable & np.isfinite(omega)
    over = float(np.mean(np.abs(omega[ok]) > th.omega_max)) if ok.any() else None
    change = np.r_[True, ~((u[1:] == u[:-1]) | (np.isnan(u[1:]) & np.isnan(u[:-1])))]
    last = np.maximum.accumulate(np.where(change, np.arange(len(t)), 0))
    since = t - t[last]
    steady = ok & np.isfinite(u) & (np.abs(u) >= th.sign_duty_min)
    steady &= since >= th.steady_after_s
    sign = (
        float(np.mean(np.sign(omega[steady]) == np.sign(u[steady])))
        if steady.any()
        else None
    )
    gap = float(np.max(np.diff(t))) if len(t) > 1 else None
    return {"omega_over": over, "sign_agreement": sign, "longest_gap_s": gap}


# ---------------------------------------------------------------- checks


def _num(rec: Mapping[str, Any], key: str) -> float | None:
    v = rec.get(key)
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def classify_trial(
    rec: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    experiment: str | None,
    motion_bias: Mapping[str, float],
    n_trials: int = 2,
    th: Thresholds = THRESHOLDS,
) -> list[Issue]:
    """Findings of one trial from its §11.2 record and `trial_metrics`.

    Args:
        rec: The trial's row of trials.parquet.
        metrics: From `trial_metrics`.
        experiment: The run's experiment in runs.csv.
        motion_bias: Median |motion - LED| clock per experiment, from
            `motion_bias_by_experiment`.
        n_trials: Number of trials in the run.
        th: Thresholds.
    """
    out: list[Issue] = []
    if rec.get("sync") is None:
        return [
            Issue(
                "warning",
                "not-synchronised",
                "The trial could not be placed on the video clock, so its input "
                "is unknown.",
            )
        ]
    n = int(rec.get("n_frames") or 0)
    if n == 0:
        return [
            Issue(
                "warning",
                "not-filmed",
                "The trial is not in the video: the run clock places it outside "
                "the recording.",
            )
        ]
    usable = int(rec.get("n_usable") or 0) / n
    if usable < th.usable_min:
        out.append(
            Issue(
                "warning",
                "few-usable",
                f"Only {_pct(usable)} of its frames are usable, below "
                f"{_pct(th.usable_min)}.",
            )
        )
    dropped = int(rec.get("n_dropped") or 0) / (n + int(rec.get("n_dropped") or 0))
    if dropped > th.dropped_note:
        out.append(
            Issue(
                "note",
                "dropped-frames",
                f"The phone dropped {_pct(dropped)} of its frames, which widens "
                "the gaps the speed is computed across.",
            )
        )
    k = _num(rec, "harm_k")
    if k is not None and k > th.k_note:
        severity = "warning" if k > th.k_warn else "note"
        out.append(
            Issue(
                severity,
                "angle-scatter",
                f"In its steady parts the angle scatters {k:.3g} times its "
                "pixel-based uncertainty (k), which indicates tracking errors.",
            )
        )
    offset = _num(rec, "offset_s")
    align = rec.get("align")
    if align == "led":
        timing = _num(rec, "timing_sd_s")
        if timing is not None and timing > th.timing_warn:
            out.append(
                Issue(
                    "warning",
                    "timing",
                    f"The LED timing uncertainty is {1e3 * timing:.2g} ms, above "
                    f"{1e3 * th.timing_warn:.2g} ms.",
                )
            )
        if rec.get("n_sync_events") == 0:
            out.append(
                Issue(
                    "note",
                    "placed-by-run",
                    "No LED edge of this trial was seen, so the other trials of "
                    "the run place it.",
                )
            )
        other = _num(rec, "offset_trial_led_s")
        if rec.get("clock") == "run" and other is not None and offset is not None:
            d = other - offset
            if abs(d) > th.led_diff_warn:
                out.append(_disagreement(rec, "its LED", d))
    elif align in MOTION_ALIGNS:
        out.append(
            Issue(
                "note",
                "no-delay",
                "The trial is synchronised by the wheel motion, not the LED, so "
                "the delay and dead time cannot be measured from it.",
            )
        )
        # On a correlation clock the correlation-only warning below says the
        # same, so the note would repeat it.
        if rec.get("n_sync_events") == 0 and align != "correlation":
            by = (
                "the other trials of the run place it"
                if n_trials > 1
                else "its clock rests on the correlation of its schedule with ω alone"
            )
            out.append(
                Issue(
                    "note",
                    "placed-by-run",
                    f"No departure from rest of its own was found, so {by}.",
                )
            )
        bias = motion_bias.get(experiment or "")
        if align == "correlation":
            off = (
                f", and in the LED runs of experiment {experiment} the motion clock "
                f"is {bias:.3g} s off in median"
                if bias is not None
                else ""
            )
            out.append(
                Issue(
                    "warning",
                    "correlation-only",
                    "Its clock rests on the correlation of the whole schedule with "
                    f"ω alone, which no departure from rest confirms{off}, so "
                    "neither its timing nor its input at each frame can be "
                    "trusted.",
                )
            )
        if bias is not None and bias > th.motion_bias_warn:
            out.append(
                Issue(
                    "warning",
                    "motion-ambiguous",
                    f"In the LED runs of experiment {experiment} the motion clock "
                    f"differs from the LED clock by {bias:.3g} s in median, so "
                    "the motion alignment of this trial may be off by a whole "
                    "step of the schedule.",
                )
            )
        other = _num(rec, "offset_trial_motion_s")
        if rec.get("clock") == "run" and other is not None and offset is not None:
            d = other - offset
            if abs(d) > th.motion_diff_warn:
                out.append(_disagreement(rec, "its first motion", d))
    over = metrics.get("omega_over")
    if over is not None and over > th.omega_over_warn:
        out.append(
            Issue(
                "warning",
                "implausible-speed",
                f"{_pct(over)} of its usable frames give a speed above "
                f"{th.omega_max:g} rad/s, which the motor cannot reach.",
            )
        )
    gap = metrics.get("longest_gap_s")
    if gap is not None and gap > th.gap_warn:
        out.append(
            Issue(
                "warning",
                "frame-gap",
                f"The phone dropped {1e3 * gap:.2g} ms of frames at once, longer "
                f"than {1e3 * th.gap_warn:.2g} ms, so ω is undetermined across "
                "that gap.",
            )
        )
    sign = metrics.get("sign_agreement")
    if sign is not None and sign < th.sign_min:
        out.append(
            Issue(
                "warning",
                "wrong-direction",
                f"In its steady parts at |u| ≥ {th.sign_duty_min:g} the wheel "
                f"turns against the duty in {_pct(1 - sign)} of the frames, so "
                "the angle or its direction is wrong.",
            )
        )
    return out


def _disagreement(rec: Mapping[str, Any], by: str, d: float) -> Issue:
    """A per-trial alignment that differs from the run clock (§10.7).

    When the run clock rests on events of the trial itself, the per-trial
    alignment, which uses less information, is the one that failed, and the
    finding is a note; without such events the run clock only extrapolates to
    the trial, and the disagreement is a warning.
    """
    own = int(rec.get("n_sync_events") or 0) > 0
    text = f"Aligned alone by {by}, the trial starts {d:+.3g} s from the run clock"
    if own:
        return Issue(
            "note",
            "sync-disagree",
            text + ", which its own events in the run clock contradict.",
        )
    return Issue(
        "warning",
        "sync-disagree",
        text + ", and no event of its own confirms the run clock.",
    )


def condition_of(trial: str, experiment: str | None) -> str:
    """The condition a trial repeats: experiment letter and tag of its ID (§5).

    A trial whose ID does not follow the six-motor pattern is its own
    condition within its run's experiment.
    """
    try:
        tid = parse_trial_id(trial)
    except ValueError:
        return f"{experiment or 'trial'} {trial}"
    return f"{tid.experiment} {tid.tag}"


def recommend(
    trials: Sequence[TrialQuality],
    *,
    motor: str,
    elsewhere: Mapping[str, Sequence[str]],
) -> None:
    """Set the recommendation of every trial with a warning.

    A trial of the same condition in the folder without warnings is
    recommended in its place; without one, redoing the experiment, naming the
    motors whose data of the condition has no warning.

    Args:
        trials: The trials of a folder.
        motor: The folder's motor identifier.
        elsewhere: Per condition, motors with a trial of it without warnings.
    """
    for tq in trials:
        if not tq.warned:
            tq.recommendation = None
            continue
        if any(i.code == "not-tracked" for i in tq.issues):
            # The video exists; only its frame table is missing.
            tq.recommendation = (
                "Track the video with scripts/track_video.py and process the "
                "folder with scripts/process_raw.py (README.md, From video to "
                "figures, steps 2 and 3)."
            )
            continue
        if any(i.code == "not-processed" for i in tq.issues):
            tq.recommendation = (
                "Process the folder with scripts/process_raw.py (README.md, "
                "From video to figures, step 3)."
            )
            continue
        alternatives = [
            o.trial
            for o in trials
            if o is not tq and o.condition == tq.condition and not o.warned
        ]
        if alternatives:
            tq.recommendation = (
                f"Use {', '.join(alternatives)} instead, the same condition "
                "without warnings."
            )
            continue
        how = (
            " with the sync LED in view"
            if any(i.code in LED_CODES for i in tq.issues)
            else ""
        )
        rec = f"Redo {tq.condition} on motor {motor}{how}."
        others = [m for m in elsewhere.get(tq.condition or "", ()) if m != motor]
        if others:
            rec += (
                f" Data of this condition without warnings exists for motor "
                f"{', '.join(others)}."
            )
        tq.recommendation = rec


# ---------------------------------------------------------------- folder


def _trials_table(processed: Path) -> list[dict[str, Any]]:
    path = processed / "trials.parquet"
    return pq.read_table(path).to_pylist() if path.is_file() else []


def motion_bias_by_experiment(raw_root: Path, processed_root: Path) -> dict[str, float]:
    """Median |motion - LED| clock per experiment over the LED runs (§10.7).

    Over every trial whose clock is the run's LED clock and that the run's
    motion clock also places, by the experiment of its run in runs.csv.
    """
    diffs: dict[str, list[float]] = defaultdict(list)
    for raw in sorted(
        p for p in raw_root.iterdir() if layout.is_motor_folder_name(p.name)
    ):
        experiments = {r.run: r.experiment for r in layout.read_runs(raw)}
        for rec in _trials_table(processed_root / raw.name):
            motion, used = _num(rec, "offset_run_motion_s"), _num(rec, "offset_s")
            if rec.get("align") == "led" and rec.get("clock") == "run":
                if motion is not None and used is not None:
                    diffs[experiments.get(rec["run"]) or ""].append(abs(motion - used))
    return {e: float(np.median(v)) for e, v in diffs.items() if v}


def _folder_issues(raw: Path, setup: layout.Setup) -> list[Issue]:
    out = []
    if not (raw / "manual" / "measurements.csv").is_file():
        out.append(
            Issue(
                "note",
                "no-measurements",
                "manual/measurements.csv is missing, so the wheel and marker radii "
                "are unknown: floor runs give no ground speed or slip.",
            )
        )
    elif layout.measured_constant(raw, "camera_readout") is None:
        out.append(
            Issue(
                "note",
                "no-readout",
                "The camera readout time is not measured, so half a frame "
                "interval is added to the timing uncertainty (§9.1).",
            )
        )
    if setup.positive_duty_turns is None:
        out.append(
            Issue(
                "note",
                "direction-inferred",
                "setup.toml gives no direction of a positive duty, so the sign of "
                "the angle is inferred from ω and the duty.",
            )
        )
    return out


def _run_issues(meta: Mapping[str, Any], th: Thresholds) -> list[Issue]:
    out = []
    k = meta.get("k")
    if k is not None and k > th.k_warn:
        out.append(
            Issue(
                "warning",
                "angle-scatter",
                f"The run's k is {k:.3g}, so the uncertainty of every angle of "
                "the run is scaled by it.",
            )
        )
    preferred, used = meta.get("sync_preferred"), meta.get("sync")
    if preferred and used and preferred.endswith("+led") and used.endswith("+motion"):
        reason = next(
            (n for n in meta.get("notes", []) if n.startswith("run LED clock")), ""
        )
        out.append(
            Issue(
                "note",
                "led-rejected",
                f"runs.csv lists {preferred}, but the LED clock was rejected"
                f"{' (' + reason.split(': ', 1)[-1] + ')' if reason else ''}, so "
                "the run is synchronised by motion.",
            )
        )
    if meta.get("mode") == "floor" and meta.get("marker_radius_m") is None:
        out.append(
            Issue(
                "note",
                "no-ground-motion",
                "The marker radius is not measured, so the floor run gives no "
                "ground position or speed.",
            )
        )
    return out


def assess_folder(
    raw: Path,
    processed: Path,
    *,
    motion_bias: Mapping[str, float],
    th: Thresholds = THRESHOLDS,
) -> FolderQuality:
    """Check every run and trial of a motor folder (without recommendations).

    Args:
        raw: ``data/raw/<folder>``.
        processed: ``data/processed/<folder>``.
        motion_bias: From `motion_bias_by_experiment`.
        th: Thresholds.
    """
    setup = layout.read_setup(raw)
    ids = layout.motor_ids(raw)
    fq = FolderQuality(
        raw.name,
        setup.robot,
        setup.motor,
        ", ".join(ids) if ids else setup.motor_label,
        _folder_issues(raw, setup),
    )
    records = defaultdict(list)
    for rec in _trials_table(processed):
        records[rec["run"]].append(rec)
    for run in layout.read_runs(raw):
        # The level used, from the processed file; none until it exists.
        rq = RunQuality(run.run, run.experiment, run.mode, None)
        path = processed / f"{run.run}.parquet"
        if not path.is_file():
            if layout.frames_path(raw, run.run).is_file():
                issue = Issue(
                    "warning",
                    "not-processed",
                    "The run is not processed yet, so nothing is computed from it.",
                )
            else:
                issue = Issue(
                    "warning",
                    "not-tracked",
                    "The run has no frame table yet, so nothing is computed from it.",
                )
            rq.issues.append(issue)
            rq.trials = [
                TrialQuality(
                    t, run.run, condition_of(t, run.experiment), run.experiment
                )
                for t in run.trials
            ]
            fq.runs.append(rq)
            continue
        table = pq.read_table(path)
        rq.meta = json.loads(table.schema.metadata[b"duckiebot"])
        rq.mode, rq.sync = rq.meta.get("mode", rq.mode), rq.meta.get("sync")
        rq.n_frames = table.num_rows
        rq.issues = _run_issues(rq.meta, th)
        recs = {r["trial"]: r for r in records[run.run]}
        n_dropped = sum(int(r.get("n_dropped") or 0) for r in recs.values())
        rq.dropped = n_dropped / (rq.n_frames + n_dropped) if rq.n_frames else None
        for trial in run.trials:
            rec = recs.get(trial, {"trial": trial, "n_frames": 0})
            metrics = trial_metrics(table, trial, th)
            rq.trials.append(
                TrialQuality(
                    trial,
                    run.run,
                    condition_of(trial, run.experiment),
                    run.experiment,
                    dict(rec),
                    metrics,
                    classify_trial(
                        rec,
                        metrics,
                        experiment=run.experiment,
                        motion_bias=motion_bias,
                        n_trials=len(run.trials),
                        th=th,
                    ),
                )
            )
        fq.runs.append(rq)
    return fq


def clean_conditions(folders: Sequence[FolderQuality]) -> dict[str, list[str]]:
    """Per condition, the motors with a trial of it without warnings."""
    out: dict[str, list[str]] = defaultdict(list)
    for fq in folders:
        for rq in fq.runs:
            for tq in rq.trials:
                if tq.condition and not tq.warned and not rq.run_warned:
                    label = fq.motor_label or fq.folder
                    if label not in out[tq.condition]:
                        out[tq.condition].append(label)
    return dict(out)


def finish(folders: Sequence[FolderQuality]) -> None:
    """Add a run's own warnings to its trials, then set every recommendation."""
    for fq in folders:
        for rq in fq.runs:
            run_warnings = [i for i in rq.issues if i.severity == "warning"]
            for tq in rq.trials:
                tq.issues = run_warnings + [
                    i for i in tq.issues if i not in run_warnings
                ]
    elsewhere = clean_conditions(folders)
    for fq in folders:
        recommend(
            [t for rq in fq.runs for t in rq.trials],
            motor=fq.motor_label or fq.folder,
            elsewhere=elsewhere,
        )


__all__ = [
    "THRESHOLDS",
    "FolderQuality",
    "Issue",
    "RunQuality",
    "Thresholds",
    "TrialQuality",
    "assess_folder",
    "classify_trial",
    "clean_conditions",
    "condition_of",
    "finish",
    "motion_bias_by_experiment",
    "recommend",
    "trial_metrics",
]
