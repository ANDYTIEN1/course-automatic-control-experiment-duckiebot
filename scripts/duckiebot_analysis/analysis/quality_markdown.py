# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""The data quality report of a motor folder in Markdown (data/README.md).

`render_report` writes ``data/raw/<folder>/quality.md``, and `readme_row` the
folder's row of the summary table between the quality-summary markers of
data/README.md, both from `quality.assess_folder` and `quality.finish`.
Each line holds one sentence (data/README.md: semantic line breaks).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

from duckiebot_analysis.analysis.quality import (
    THRESHOLDS,
    FolderQuality,
    RunQuality,
    Thresholds,
    _num,
    _pct,
    _secs,
)

REPORT_NAME = "quality.md"
REPORT_VERSION = "1.0.0"
README_START = "<!-- quality-summary: start -->"
README_END = "<!-- quality-summary: end -->"
# Short labels for the summary in data/README.md.
SHORT = {
    "not-processed": "not tracked yet",
    "not-filmed": "trials outside the video",
    "not-synchronised": "trials not synchronised",
    "few-usable": "few usable frames",
    "dropped-frames": "dropped frames",
    "frame-gap": "long frame gaps",
    "angle-scatter": "angle scatter",
    "timing": "imprecise LED timing",
    "sync-disagree": "synchronisations disagree",
    "motion-ambiguous": "ambiguous motion alignment",
    "correlation-only": "clock on the correlation alone",
    "implausible-speed": "implausible speeds",
    "wrong-direction": "wheel turns against the duty",
}

# ---------------------------------------------------------------- report


def replace_marked(text: str, start: str, end: str, new: str) -> str:
    """Replace what lies between the lines `start` and `end`, keeping them.

    Raises:
        ValueError: If a marker is missing or the end comes first.
    """
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ValueError(f"marker {start!r} or {end!r} missing or out of order")
    head = text[: i + len(start)]
    return head + "\n" + new + text[j:]


def _sentences(text: str, indent: str) -> str:
    """One sentence per line (data/README.md: semantic line breaks)."""
    parts = [p.strip() for p in text.replace(". ", ".\n").splitlines() if p.strip()]
    return ("\n" + indent).join(parts)


def _fmt(x: float | None, spec: str) -> str:
    return "–" if x is None else format(x, spec)


def _verdict(rq: RunQuality) -> str:
    if rq.warned:
        return "**Warning**"
    return "Notes only" if any(t.issues for t in rq.trials) or rq.issues else "OK"


def _warning_codes(fq: FolderQuality) -> Counter[str]:
    codes: Counter[str] = Counter()
    for rq in fq.runs:
        for code in {i.code for i in rq.issues if i.severity == "warning"}:
            codes[code] += 1
        for tq in rq.trials:
            for code in {i.code for i in tq.issues if i.severity == "warning"}:
                if code not in {i.code for i in rq.issues}:
                    codes[code] += 1
    return codes


def _thresholds_table(th: Thresholds) -> list[str]:
    return [
        "| Check | Warning | Note |",
        "|---|---|---|",
        f"| Angle scatter in steady parts over its pixel-based uncertainty, k "
        f"(schema.md §9.6) | > {th.k_warn:g} | > {th.k_note:g} |",
        f"| Usable frames (§9.2) | < {_pct(th.usable_min)} | – |",
        f"| Frames dropped by the phone | – | > {_pct(th.dropped_note)} |",
        f"| Longest gap between frames, so ω is undetermined across it (§9.7) "
        f"| > {1e3 * th.gap_warn:g} ms | – |",
        f"| LED timing uncertainty σ_τ (§10.3) | > {1e3 * th.timing_warn:g} ms | – |",
        f"| Trial aligned alone by its LED against the run's LED clock (§10.7), "
        "a warning when no event of the trial supports the run clock "
        f"| > {1e3 * th.led_diff_warn:g} ms | > {1e3 * th.led_diff_warn:g} ms |",
        "| Trial aligned alone by its motion against the run's motion clock, "
        "a warning when no event of the trial supports the run clock "
        f"| > {th.motion_diff_warn:g} s | > {th.motion_diff_warn:g} s |",
        f"| Median motion against LED clock in the LED runs of the experiment "
        f"| > {th.motion_bias_warn:g} s | – |",
        f"| Usable frames faster than {th.omega_max:g} rad/s | "
        f"> {_pct(th.omega_over_warn)} | – |",
        f"| Steady frames at \\|u\\| ≥ {th.sign_duty_min:g} turning against the duty "
        f"| > {_pct(1 - th.sign_min)} | – |",
        "| Clock on the correlation of the schedule with ω alone, no departure "
        "from rest confirming it (§10.5) | always | – |",
        "| Synchronised by motion, not the LED | – | always |",
        "| Trial without a sync event of its own | – | always |",
    ]


def render_report(
    fq: FolderQuality,
    *,
    date: str,
    motion_bias: Mapping[str, float],
    th: Thresholds = THRESHOLDS,
) -> str:
    """The Markdown report ``quality.md`` of a motor folder."""
    trials = [t for rq in fq.runs for t in rq.trials]
    warned = [rq for rq in fq.runs if rq.warned]
    lines = [
        "---",
        f'title: "Duckiebot wheel data quality of {fq.folder}"',
        'author: "Torbjörn E. M. Nordling"',
        f'date: "{date}"',
        f"version: {REPORT_VERSION}",
        'license: "Apache-2.0"',
        f'purpose: "Checks of the measured data of motor folder {fq.folder}, '
        "their thresholds, and the warnings, notes, and recommendations for each "
        'run and trial."',
        "---",
        "",
        f"<!-- Generated by scripts/quality_report.py from data/raw/{fq.folder}/ "
        f"and data/processed/{fq.folder}/: edit the script, not this file. -->",
        "",
        f"# Data quality of `{fq.folder}`",
        "",
        f"Motor {fq.motor_label or '(no identifier)'}, tested in the {fq.motor} "
        f"position of {fq.robot.capitalize()}.",
        f"Of {len(fq.runs)} runs with {len(trials)} trials, {len(warned)} have "
        "warnings.",
        "A warning means that the data should not be used for the quantities it "
        "affects, and a note states a limitation.",
        "",
    ]
    if warned:
        lines += [
            "> [!WARNING]",
            f"> **{len(warned)} of {len(fq.runs)} runs have data "
            "that should not be used as it is.**",
        ]
        for rq in warned:
            codes = sorted(
                {i.code for i in rq.issues if i.severity == "warning"}
                | {
                    i.code
                    for t in rq.trials
                    for i in t.issues
                    if i.severity == "warning"
                }
            )
            n = sum(t.warned for t in rq.trials)
            labels = ", ".join(SHORT.get(c, c) for c in codes)
            lines.append(
                f"> - `{rq.run}`: {labels}, in {n} of {len(rq.trials)} trials."
            )
        lines += [
            "",
            "Each warning is given with its recommendation under its run below.",
            "",
        ]
    lines += ["## Checks", "", *_thresholds_table(th), ""]
    if motion_bias:
        lines += [
            "Median motion against LED clock by experiment, over the LED runs of "
            "all motor folders (schema.md §10.7):",
            "",
            "| Experiment | Median difference |",
            "|---|---|",
            *[f"| {e or '–'} | {b:.3g} s |" for e, b in sorted(motion_bias.items())],
            "",
        ]
    lines += ["## Folder", ""]
    if fq.issues:
        lines += [
            f"- **{i.severity.capitalize()}:** {_sentences(i.text, '  ')}"
            for i in fq.issues
        ]
    else:
        lines.append("The folder has no findings of its own.")
    lines += [
        "",
        "## Runs",
        "",
        "| Run | Experiment | Mode | Synchronised by | Frames | Dropped | k "
        "| Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rq in fq.runs:
        k = rq.meta.get("k")
        dropped = "–" if rq.dropped is None else _pct(rq.dropped)
        lines.append(
            f"| `{rq.run}` | {rq.experiment or '–'} | {rq.mode} | {rq.sync or '–'} "
            f"| {rq.n_frames} | {dropped} | {_fmt(k, '.3g')} | {_verdict(rq)} |"
        )
    by_exp: dict[str, list[RunQuality]] = defaultdict(list)
    for rq in fq.runs:
        by_exp[rq.experiment or "–"].append(rq)
    for exp in sorted(by_exp):
        lines += ["", f"## Experiment {exp}"]
        for rq in by_exp[exp]:
            lines += ["", f"### `{rq.run}`", ""]
            for i in rq.issues:
                lines.append(
                    f"- **{i.severity.capitalize()}:** {_sentences(i.text, '  ')}"
                )
            for pair, c in (rq.meta.get("sync_comparison") or {}).items():
                lines.append(
                    f"- Synchronisation {pair.replace('_', ' ')}: median "
                    f"{_secs(c['median_s'])}, largest difference "
                    f"{_secs(c['max_abs_s']).lstrip('+')} over {c['n']} trials."
                )
            if rq.issues or rq.meta.get("sync_comparison"):
                lines.append("")
            lines += [
                "| Trial | Frames | Usable | Sync events | k | Findings |",
                "|---|---|---|---|---|---|",
            ]
            for tq in rq.trials:
                rec = tq.record
                n = int(rec.get("n_frames") or 0)
                usable = _pct(int(rec.get("n_usable") or 0) / n) if n else "–"
                w = sum(i.severity == "warning" for i in tq.issues)
                notes = sum(i.severity == "note" for i in tq.issues)
                found = (
                    ", ".join(
                        s
                        for s in (
                            f"{w} warning{'s' * (w > 1)}" if w else "",
                            f"{notes} note{'s' * (notes > 1)}" if notes else "",
                        )
                        if s
                    )
                    or "–"
                )
                sync_ev = rec.get("n_sync_events")
                lines.append(
                    f"| `{tq.trial}` | {n} | {usable} | "
                    f"{'–' if sync_ev is None else sync_ev} | "
                    f"{_fmt(_num(rec, 'harm_k'), '.3g')} | {found} |"
                )
            flagged = [t for t in rq.trials if t.issues]
            if flagged:
                lines.append("")
            for tq in flagged:
                own = [i for i in tq.issues if i not in rq.issues]
                if not own and not tq.recommendation:
                    continue
                lines.append(f"- `{tq.trial}`")
                for i in own:
                    lines.append(
                        f"  - **{i.severity.capitalize()}:** "
                        f"{_sentences(i.text, '    ')}"
                    )
                if tq.recommendation:
                    lines.append(
                        "  - **Recommendation:** "
                        f"{_sentences(tq.recommendation, '    ')}"
                    )
    return "\n".join(lines) + "\n"


def readme_header() -> list[str]:
    """Header of the summary table in data/README.md."""
    return [
        "| Folder | Motor | Runs | Trials | Synchronised by | Quality |",
        "|---|---|---|---|---|---|",
    ]


def readme_row(fq: FolderQuality) -> str:
    """The folder's row of the summary table in data/README.md."""
    trials = [t for rq in fq.runs for t in rq.trials]
    syncs = Counter(
        (t.record.get("sync") or "").split("+")[-1] or "none"
        for t in trials
        if t.record.get("n_frames")
    )
    by = ", ".join(f"{k} {v}" for k, v in sorted(syncs.items())) or "–"
    codes = _warning_codes(fq)
    if codes:
        labels = ", ".join(
            f"{SHORT.get(c, c)} ({n})"
            for c, n in sorted(
                codes.items(), key=lambda cn: (-cn[1], SHORT.get(cn[0], cn[0]))
            )
        )
        quality = f"**Warning:** {labels}, see `{fq.folder}/{REPORT_NAME}`"
    else:
        quality = f"No warnings, see `{fq.folder}/{REPORT_NAME}`"
    return (
        f"| `{fq.folder}` | {fq.motor_label or '–'} | {len(fq.runs)} | {len(trials)} "
        f"| {by} | {quality} |"
    )


__all__ = [
    "README_END",
    "README_START",
    "REPORT_NAME",
    "readme_header",
    "readme_row",
    "render_report",
    "replace_marked",
]
