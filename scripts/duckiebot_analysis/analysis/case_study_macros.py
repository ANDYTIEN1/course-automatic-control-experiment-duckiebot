# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""LaTeX macros holding every number the CSME 2026 case study reports.

The article inputs the generated file, so no number in it is copied by hand.
An uncertainty is rounded to two significant figures and its value to the
same decimal place (doc-writing guide); a time is given in ms.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from duckiebot_analysis.analysis.case_study import (
    FINAL_S,
    REST_EXCEEDANCE,
    CaseStudy,
    StepResult,
)
from duckiebot_analysis.analysis.step_response import SETTLING_BAND

MS = 1e3
EARLY_S = 0.025  # start of a step over which a late start is measured
STEP_NAMES = {
    "up06": "UpSix",
    "up10": "UpTen",
    "down06": "DownSix",
    "down0": "DownZero",
}


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"cannot format a non-finite number: {values}")


def fmt(value: float, decimals: int) -> str:
    """Return value with the given number of decimals, never as -0."""
    _finite(value)
    text = f"{value:.{max(decimals, 0)}f}"
    return text[1:] if re.fullmatch(r"-0(\.0*)?", text) else text


def fmt_pm(value: float, sd: float) -> str:
    r"""Return ``value \pm sd`` with sd to two significant figures."""
    _finite(value, sd)
    if sd <= 0:
        return fmt(value, 2)
    decimals = 1 - math.floor(math.log10(sd))
    value_text = fmt(round(value, decimals), decimals)
    return f"{value_text} \\pm {fmt(round(sd, decimals), decimals)}"


def write_macros(path: Path, macros: dict[str, str], command: str) -> None:
    """Write one \\newcommand per macro, under a header naming the command."""
    bad = [k for k in macros if not re.fullmatch(r"[A-Za-z]+", k)]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    lines = [
        "% Generated; do not edit. Every number of the article, computed from",
        "% data/processed/ and data/raw/ by:",
        f"%   {command}",
    ]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _step_macros(prefix: str, s: StepResult) -> dict[str, str]:
    i, f, fd = s.info, s.fit, s.fit_delay
    change = abs(i.final - i.initial)
    noise = np.concatenate([r.omega_sd[r.t >= 0] for r in s.reps])
    out = {
        "Initial": fmt(i.initial, 1),
        "Final": fmt(i.final, 1),
        "Rise": fmt(i.rise_time_s * MS, 0),
        "Gain": fmt_pm(f.gain, f.gain_sd),
        "Tau": fmt_pm(f.tau_s * MS, f.tau_sd * MS),
        "Rms": fmt(f.rms_rel * 100, 1),
        "SettleModel": fmt(f.tau_s * math.log(1 / SETTLING_BAND) * MS, 0),
        "GainD": fmt_pm(fd.gain, fd.gain_sd),
        "TauD": fmt_pm(fd.tau_s * MS, fd.tau_sd * MS),
        "DelayD": fmt_pm(fd.delay_s * MS, fd.delay_sd * MS),
        "RmsD": fmt(fd.rms_rel * 100, 1),
        "Change": fmt(change, 1),
        "Band": fmt(SETTLING_BAND * change, 2),
        "NoiseRms": fmt(float(np.sqrt(np.nanmean(noise**2))) / change * 100, 1),
    }
    if math.isfinite(i.settling_time_s):
        out["Settle"] = fmt(i.settling_time_s * MS, 0)
    if math.isfinite(i.overshoot):
        out["Overshoot"] = fmt(i.overshoot * 100, 1)
    return {prefix + k: v for k, v in out.items()}


def _early_lag(s: StepResult, window_s: float) -> float:
    """Largest amount by which a repetition lags the first-order fit early on."""
    lags = []
    for k, r in enumerate(s.reps):
        sel = (r.t >= 0) & (r.t <= window_s)
        res = r.omega[sel] - s.fit.predict(r.t[sel], s.fit.omega0[k])
        # A lagging wheel is below the model on a step up, above it on a step down.
        lags.append(np.nanmax(-np.sign(s.u_to - s.u_from) * res))
    return float(max(lags))


def _ripple_sd(s: StepResult) -> float:
    """Mean over repetitions of the sd of the speed over the last FINAL_S."""
    return float(
        np.mean([np.nanstd(r.omega[r.t >= s.hold_s - FINAL_S], ddof=1) for r in s.reps])
    )


def _level(cs: CaseStudy, duty: float, rising: bool) -> tuple[float, float]:
    for lv in cs.e1_levels:
        if math.isclose(lv.duty, duty) and lv.rising == rising:
            return lv.mean, lv.sd
    raise ValueError(f"no {'rising' if rising else 'falling'} level at {duty}")


def _dropped(cs: CaseStudy, trials: list[str]) -> float:
    rows = [cs.trials[t] for t in trials]
    lost = sum(r["n_dropped"] for r in rows)
    return float(100 * lost / (lost + sum(r["n_frames"] for r in rows)))


def case_study_macros(cs: CaseStudy) -> dict[str, str]:
    """Return the macros of every number the article reports."""
    m: dict[str, str] = {}
    for key, prefix in STEP_NAMES.items():
        m |= _step_macros(prefix, cs.steps[key])
    up06, up10 = cs.steps["up06"], cs.steps["up10"]
    down06, down0 = cs.steps["down06"], cs.steps["down0"]
    w02, sd02 = _level(cs, 0.2, rising=True)
    w_start, sd_start = _level(cs, cs.e1_start, rising=True)
    rows = list(cs.trials.values())
    w_min = cs.omega_min
    turning_down = [
        abs(x.duty) for x in cs.e1_levels if not x.rising and abs(x.mean) > w_min
    ]
    speeds = [abs(x.mean) for x in cs.e1_levels + cs.e1_neg_levels]
    m |= {
        "OmegaMin": fmt(w_min, 2),
        "RestShare": fmt(REST_EXCEEDANCE * 100, 0),
        "RestFrames": str(len(cs.rest_noise)),
        "RestSd": fmt(float(np.std(cs.rest_noise, ddof=1)), 2),
        "RestBeyond": str(int(np.sum(np.abs(cs.rest_noise) > w_min))),
        "RestMax": fmt(float(np.max(np.abs(cs.rest_noise))), 2),
        "RestLevelMax": fmt(max(w for w in speeds if w <= w_min), 2),
        "TurnLevelMin": fmt(min(w for w in speeds if w > w_min), 1),
        "LastTurningDuty": fmt(min(turning_down), 2),
        "EarlyLagMs": fmt(EARLY_S * MS, 0),
        "UpSixLag": fmt(_early_lag(up06, EARLY_S), 1),
        "RippleSd": fmt(_ripple_sd(up10), 1),
        "StartDuty": fmt(cs.e1_start, 2),
        "StopDuty": fmt(cs.e1_stop, 2),
        "NegStartDuty": fmt(cs.e1_neg_start, 2),
        "NegStopDuty": fmt(cs.e1_neg_stop, 2),
        "SpeedAtTwo": fmt_pm(w02, sd02),
        "SpeedAtStart": fmt_pm(w_start, sd_start),
        "PredTwo": fmt(up06.fit.gain * 0.2, 1),
        "GainDrop": fmt(100 * (1 - up10.fit_delay.gain / up06.fit.gain), 0),
        "PredOne": fmt(up06.fit.gain * 1.0, 1),
        "MeasOne": fmt(up10.info.final, 1),
        "TauRatio": fmt(up06.fit.tau_s / up10.fit_delay.tau_s, 1),
        # The article gives the dead-time fit of up10, so its time constant
        # is the one every number about that step uses.
        "UpTenSettleModel": fmt(
            up10.fit_delay.tau_s * math.log(1 / SETTLING_BAND) * MS, 0
        ),
        "DownUpRatio": fmt(down06.fit.tau_s / up10.fit_delay.tau_s, 1),
        "StartStopRatio": fmt(up06.fit.tau_s / down0.fit.tau_s, 1),
        "NTrials": str(len(rows)),
        "NTrialsLed": str(sum(r["sync"] == "log+led" for r in rows)),
        "DroppedAll": fmt(_dropped(cs, list(cs.trials)), 1),
        # E1 is the forward and the backward sweep, both reported.
        "DroppedEOne": fmt(_dropped(cs, [cs.e1.trial, cs.e1_neg_trial]), 1),
        "DroppedETwo": fmt(_dropped(cs, cs.e2_trials), 1),
        "DroppedETwoMax": fmt(max(_dropped(cs, [t]) for t in cs.e2_trials), 0),
        "TimingSd": fmt(cs.trials[cs.e2_trials[0]]["timing_sd_s"] * MS, 1),
        "KETwo": fmt(cs.meta_e2["k"], 2),
        "KEOne": fmt(cs.meta_e1["k"], 2),
        "CallMs": fmt(cs.call_s * MS, 1),
        "NMotors": str(len(cs.speeds_06)),
        "SpeedSixMin": fmt(min(cs.speeds_06.values()), 1),
        "SpeedSixMax": fmt(max(cs.speeds_06.values()), 1),
        "OmegaSd": fmt(
            float(np.nanmedian(np.concatenate([r.omega_sd for r in up10.reps]))), 2
        ),
    }
    return m
