# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Figures and number macros of the CSME 2026 case study of motor m5.

Draws the four figures of
the CSME 2026 article from the results
of analysis.case_study, one CSME column wide, and writes the macro file that
holds every number of the text. Each step response is shown as angular
velocity against time from the logged command, with the rise time, overshoot
and settling time marked where the response has them (Franklin et al., 2019,
§3.4).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from duckiebot_analysis.analysis.case_study import (  # noqa: E402
    CaseStudy,
    StepResult,
)
from duckiebot_analysis.analysis.case_study_macros import (  # noqa: E402
    case_study_macros,
    write_macros,
)
from duckiebot_analysis.analysis.step_response import (  # noqa: E402
    SETTLING_BAND,
    StepInfo,
    first_order_step,
    mean_ci95,
)

COLUMN_IN = 8.1 / 2.54  # CSME column: (21 - 2 - 2 - 0.8) cm / 2
# Categorical slots 1-3 of the dataviz reference palette, validated for
# colour-vision deficiency; the marks are in the secondary ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
OMEGA_LABEL = r"$\omega$ (rad/s)"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _interp(s: StepResult, t: float, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    return float(np.interp(t, s.grid[ok], y[ok]))


def mark_step(
    ax: Axes,
    s: StepResult,
    info: StepInfo,
    y: np.ndarray,
    sign: int = 1,
    bar_frac: float = 0.1,
    offset: float = 0.0,
) -> None:
    """Mark rise time, overshoot and settling time of a response on ax.

    Call after setting the x limits; a mark beyond them is left out. y is the
    plotted response on s.grid, offset + sign times the response of info;
    sign = -1 for a mirrored step down. The rise time is drawn as an arrow at
    bar_frac of the change.
    """
    if not info.moved:
        return
    x0, x1 = ax.get_xlim()
    lo, hi = info.t_rise_start_s, info.t_rise_end_s
    y_lo, y_hi = _interp(s, lo, y), _interp(s, hi, y)
    ax.plot([lo, hi], [y_lo, y_hi], "o", ms=3, mfc="white", mec=INK, mew=0.7, zorder=5)
    y_bar = y[np.isfinite(y)][0] + bar_frac * abs(info.final - info.initial)
    for t_k, y_k in ((lo, y_lo), (hi, y_hi)):
        ax.vlines(
            t_k, min(y_bar, y_k), max(y_bar, y_k), color=INK2, lw=0.5, ls=(0, (2, 2))
        )
    ax.annotate(
        "",
        xy=(hi, y_bar),
        xytext=(lo, y_bar),
        arrowprops={"arrowstyle": "<->", "color": INK2, "lw": 0.6, "shrinkA": 0},
    )
    ax.text(hi + 0.01 * (x1 - x0), y_bar, r"$t_r$", va="center", color=INK)
    # The band step_info used, about its final value, not about the first
    # plotted sample, which carries that sample's noise.
    final = offset + sign * info.final
    if math.isfinite(info.overshoot) and info.peak_time_s < x1:
        yp = _interp(s, info.peak_time_s, y)
        ax.plot(info.peak_time_s, yp, "v", ms=3.5, color=INK, zorder=5)
        ax.text(info.peak_time_s, yp, r" $M_p$", va="bottom", color=INK)
    if math.isfinite(info.settling_time_s) and info.settling_time_s < x1:
        band = SETTLING_BAND * abs(info.final - info.initial)
        for v in (final - band, final + band):
            ax.hlines(v, info.settling_time_s, x1, color=INK2, lw=0.5)
        ax.vlines(
            info.settling_time_s, final - 6 * band, final + band, color=INK2, lw=0.6
        )
        ax.text(
            info.settling_time_s,
            final - 6 * band,
            r" $t_s$",
            va="top",
            ha="left",
            color=INK,
        )


def _reps(ax: Axes, s: StepResult, label: str) -> None:
    for k, r in enumerate(s.reps):
        ax.plot(
            r.t,
            r.omega,
            ".",
            ms=1.2,
            color=BLUE,
            alpha=0.45,
            mew=0,
            label=label if k == 0 else None,
            rasterized=True,
        )


def fig_small(cs: CaseStudy, path: Path) -> None:
    """Duty and wheel speed against time for E1."""
    e1 = cs.e1
    fig, (au, aw) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(COLUMN_IN, 2.4),
        gridspec_kw={"height_ratios": [1, 2]},
    )
    au.plot(e1.t, e1.u_cmd, color=INK2, lw=0.8, drawstyle="steps-post")
    au.set_ylabel("Duty $u$")
    aw.plot(e1.t, e1.omega, color=BLUE, lw=0.4, rasterized=True)
    aw.set_ylabel(OMEGA_LABEL)
    aw.set_xlabel("Time (s)")
    two = next(lv for lv in cs.e1_levels if lv.rising and math.isclose(lv.duty, 0.2))
    for ax in (au, aw):
        ax.axvspan(two.t_start_s, two.t_end_s, color=AQUA, alpha=0.2, lw=0)
    aw.text(two.t_end_s + 0.5, 9.5, "$u$ = 0.20", color=INK, ha="left", va="top")
    start = next(lv for lv in cs.e1_levels if lv.rising and lv.duty == cs.e1_start)
    stop = next(lv for lv in cs.e1_levels if not lv.rising and lv.duty == cs.e1_stop)
    for lv, text, ha in ((start, "start", "right"), (stop, "stop", "left")):
        aw.annotate(
            f"{text} at {abs(lv.duty):.2f}",
            xy=(lv.t_start_s, 0.5),
            xytext=(lv.t_start_s + (-2 if ha == "right" else 2), 6.5),
            ha=ha,
            arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.6},
        )
    au.set_ylim(0, 0.45)
    aw.set_xlim(0, 60)
    aw.set_ylim(-0.8, 10.5)
    fig.align_ylabels()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def fig_step(
    s: StepResult,
    path: Path,
    *,
    delay: bool,
    x_max: float,
    compare: tuple[StepResult, str] | None = None,
) -> None:
    """Measured step response with its fitted model and the residuals."""
    fit = s.fit_delay if delay else s.fit
    fig, (ax, ar) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(COLUMN_IN, 2.7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    _reps(ax, s, f"Measured, {len(s.reps)} repetitions")
    t = s.grid[s.grid <= x_max]
    w0 = float(np.mean(fit.omega0))
    model = "First-order model" + (" with dead time" if delay else "")
    ax.plot(t, fit.predict(t, w0), color=ORANGE, lw=1.1, label=model)
    if compare is not None:
        other, label = compare
        f = other.fit
        ax.plot(
            t,
            first_order_step(t, w0, f.gain, s.u_to - s.u_from, f.tau_s, f.delay_s),
            color=AQUA,
            lw=1.1,
            ls=(0, (4, 2)),
            label=label,
        )
    ax.set_xlim(-0.05, x_max)
    mark_step(ax, s, s.info, s.smooth)
    ax.set_ylabel(OMEGA_LABEL)
    ax.legend(loc="lower right", handlelength=1.6, markerscale=5)
    for k, r in enumerate(s.reps):
        ar.plot(
            r.t,
            r.omega - fit.predict(r.t, fit.omega0[k]),
            ".",
            ms=1.0,
            color=BLUE,
            alpha=0.45,
            mew=0,
            rasterized=True,
        )
    ar.axhline(0, color=ORANGE, lw=0.8)
    ar.set_ylim(-1.5, 1.5)
    ar.set_ylabel("Residual\n(rad/s)")
    ar.set_xlabel("Time from the command (s)")
    fig.align_ylabels()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def fig_hysteresis(cs: CaseStudy, path: Path) -> None:
    """(a) Static characteristic raised and lowered; (b) step up and down."""
    fig, (aa, ab) = plt.subplots(2, 1, figsize=(COLUMN_IN, 4.3))
    for rising, color, marker, label in (
        (True, BLUE, "o", "Raised"),
        (False, ORANGE, "s", "Lowered"),
    ):
        lv = [x for x in cs.e1_levels if x.rising == rising]
        aa.errorbar(
            [x.duty for x in lv],
            [x.mean for x in lv],
            yerr=[mean_ci95(x) for x in lv],
            color=color,
            marker=marker,
            ms=3.5,
            mfc=color if rising else "white",
            lw=0.8,
            elinewidth=0.6,
            capsize=1.5,
            label=label,
        )
    u_line = np.array([0.08, 0.37])
    aa.plot(
        u_line,
        cs.steps["up06"].fit.gain * u_line,
        color=INK2,
        lw=0.8,
        ls=(0, (4, 2)),
        label=r"$\omega = G u$, $G$ from the step to 0.6",
    )
    marks = (
        (cs.e1_start, True, "start", (0.375, 3.5)),
        (cs.e1_stop, False, "stop", (0.16, 3.0)),
    )
    for duty, rising, text, xy_text in marks:
        level = next(
            x for x in cs.e1_levels if x.rising == rising and math.isclose(x.duty, duty)
        )
        aa.annotate(
            f"{text}\n{duty:.2f}",
            xy=(duty, level.mean),
            xytext=xy_text,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.6},
        )
    aa.set_xlabel("Duty $u$")
    aa.set_ylabel(r"Steady $\omega$ (rad/s)")
    aa.set_xlim(0.08, 0.40)
    aa.set_ylim(-0.6, 11.5)
    aa.legend(loc="upper left")
    aa.set_title("(a)", loc="left", fontsize=8)
    up, down = cs.steps["up06"], cs.steps["down0"]
    t_max = 1.2
    ab.plot(up.grid, up.smooth, color=BLUE, label="0 to 0.6")
    mirrored = down.info.initial - down.smooth
    ab.plot(down.grid, mirrored, color=ORANGE, label="0.6 to 0, mirrored")
    w0_up = float(np.mean(up.fit.omega0))
    w0_down = float(np.mean(down.fit_delay.omega0))
    ab.plot(
        up.grid,
        up.fit.predict(up.grid, w0_up) - w0_up,
        color=INK,
        lw=0.7,
        ls=(0, (4, 2)),
        label="Fitted models",
    )
    ab.plot(
        down.grid,
        w0_down - down.fit_delay.predict(down.grid, w0_down),
        color=INK,
        lw=0.7,
        ls=(0, (4, 2)),
    )
    ab.set_xlim(-0.05, t_max)
    mark_step(ab, up, up.info, up.smooth, bar_frac=0.05)
    mark_step(
        ab, down, down.info, mirrored, sign=-1, bar_frac=0.35, offset=down.info.initial
    )
    ab.set_xlabel("Time from the command (s)")
    ab.set_ylabel(r"$|\omega - \omega(0)|$ (rad/s)")
    ab.legend(loc="lower right")
    ab.set_title("(b)", loc="left", fontsize=8)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def make_all(cs: CaseStudy, figures: Path, macros: Path, command: str) -> list[Path]:
    """Write the four figures into `figures` and the macro file to `macros`.

    The figures go under results/figures/, which the article's \\graphicspath
    names, because doc/ ignores PDFs as build output; the macro file is
    \\input from the article folder.

    Returns:
        The paths written.
    """
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "small": figures / "fig-small-duty.pdf",
        "step06": figures / "fig-step-06.pdf",
        "step10": figures / "fig-step-10.pdf",
        "hyst": figures / "fig-hysteresis.pdf",
        "macros": macros,
    }
    fig_small(cs, paths["small"])
    fig_step(cs.steps["up06"], paths["step06"], delay=False, x_max=1.5)
    fig_step(
        cs.steps["up10"],
        paths["step10"],
        delay=True,
        x_max=2.0,
        compare=(cs.steps["up06"], "Model from the step to 0.6"),
    )
    fig_hysteresis(cs, paths["hyst"])
    write_macros(paths["macros"], case_study_macros(cs), command)
    return list(paths.values())
