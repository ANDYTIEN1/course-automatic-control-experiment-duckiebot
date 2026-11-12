# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Schematic figure of S3-S5 of the CSME 2026 supplement.

Illustrates, with the symbols of
the CSME 2026 supplement,
(a) the once-per-turn correction and the factor k (S3), (b) the speed as the
slope of a local quadratic (S4), (c) the time of an LED edge and of a call
logged by the Pi (S5), and (d) the clock fit t = a + (1 + delta) tau (S5).
The example data are synthetic, so the figure shows the method, not a
measurement, but they pass through the pipeline's own functions
(harmonics.harmonic_correct, derivatives.local_quadratic, sync.fit_clock),
so every fitted curve is what the code computes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from duckiebot_analysis.analysis.case_study import DT  # noqa: E402
from duckiebot_analysis.analysis.derivatives import (  # noqa: E402
    W_OMEGA,
    local_quadratic,
)
from duckiebot_analysis.analysis.harmonics import (  # noqa: E402
    TrialWindows,
    harmonic_correct,
)
from duckiebot_analysis.analysis.sync import ClockFit, fit_clock  # noqa: E402
from duckiebot_analysis.visualization.case_study_figures import (  # noqa: E402
    AQUA,
    BLUE,
    INK,
    INK2,
    ORANGE,
)

PAGE_IN = 17.0 / 2.54  # CSME text width: 21 cm less 2 cm margins
SEED = 20261031
# (a) Two steady parts at constant speed, a once-per-turn error a1, b1, a2,
# b2, a3, b3 (rad), and noise NOISE_FACTOR times its stated sd, so k > 1.
SPEEDS = (20.0, 28.0)  # rad/s, about duty 0.6 and 1.0
TRUE_COEF = np.array([0.030, -0.020, 0.010, 0.006, 0.0, 0.003])
THETA_SD = 0.010  # rad, stated
NOISE_FACTOR = 1.3
# (b) An accelerating wheel, theta = OMEGA0 t + ALPHA t^2 / 2, with one
# unusable frame inside the window. The noise is exaggerated to be visible.
OMEGA0, ALPHA = 10.0, 400.0  # rad/s, rad/s^2
T_I = 0.05  # s
THETA_SD_B = 0.015  # rad
# (c, d) One LED edge with a dropped frame after it, and K edges over a run.
CALL_S = 2.4e-3  # duration of a logged call (CallMs of results.tex)
CLOCK_A, CLOCK_DELTA = 3.864, 3.0e-5  # s, relative drift
N_EDGES, SPAN_S = 12, 72.0
USED_LABEL = r"$\theta^c_l \pm \sigma_{\theta^c,l}$, used"


@dataclass(frozen=True)
class SpeedExample:
    """Frames around t_i and the local quadratic fitted at t_i (S4)."""

    t: np.ndarray
    theta: np.ndarray
    theta_sd: np.ndarray
    usable: np.ndarray
    t_i: float
    beta: np.ndarray
    omega: float
    omega_sd: float


@dataclass(frozen=True)
class EdgeExample:
    """One LED edge on the video clock and the call that switched it (S5)."""

    frames: np.ndarray  # times of the recorded frames, s
    led_on: np.ndarray  # LED state in each recorded frame
    t_last_old: float
    t_first_new: float
    t_k: float
    dt_k: float
    tau: float  # call start, on the video clock
    tau_after: float  # call end


@dataclass(frozen=True)
class ClockExample:
    """K edges and the clock fit through them (S5)."""

    tau_bar: np.ndarray
    t_video: np.ndarray
    s_k: np.ndarray
    fit: ClockFit


@dataclass(frozen=True)
class MethodsExample:
    """Everything the figure draws."""

    theta_mod: np.ndarray  # wrapped angle of the steady frames
    resid: np.ndarray  # theta - theta_0j - omega_j t, rad
    part: np.ndarray  # steady part of each frame, 0 or 1
    theta_grid: np.ndarray
    h_grid: np.ndarray  # fitted h(theta), rad
    k: float
    speed: SpeedExample
    edge: EdgeExample
    clock: ClockExample


def harmonic_sum(theta: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Return h(theta) = sum_m a_m cos(m theta) + b_m sin(m theta)."""
    m = np.arange(1, coef.size // 2 + 1)
    mt = np.asarray(theta)[..., None] * m
    return np.asarray(np.cos(mt) @ coef[0::2] + np.sin(mt) @ coef[1::2], float)


def _harmonics_example(rng: np.random.Generator) -> dict:
    t = np.arange(0.0, 6.0, DT)
    true = np.where(t < 3.0, SPEEDS[0] * t, SPEEDS[0] * 3.0 + SPEEDS[1] * (t - 3.0))
    theta = true + harmonic_sum(true, TRUE_COEF)
    theta += rng.normal(0.0, NOISE_FACTOR * THETA_SD, t.size)
    sd = np.full_like(t, THETA_SD)
    steady = ((1.2, 3.0 - DT), (4.2, 6.0))
    res = harmonic_correct(
        t, theta, sd, np.zeros(t.size, np.int64), [TrialWindows((0.0, 6.0), steady)]
    )
    coef = res.trials[0].coef
    thetas, resid, part = [], [], []
    for j, (a, b) in enumerate(steady):
        s = (t >= a) & (t <= b)
        line = np.polyval(np.polyfit(t[s], res.theta[s], 1), t[s])
        thetas.append(theta[s])
        resid.append(theta[s] - line)
        part.append(np.full(s.sum(), j))
    grid = np.linspace(0.0, 2 * np.pi, 361)
    return {
        "theta_mod": np.mod(np.concatenate(thetas), 2 * np.pi),
        "resid": np.concatenate(resid),
        "part": np.concatenate(part),
        "theta_grid": grid,
        "h_grid": harmonic_sum(grid, coef),
        "k": float(res.k),
    }


def _speed_example(rng: np.random.Generator) -> SpeedExample:
    n = round(W_OMEGA / DT) + 3
    t = T_I + DT * np.arange(-n, n + 1)
    theta = OMEGA0 * t + 0.5 * ALPHA * t**2 + rng.normal(0.0, THETA_SD_B, t.size)
    sd = np.full_like(t, THETA_SD_B)
    usable = np.ones(t.size, bool)
    usable[n + 3] = False
    fit = local_quadratic(t, theta, sd, W_OMEGA, usable=usable)
    return SpeedExample(
        t,
        theta,
        sd,
        usable,
        float(t[n]),
        fit.beta[n],
        fit.beta[n, 1],
        fit.beta_sd[n, 1],
    )


def _edge_example() -> EdgeExample:
    frames = DT * np.arange(-5, 7) + 0.4 * DT
    switch = 0.0  # the LED changes at the middle of the call
    keep = np.ones(frames.size, bool)
    first_new = int(np.searchsorted(frames, switch))
    keep[first_new] = False  # dropped frame
    frames = frames[keep]
    led_on = frames > switch
    last_old = float(frames[~led_on][-1])
    first = float(frames[led_on][0])
    return EdgeExample(
        frames,
        led_on,
        last_old,
        first,
        (last_old + first) / 2,
        first - last_old,
        switch - CALL_S / 2,
        switch + CALL_S / 2,
    )


def _clock_example(rng: np.random.Generator) -> ClockExample:
    tau_bar = np.sort(rng.uniform(0.0, SPAN_S, N_EDGES))
    dt_k = np.where(rng.random(N_EDGES) < 0.25, 2 * DT, DT)
    sd = dt_k / np.sqrt(12)
    t_video = CLOCK_A + (1 + CLOCK_DELTA) * tau_bar + rng.normal(0.0, sd)
    fit = fit_clock(
        tau_bar - CALL_S / 2,
        t_video,
        sd,
        tau_after_s=tau_bar + CALL_S / 2,
        frame_interval_s=DT,
    )
    s_k = np.sqrt((dt_k**2 + CALL_S**2) / 12)
    return ClockExample(tau_bar, t_video, s_k, fit)


def methods_example(seed: int = SEED) -> MethodsExample:
    """Generate the synthetic example and process it with the pipeline."""
    rng = np.random.default_rng(seed)
    return MethodsExample(
        **_harmonics_example(rng),
        speed=_speed_example(rng),
        edge=_edge_example(),
        clock=_clock_example(rng),
    )


def _panel_harmonics(ax: Axes, ex: MethodsExample) -> None:
    for j, colour in enumerate((BLUE, ORANGE)):
        s = (ex.part == j) & (np.arange(ex.part.size) % 3 == 0)
        ax.plot(
            ex.theta_mod[s],
            1e3 * ex.resid[s],
            ".",
            ms=1.5,
            color=colour,
            label=rf"steady part $j={j + 1}$",
        )
    ax.plot(
        ex.theta_grid, 1e3 * ex.h_grid, color=INK, lw=1.2, label=r"fitted $h(\theta)$"
    )
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(
        np.pi * np.arange(5) / 2, ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    )
    ax.set_xlabel(r"$\theta_i\ \mathrm{mod}\ 2\pi$ (rad)")
    ax.set_ylabel(r"$\theta_i - \theta_{0,j} - \omega_j t_i$ (mrad)")
    ax.set_ylim(-80, 135)
    ax.text(
        0.02,
        0.97,
        r"$\theta^c_i = \theta_i - h(\theta_i)$"
        "\n"
        rf"$k^2 = \max(1, \sum_j\chi^2_j / \sum_j\nu_j)$, $k = {ex.k:.2f}$",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
    )
    ax.legend(loc="upper right", markerscale=4, fontsize=6)
    ax.set_title("(a) S3: once-per-turn correction", loc="left")


def _panel_speed(ax: Axes, ex: MethodsExample) -> None:
    s = ex.speed
    inside = np.abs(s.t - s.t_i) <= W_OMEGA + 1e-12
    x = 1e3 * (s.t - s.t_i)
    used = inside & s.usable
    ax.axvspan(-1e3 * W_OMEGA, 1e3 * W_OMEGA, color=AQUA, alpha=0.12, lw=0)
    ax.errorbar(
        x[used],
        s.theta[used],
        s.theta_sd[used],
        fmt="o",
        ms=2.5,
        color=BLUE,
        elinewidth=0.6,
        label=USED_LABEL,
    )
    ax.plot(
        x[~used & s.usable],
        s.theta[~used & s.usable],
        "o",
        ms=2.5,
        mfc="none",
        color=INK2,
        label="outside the window",
    )
    ax.plot(
        x[~s.usable], s.theta[~s.usable], "x", ms=4, color=ORANGE, label="not usable"
    )
    dt = np.linspace(-W_OMEGA, W_OMEGA, 100)
    b0, b1, b2 = s.beta
    ax.plot(
        1e3 * dt,
        b0 + b1 * dt + b2 * dt**2,
        color=INK,
        lw=1.0,
        label=r"$\beta_0 + \beta_1(t-t_i) + \beta_2(t-t_i)^2$",
    )
    dt2 = np.array([-0.6, 0.6]) * W_OMEGA
    ax.plot(
        1e3 * dt2,
        b0 + b1 * dt2,
        "--",
        color=ORANGE,
        lw=1.0,
        label=(
            rf"slope $\omega_i = \beta_1 = {b1:.1f}"
            rf" \pm {s.omega_sd:.1f}$ rad/s"
        ),
    )
    ax.annotate(
        "",
        xy=(1e3 * W_OMEGA, 0.98),
        xytext=(0, 0.98),
        xycoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "<->", "color": INK2, "lw": 0.6},
    )
    ax.text(
        0.5e3 * W_OMEGA,
        0.95,
        r"$w = 23$ ms",
        ha="center",
        va="top",
        fontsize=7,
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xlabel(r"$t_l - t_i$ (ms)")
    ax.set_ylabel(r"$\theta^c$ (rad)")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_ylim(top=1.1 * float(np.max(s.theta)) + 0.6)
    ax.set_title("(b) S4: speed of frame $i$", loc="left")


def _panel_edge(ax: Axes, ex: MethodsExample) -> None:
    e = ex.edge
    ms = 1e3
    y_pi, y_vid = 1.0, 0.0
    ax.axhline(y_pi, color=INK2, lw=0.5)
    ax.axhline(y_vid, color=INK2, lw=0.5)
    ax.add_patch(
        plt.Rectangle(
            (ms * e.tau, y_pi - 0.08),
            ms * (e.tau_after - e.tau),
            0.16,
            color=BLUE,
            alpha=0.3,
            lw=0,
        )
    )
    tau_bar = (e.tau + e.tau_after) / 2
    for x, lab in ((e.tau, r"$\tau$"), (e.tau_after, r"$\tau'$")):
        ax.text(ms * x, y_pi + 0.12, lab, ha="center", fontsize=7)
    ax.plot(ms * tau_bar, y_pi, "|", color=BLUE, ms=10, mew=1.2)
    ax.text(
        ms * tau_bar,
        y_pi - 0.2,
        r"$\bar\tau_k$, var $(\tau'-\tau)^2/12$",
        ha="center",
        va="top",
        fontsize=7,
        color=BLUE,
    )
    for x, on in zip(e.frames, e.led_on, strict=True):
        ax.plot(
            ms * x,
            y_vid,
            "s",
            ms=5,
            color=ORANGE if on else INK2,
            mfc=ORANGE if on else "white",
        )
    missing = e.t_first_new - DT
    ax.plot(ms * missing, y_vid, "x", ms=5, color=INK2)
    ax.text(ms * missing, y_vid - 0.12, "dropped", ha="center", va="top", fontsize=6)
    ax.annotate(
        "",
        xy=(ms * e.t_first_new, y_vid + 0.22),
        xytext=(ms * e.t_last_old, y_vid + 0.22),
        arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.6},
    )
    ax.text(ms * e.t_k, y_vid + 0.27, r"$\Delta t_k$", ha="center", fontsize=7)
    ax.errorbar(
        ms * e.t_k,
        y_vid - 0.35,
        xerr=ms * e.dt_k / np.sqrt(12),
        fmt="|",
        color=ORANGE,
        ms=8,
        capsize=2,
        elinewidth=0.8,
    )
    ax.text(
        ms * e.t_k,
        y_vid - 0.45,
        r"$t_k \pm \Delta t_k/\sqrt{12}$",
        ha="center",
        va="top",
        fontsize=7,
        color=ORANGE,
    )
    ax.text(
        ms * e.frames[-1],
        0.5,
        r"$t_k = a + (1+\delta)\,\bar\tau_k$",
        ha="right",
        va="center",
        fontsize=7,
    )
    ax.set_yticks([y_vid, y_pi], ["video\nframes", "Pi log"])
    ax.set_ylim(-0.75, 1.35)
    ax.grid(False)
    ax.set_xlabel(r"time on the video clock, from the switch (ms)")
    ax.set_title("(c) S5: one LED edge and its call", loc="left")


def _panel_clock(ax: Axes, ex: MethodsExample) -> None:
    c = ex.clock
    f = c.fit
    y = 1e3 * (c.t_video - c.tau_bar - f.a)
    ax.errorbar(
        c.tau_bar,
        y,
        1e3 * c.s_k,
        fmt="o",
        ms=2.5,
        color=BLUE,
        elinewidth=0.6,
        label=r"edge $k$: $t_k \pm s_k$",
    )
    x = np.array([0.0, SPAN_S])
    ax.plot(
        x,
        1e3 * f.delta * x,
        color=INK,
        label=(
            r"fit: $a + \delta\,\bar\tau$, "
            rf"$\delta = {f.delta * 1e6:.0f}\times10^{{-6}}$"
        ),
    )
    i = int(np.argmax(np.abs(f.residuals_s)))
    ax.annotate(
        "",
        xy=(c.tau_bar[i], y[i]),
        xytext=(c.tau_bar[i], 1e3 * f.delta * c.tau_bar[i]),
        arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 0.8},
    )
    ax.text(
        c.tau_bar[i] + 1.5,
        (y[i] + 1e3 * f.delta * c.tau_bar[i]) / 2,
        r"$e_k$",
        color=ORANGE,
        fontsize=7,
        va="center",
    )
    ax.text(
        0.02,
        0.03,
        rf"$K = {f.n_edges}$, "
        rf"$\sigma_\tau = {1e3 * f.timing_sd_s:.1f}$ ms",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.set_xlabel(r"$\bar\tau_k$, Pi time from the run start (s)")
    ax.set_ylabel(r"$t_k - \bar\tau_k - a$ (ms)")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title("(d) S5: clock fit over a run", loc="left")


def draw_methods(ex: MethodsExample) -> Figure:
    """Draw the four panels and return the figure."""
    fig, axes = plt.subplots(
        2, 2, figsize=(PAGE_IN, 0.62 * PAGE_IN), layout="constrained"
    )
    _panel_harmonics(axes[0, 0], ex)
    _panel_speed(axes[0, 1], ex)
    _panel_edge(axes[1, 0], ex)
    _panel_clock(axes[1, 1], ex)
    return fig


def fig_methods(path: Path, ex: MethodsExample | None = None) -> None:
    """Write the schematic figure of S3-S5 to `path`."""
    fig = draw_methods(methods_example() if ex is None else ex)
    fig.savefig(path)
    plt.close(fig)
