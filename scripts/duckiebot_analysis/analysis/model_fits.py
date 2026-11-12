# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Fit the reduced wheel-speed models to static curves and step responses.

Three models share reduced_motor.MotorParams:

* ``AVG``: averaged PWM, the structure of the Nordling 2026 model; affine
  static curve, one time constant for every amplitude;
* ``QUAD``: AVG with a friction term f2 x^2, an empirical way to bend the
  static curve and shorten the time constant at high speed;
* ``FAST``: fast-decay PWM with viscous friction f1, the drive the DB17
  actually uses.

The static parameters come from steady speeds; the mechanical time scale and
the delay then come from step responses with the static parameters held. The
delay is fitted separately for LED-timed trials, where it is physical, and
for motion-timed trials, whose time origin was placed by the motion itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from duckiebot_analysis.models.reduced_motor import (
    MotorParams,
    drive_current,
    simulate,
    steady_speed,
)

FloatArray = npt.NDArray[np.float64]
DRIVE = {"AVG": "avg", "QUAD": "avg", "FAST": "fast"}
STATIC = {
    "AVG": ("omega0_pos", "omega0_neg", "f0_pos", "f0_neg"),
    "QUAD": ("omega0_pos", "omega0_neg", "f0_pos", "f0_neg", "f2"),
    "FAST": ("omega0_pos", "omega0_neg", "f0_pos", "f0_neg", "f1", "rho"),
}
DT = 2e-3


def make_params(model: str, values: dict) -> MotorParams:
    """Return MotorParams of a model, with s0 = f0 and T = 0.1 s by default."""
    v = dict(omega0_pos=30.0, omega0_neg=30.0, f0_pos=0.1, f0_neg=0.1, t_mech=0.1)
    v |= values
    v.setdefault("s0_pos", v["f0_pos"])
    v.setdefault("s0_neg", v["f0_neg"])
    return MotorParams(drive=DRIVE[model], **v)


def _start(model: str, u: FloatArray, w: FloatArray) -> list[dict]:
    top_pos = np.max(np.abs(w[u > 0]), initial=20.0)
    top_neg = np.max(np.abs(w[u < 0]), initial=20.0)
    if model == "FAST":
        return [
            dict(
                omega0_pos=1.05 * top_pos,
                omega0_neg=1.05 * top_neg,
                f0_pos=0.01,
                f0_neg=0.01,
                f1=0.02,
                rho=rho,
            )
            for rho in (0.5, 1.5, 4.0)
        ]
    base = dict(
        omega0_pos=1.3 * top_pos, omega0_neg=1.3 * top_neg, f0_pos=0.2, f0_neg=0.2
    )
    return [base | dict(f2=0.5)] if model == "QUAD" else [base]


def fit_static(
    model: str,
    u: npt.ArrayLike,
    omega: npt.ArrayLike,
    fixed: dict | None = None,
    start: dict | None = None,
) -> tuple[MotorParams, dict]:
    """Fit the static parameters of a model to steady speeds of a turning wheel.

    Args:
        model: ``AVG``, ``QUAD`` or ``FAST``.
        u: Duty of each steady speed.
        omega: Steady wheel speed in rad/s, nonzero.
        fixed: Parameters held during the fit, such as an extension r_on.
        start: The one starting point of the fitted parameters; by default
            several generic ones are tried.

    Returns:
        The fitted parameters and a dict with rms (rad/s), aic, n, k, the
        standard errors ``se`` and the correlation matrix ``corr``.
    """
    u = np.asarray(u, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if u.size == 0 or u.shape != omega.shape:
        raise ValueError("need equally long, non-empty u and omega")
    names = STATIC[model]
    logs = {"rho", "omega0_pos", "omega0_neg"}

    def unpack(z: FloatArray) -> dict:
        return {
            n: (np.exp(v) if n in logs else v) for n, v in zip(names, z, strict=True)
        }

    held = fixed or {}

    def resid(z: FloatArray) -> FloatArray:
        return steady_speed(make_params(model, unpack(z) | held), u) - omega

    lower = [-np.inf if n in logs else 0.0 for n in names]
    best = None
    for s in [start] if start else _start(model, u, omega):
        z0 = [np.log(s[n]) if n in logs else s[n] for n in names]
        r = least_squares(resid, z0, bounds=(lower, np.inf), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    p = make_params(model, unpack(best.x) | held)
    n, k = u.size, len(names)
    rss = float(np.sum(best.fun**2))
    info = {"rms": np.sqrt(rss / n), "n": n, "k": k, "aic": n * np.log(rss / n) + 2 * k}
    # Standard errors on the natural scale, by the delta method for the logs:
    # dr/dtheta = (dr/dz) / theta for z = log theta.
    jac = best.jac / np.array(
        [np.exp(v) if nm in logs else 1.0 for nm, v in zip(names, best.x, strict=True)]
    )
    try:
        cov = np.linalg.inv(jac.T @ jac) * rss / max(n - k, 1)
        se = np.sqrt(np.diag(cov))
        info["se"] = dict(zip(names, se, strict=True))
        info["corr"] = cov / np.outer(se, se)
    except np.linalg.LinAlgError:
        info["se"] = dict.fromkeys(names, np.nan)
        info["corr"] = np.full((k, k), np.nan)
    info["cond"] = float(np.linalg.cond(jac))
    return p, info


def with_supply(p: MotorParams, k: float) -> MotorParams:
    """Return p with the supply V_s scaled by k.

    The speed scale grows with V_s, and the friction in units of K V_s / R
    shrinks with it; f2 is per (ω/ω0)², so it grows with ω0.
    """
    if k == 1.0:
        return p
    return replace(
        p,
        omega0_pos=k * p.omega0_pos,
        omega0_neg=k * p.omega0_neg,
        f0_pos=p.f0_pos / k,
        f0_neg=p.f0_neg / k,
        s0_pos=p.s0_pos / k,
        s0_neg=p.s0_neg / k,
        f2=p.f2 * k,
    )


def floor_friction(
    p: MotorParams, supply: float, speed: float, duty: float = 0.6
) -> float:
    """Return the extra Coulomb friction that slows the wheel to speed.

    The wheel turns on the floor with both motors driven at duty, so the
    supply is the one found in the air with both wheels driven. The extra
    friction is added to f0 in units of K V_s / R at the nominal supply, as
    f0 is; it is the load torque T_L of the floor in these units. It may be
    negative down to -f0, where the data say the floor helps.

    Args:
        p: Parameters fitted in the air (A to D), at the nominal supply.
        supply: Factor on V_s with both wheels driven (E in the air).
        speed: Steady wheel speed on the floor in rad/s, positive.
        duty: Duty of the floor runs.

    Returns:
        The extra friction, NaN when no value in [-f0, 1] gives the speed.
    """
    if not (np.isfinite(supply) and supply > 0 and np.isfinite(speed) and speed > 0):
        return np.nan
    q = with_supply(p, supply)

    def model(extra: float) -> float:
        f0_pos = q.f0_pos + extra / supply
        f0_neg = q.f0_neg + extra / supply
        loaded = replace(
            q,
            f0_pos=f0_pos,
            f0_neg=f0_neg,
            s0_pos=max(q.s0_pos, f0_pos),
            s0_neg=max(q.s0_neg, f0_neg),
        )
        return float(steady_speed(loaded, np.array([duty]))[0])

    lo, hi = -p.f0_pos, 1.0
    if not model(hi) <= speed <= model(lo):
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if model(mid) > speed else (lo, mid)
    return 0.5 * (lo + hi)


def with_start_duties(
    p: MotorParams, start_pos: float, start_neg: float
) -> MotorParams:
    """Set the static friction so that the wheel starts at the measured duties.

    The static friction is the current at rest at the start duty, less the
    share mu |j| of the current friction; it is not allowed below the Coulomb
    friction, which a model needs to stop the wheel.
    """
    keep = 1 - p.mu
    j_pos = keep * float(drive_current(p, np.array([start_pos]), np.zeros(1))[0])
    j_neg = -keep * float(drive_current(p, np.array([-start_neg]), np.zeros(1))[0])
    return replace(p, s0_pos=max(j_pos, p.f0_pos), s0_neg=max(j_neg, p.f0_neg))


@dataclass
class Trial:
    """One measured response: speed at frame times and the applied duty."""

    t: FloatArray
    omega: FloatArray
    t_u: FloatArray
    u: FloatArray
    led: bool
    omega_init: float


def _input(tr: Trial, grid: FloatArray, delay: float) -> FloatArray:
    # Exact mean of the held, delayed duty over each step, from its running
    # integral, which is continuous in the delay. Before the first command the
    # duty is that command, as the trial starts steady at it.
    tb = np.r_[tr.t_u[0] - 1e3, tr.t_u, tr.t_u[-1] + 1e3] + delay
    ub = np.r_[tr.u[0], tr.u, tr.u[-1]]
    integral = np.r_[0.0, np.cumsum(ub[:-1] * np.diff(tb))]
    edges = np.r_[grid, grid[-1] + DT]
    return np.diff(np.interp(edges, tb, integral)) / DT


def simulate_trials(
    p: MotorParams, trials: list[Trial], delay_led: float, delay_motion: float
) -> list[FloatArray]:
    """Return the simulated speed of each trial at its frame times."""
    if not trials:
        return []
    t0 = min(tr.t[0] for tr in trials)
    t1 = max(tr.t[-1] for tr in trials)
    grid = np.arange(t0, t1 + DT, DT)
    u = np.stack(
        [_input(tr, grid, delay_led if tr.led else delay_motion) for tr in trials]
    )
    w = simulate(p, u, DT, np.array([tr.omega_init for tr in trials]))
    tg = np.r_[grid, grid[-1] + DT]
    return [np.interp(tr.t, tg, w[i]) for i, tr in enumerate(trials)]


def fit_dynamic(p: MotorParams, trials: list[Trial]) -> tuple[MotorParams, dict]:
    """Fit the time scale and the two delays with the static parameters held."""
    if not trials:
        raise ValueError("no trials to fit")

    def resid(z: FloatArray) -> FloatArray:
        q = replace(p, t_mech=float(np.exp(z[0])))
        sim = simulate_trials(q, trials, z[1], z[2])
        return np.concatenate([s - tr.omega for s, tr in zip(sim, trials, strict=True)])

    r = least_squares(
        resid,
        [np.log(p.t_mech), 0.005, 0.005],
        bounds=([np.log(0.005), -0.05, -0.1], [np.log(5.0), 0.1, 0.2]),
        diff_step=[1e-3, 0.02, 0.02],
    )
    q = replace(p, t_mech=float(np.exp(r.x[0])), delay=float(r.x[1]))
    return q, {
        "rms": float(np.sqrt(np.mean(r.fun**2))),
        "delay_led": float(r.x[1]),
        "delay_motion": float(r.x[2]),
        "n": r.fun.size,
    }
