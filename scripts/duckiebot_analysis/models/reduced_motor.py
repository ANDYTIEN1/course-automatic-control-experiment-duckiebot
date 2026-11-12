# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Reduced wheel-speed models of a PWM-driven gearmotor, for identification.

Measuring only the wheel speed cannot separate the parameters of the full
model in duckiebot_motor.py, so these models keep the identifiable groups.
With the back EMF x = omega / omega0 in units of the supply voltage, where
omega0 = V_s / (K N) is the wheel speed at which the back EMF equals V_s,

    T dx/dt = j(u, x) - F(omega),

where j is the period-mean armature current in units of V_s / R and F the
friction torque in the same units:

    F = sgn(omega) (f0 + f1 |x| + f2 x^2) while turning,

with Karnopp sticking at rest until |j(u, 0)| exceeds the static friction s0.
The drive is either

* ``avg``: the averaged PWM of the Nordling 2026 model, j = u - x, whose
  static curve is affine above the dead zone; or
* ``fast``: fast-decay PWM (fast_decay_pwm.mean_current), in which the
  current stops within each period at light load.

omega0 and f0, s0 may differ between the directions, as the measured motors
do; omega0 is taken from the sign of the duty, f0 and s0 from the sign of
the speed. f1 and f2 act on x = omega / mean(omega0) so that they do not jump
when the duty changes sign.

Four optional extensions, each zero by default, test where FAST fails:

* ``mu``: friction mu |j| proportional to the current, the mu K |i| of the
  full model, added to F while turning. At rest the wheel starts when
  (1 - mu) |j| exceeds s0, so s0 is the static friction at zero current.
* ``r_on``: on-resistance of the bridge in units of R
  (fast_decay_pwm.mean_current); fast drive only.
* ``kappa``: a supply V_s (1 - kappa |j|) that sags with the mean current;
  x and j stay in units of the unloaded supply.
* ``tau_j``: a time scale T (1 + tau_j |j|) that depends on the current.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from duckiebot_analysis.models.fast_decay_pwm import mean_current

FloatArray = npt.NDArray[np.float64]
DRIVES = ("avg", "fast")
# Grid of the fast-decay current table; the step bounds the interpolation
# error of the current to about 1e-3 of V_s / R, largest near |u| = 1.
GRID_N = 801
# Damped fixed-point steps that solve for the current under a sagging supply;
# each shrinks the error by kappa / (1 + kappa) or less.
SAG_STEPS = 12
# Floor on 1 + tau_j |j|, so that a negative tau_j cannot make T vanish.
MIN_T_FACTOR = 0.05


@dataclass(frozen=True)
class MotorParams:
    """Parameters of a reduced model; see the module docstring."""

    drive: str
    omega0_pos: float
    omega0_neg: float
    f0_pos: float
    f0_neg: float
    s0_pos: float
    s0_neg: float
    t_mech: float
    f1: float = 0.0
    f2: float = 0.0
    rho: float = 1.5
    delay: float = 0.0
    mu: float = 0.0
    r_on: float = 0.0
    kappa: float = 0.0
    tau_j: float = 0.0

    def __post_init__(self) -> None:
        if self.drive not in DRIVES:
            raise ValueError(f"drive must be one of {DRIVES}, got {self.drive!r}")
        if not self.t_mech > 0 or not self.rho > 0:
            raise ValueError("t_mech and rho must be positive")
        if min(self.omega0_pos, self.omega0_neg) <= 0:
            raise ValueError("omega0 must be positive")
        if self.s0_pos < self.f0_pos or self.s0_neg < self.f0_neg:
            raise ValueError("static friction s0 must not be below f0")
        if not (0 <= self.mu < 1 and 0 <= self.kappa < 1 and self.r_on >= 0):
            raise ValueError("need 0 <= mu < 1, 0 <= kappa < 1 and r_on >= 0")
        if not self.tau_j > -1:
            raise ValueError("tau_j must exceed -1")
        if self.r_on and self.drive != "fast":
            raise ValueError("the on-resistance needs the fast drive")

    @property
    def omega0_ref(self) -> float:
        """Mean of the two speed scales, used by the speed-dependent friction."""
        return 0.5 * (self.omega0_pos + self.omega0_neg)


def _sagged(current, u: FloatArray, x: FloatArray, kappa: float) -> FloatArray:
    # With the supply at v = 1 - kappa |j| of V_s, the back EMF is x / v of the
    # supply and the current v current(u, x / v) in units of V_s / R.
    j = current(u, x)
    w = 1 / (1 + kappa)
    for _ in range(SAG_STEPS):
        # The floor keeps the supply positive where |j| nears 1 / kappa, as in
        # a reversal, where |j| reaches about 2.
        v = np.maximum(1 - kappa * np.abs(j), 0.1)
        # No clip of x / v: the fast-decay lookup clips its own inputs, and
        # the averaged current u - x / v holds for any x, as in a reversal.
        j = (1 - w) * j + w * v * current(u, x / v)
    return j


# A table is GRID_N^2 float64, about 5 MB; 32 hold the cold and hot tables
# of the rho, r_on and kappa values of one discrimination run.
@lru_cache(maxsize=32)
def _current_table(rho: float, r_on: float = 0.0, kappa: float = 0.0) -> FloatArray:
    grid = np.linspace(-1, 1, GRID_N)
    uu, xx = np.meshgrid(grid, grid, indexing="ij")
    # The u = 0 row holds the limit u -> 0 of conduction (no current, as the
    # open bridge), so that interpolation does not blend the brake into the
    # small duties; drive_current applies the brake at exactly u = 0.
    base = mean_current(uu, xx, rho, r_on, zero_duty="coast")
    if not kappa:
        return base
    return _sagged(lambda u, x: _interpolate(base, u, x), uu, xx, kappa)


def drive_current(p: MotorParams, u: FloatArray, x: FloatArray) -> FloatArray:
    """Return the period-mean current j(u, x) of the model's drive."""
    if p.drive == "avg":
        if p.kappa:
            # As in the fast drive, the short circuit at u = 0 escapes the sag.
            return np.where(u == 0, -x, _sagged(lambda u, x: u - x, u, x, p.kappa))
        return u - x
    j = _interpolate(_current_table(p.rho, p.r_on, p.kappa), u, x)
    # Throttle 0 shorts the motor (data/schema.md §10.1); the short circuit
    # does not draw from the supply, so its sag does not act.
    return np.where(u == 0, -np.clip(x, -1, 1) / (1 + p.r_on), j)


def _interpolate(table: FloatArray, u: FloatArray, x: FloatArray) -> FloatArray:
    h = 2 / (GRID_N - 1)
    fu = (np.clip(u, -1, 1) + 1) / h
    fx = (np.clip(x, -1, 1) + 1) / h
    iu = np.minimum(fu.astype(int), GRID_N - 2)
    ix = np.minimum(fx.astype(int), GRID_N - 2)
    au, ax = fu - iu, fx - ix
    return (
        table[iu, ix] * (1 - au) * (1 - ax)
        + table[iu + 1, ix] * au * (1 - ax)
        + table[iu, ix + 1] * (1 - au) * ax
        + table[iu + 1, ix + 1] * au * ax
    )


def _moving_friction(p: MotorParams, w: FloatArray, j: FloatArray) -> FloatArray:
    xr = np.abs(w) / p.omega0_ref
    f0 = np.where(w >= 0, p.f0_pos, p.f0_neg)
    return np.sign(w) * (f0 + p.f1 * xr + p.f2 * xr**2 + p.mu * np.abs(j))


def _omega0(p: MotorParams, u: FloatArray) -> FloatArray:
    return np.where(u >= 0, p.omega0_pos, p.omega0_neg)


def steady_speed(p: MotorParams, u: npt.ArrayLike) -> FloatArray:
    """Return the steady wheel speed reached from motion at duty u."""
    u = np.asarray(u, dtype=float)
    s = np.where(u >= 0, 1.0, -1.0)
    om0 = _omega0(p, u)
    lo, hi = np.zeros_like(u), np.ones_like(u)

    def excess(x: FloatArray) -> FloatArray:
        w = s * x * om0
        j = drive_current(p, u, s * x)
        return s * (j - _moving_friction(p, w + s * 1e-12, j))

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = excess(mid) > 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    x = 0.5 * (lo + hi)
    return np.where(excess(np.zeros_like(u)) > 0, s * x * om0, 0.0)


def _threshold(p: MotorParams, sign: int, level: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        j = sign * drive_current(p, np.array([sign * mid]), np.zeros(1))[0]
        # The current friction mu |j| takes its share of the drive at rest.
        lo, hi = (lo, mid) if (1 - p.mu) * j > level else (mid, hi)
    return 0.5 * (lo + hi)


def start_duty(p: MotorParams, sign: int) -> float:
    """Return the smallest |u| that starts the wheel from rest in direction sign."""
    return _threshold(p, sign, p.s0_pos if sign > 0 else p.s0_neg)


def stop_duty(p: MotorParams, sign: int) -> float:
    """Return the |u| below which a turning wheel stops."""
    return _threshold(p, sign, p.f0_pos if sign > 0 else p.f0_neg)


def sample_input(
    t_frames: FloatArray, u_frames: FloatArray, t_grid: FloatArray, delay: float
) -> FloatArray:
    """Return the duty applied at t_grid, the frame duty held and delayed."""
    k = np.searchsorted(t_frames, t_grid - delay, side="right") - 1
    return np.where(k >= 0, u_frames[np.clip(k, 0, None)], 0.0)


def simulate(
    p: MotorParams, u: FloatArray, dt: float, omega_init: FloatArray
) -> FloatArray:
    """Integrate the model for a batch of inputs.

    Args:
        p: Model parameters; the delay is applied by sample_input, not here.
        u: Duty, shape (trials, steps), held over each step of length dt.
        dt: Step in s.
        omega_init: Wheel speed at the start of each trial in rad/s.

    Returns:
        Wheel speed in rad/s, shape (trials, steps + 1).
    """
    u = np.asarray(u, dtype=float)
    out = np.empty((u.shape[0], u.shape[1] + 1))
    w = np.asarray(omega_init, dtype=float).copy()
    out[:, 0] = w
    s0 = (p.s0_pos, p.s0_neg)
    f0 = (p.f0_pos, p.f0_neg)
    for k in range(u.shape[1]):
        uk = u[:, k]
        om0 = _omega0(p, uk)
        j = drive_current(p, uk, w / om0)
        j0 = drive_current(p, uk, np.zeros_like(uk))
        rest = w == 0
        starts = (1 - p.mu) * np.abs(j0) > np.where(j0 >= 0, *s0)
        f_rest = np.sign(j0) * (np.where(j0 >= 0, *f0) + p.mu * np.abs(j0))
        friction = np.where(rest, f_rest, _moving_friction(p, w, j))
        t_eff = p.t_mech * np.maximum(1 + p.tau_j * np.abs(j), MIN_T_FACTOR)
        w_new = w + dt * om0 / t_eff * (j - friction)
        crossed = ~rest & (np.sign(w_new) != np.sign(w))
        w = np.where((rest & ~starts) | crossed, 0.0, w_new)
        out[:, k + 1] = w
    return out
