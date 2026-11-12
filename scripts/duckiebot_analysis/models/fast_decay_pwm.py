# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Period-mean armature current of a DC motor driven by fast-decay PWM.

The DB17 drives its motors through adafruit_motorkit, whose DCMotor defaults
to fast decay: during the off part of each PWM period both TB6612 inputs are
low, the bridge is open, and the current returns to the supply through the
body diodes against -V_s until it reaches zero, where it stays while the back
EMF is below V_s. At light load the current therefore stops in every period
(discontinuous conduction), and the mean current is not the (u V_s - E) / R of
an averaged PWM.

A duty of exactly 0 is not the limit of small duties: adafruit_motor's
throttle 0 sets both TB6612 inputs high, which shorts the motor and brakes
it, i = -E / (R (1 + r_on)) (data/schema.md §10.1). Only throttle None opens
the bridge and lets the wheel coast; ``zero_duty="coast"`` models that.

Everything is normalised: the back EMF x = E / V_s, the current in units of
V_s / R, time in units of the electrical time constant L / R, so the PWM period
is rho = P R / L. The motor speed is taken constant over one period, which
holds because the mechanical time constant is about 100 times the period.

An optional on-resistance r_on R of the bridge acts in the on phase and the
brake; in the off phase the current passes through the body diodes. A resistance
in both phases would only be a larger R: substituting t' = (1 + r) t and
i' = (1 + r) i gives the same equations with rho (1 + r), so it is absorbed
by rho and the units of the current and cannot be told apart from speed data.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _segment(
    y0: FloatArray, y_inf: FloatArray, length: FloatArray, rate: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Return the end value and the integral of y' = rate (y_inf - y)."""
    decay = np.exp(-rate * length)
    integral = y_inf * length + (y0 - y_inf) * (1 - decay) / rate
    return y_inf + (y0 - y_inf) * decay, integral


def mean_current(
    u: npt.ArrayLike,
    x: npt.ArrayLike,
    rho: float,
    r_on: float = 0.0,
    zero_duty: str = "brake",
) -> FloatArray:
    """Return the period-mean current in the periodic steady state.

    Args:
        u: Duty cycle in [-1, 1]; its sign selects the bridge diagonal.
        x: Back EMF divided by the supply voltage, in [-1, 1].
        rho: PWM period divided by the electrical time constant L / R.
        r_on: On-resistance of the bridge in units of R, in the on phase and
            the brake.
        zero_duty: At u = 0, ``brake`` for throttle 0 (motor shorted) or
            ``coast`` for throttle None (bridge open).

    Returns:
        Mean current in units of V_s / R, with the shape of ``u``.

    Raises:
        ValueError: If the shapes differ, |u| > 1, |x| > 1, or rho is not
            positive and finite, r_on is negative, or zero_duty is unknown.
    """
    u = np.asarray(u, dtype=float)
    x = np.asarray(x, dtype=float)
    if u.shape != x.shape:
        raise ValueError(f"u and x differ in shape: {u.shape} and {x.shape}")
    if not (np.isfinite(rho) and rho > 0):
        raise ValueError(f"rho must be positive and finite, got {rho}")
    if not (np.isfinite(r_on) and r_on >= 0):
        raise ValueError(f"r_on must be non-negative and finite, got {r_on}")
    if np.any(np.abs(u) > 1) or np.any(np.abs(x) > 1):
        raise ValueError("|u| and |x| must not exceed 1")
    if zero_duty not in ("brake", "coast"):
        raise ValueError(f"zero_duty must be 'brake' or 'coast', got {zero_duty!r}")
    s = np.sign(u)
    t_on = np.abs(u) * rho
    t_off = rho - t_on
    # In the on phase, L di/dt = s V_s - E - R (1 + r_on) i.
    a_on = 1.0 + r_on
    y_on = (s - x) / a_on
    y_off = -s - x

    # Continuous conduction: one period is an affine map of the start current,
    # so its fixed point is explicit. It holds if the current keeps the sign
    # of the on-phase drive throughout the period.
    end_on_from_0, _ = _segment(np.zeros_like(u), y_on, t_on, a_on)
    end_from_0, _ = _segment(end_on_from_0, y_off, t_off)
    with np.errstate(invalid="ignore", divide="ignore"):
        # a_on t_on + t_off, written so that it is exactly rho at r_on = 0
        y0 = end_from_0 / (1 - np.exp(-(rho + r_on * t_on)))
    y1, int_on = _segment(y0, y_on, t_on, a_on)
    y2, int_off = _segment(y1, y_off, t_off)
    ccm = (s * y0 > 0) & (s * y1 > 0) & (s * y2 > 0)
    mean_ccm = (int_on + int_off) / rho

    # Discontinuous conduction: the current starts each period at zero and
    # decays to zero in the off phase, after which the open bridge holds it.
    y1d, int_on_d = _segment(np.zeros_like(u), y_on, t_on, a_on)
    v_off = -np.sign(y1d)
    y_inf = v_off - x
    with np.errstate(invalid="ignore", divide="ignore"):
        t_zero = np.where(y1d != 0, np.log((y1d - y_inf) / (-y_inf)), 0.0)
    t_cond = np.minimum(np.nan_to_num(t_zero, nan=0.0), t_off)
    _, int_off_d = _segment(y1d, y_inf, t_cond)
    mean_dcm = (int_on_d + int_off_d) / rho

    zero = -x / (1 + r_on) if zero_duty == "brake" else np.zeros_like(x)
    return np.where(u == 0, zero, np.where(ccm, mean_ccm, mean_dcm))
