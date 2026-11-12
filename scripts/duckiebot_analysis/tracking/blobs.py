# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""Pixel classes and their image moments (data/schema.md §3.1).

A colour rule is a conjunction of linear inequalities in the red, green, and
blue values of a pixel, written as in the frame-table metadata (§3.3), such as
``"r>170 & g>170 & b>160"`` or ``"r-b>80 & abs(r-g)<55"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# Named rules. Each value is a rule string, so the metadata always records the
# inequalities themselves, never only a name.
PRESETS: dict[str, str] = {
    # White marker of the six-motor campaign (Saloni's track_all.py).
    "white": "r>170 & g>170 & b>160",
    # Yellow hub of the six-motor campaign (Saloni's track_all.py).
    "yellow": "r>150 & g>125 & b<110 & r-b>80 & abs(r-g)<55",
    # Blue marker of Sam's daisy videos (task1_analyse.py). His tracker took an
    # HSV window around a clicked pixel: hue +-12 (OpenCV units, +-24 degrees)
    # and saturation and value down to 70 below the clicked pixel. For a blue
    # marker (hue about 110, saturation and value about 200) that window is
    # approximated by these RGB inequalities: blue bright, and well above red
    # and green. Check it on one of his frames before a campaign is tracked.
    "blue": "b>130 & b-r>70 & b-g>40",
}

_OPS = ("<=", ">=", "<", ">")
_CHANNELS = "rgb"
_CONDITION = re.compile(r"^(?P<lhs>.+?)(?P<op><=|>=|<|>)(?P<rhs>[+-]?\d+)$")
_ABS = re.compile(r"^abs\((?P<inner>.*)\)$")
_TERM = re.compile(r"([+-]?)(?:(\d+)\*)?([rgb])")


@dataclass(frozen=True)
class Condition:
    """One inequality ``[abs](cr*r + cg*g + cb*b) op threshold``."""

    coefficients: tuple[int, int, int]
    absolute: bool
    op: str
    threshold: int

    def __str__(self) -> str:
        text = ""
        for coef, name in zip(self.coefficients, _CHANNELS, strict=True):
            if coef == 0:
                continue
            sign = "-" if coef < 0 else ("+" if text else "")
            mag = "" if abs(coef) == 1 else f"{abs(coef)}*"
            text += f"{sign}{mag}{name}"
        if self.absolute:
            text = f"abs({text})"
        return f"{text}{self.op}{self.threshold}"

    def evaluate(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the boolean result for integer channel arrays."""
        cr, cg, cb = self.coefficients
        value = cr * r + cg * g + cb * b
        if self.absolute:
            value = np.abs(value)
        if self.op == "<":
            return np.asarray(value < self.threshold)
        if self.op == "<=":
            return np.asarray(value <= self.threshold)
        if self.op == ">":
            return np.asarray(value > self.threshold)
        return np.asarray(value >= self.threshold)


@dataclass(frozen=True)
class ColourRule:
    """Conjunction of conditions; ``str()`` gives the canonical rule string."""

    conditions: tuple[Condition, ...]

    def __str__(self) -> str:
        return " & ".join(str(c) for c in self.conditions)

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        """Classify the pixels of an (..., 3) uint8 RGB array.

        Args:
            rgb: Array whose last axis holds red, green, and blue.

        Returns:
            Boolean array of the shape of ``rgb`` without its last axis.
        """
        a = np.asarray(rgb).astype(np.int32)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        out = np.ones(a.shape[:-1], bool)
        for cond in self.conditions:
            out &= cond.evaluate(r, g, b)
        return out


def _parse_linear(text: str, condition: str) -> tuple[int, int, int]:
    if not text:
        raise ValueError(f"empty left-hand side in colour condition {condition!r}")
    coefs = [0, 0, 0]
    pos = 0
    first = True
    while pos < len(text):
        m = _TERM.match(text, pos)
        if m is None or (not first and not m.group(1)):
            raise ValueError(
                f"cannot parse {text[pos:]!r} in colour condition {condition!r}"
            )
        sign = -1 if m.group(1) == "-" else 1
        mag = int(m.group(2)) if m.group(2) else 1
        coefs[_CHANNELS.index(m.group(3))] += sign * mag
        pos = m.end()
        first = False
    if not any(coefs):
        raise ValueError(
            f"colour condition {condition!r} does not depend on r, g, or b"
        )
    return coefs[0], coefs[1], coefs[2]


def _parse_condition(text: str) -> Condition:
    compact = "".join(text.split())
    m = _CONDITION.match(compact)
    if m is None:
        raise ValueError(
            f"colour condition {text!r} is not of the form 'expr op integer'"
        )
    lhs = m.group("lhs")
    absolute = False
    am = _ABS.match(lhs)
    if am is not None:
        absolute = True
        lhs = am.group("inner")
    coefs = _parse_linear(lhs, text)
    return Condition(coefs, absolute, m.group("op"), int(m.group("rhs")))


def parse_colour_rule(text: str) -> ColourRule:
    """Parse a rule such as ``"r>170 & g>170 & b>160"``.

    Each condition is ``expr op integer`` or ``abs(expr) op integer``, where
    ``expr`` is a sum of terms ``[k*]r``, ``[k*]g``, ``[k*]b`` with signs and
    integer coefficients ``k``, and ``op`` is ``<``, ``<=``, ``>``, or ``>=``.
    Conditions are joined by ``&``. Whitespace is ignored.

    Args:
        text: The rule string.

    Returns:
        The parsed rule.

    Raises:
        ValueError: If the string is not a valid rule.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty colour rule")
    parts = text.split("&")
    return ColourRule(tuple(_parse_condition(p) for p in parts))


def resolve_colour_rule(text: str) -> ColourRule:
    """Return the preset named ``text``, or parse ``text`` as a rule string."""
    return parse_colour_rule(PRESETS.get(text.strip(), text))


class BlobMoments(NamedTuple):
    """Count, centroid, and covariance of a pixel set, in full-frame px."""

    n: int
    x: float
    y: float
    cxx: float
    cyy: float
    cxy: float


def blob_moments(mask: np.ndarray, x0: int = 0, y0: int = 0) -> BlobMoments:
    """Compute the zeroth, first, and central second moments of a mask.

    No threshold is applied: a single pixel gives a centroid and zero
    covariance, and an empty mask gives ``n = 0`` and NaN elsewhere (§3.1).
    The covariance divides by ``n``, the second central moment of the set.

    Args:
        mask: Boolean array, rows ``y`` and columns ``x``.
        x0: Full-frame column of ``mask[:, 0]``.
        y0: Full-frame row of ``mask[0, :]``.

    Returns:
        The moments, with positions in full-frame coordinates.
    """
    ys, xs = np.nonzero(mask)
    n = int(xs.size)
    if n == 0:
        nan = float("nan")
        return BlobMoments(0, nan, nan, nan, nan, nan)
    xf = xs.astype(np.float64)
    yf = ys.astype(np.float64)
    mx = float(xf.mean())
    my = float(yf.mean())
    dx = xf - mx
    dy = yf - my
    return BlobMoments(
        n,
        mx + x0,
        my + y0,
        float(np.mean(dx * dx)),
        float(np.mean(dy * dy)),
        float(np.mean(dx * dy)),
    )


def disk_mask(
    shape: tuple[int, int], cx: float, cy: float, r: float, x0: int = 0, y0: int = 0
) -> np.ndarray:
    """Return the pixels of an array within distance ``r`` of (``cx``, ``cy``).

    Args:
        shape: (rows, columns) of the array.
        cx: Full-frame x of the disk centre.
        cy: Full-frame y of the disk centre.
        r: Radius in px; a pixel at exactly ``r`` is inside.
        x0: Full-frame column of the array's first column.
        y0: Full-frame row of the array's first row.
    """
    yy = np.arange(shape[0], dtype=np.float64)[:, None] + y0 - cy
    xx = np.arange(shape[1], dtype=np.float64)[None, :] + x0 - cx
    return np.asarray(xx * xx + yy * yy <= r * r)


LED_BLOCK_PX = 8
LED_SPACING_PX = 8


# Sync LED colour -> (RGB channel that rises when the LED turns on, channel
# subtracted). A red LED gives red minus green and a green LED green minus red,
# so the led_NN values rise when either LED turns on and later steps read them
# the same way. The frame-table metadata records the signal used.
LED_SIGNALS = {"red": (0, 1), "green": (1, 0)}
LED_SIGNAL_TEXT = {"red": "red minus green", "green": "green minus red"}


def led_channels(colour: str) -> tuple[int, int]:
    """Return the (on, off) RGB channel indices of a sync LED colour.

    Raises:
        ValueError: If the colour is not red or green.
    """
    try:
        return LED_SIGNALS[colour]
    except KeyError:
        raise ValueError(
            f"LED colour {colour!r} not one of {', '.join(LED_SIGNALS)}"
        ) from None


def led_grid(
    frame: np.ndarray,
    led_x: float,
    led_y: float,
    x0: int = 0,
    y0: int = 0,
    block: int = LED_BLOCK_PX,
    spacing: int = LED_SPACING_PX,
    colour: str = "red",
) -> np.ndarray:
    """Sample the 5 x 5 LED grid of §3.1.

    Block ``5i + j`` is centred at ``(led_x + spacing*(j-2), led_y +
    spacing*(i-2))``. A block of even size is centred between pixels, so pass a
    centre ending in .5 for an exact centre; other centres are rounded to the
    nearest block. A block partly outside the frame is averaged over the part
    inside.

    Args:
        frame: (rows, columns, 3) uint8 RGB array.
        led_x: Full-frame x of the grid centre.
        led_y: Full-frame y of the grid centre.
        x0: Full-frame column of ``frame[:, 0]``.
        y0: Full-frame row of ``frame[0, :]``.
        block: Block size in px.
        spacing: Distance between block centres in px.
        colour: LED colour, ``red`` or ``green``; see LED_SIGNALS.

    Returns:
        uint8 array of 25 values: mean of the LED colour minus the other of red
        and green (red minus green for a red LED), plus 128, clipped
        to 0-255 and rounded.

    Raises:
        ValueError: If a block lies entirely outside the frame.
    """
    on, off = led_channels(colour)
    h, w = frame.shape[:2]
    half = (block - 1) / 2
    out = np.empty(25, np.uint8)
    for i in range(5):
        top = int(round(led_y + spacing * (i - 2) - half)) - y0
        for j in range(5):
            left = int(round(led_x + spacing * (j - 2) - half)) - x0
            ys = slice(max(top, 0), min(top + block, h))
            xs = slice(max(left, 0), min(left + block, w))
            patch = frame[ys, xs]
            if patch.size == 0:
                raise ValueError(f"LED block {5 * i + j} lies outside the frame")
            diff = patch[..., on].astype(np.float64) - patch[..., off].astype(
                np.float64
            )
            out[5 * i + j] = np.clip(np.rint(diff.mean() + 128), 0, 255)
    return out
