# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Figure of the speed noise of a wheel at rest and the rest threshold.

Fig. S4 of the CSME 2026 supplement shows (a) the empirical distribution of
the speeds of a wheel at rest, with the threshold +-omega_min, and (b) the
share of their magnitudes above each value, which falls to the exceedance
(1 %) at omega_min, so the threshold and its risk are read off the data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

from duckiebot_analysis.visualization.case_study_figures import (  # noqa: E402
    BLUE,
    COLUMN_IN,
    INK2,
    OMEGA_LABEL,
    ORANGE,
)

FloatArray = npt.NDArray[np.float64]


def exceedance_curve(speeds: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return the sorted magnitudes and the share of magnitudes above each."""
    a = np.sort(np.abs(np.asarray(speeds, float)))
    # Share strictly above a[j]: count of later magnitudes, ties counted once.
    above = a.size - np.searchsorted(a, a, side="right")
    return a, above / a.size


def fig_rest_noise(
    speeds: npt.ArrayLike, omega_min: float, exceedance: float, path: Path
) -> None:
    """Draw the histogram and the exceedance curve of the rest speeds.

    Raises:
        ValueError: If speeds is empty.
    """
    w = np.asarray(speeds, float)
    if w.size == 0:
        raise ValueError("no rest speeds to draw")
    fig, (aa, ab) = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.6))
    lim = 1.15 * float(np.max(np.abs(w)))
    # Bins of omega_min / 8, so that +-omega_min falls on a bin edge.
    width = omega_min / 8 if omega_min > 0 else lim / 30
    n_bins = int(np.ceil(lim / width))
    edges = np.arange(-n_bins, n_bins + 1) * width
    inside = np.abs(w) <= omega_min
    aa.hist(
        [w[inside], w[~inside]],
        bins=edges,
        stacked=True,
        color=[BLUE, ORANGE],
        label=[r"$|\omega| \leq \omega_{\min}$", r"$|\omega| > \omega_{\min}$"],
    )
    for edge in (-omega_min, omega_min):
        aa.axvline(edge, color=INK2, lw=0.8, ls=(0, (4, 2)))
    aa.set_yscale("log")
    aa.set_xlim(-lim, lim)
    aa.set_xlabel(OMEGA_LABEL + " at rest")
    aa.set_ylabel("Frames")
    aa.legend(loc="upper left")
    aa.set_title("(a)", loc="left", fontsize=8)
    x, share = exceedance_curve(w)
    ab.step(x, share * 100, where="post", color=BLUE)
    ab.axhline(exceedance * 100, color=INK2, lw=0.6)
    ab.axvline(omega_min, color=INK2, lw=0.8, ls=(0, (4, 2)))
    ab.annotate(
        rf"$\omega_{{\min}} = {omega_min:.2f}$ rad/s",
        xy=(omega_min, exceedance * 100),
        xytext=(omega_min + 0.1 * lim, 10 * exceedance * 100),
        arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.6},
    )
    ab.set_yscale("log")
    ab.set_xlim(0, lim)
    ab.set_ylim(0.5 * 100 / w.size, 100)
    ab.set_xlabel(r"$|\omega|$ at rest (rad/s)")
    ab.set_ylabel(r"Frames above $|\omega|$ (%)")
    ab.set_title("(b)", loc="left", fontsize=8)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(path, dpi=600)
    plt.close(fig)
