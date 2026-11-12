# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Array and argument types shared by the angle modules (data/schema.md §9)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
Frames = Mapping[str, npt.ArrayLike]
Mode = Literal["stand", "floor"]
Direction = Literal["ccw", "cw"]
# A window is a boolean mask over the frames or a closed interval of t in s.
Window = BoolArray | tuple[float, float]
