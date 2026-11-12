# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests of the student-kit changes: trial labels, green sync LED, motor choice.

Run from the repository root: python -m pytest scripts/tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckiebot_analysis.analysis import case_study as cs  # noqa: E402
from duckiebot_analysis.data.layout import parse_motor_folder  # noqa: E402
from duckiebot_analysis.data.pilogs import parse_trial_id  # noqa: E402
from duckiebot_analysis.tracking.blobs import led_grid  # noqa: E402


@pytest.mark.parametrize(
    ("text", "label", "side", "motors", "channels"),
    [
        (
            "R5-L-m9_B_p050_rep03_20261024_113544",
            "R5-L-m9",
            "L",
            ("m9",),
            ("motor_left",),
        ),
        (
            "R8-R-m16_C_b060p040_rep01_20261024_113544",
            "R8-R-m16",
            "R",
            ("m16",),
            ("motor_right",),
        ),
        (
            "R5-both-m9-m10_E_air_p060_filmedLeft_rep01_20261024_113544",
            "R5-both-m9-m10",
            "both",
            ("m9", "m10"),
            ("motor_left", "motor_right"),
        ),
        (
            "R2-m4_A_sweep_forward_010to036by020_rep01_20261031_101500",
            "R2-m4",
            "R",
            ("m4",),
            ("motor_right",),
        ),
        (
            "R3-L_A_sweep_forward_010to036by020_rep01_20261024_211603",
            "R3-L",
            "L",
            (),
            ("motor_left",),
        ),
        (
            "R3-both_E_air_p060_filmedLeft_rep01_20261025_101500",
            "R3-both",
            "both",
            (),
            ("motor_left", "motor_right"),
        ),
    ],
)
def test_trial_ids_round_trip(text, label, side, motors, channels):
    tid = parse_trial_id(text)
    assert (tid.label, tid.side, tid.motors) == (label, side, motors)
    assert tid.channels == channels
    assert str(tid) == text


@pytest.mark.parametrize(
    "text",
    [
        "R9-L-m1_B_p050_rep01_20261024_113544",
        "R1-L-m17_B_p050_rep01_20261024_113544",
        "R1-X_B_p050_rep01_20261024_113544",
        "",
    ],
)
def test_invalid_trial_ids_rejected(text):
    with pytest.raises(ValueError):
        parse_trial_id(text)


def test_student_folder_name():
    assert parse_motor_folder("db17-r5-left-m9") == ("r5", "left")


def test_led_signal_rises_for_its_own_colour():
    frame = np.zeros((40, 40, 3), np.uint8)
    off = led_grid(frame, 20.5, 20.5, colour="green")
    frame[..., 1] = 200  # a green LED turns on
    assert led_grid(frame, 20.5, 20.5, colour="green").min() > off.max()
    assert led_grid(frame, 20.5, 20.5, colour="red").max() < off.min()
    with pytest.raises(ValueError):
        led_grid(frame, 20.5, 20.5, colour="blue")


def test_configure_points_at_one_motor(tmp_path):
    saved = (
        cs.MOTOR,
        cs.LABEL,
        cs.MOTOR_CHANNEL,
        cs.E1_RUN,
        cs.E1_TRIAL,
        cs.E1_TRIAL_NEG,
        cs.E2_RUN,
        cs.E2_PREFIX,
    )
    try:
        cs.configure("db17-r2-right-m4", "R2-m4")
        assert cs.MOTOR_CHANNEL == "motor_right"
        assert cs.E2_RUN == "R2-m4_C_b060p040*.parquet"
        with pytest.raises(ValueError):
            cs.configure("db17-r1-both-m1-m2", "R1-both-m1-m2")
        (tmp_path / "R2-m4_C_b060p040-m055_a.parquet").touch()
        (tmp_path / "R2-m4_C_b060p040-m055_b.parquet").touch()
        with pytest.raises(FileNotFoundError):
            cs._one_run(tmp_path, cs.E2_RUN)
    finally:
        (
            cs.MOTOR,
            cs.LABEL,
            cs.MOTOR_CHANNEL,
            cs.E1_RUN,
            cs.E1_TRIAL,
            cs.E1_TRIAL_NEG,
            cs.E2_RUN,
            cs.E2_PREFIX,
        ) = saved
