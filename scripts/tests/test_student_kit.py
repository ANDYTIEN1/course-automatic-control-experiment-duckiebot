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


# ------------------------------------------------- case study input checks

_RUNS_HEADER = (
    "run,video_file,video_sha256,video_bytes,duration_s,resolution,fps_nominal,"
    "mode,experiment,sync,pi_log,trials,notes\n"
)


def _trace_columns(trial: str, duty: list[float]) -> dict[str, list]:
    n = len(duty)
    return {
        "trial": [trial] * n,
        "t_trial_s": [k / 240 for k in range(n)],
        "omega_rad_s": [0.0] * n,
        "omega_sd_rad_s": [0.1] * n,
        "u_cmd": duty,
    }


def _write_run(path: Path, trials: list[tuple[str, list[float]]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols: dict[str, list] = {}
    for name, duty in trials:
        for k, v in _trace_columns(name, duty).items():
            cols.setdefault(k, []).extend(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


def _motion_synced_data(root: Path) -> str:
    """Write processed runs of the default motor whose E2 trial lacks LED sync."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    proc = root / "processed" / cs.MOTOR
    e1 = cs.E1_TRIAL + "20261024_211603"
    neg = cs.E1_TRIAL_NEG + "20261024_211708"
    e2 = cs.E2_PREFIX + "01_20261024_212000"
    _write_run(
        proc / f"{cs.LABEL}_A_sweep_dense_x.parquet", [(e1, [0.0]), (neg, [0.0])]
    )
    _write_run(proc / f"{cs.LABEL}_C_b060p040_x.parquet", [(e2, [0.0] * 10)])
    rows = [
        {"trial": e1, "sync": "log+led", "drift": 1e-5, "timing_sd_s": 1e-4},
        {"trial": neg, "sync": "log+led", "drift": 1e-5, "timing_sd_s": 1e-4},
        {"trial": e2, "sync": "log+motion", "drift": None, "timing_sd_s": None},
    ]
    pq.write_table(pa.Table.from_pylist(rows), proc / "trials.parquet")
    return e2


def test_case_study_needs_led_synchronisation(tmp_path):
    trial = _motion_synced_data(tmp_path)
    with pytest.raises(ValueError, match="LED") as exc:
        cs.case_study(tmp_path)
    assert trial in str(exc.value)
    assert "log+led" in str(exc.value) and "script+led" in str(exc.value)


def test_plot_script_reports_motion_sync_without_traceback(tmp_path):
    import subprocess

    trial = _motion_synced_data(tmp_path)
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [
            sys.executable,
            "scripts/plot_case_study_csme2026.py",
            "--data",
            str(tmp_path),
            "--figures",
            str(tmp_path / "figures"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode != 0
    assert "Traceback" not in out.stderr
    assert trial in out.stderr and "LED" in out.stderr


def test_speeds_at_06_skips_trials_without_a_hold_at_06(tmp_path):
    hold = [0.0] * 240 + [0.6] * 480 + [1.0] * 240
    short = [0.0] * 240 + [0.6] * 120 + [1.0] * 240
    good = tmp_path / "db17-a-left-m1" / "R1-L-m1_C_b060p040_x.parquet"
    _write_run(good, [("R1-L-m1_C_b060p040_rep01_20261024_100000", hold)])
    bad = tmp_path / "db17-b-right-m2" / "R2-R-m2_C_b060p040_x.parquet"
    _write_run(bad, [("R2-R-m2_C_b060p040_rep01_20261024_100000", short)])
    assert cs.speeds_at_06(tmp_path) == {"m1": 0.0}


def test_speeds_at_06_with_no_usable_trial_is_empty(tmp_path):
    bad = tmp_path / "db17-b-right-m2" / "R2-R-m2_C_b060p040_x.parquet"
    _write_run(bad, [("R2-R-m2_C_b060p040_rep01_20261024_100000", [0.6] * 10)])
    assert cs.speeds_at_06(tmp_path) == {}


def test_case_study_refuses_a_motor_other_than_the_configured_one(tmp_path):
    with pytest.raises(ValueError, match="configure"):
        cs.case_study(tmp_path, motor="db17-r5-left-m9")


def test_quality_tells_to_process_a_tracked_run(tmp_path):
    from duckiebot_analysis.analysis import quality

    raw = tmp_path / "raw" / "db17-r5-left-m9"
    raw.mkdir(parents=True)
    (raw / "setup.toml").write_text(
        'robot = "r5"\nmotor = "left"\nmotor_label = "m9"\n', encoding="utf-8"
    )
    sha = "0" * 64
    rows = [
        f"{run},{run}.mp4,{sha},1,1,1280x720,240,stand,C,log+led,"
        f"pi/{run}_t_events.csv,{run}_t,\n"
        for run in ("R5-L-m9_C_tracked", "R5-L-m9_C_untracked")
    ]
    (raw / "runs.csv").write_text(_RUNS_HEADER + "".join(rows), encoding="utf-8")
    (raw / "frames").mkdir()
    (raw / "frames" / "R5-L-m9_C_tracked.parquet").touch()
    fq = quality.assess_folder(raw, tmp_path / "processed" / raw.name, motion_bias={})
    quality.finish([fq])
    tracked, untracked = fq.runs
    message = " ".join(i.text for i in tracked.issues)
    assert "not processed" in message and "frame table" not in message
    assert "process_raw.py" in tracked.trials[0].recommendation
    assert "track_video.py" not in tracked.trials[0].recommendation
    assert "no frame table" in " ".join(i.text for i in untracked.issues)
    assert "track_video.py" in untracked.trials[0].recommendation
    for rec in (tracked.trials[0].recommendation, untracked.trials[0].recommendation):
        assert "data/README.md" not in rec


def test_student_scripts_cite_no_missing_data_readme():
    scripts = Path(__file__).resolve().parents[1]
    paths = [p for p in scripts.rglob("*.py") if "tests" not in p.parts]
    if not (scripts.parent / "data" / "README.md").exists():
        cited = [p.name for p in paths if "data/README.md" in p.read_text("utf-8")]
        assert cited == []
