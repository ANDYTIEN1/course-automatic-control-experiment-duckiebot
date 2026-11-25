# db17-r1-left-m1

Robot R1, left wheel, motor m1, on the stand with the wheels free (Pi label `R1-L-m1`).

- **Group:** 8
- **People:** 戴子宇 E14135065, 陳國靖 E14135031, 譚凱茹 E14135057, 李恩祈 E14135332, 黃宇均 E14131249
- **Recorded:** 2026-11-14, Xiaomi 15T Pro, slow motion 1280x720 240 fps
- **Pi script:** `motor_trial.py`, commit `d81c76229121`

## Runs

| Run | Phone clip | Experiment | Trials |
|---|---|---|---|
| `R1-L-m1_A_sweep_forward_005to100by050_rep01_20261114` | LA1 | A, coarse sweep forward | 1 |
| `R1-L-m1_A_sweep_backward_005to100by050_rep01_20261114` | LA2 | A, coarse sweep backward | 1 |
| `R1-L-m1_A_sweep_forward_010to036by020_rep01_20261114` | LA3 | A, dense sweep (E1) forward | 1 |
| `R1-L-m1_A_sweep_backward_010to036by020_rep01_20261114` | LA4 | A, dense sweep (E1) backward | 1 |
| `R1-L-m1_C_b060p040_rep01-03_20261114` | LC1 | C, steps 0 → 0.6 → 1.0 → 0.6 → 0 (E2) | 3 |
| `R1-L-m1_C_b060m040_rep01-03_20261114` | LC2 | C, steps 0 → 0.6 → 0.2 → 0.6 → 0 | 3 |

## Notes

- The dense sweep (E1) was filmed as two clips, LA3 (forward trial) and LA4 (backward trial), with the camera stopped for about 30 s between them, so each clip is a run of its own.
  Joining the two clips into one run was tried and dropped: joining removes the pause from the video time but not from the Pi clock, so the single clock fitted to the run (schema §10.6) placed one of the two trials tens of seconds off.
  `case_study.py` reads the two sweeps from their own runs when no `_A_sweep_dense_` run exists.
- The marker is cyan tape, which the kit's default blue rule misses;
  every run was tracked with `--marker-colour "b>100 & g>80 & b-r>80 & b-g>12"`.
- `positive_duty_turns = "ccw"` was read from the video: at duty +1.0 the marker turns counterclockwise on screen, about 5° per frame.
- Experiment B (clips LB1, LB2) was recorded but is not part of this hand-in; its Pi logs stay in `pi/` unchanged.
