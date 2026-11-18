# db17-r5-right-m10

Robot R5 (host name "orange"), right wheel, motor m10, Experiment 2 of the Automatic Control course.

## Group

Group 05.

| Member | Student ID |
|---|---|
| 田祐安 | E14136388 |
| 林有誼 | E14131574 |
| 李柏欣 | E14131697 |
| 潘冠宇 | E14131786 |
| 盧宣嘉 | E14131867 |
| 彭昭叡 | E14131914 |
| 鄧永林 | E14137025 |

## Recording

- Date: 2026-11-18, by group 05 in the course laboratory session.
- Mode: stand, wheels free; robot R5 (Pi host name rpi-trixie-lite), motor m10 on HAT channel M1.
- Camera: iPhone slow motion, 1920x1080 at 120 frames/s (the course asks for 240 frames/s; the phone was set to 1080p 120 frames/s by mistake), side-on to the right wheel.
- Pi script: `motor_trial.py` (commit d81c76229121), arguments in `trials.csv`.

| Run (video) | Original phone file | Trials |
|---|---|---|
| `R5-R-m10_A_sweep_dense_forward_20261118` | IMG_1098.MOV | E1 forward sweep 184940 |
| `R5-R-m10_A_sweep_dense_backward_20261118` | IMG_1099.MOV | E1 backward sweep 185116 |
| `R5-R-m10_C_b060p040_rep01-03_20261118` | IMG_1100.MOV | E2 steps 0 → 0.6 → 1.0 → 0.6 → 0, repetitions 185333, 185346, 185359 |

The videos are kept outside the repository; `runs.csv` names each by its SHA-256 checksum.

## Notes on the runs

- E1 was filmed as two videos, one per sweep direction, so both runs carry `A_sweep_dense` in their names.
- An earlier E1 forward trial (184658) and six E2 test trials (183944 to 184501) were not filmed for analysis and are not included; dry runs (`DRY`) are not included.
- The camera drifted by up to about 100 px during the runs and the robot vibrated on the stand, so the frames were tracked with search disks that follow the hub and an LED grid that follows the LED (`track_video.py --wheel-follow 150 --led-follow 150`, start positions `--hub-region`, `--marker-region`, `--led-xy` read from the first frame). With fixed regions the tracker located the marker on the laptop screen in the background and missed the LED.
- The marker tape looks cyan in these videos, so the marker colour rule was widened to `b>130 & b-r>60 & b-g>15`.
- The Pi log files were copied with `scp` and checked against SHA-256 checksums taken on the Pi; git stores the CSV and JSON files with LF line ends (`.gitattributes`), so the checksums of the committed copies differ from those on the Pi only by the line ends.
