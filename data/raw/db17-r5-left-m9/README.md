# db17-r5-left-m9

Robot R5 (host name "orange"), left wheel, motor m9, Experiment 2 of the Automatic Control course.

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
- Mode: stand, wheels free; robot R5 (Pi host name rpi-trixie-lite), motor m9 on HAT channel M2.
- Camera: Android phone slow motion, 1920x1080, captured at 240 frames/s and stored for playback at 30 frames/s, side-on to the left wheel.
- Pi script: `motor_trial.py` (commit d81c76229121), arguments in `trials.csv`.

| Run (video) | Original phone file | Trials |
|---|---|---|
| `R5-L-m9_A_sweep_dense_forward_20261118` | 20261118_194706.mp4 | E1 forward sweep 194711 |
| `R5-L-m9_A_sweep_dense_backward_20261118` | 20261118_194958.mp4 | E1 backward sweep 195003 |
| `R5-L-m9_C_b060p040_rep01-03_20261118` | 20261118_195120.mp4 | E2 steps 0 → 0.6 → 1.0 → 0.6 → 0, repetitions 195125, 195138, 195151 |

The videos are kept outside the repository; `runs.csv` names each tracked file by its SHA-256 checksum.

## Notes on the runs

- The phone stores slow motion for playback at 30 frames/s, so the timestamps of the video run 8 times slower than real time. Each video was therefore copied with its timestamps rescaled by 1/8 and without re-encoding (`ffmpeg -itsscale 0.125 -i <phone file> -map 0:v -c copy <run>.mp4`), and the copy was tracked; `runs.csv` gives the checksum of the copy. The E2 run confirms the rescaling: its LED and motion clocks agree within 10 ms.
- E1 was filmed as two videos, one per sweep direction, so both runs carry `A_sweep_dense` in their names.
- E1 forward trial `R5-L-m9_A_sweep_forward_010to036by020_rep01_20261118_194711` is the filmed run, but the Pi wrote both its log files empty (0 bytes), so they are not included. Its command schedule is reconstructed from the trial identifier, its duration of 61.0 s, and its 56 changes, as for the identical trials 194533 (left) and 184940 (right), and the video is aligned to it by the LED (sync `script+led`, schema §10.4).
- E1 forward trial 194533 was not filmed in full (the video stopped after about 34 s) and is not included.
- The camera drifted during the runs, so the frames were tracked with search disks that follow the hub and an LED grid that follows the LED (`track_video.py --wheel-follow --led-follow`). For the backward E1 run, the LED grid was placed 40 px above the followed LED tint, on the lit core of the LED (`--led-offset 0 -40`), and averaged over two frames to cancel the 120 Hz flicker of the room lighting, which the 240 frames/s camera samples in step (`--led-average 2`); without these, only 3 of its 54 LED edges were found.
- The marker tape looks cyan in these videos, so the marker colour rule was widened to `b>130 & b-r>60 & b-g>15`.
- The backward E1 run keeps a warning in `quality.md`: its angle scatters 10.4 times its pixel-based uncertainty in the steady parts, so the uncertainties of that run are scaled by 10.4.
- pi/motor_mapping.json is the robot's mapping file, the same file as in db17-r5-right-m10.
- The Pi log files were copied with `scp` and checked against SHA-256 checksums taken on the Pi; git stores the CSV and JSON files with LF line ends (`.gitattributes`), so the checksums of the committed copies differ from those on the Pi only by the line ends.
