---
title: "control-motor-duckiebot - Data Schema"
author: "Torbjörn E. M. Nordling"
date: "2026-10-26"
version: 2.2.0
license: "Apache-2.0"
purpose: "Every stored variable of the DB17 motor measurements, its format and motivation, and how each derived quantity and its uncertainty is computed from them."
---

# Data Schema

The section Data of the repository's `README.md` describes the folders.
This file specifies the content of each stored file, then the derivation of every computed quantity.
The code refers to the section numbers.

## 1. Conventions

- **Units.**
  SI throughout:
  seconds, metres, radians.
  Pixel quantities are in pixels (px) of the full video frame.
- **Image coordinates.**
  Origin at the centre of the top-left pixel, `x` to the right, `y` down, in the frame as stored in the file, before any display rotation.
  The rotation flag of the video is recorded in the file metadata (§3.3).
- **Duty.**
  The duty `u` is the commanded throttle in [−1, 1], with the sign the script used, before any sign change the script applies for a motor's wiring.
  A positive duty may turn a wheel backward;
  `setup.toml` records the direction (§6).
- **Missing values.**
  Null in Parquet, empty in CSV.
  Zero is a measurement, never a placeholder.
- **Text files.**
  CSV in UTF-8 with a header row and LF line endings;
  TOML for constants.

## 2. Files of a motor folder

| File | Format | One row per | Section |
|---|---|---|---|
| `setup.toml` | TOML | – | §6 |
| `runs.csv` | CSV | video | §4 |
| `trials.csv` | CSV | trial | §5 |
| `frames/<run>.parquet` | Parquet | video frame | §3 |
| `pi/<log>` | as written by the Pi | logged event | §7 |
| `manual/` | CSV, plus the scanned sheet | hand-read value | §8 |

## 3. Frame table: `frames/<run>.parquet`

One file per video.
It holds what the tracker measured in each decoded frame, and nothing computed from more than one frame.

### 3.1 Columns

| Column | Type | Unit | Content | Motivation |
|---|---|---|---|---|
| `frame` | uint32 | – | Index of the decoded frame, from 0 | Shows frames the decoder skipped |
| `pts` | int64 | time base | Presentation timestamp of the frame, as stored in the video | Exact frame time; the phone drops frames in bursts, so the frame rate cannot be assumed |
| `marker_n` | uint32 | px | Number of marker-coloured pixels inside the marker search region | Visibility of the marker, and the sample size of its centroid |
| `marker_x`, `marker_y` | float32 | px | Mean position of those pixels | The measurement the wheel angle is computed from |
| `marker_cxx`, `marker_cyy`, `marker_cxy` | float16 | px² | Covariance of the positions of those pixels | Uncertainty of the centroid, and its elongation by motion blur; two significant digits suffice for an uncertainty |
| `hub_n` | uint32 | px | Number of hub-coloured pixels inside the hub search region | Visibility of the hub |
| `hub_x`, `hub_y` | float32 | px | Mean position of those pixels | Reference point that moves with the wheel axle, so the angle stays valid when the camera or the robot moves |
| `hub_cxx`, `hub_cyy`, `hub_cxy` | float16 | px² | Covariance of the positions of those pixels | Uncertainty of the hub centroid |
| `led_x`, `led_y` | float32 | px | Centre of the LED sampling grid in this frame | Constant on the stand; follows the robot on the floor |
| `led_00` … `led_24` | uint8 | – | Mean of the LED colour minus the other colour over an 8 × 8 px block, plus 128, clipped to 0–255: green minus red for a green LED, the kit's default, and red minus green for a red LED; `led_grid.signal` in the file metadata records which | The sync LED's brightness; the grid makes the edges detectable even if the LED was located a few pixels off |

Block `led_{5i+j}` is centred at (`led_x` + 8(j − 2), `led_y` + 8(i − 2)) for i, j = 0 … 4, so `led_12` is the centre block.
`led_x` and `led_y` lie on a half pixel, so each 8 × 8 block covers whole pixels.
A run without a sync LED has null LED columns;
the reader then returns them as float32 with NaN, since uint8 holds no missing value.

A floor run tracked with the spoke tracker has three more columns, after the LED columns:

| Column | Type | Unit | Content | Why it is stored |
|---|---|---|---|---|
| `spoke_angle` | float32 | rad | Angle of the five yellow spokes, modulo 2π/5, in (−π/5, π/5], growing clockwise on screen | The wheel angle where the tape is hidden (§9.5) |
| `spoke_angle_sd` | float32 | rad | Its standard deviation | Weight of the frame |
| `spoke_contrast` | float32 | – | \|c₅\| / Σ of the ring samples, 0 when no hub pixel lies on the ring | Whether the spokes were seen |

The hub pixels, the yellow wheel, are sampled at 360 angles on a ring at 0.55 of the wheel radius (the 98th percentile of the hub pixels' distance from the hub centroid) around the hub centroid, which crosses every spoke and not the rim.
The phase of the ring's 5th angular harmonic c₅ gives the angle, −arg(c₅)/5, and its standard deviation 0.5 · √(360/2) / (5 |c₅|), with 0.5 the largest standard deviation of a binary sample.
Frame tables tracked before the spoke tracker have no spoke columns.

Rules for the tracker:

- Apply no quality threshold.
  Store the count and the centroid however few pixels were found, and null only when the count is zero.
  Whether a frame is usable is decided in the derivation (§9.2), where the threshold can be changed.
- Store image moments only.
  The centroid and the covariance are the first and second moments of the pixel set;
  together with the count they are what the pixel set tells about the marker position.

### 3.2 Storage

- Compression zstd, level 9.
- `frame` and `pts` with delta encoding, which stores a regular sequence in about one byte per value.
- Float columns with byte-stream-split encoding and no dictionary, which compresses measurement noise best.
- One row group per file.

### 3.3 File metadata

The Parquet key-value metadata holds one key, `duckiebot`, whose value is a JSON object:

| Key | Content |
|---|---|
| `schema_version` | `"2.0.0"` |
| `run` | Run identifier, equal to the file name stem |
| `video_file`, `video_sha256`, `video_bytes` | Name, SHA-256 checksum, and size of the source video |
| `width`, `height`, `rotation` | Frame size in px, and the display rotation in degrees from the container |
| `time_base` | Seconds per `pts` unit, as a fraction string such as `"1/90000"` |
| `fps_nominal` | Frame rate from the container |
| `camera` | Phone model and recording mode, such as `"Samsung S7 Edge, slow motion 1280x720 240 fps"` |
| `mode` | `"stand"` (wheel fixed in the picture) or `"floor"` (robot moves through the picture) |
| `marker_colour`, `hub_colour` | Colour rule of each pixel class, as RGB inequalities, such as `"r>170 & g>170 & b>160"` |
| `marker_region`, `hub_region` | Search regions: on the stand a disk (`cx`, `cy`, `r` in px), with the marker disk 1.1 times the radius swept by the marker so its outer edge is not cut; on the floor a disk of radius `r` around the previous frame's hub, or around the largest hub-coloured blob in the first frame and after the hub was lost |
| `led_grid` | Block size and spacing in px, and how `led_x`, `led_y` were found: on the stand from the changes of the LED colour in key frames, on the floor as the centroid of the pixels within 10 % of the strongest LED colour near the hub; `signal` records the colour difference used |
| `decoder` | FFmpeg version |
| `tracker` | Script path and git commit of the tracker, and whether the working tree had uncommitted changes |
| `tracked` | ISO 8601 date and time of tracking |
| `spokes` | Floor runs with spoke columns only: number of spokes, ring fraction, and method of the spoke angle |

## 4. Run list: `runs.csv`

| Column | Type | Content |
|---|---|---|
| `run` | string | Run identifier, the video file name without extension |
| `video_file` | string | Video file name |
| `video_sha256` | string | SHA-256 checksum of the video, 64 hex digits |
| `video_bytes` | integer | Size of the video in bytes |
| `duration_s` | float | Duration from the container |
| `resolution` | string | Width x height in px |
| `fps_nominal` | float | Frame rate from the container |
| `mode` | string | `stand` or `floor` |
| `experiment` | string | Experiment or test name, such as `B` or `hyst-pos` |
| `sync` | string | Synchronisation level: `log+led`, `script+led`, `log+motion`, or `script+motion` (§10) |
| `pi_log` | string | Path of the Pi log relative to the motor folder, or empty |
| `trials` | string | Trial identifiers in the run, separated by `;` |
| `notes` | string | Anything that affects the use of the run |

## 5. Trial list: `trials.csv`

The trial list states what the Pi was asked to do.
It is the backup source of the command schedule when the Pi log is missing (§10.4).
`duration_s` and `changes` identify the schedule of a script whose arguments were not recorded, such as the six-motor campaign's `motor_trial.py`.

| Column | Type | Content |
|---|---|---|
| `trial` | string | Trial identifier, such as `R1-L_B_p050_rep03_20261024_113544` |
| `run` | string | Run identifier of the video that shows it |
| `script` | string | Path of the Pi script in the repository, such as `scripts/robot/motor_trial.py` |
| `script_commit` | string | Git commit of the script as it ran on the Pi |
| `args` | string | Command-line arguments of the script, exactly as typed |
| `start_pi` | string | Start on the Pi wall clock, ISO 8601, at the resolution known |
| `duration_s` | float | Duration of the trial as the Pi recorded it, or empty |
| `changes` | integer | Number of output changes the Pi recorded, or empty |
| `status` | string | `completed`, `aborted`, or `failed` |
| `notes` | string | Observations during the trial |

## 6. Setup constants: `setup.toml`

```toml
robot = "cherry"                # lower-case name, as in the folder name db17-cherry-left-m5
motor = "left"                  # left, right, or both: the position during the test
motor_label = "m5"              # identifier written on the motor, as in the folder name;
                                # for both, the left then the right motor: "m5, m6"
hat = "Adafruit DC Motor HAT, PCA9685 at 0x60"
hat_channel = "M1"
pwm_frequency_hz = 1600         # nominal; the PCA9685 prescaler sets the actual value
driver = "adafruit_motorkit 3.4.13, adafruit_pca9685 3.4.16"  # versions on the Pi
quantisation = "pca9685"        # or "count255" if the script rounded to 8 bits
supply = "LiPo 3.7 V 3800 mAh"
positive_duty_turns = "ccw"     # direction of a positive duty, as seen by the camera
```

The values above illustrate the format.
`setup.toml` describes the equipment;
quantities measured on it, such as the wheel radius, are stored as repeated measurements in `manual/measurements.csv` (§8.2).
A fact that is not known is left out;
the code then treats it as unknown and reports the quantities that need it as missing.

## 7. Pi logs: `pi/<log>`

The Pi log is the record of the actual input.
It is committed exactly as the Pi wrote it, so no conversion can lose or alter a timestamp.
The code reads the following formats.

### 7.1 Format for new runs

One CSV per trial, one row per output change, written by the Pi script.
The file opens with a comment line giving the Pi wall clock at `t_s = 0`, such as `# start 2026-10-24T11:35:44.123456+08:00`, followed by the header:

| Column | Type | Content |
|---|---|---|
| `trial` | string | Trial identifier |
| `t_s` | float | `time.monotonic()` just before the output call, seconds from the trial start, six decimals |
| `t_after_s` | float | `time.monotonic()` just after the call returned |
| `channel` | string | `motor_left`, `motor_right`, or `led` |
| `value` | float | Throttle passed to the driver, or 0 and 1 for the LED |
| `event` | string | Name of the step in the schedule, such as `step_on` |

`t_after_s − t_s` bounds the time the call took, which enters the timing uncertainty (§10.3).
A start given to a fraction of a second also places the trials of a run on one Pi clock (§10.6);
a start to the second does not.

### 7.2 Formats of existing logs

| Format | Columns | Written by |
|---|---|---|
| `daisy-task1` | `t_s,event,u,count`; `count` is the duty × 255 actually written | Sam's `task1_static.py` |
| `cherry-steps` | `repeat,duty,on_s,off_s,wheel,note`; LED and motor switched together | Saloni's `step_series.py` and related scripts |
| `motor-trial` | `t_s,t_monotonic_s,event,label,requested_duty,written_throttle,led`; one row per call, logged just after it returned, LED first and then each motor (`motor_L`, `motor_R`); `requested_duty` is the commanded duty, `written_throttle` the value after the wiring sign; `start`, `end`, and `stop` mark the trial, and `stop` switches all outputs off; `t_monotonic_s − t_s` is the same on every row, to 10 µs, and is the trial's start on the Pi's monotonic clock (§10.6) | Saloni's `scripts/robot/motor_trial.py` (six-motor campaign), as `<trial>_events.csv` beside `<trial>_meta.json` |

## 8. Hand-read values: `manual/`

### 8.1 Results of runs

Values measured by hand during runs, such as floor distances, are stored as CSV with a column per measured quantity and its unit in the header, such as `distance_m`.
The scanned or photographed sheet they were read from is stored beside the CSV, and `README.md` of the motor folder says how unclear entries were read.

### 8.2 Measured constants: `manual/measurements.csv`

Every quantity measured on the equipment, such as the wheel radius, the marker's distance from the axle, the robot's mass, or the supply voltage, is measured ten times, and all ten values are stored, one row per measurement:

| Column | Type | Content |
|---|---|---|
| `quantity` | string | Name, such as `wheel_radius`, `marker_radius`, `robot_mass`, `supply_voltage`, or `camera_readout` |
| `repeat` | integer | 1 to 10 |
| `value` | float | The reading, as read |
| `unit` | string | SI unit of `value`, such as `m`, `kg`, `V`, or `s` |
| `instrument` | string | Instrument and its resolution, such as `caliper, 0.01 mm` |
| `resolution` | float | Resolution of the instrument in `unit` |
| `date` | string | ISO 8601 date |
| `by` | string | Person who measured |
| `notes` | string | How it was measured, such as `diameter across the tyre, halved` |

The value and its standard uncertainty are computed, following the GUM (JCGM, 2008), as the mean x̄ of the n readings and

  u(x̄) = √(s² / n + r² / 12),

where s is the sample standard deviation and r the resolution:
the first term is the scatter of the readings, the second the rounding of each reading.
The derivations use `wheel_radius`, `marker_radius` (§9.8), and `camera_readout` (§9.1) from this file.

## 9. Derived quantities per frame

Every derived quantity is computed by `control_motor_duckiebot.analysis` and written to `data/processed/db17-<robot>-<motor>-<id>/<run>.parquet` (§11).
For frame i, write the marker centroid **m**ᵢ = (`marker_x`, `marker_y`), its pixel covariance **C**ᵢ, and its count nᵢ, and the same with **h** for the hub.

### 9.1 Time

tᵢ = (`pts`ᵢ − `pts`₀) · `time_base`, in seconds from the first frame.
A gap tᵢ − tᵢ₋₁ greater than 1.5 times the median frame interval marks dropped frames.
The timestamp is exact to the time base.
A rolling shutter reads rows at different times, so a feature at row y is seen at tᵢ + (y / `height`) · t_r, with the readout time t_r = `camera_readout` (§8.2);
when the readout time is unknown, half a frame interval is added to the uncertainty of events compared across rows (§10.3).

### 9.2 Usable frames

Frame i is usable when nᵢ ≥ n_min and the hub is found (`hub_n` ≥ n_hub,min), or, on the stand, imputed (§9.4).
The thresholds are parameters of the processing, stored with the processed file;
the defaults are 25 px for the marker and 50 px for the hub, the values used for the six-motor campaign.

### 9.3 Centroid uncertainty

The covariance of a centroid estimated from n pixels with spread **C** is

  **Σ**ₘ,ᵢ = k² **C**ᵢ / nᵢ,

and likewise **Σ**ₕ,ᵢ for the hub.
The factor k ≥ 1 accounts for what the pixel spread does not capture, such as colour threshold noise and compression artefacts.
It is calibrated per run from the residuals of the angle in steady rotation (§9.6):
the angle is computed first with k = 1, then recomputed with the calibrated k.

### 9.4 Position relative to the hub

The marker is measured relative to the hub,

  **r**ᵢ = **m**ᵢ − **h**ᵢ,  **Σ**ᵣ,ᵢ = **Σ**ₘ,ᵢ + **Σ**ₕ,ᵢ,

so a translation of the camera or the robot cancels.
On the stand, where the hub should not move, the frame is flagged `hub_moved` when, on either axis, |hᵢ − median(h)| exceeds three times the robust standard deviation (1.4826 times the median absolute deviation, but at least the centroid's own standard error);
the measurement remains valid through the subtraction, and the flag shows where it was needed.
A uniform drift over the whole run raises the robust deviation with it and is not flagged;
the subtraction corrects it all the same.
On the floor, where the hub moves by design, the flag is not set.
When the hub is not found in a stand frame, **h**ᵢ is replaced by the median hub position of the run and the frame is flagged `hub_imputed`, with **Σ**ₕ,ᵢ the sample covariance of the found hub positions;
on the floor such a frame is not usable, and a stand run without any hub has no usable frame.

### 9.5 Angle

The marker moves on a circle that the camera sees as an ellipse.
A direct least-squares ellipse fit (Halíř & Flusser, 1998) to the usable **r**ᵢ of the run gives the ellipse centre **c**, and the matrix **A** = **D**^½ **V**ᵀ, from the eigenvalues **D** and eigenvectors **V** of the conic's quadratic form, that maps the ellipse onto a circle.
For each frame

  **q**ᵢ = **A**(**r**ᵢ − **c**),  θᵢ = s · unwrap(atan2(qᵢ,y, qᵢ,x)),

where s = ±1 makes a positive duty give a positive angle (§6, `positive_duty_turns`).
Unwrapping assumes less than half a turn between usable frames.
A jump of half a turn or more cannot be seen in the wrapped angle, so it is predicted:
a new segment starts where the running median of the speed over 11 intervals, times the interval, reaches π.
θ stays continuous across a break, so the correction of §9.6 still depends only on the wheel position, but differences of θ across segments are not used.

The uncertainty follows from the gradient of θ with respect to **r**:

  **g**ᵢ = (−qᵢ,y, qᵢ,x) **A** / |**q**ᵢ|²,  σ²_θ,ᵢ = **g**ᵢ **Σ**ᵣ,ᵢ **g**ᵢᵀ.

σ_θ leaves out the uncertainty of the ellipse fit, which is the same for every frame of the run and shows up as the once-per-turn error that §9.6 removes.

On the floor, where the perspective changes as the robot moves, the ellipse is refitted in windows of one wheel turn.
Each window's map is fixed only up to a rotation, so it is aligned to the previous window by the mean difference of the two phases;
a window with fewer than 20 points, or with points in fewer than 10 of 12 sectors of the turn, uses the run's ellipse.

When the frame table has spoke columns (§3.1), the wheel angle of a floor run is taken from the spokes instead, and the marker angle is kept only for the pixel scale of §9.8.
θᵢ is s times the sum of the frame-to-frame increments of `spoke_angle`, each wrapped to (−π/5, π/5], with σ_θ,ᵢ = `spoke_angle_sd`.
A frame is usable when its spoke contrast is at least 0.3.
A new segment starts where more than four median frame intervals separate two usable frames, which keeps every increment below half a spoke period up to 37 rad/s at 240 frames per second.
The spokes are symmetric, so this angle needs no once-per-turn correction (§9.6).
Without `positive_duty_turns` in `setup.toml`, s is taken from the sign of ω under the duty, and the motion alignment of §10.5 uses ω from the spokes.

### 9.6 Once-per-turn correction

Tracking errors that depend on the wheel position, such as a marker that is not a point, repeat once per turn.
In the steady parts of each trial (constant duty for at least 1.6 s, from 1.2 s after the change), the model

  θᵢ = θ₀,ⱼ + ωⱼ tᵢ + Σₖ₌₁³ [aₖ cos(kθᵢ) + bₖ sin(kθᵢ)]

is fitted by weighted least squares with weights 1/σ²_θ,ᵢ, with an offset θ₀,ⱼ and a speed ωⱼ for each steady part j within a segment, and harmonics shared by the whole trial.
A steady part that sweeps at least four turns also gets terms in (t − t̄ⱼ)² and (t − t̄ⱼ)³, because at constant duty the speed of some motors drifts, by up to 40 % over 7 s on motor m6 backward;
with a constant ωⱼ that motion would enter χ² and k and bend the harmonics.
Over four turns a cubic in time can take up at most 11 % of the variance of a once-per-turn term, so the harmonics stay determined;
a shorter part keeps a constant speed.
The harmonics are fitted per trial because the phase of the error drifts between trials on some motors.
A trial is corrected only when its steady parts together cover at least one turn;
otherwise the harmonics are not determined.
The corrected angle is θᶜᵢ = θᵢ − h(θᵢ), with h the fitted harmonic sum, and to first order

  σ²_θᶜ,ᵢ = (1 − h′(θᵢ))² σ²_θ,ᵢ + **H**ᵢ **Cov**(a, b) **H**ᵢᵀ,

with **H**ᵢ the gradient of h with respect to the coefficients.
The errors of θᶜ are correlated across the frames of a trial through the shared coefficients.
The χ² of the fits calibrates the factor of §9.3 once per run, over the corrected trials:
k² = max(1, Σχ² / Σν).
The per-trial value is stored as well (§11.2);
when no trial is corrected, k = 1.

### 9.7 Angular velocity and acceleration

A second-degree polynomial is fitted by weighted least squares to θᶜ against t over the usable frames within ±w of tᵢ:

  θᶜ(t) ≈ β₀ + β₁(t − tᵢ) + β₂(t − tᵢ)²,  ωᵢ = β₁,  αᵢ = 2β₂.

The covariance of β is (**X**ᵀ**W****X**)⁻¹ · max(1, χ²/ν), which gives σ_ω,ᵢ and σ_α,ᵢ.
The window is set in seconds so runs at different frame rates are comparable:
w_ω = 23 ms (11 frames at 240 frames/s) for ω and w_α = 50 ms for α.
A fit needs at least five usable frames with at least two on each side of tᵢ, all in the same segment;
otherwise ω and α are missing.
A frame that is not usable but lies inside a segment gets ω and α from its neighbours.
Values from overlapping windows are correlated;
a fit across frames must use the angle, not the frame-wise speed.

### 9.8 Ground motion (floor runs)

The pixel scale at the marker is ρ = `marker_radius` / a, where a is the ellipse's semi-major axis in px (§9.5), with its uncertainty from both.
The ground position xᵢ is the sum, over the steps from the first frame, of ρ times the hub displacement projected onto the principal axis of travel, signed so that the net displacement is positive;
summing per step applies the scale where it holds as the perspective changes.
The ground speed v is its local slope as in §9.7.
The slip is 1 − v / (`wheel_radius` · ω), with both radii from §8.2.

## 10. Input and synchronisation

### 10.1 Applied duty

The duty that reaches the motor is the commanded throttle quantised by the driver.
With `quantisation = "pca9685"`, `adafruit_motor` sets the 16-bit duty cycle d₁₆ = ⌊65535 · |u|⌋ and the PCA9685 keeps 12 bits, so the applied duty is

  u_a = sign(u) · ⌊(d₁₆ + 1) / 16⌋ / 4096,

which gives u_a = sign(u) for |u| = 1.
With `quantisation = "count255"`, the script rounded to u = round(255 u)/255 before the driver, and the formula above is applied to that value;
Python's `round`, like numpy's, rounds halves to even.
A duty of zero is a brake, not an open circuit:
`adafruit_motor` then switches both driver inputs fully on, which shorts the motor, while a non-zero duty lets the motor coast during the off part of each PWM period (fast decay).
The quantisation step, 1/4096, is the resolution of u_a;
the motor voltage that results is not measured, so the supply's variation is not part of u_a.
The formula is to be checked against the library versions in `setup.toml`.

### 10.2 LED edges in the video

For each LED block, the step signal is the mean of the next four frames minus the mean of the previous four.
The block whose edges best match the Pi's LED schedule is used for the run.
An edge lies between the last frame at the old level and the first at the new;
its time is the midpoint, with the standard deviation of a uniform distribution over the interval, Δt / √12.
A frame at an intermediate level was exposed while the LED switched, and narrows the interval when the exposure time is known.

### 10.3 Clock mapping

With Pi edge times τₖ from the log and video edge times tₖ, the two clocks are related by

  tₖ = a + (1 + δ) τₖ,

fitted by weighted least squares over all edges of the run.
δ is the relative drift of the clocks;
with edges spanning less than 10 s, δ is set to 0 with a standard deviation of 1 × 10⁻⁴, the sum of the tolerances of two ordinary quartz clocks.
Over a longer span, δ is fitted with that tolerance as a prior, δ ~ N(0, (1 × 10⁻⁴)²), counted as one observation, and the weights are divided by max(1, χ²/ν) of the fit until it settles;
events that scatter more than their stated uncertainties, such as departures whose bias varies with the duty (§10.5), then cannot give a drift that no pair of quartz clocks has, while many LED edges over a long span outweigh the prior.
The residuals give the timing uncertainty σ_τ, which includes the time the Pi call took (§7.1) and the rolling shutter (§9.1).
Each logged command at Pi time τ is placed at video time a + (1 + δ) τ, with the uncertainty from the fit.

### 10.4 Backup without a Pi log

When the log is missing, the command schedule is reconstructed from the trial's script, commit, and arguments in `trials.csv`, by the same function the script calls, with the script's nominal times.
The Pi's sleep timing makes these times uncertain by a few milliseconds per change;
the code takes 2 ms unless the script states better.
The schedule is then aligned by the LED as in §10.3 (`script+led`).

### 10.5 Backup without a sync LED

Without a readable LED, the schedule is aligned by the motion it causes:
the first frame where ω leaves its rest level by more than five times σ_ω after an anchor pulse or a step (`log+motion` or `script+motion`).
The alignment then absorbs the motor's delay, so the delay and the zero-crossing time are not computed for these runs.
For logs without an anchor pulse, the alignment maximises the correlation between ω and the logged duty staircase.

### 10.6 One clock per run

The trials of a run were sent by one Pi, whose clock runs on between them.
When every trial's log gives its start on that clock (`t_monotonic_s` of a `motor-trial` log, §7.2, or a `# start` to a fraction of a second, §7.1), the trials are placed on one time axis, and one clock (§10.3) is fitted for the whole run:

- **LED.**
  The LED schedules of all trials are matched to the video's LED edges as one schedule (§10.2), so every edge of the run constrains the offset and the drift, and a trial whose LED was not seen is placed by the others.
  The clock is accepted only if it matches at least half of the LED edges of the run's trials, which `runs.csv` lists as filmed;
  a schedule of identical trials repeats, so a few edges fit a shift by whole trials as well as the truth.
- **Motion.**
  The run's whole duty schedule is aligned to ω by the correlation of §10.5, without regard to the sign of ω, so the wheel must follow every trial at once;
  only lags that keep the schedule inside the video, give or take 1 s at each end, are searched, since `runs.csv` lists the trials a video shows.
  Each trial's first command is then placed by the departure of ω from rest (§10.5) found within 0.4 s of where that alignment puts it, and the clock is fitted to the departures found;
  with none, the correlation is the clock.

The run clock is used whenever it exists.
A trial that the run clock places entirely outside the video has no frames.
When the run's LED clock fails and every trial has its log, the motion clock of the run is used, not a per-trial LED alignment, and not a schedule rebuilt from `trials.csv` (§10.4), which stands in only for lost logs.
Logs without a shared clock, such as `daisy-task1` and the cherry formats (§7.2), keep the per-trial alignment of §10.2 to §10.5.

### 10.7 Comparison of the synchronisations

Every run is also synchronised in each other way its sources allow, trial by trial:
by the LED per trial (§10.2, for runs whose `sync` in `runs.csv` is an LED level), by the run's motion clock (§10.6), and by motion per trial (§10.5).
For each, the difference of the trial offsets from the clock used, the other minus the used, is recorded per trial (§11.2) and summarised per run in the file metadata (`sync_comparison`: number of trials, median, largest absolute difference, and the trial with it) and in the processing log.
Two LED alignments of a correct run agree to a few milliseconds.
A motion alignment absorbs the motor's delay (§10.5), so against the LED it lags by the delay plus the rise, a few tens of milliseconds;
a difference of seconds marks a wrong alignment, such as an LED clock shifted by whole trials or a motion clock that took a later level of a sweep for the first that moved the wheel.

## 11. Processed data

### 11.1 Per frame: `data/processed/db17-<robot>-<motor>-<id>/<run>.parquet`

| Column | Type | Unit | Content | Section |
|---|---|---|---|---|
| `frame` | uint32 | – | Frame index, as in the frame table | §3 |
| `t_s` | float64 | s | Time from the first frame | §9.1 |
| `trial` | string | – | Trial running at this frame, or null | §10 |
| `t_trial_s` | float64 | s | Time from the start of that trial on the video clock | §10.3 |
| `usable` | bool | – | Marker found, and hub found or, on the stand, imputed; on a floor run with spoke columns, spokes seen | §9.2, §9.5 |
| `hub_moved`, `hub_imputed` | bool | – | Hub flags | §9.4 |
| `theta_rad`, `theta_sd_rad` | float64 | rad | Corrected angle and its standard deviation | §9.5, §9.6 |
| `omega_rad_s`, `omega_sd_rad_s` | float64 | rad/s | Angular velocity | §9.7 |
| `alpha_rad_s2`, `alpha_sd_rad_s2` | float64 | rad/s² | Angular acceleration | §9.7 |
| `u_cmd` | float64 | – | Commanded duty at this frame | §10 |
| `u` | float64 | – | Applied duty at this frame | §10.1 |
| `x_ground_m`, `x_ground_sd_m` | float64 | m | Ground position, floor runs only | §9.8 |
| `v_ground_m_s`, `v_ground_sd_m_s` | float64 | m/s | Ground speed, floor runs only | §9.8 |

The file metadata records the processing parameters (thresholds, windows), k, the ellipse, the clock used (`clock`: scope, method, and for a run clock its offset, drift, timing, number of events, and span), the comparison of §10.7 (`sync_comparison`), and the git commit of the code.
A floor run's metadata also records the source of its wheel angle (`wheel_angle`: `"spokes"` or `"marker"`, §9.5).
Its `schema_version` is `"2.1.0"`.

### 11.2 Per trial: `data/processed/db17-<robot>-<motor>-<id>/trials.parquet`

| Column | Content |
|---|---|
| `trial`, `run` | Identifiers |
| `sync` | Synchronisation level used (§10) |
| `align` | Alignment method: `led`, `motion`, or `correlation` (§10.2, §10.5) |
| `offset_s`, `offset_sd_s` | Video time of the trial start, a + (1 + δ) τ₀ |
| `drift`, `drift_sd` | Clock drift δ |
| `timing_sd_s` | Residual timing uncertainty σ_τ |
| `led_block` | LED block used |
| `clock` | `run` for one clock for the whole run (§10.6), `trial` for a clock per trial |
| `n_sync_events` | Events of this trial that the clock was fitted to: its matched LED edges, or 1 for its departure from rest; 0 when the other trials of the run placed it or its clock rests on the correlation alone (§10.5) |
| `offset_run_led_s`, `offset_trial_led_s`, `offset_run_motion_s`, `offset_trial_motion_s` | Video time of the trial start by each way of §10.7, null where that way was not possible or failed; the way used equals `offset_s` |
| `harm_a1` … `harm_b3`, `harm_k` | Once-per-turn coefficients and k |
| `n_frames`, `n_usable`, `n_dropped` | Frame counts |

Quantities computed from these, such as steady speeds, time constants, thresholds, and model fits, are results:
the analysis scripts compute them into `build/`, and they are not stored.

## References

Halíř, R., & Flusser, J. (1998). Numerically stable direct least squares fitting of ellipses. In *Proceedings of the 6th International Conference in Central Europe on Computer Graphics and Visualization (WSCG '98)* (pp. 125–132). https://autotrace.sourceforge.net/WSCG98.pdf

Joint Committee for Guides in Metrology. (2008). *Evaluation of measurement data: Guide to the expression of uncertainty in measurement* (JCGM 100:2008). BIPM. https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf
