# R8 right wheel, motor m16

- Group: 04.
- Members (registered names): 陳晞, 賴鉅元, 林宇恒, 王文中, 陳易謙, 楊舒翔, 陳文康.
- Recording date: 2026-11-18.
- Setup: right wheel suspended, blue tape on the yellow wheel, red synchronisation LED visible on video. The Raspberry Pi used a laptop micro-USB connection. The motor supply voltage was not measured.
- E1 is the recorded forward and backward duty sweep, with 0.02 increments and a top magnitude of 0.36.
- E2 comprises three repeats of the step sequence 0, 0.60, 1.00, 0.60, 0.

The MP4s are retained outside Git under the course clip names in `runs.csv`.
Their SHA-256 checksums identify the original video bytes.
The frame tables store the original presentation times. Their pixel measurements were first tracked on a timestamp-only remux, and the original timestamps were restored after all encoded video packets were matched by SHA-256.

The MP4s play about eight times slower than recording time. Use real-time timestamps for physical analysis.
For E1, a single time-scale factor did not align the full sweep. The preliminary local analysis also used piecewise LED alignment against the Pi, with the timing between LED events estimated by interpolation.
No hand measurements are available.
