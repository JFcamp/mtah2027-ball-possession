# Error analysis of v1 on the public window

Frames 11185-18684 (7500 graded frames, 300.0 s). Prediction = `submissions/v1.csv` pipeline. The public ground truth is read only to understand how cases are labelled; nothing in the pipeline changed and no threshold value is proposed here.

## 1. Dead-ball stretches of the ground truth

4 stretches. 'covered' is the share of the stretch that v1 also calls D.

| stretch (frames) | length (s) | covered % | pred D start - GT start (f) | pred D end - GT end (f) | nearest set-piece tag (start - GT start) | main reason for the missed part |
|---|---|---|---|---|---|---|
| 12851-12988 | 5.5 | 46 | +64 | -10 | none within 60 s | ball unseen (owner kept) |
| 15889-16399 | 20.4 | 0 | - | - | none within 60 s | ball seen, moving, nobody controls it |
| 17380-17837 | 18.3 | 80 | +78 | -12 | Free Kick (+153 f) | ball seen, moving, nobody controls it |
| 18027-18388 | 14.5 | 72 | +165 | -34 | Goal Kick (+293 f) | ball seen, moving, nobody controls it |

Remainder of the stretches (frames that the GT calls D and v1 does not):

| what the tracking shows at those frames | frames | % |
|---|---|---|
| ball seen, moving, nobody controls it | 522 | 67.1 |
| ball seen, moving, a team controls it | 182 | 23.4 |
| ball unseen (owner kept) | 64 | 8.2 |
| ball seen, still (<0.5 m/s) but not long enough for R6 | 10 | 1.3 |

Frames both call D, credited to the first rule (R1, R4, R3, R2, R6, R5) that marks them:

| rule | frames |
|---|---|
| R1 | 0 |
| R4 | 0 |
| R3 | 326 |
| R2 | 96 |
| R6 | 0 |
| R5 | 269 |

v1 calls D where the ground truth says H or A on 69 frames, by rule: R2 1, R5 68.

## 2. Possession switches H<->A of the ground truth

12 direct switches in the ground truth, 14 direct switches predicted in the same window. Delay = predicted frame - true frame (negative = v1 early).

| GT switch frame | GT | nearest predicted switch | delay (f) | result | what the tracking shows |
|---|---|---|---|---|---|
| 11351 | A->H | 11469 | +118 | switch predicted late by 4.7 s | ball seen but no control episode by the new team in -2..+4 s |
| 11553 | H->A | 11469 | -84 | switch predicted early by 3.4 s | new team touched it briefly (5 f at most), under the 0.5 s hysteresis |
| 11624 | A->H | 11706 | +82 | switch predicted late by 3.3 s | new team had a control episode >= 0.5 s (first at -43 f); owner already new or switch placed elsewhere |
| 12665 | H->A | 12626 | -39 | switch predicted early by 1.6 s | new team had a control episode >= 0.5 s (first at -39 f); owner already new or switch placed elsewhere |
| 13704 | H->A | 13671 | -33 | switch predicted early by 1.3 s | new team had a control episode >= 0.5 s (first at -33 f); owner already new or switch placed elsewhere |
| 13785 | A->H | 13840 | +55 | switch predicted late by 2.2 s | new team had a control episode >= 0.5 s (first at +55 f); owner already new or switch placed elsewhere |
| 14846 | H->A | 14845 | -1 | matched within 1 s |  |
| 15346 | A->H | 15651 | +305 | no switch predicted within 10 s | new team touched it briefly (7 f at most), under the 0.5 s hysteresis |
| 15365 | H->A | 15651 | +286 | no switch predicted within 10 s | new team had a control episode >= 0.5 s (first at +28 f); owner already new or switch placed elsewhere |
| 15766 | A->H | 15707 | -59 | switch predicted early by 2.4 s | ball seen but no control episode by the new team in -2..+4 s |
| 15799 | H->A | 15707 | -92 | switch predicted early by 3.7 s | new team had a control episode >= 0.5 s (first at -22 f); owner already new or switch placed elsewhere |
| 17293 | A->H | 16288 | -1005 | no switch predicted within 10 s | new team touched it briefly (10 f at most), under the 0.5 s hysteresis |

## 3. Predicted switches with no ground-truth switch within 1 s

| frame | pred | what triggered it | ground truth at that frame |
|---|---|---|---|
| 11469 | A->H | new team controlled the ball for >= 0.5 s (47 f) | GT already has H (v1 late/early by more than 1 s) |
| 11706 | H->A | new team controlled the ball for >= 0.5 s (19 f) | GT keeps H |
| 11842 | A->H | new team controlled the ball for >= 0.5 s (32 f) | GT already has H (v1 late/early by more than 1 s) |
| 12434 | H->A | new team controlled the ball for >= 0.5 s (22 f) | GT keeps H |
| 12479 | A->H | new team controlled the ball for >= 0.5 s (36 f) | GT already has H (v1 late/early by more than 1 s) |
| 12626 | H->A | two touches / short touch followed by control (first touch 3 f) | GT keeps H |
| 12733 | A->H | two touches / short touch followed by control (first touch 2 f) | GT keeps A |
| 13671 | H->A | two touches / short touch followed by control (first touch 7 f) | GT keeps H |
| 13840 | A->H | new team controlled the ball for >= 0.5 s (34 f) | GT already has H (v1 late/early by more than 1 s) |
| 15651 | A->H | new team controlled the ball for >= 0.5 s (35 f) | GT keeps A |
| 15707 | H->A | two touches / short touch followed by control (first touch 3 f) | GT already has A (v1 late/early by more than 1 s) |
| 16044 | A->H | new team controlled the ball for >= 0.5 s (31 f) | GT has a dead ball |
| 16288 | H->A | new team controlled the ball for >= 0.5 s (30 f) | GT has a dead ball |

## 4. Summary by category

**Ground-truth dead-ball stretches (4)**

| category | stretches |
|---|---|
| partly covered (30-90%) | 3 |
| mostly missed (< 30%) | 1 |

**Frames of the missed dead ball by what the tracking shows**

| category | frames | % |
|---|---|---|
| ball seen, moving, nobody controls it | 522 | 67.1 |
| ball seen, moving, a team controls it | 182 | 23.4 |
| ball unseen (owner kept) | 64 | 8.2 |
| ball seen, still (<0.5 m/s) but not long enough for R6 | 10 | 1.3 |

**Ground-truth possession switches (12)**

| result | what the tracking shows | switches |
|---|---|---|
| matched within 1 s | - | 1 |
| none predicted within 10 s | new team controlled it >= 0.5 s: timing / placement of the switch | 1 |
| none predicted within 10 s | new team only touched it briefly (hysteresis) | 2 |
| predicted 1-10 s away | ball seen, new team never within control distance | 2 |
| predicted 1-10 s away | new team controlled it >= 0.5 s: timing / placement of the switch | 5 |
| predicted 1-10 s away | new team only touched it briefly (hysteresis) | 1 |

**Predicted switches without a match (13)**

| trigger | ground truth at that frame | switches |
|---|---|---|
| new team controlled the ball for >= 0.5 s | GT already has H | 4 |
| new team controlled the ball for >= 0.5 s | GT keeps H | 2 |
| two touches / short touch followed by control | GT keeps H | 2 |
| new team controlled the ball for >= 0.5 s | GT has a dead ball | 2 |
| two touches / short touch followed by control | GT keeps A | 1 |
| new team controlled the ball for >= 0.5 s | GT keeps A | 1 |
| two touches / short touch followed by control | GT already has A | 1 |
