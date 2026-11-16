# branca analytics: ball possession, Phase 1

Submitted file: `submissions/branca analytics.csv`, produced by pipeline **v4** (`python -m src.run`).
The earlier versions are kept for the record (sections 5b-5d). Public 5-minute window, reported only
and never used to choose anything (17 possessions, so differences below ~0.3 spellF1 are noise):
v4 macroF1 0.748, spellF1 0.489; v1 macroF1 0.751, spellF1 0.432. Whole match v4: 353 possessions, mean
10.5 s, 33.8% H / 24.7% A / 41.5% D of the graded frames.

## 1. Approach and why

The label of a frame is decided in two steps, because the metric ranks on possessions (`spellF1`) and
the two questions fail in different ways.

1. **Is the ball alive?** A dead-ball mask from six rules (half boundaries, ball out of the field,
   Smart Tagging set pieces, goals, long ball gaps, a ball that stays still). Stoppages are long, so
   they are easy once the restart is located; they split the match into live segments. In the long
   gaps where the ball is not tracked (5 s or more), a small logistic model of the **players only**
   (speeds, how many walk or run, team movement and compactness over 1 s and 5 s windows) decides
   whether play is stopped.
2. **Who has the ball in the live segments?** A control-and-hysteresis state machine. A team controls
   the ball when its nearest player is within 1.0 m and at least 0.3 m closer than the nearest
   opponent. Possession changes after 6 frames (0.24 s) of control by the other team or two separate
   touches. When the ball is not seen the owner is kept.

Rules provide the structure and the definition of the task (`CHALLENGE.md`); the only learned part is
the stopped-ball model, trained on other matches (section 3), and the possession parameters were
chosen by validation on those matches.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (`*_tactical_atd.txt`, `.xml`) | ball and 46 player slots in metres; team of each slot from the metadata |
| tactical Smart Tagging | set-piece and goal anchors (start, team), phase tags as a weak possession cue |
| tactical homography | not used |
| tactical / broadcast video | not used (the pipeline works on the tracking) |
| broadcast ATD / homography | tried in v3 (ball fusion) and not kept: aligned to the tactical timeline by player positions only (91% of the valid broadcast frames), but it adds only 3.9 points of ball coverage |
| Metrica sample-data, games 1-3 | other matches, used for validation and to train the stopped-ball model (section 3) |

The ATD `speed` field is not used (checked against positions: same unit, ratio 0.99; positions give a
smoothing we control).

## 3. Training data

- **Source**: the three public sample games of Metrica Sports,
  https://github.com/metrica-sports/sample-data (Sample_Game_1, 2 and 3; two in the CSV format, one
  in the EPTS FIFA format). They are other matches, not BAR-ATH. The data is downloaded to
  `data_external/` (git-ignored, not redistributed).
- **Who labelled**: nobody by hand on our side. The tracking is the data provider's corrected tracking
  and the events are the provider's. The labels used are derived by code (`src/external.py`):
  *stopped* = the ball is absent from the corrected tracking inside a period; *H/A* = the team of the
  latest ball event (pass, shot, recovery, carry, set piece) of the live spell.
- **How much**: 3 games, 429,920 graded frames at 25 fps (about 4.8 hours), 37.4% stopped.
- **What for**: (a) the stopped-ball model (logistic regression, 20 player features, balanced classes,
  Newton iterations, no randomness) is trained on all three games with the *stopped* label, and the
  trained weights are in `models/dead_model.json` (regenerate with `python scripts/train_dead_model.py`
  after cloning the sample data); (b) the possession parameters (control radius, margin, minimum
  control, two touches, back-dating, minimum possession) were selected on these games, choosing on two
  and testing on the third (324 configurations, with the true ball and with 40% of the live-play ball
  frames erased in long gaps); (c) validation of the stopped-ball model against the pace rule it
  replaces, with the same leave-one-game-out scheme.
- **Not used**: no label, event or correction of BAR-ATH, nothing from its private ground truth, and
  the public 5 minutes were never used to select a parameter or a version; they are only reported.
- Definitions differ from `CHALLENGE.md` in details (events are touches, not necessarily controlled
  possession; the dead ball starts when the corrected tracking drops the ball, which can be seconds
  after it leaves the pitch). See `docs/external_data.md`.

## 4. Possession rules implemented (v4)

- **Dead ball (`src/dead_ball.py`)**: R1 start of each half until the kick-off; R2 ball more than
  0.5 m outside the lines until play restarts (ball back inside and moving above 2 m/s for 0.5 s);
  R3 set pieces: the tag start sits in the middle of the stoppage, so D runs from the last live
  frame before it (ball moving, or players jogging) to the restart; R4 goals: from the goal to the
  restart of the kick-off; R5 ball unseen for at least 5 s: D where the stopped-ball model gives
  P(stopped) >= 0.5 (v1 used the players' median speed below 1.4 m/s); R6 ball still for at least
  2 s. D runs shorter than 2 s are dropped.
- **Possession (`src/possession.py`)**: control radius 1.0 m with 0.3 m margin over the opponent;
  switch after 6 frames of control or two touches within 3 s; the switch is placed when the evidence
  is complete (no back-dating to the first touch); ball unseen: keep the owner, and place a change
  that happens inside a gap at the phase change if there is one, else in the middle of the gap; after
  a restart the taker is the Team of the set piece, or the first team to control the ball when it is
  N/A (throw-ins, first kick-off).
- **Loose balls, challenges, restarts**: a touch, a deflection or an aerial duel does not change the
  owner; a possession shorter than 25 frames (1 s) is absorbed by its neighbour
  (`src/postprocess.py`).

Thresholds and their justification are in the module docstrings (`src/ball.py`, `src/features.py`,
`src/dead_ball.py`, `src/possession.py`, `src/postprocess.py`); the values changed in v4 and why are in
`docs/report_v4.md`.

## 5. Limitations

- The ball is missing in 41% of graded frames, mostly in gaps longer than 5 s. Inside them the owner is
  kept and dead ball is inferred from the players only, with a model that reaches AUC 0.92-0.95 on
  other games, not perfect.
- The external games have corrected tracking and event-based labels; BAR-ATH has automatic tracking
  (noisier positions and speeds) and its labels come from corrected tracking. The model and the
  possession parameters were validated on the first kind of data and applied to the second. A control
  radius of 1.0 m, the lowest of the grid, was chosen; with noisier positions it may be too tight.
  The best values of three parameters lie at the edge of the grid, which was not extended.
- The stoppage start is inferred from an anchor that is itself automatic (Smart Tagging), and the
  restart is located from the ball only when it is seen. Restart times can be off by seconds.
- The Smart Tagging `Team` of corners is not confirmed by the tracking (the first controller after a
  corner is the tagged team in 3 of 10 cases); it is used as given.
- With 17 possessions the public window cannot rank versions; the choices were made on other games.

## 5b. v2 attempted and discarded

v2 replaced the pace rule R5 and the fixed look-back / stoppage caps by a learned live/stopped model:
algorithmic pseudo-labels (stopped: ball out for 1 s, ball still for 2 s, set-piece tag start to
restart, goal to kick-off, opening of each half; live: ball inside, moving, someone controlling it), a
logistic regression on player-only features and a two-state Viterbi HMM with football-duration
transitions (stopped 25 s, live 40 s on average). The decision rule was fixed beforehand: replace v1
only if the model beats R5 on the pseudo-labels of the held-out half in both directions. It lost in
one direction (AUC 0.898 vs 0.909), so v1 stayed. The idea survived in v4, trained on real labels
from other games. Report: `docs/report_v2.md`.

## 5c. Error analysis and broadcast ball fusion (v3), discarded

`docs/error_analysis_v1.md` compares v1 with the public 5 minutes (read only to understand the cases,
no threshold changed). The missed dead-ball frames are mostly frames where the ball is visible and
moving but nobody controls it (67%) or a team controls it (23%); possession switches are missed or
shifted mostly because the new team controlled the ball earlier or later than the predicted switch.
v3 completed the tactical ball with the broadcast ball, aligned to the tactical frames by player
positions only (offsets constant to within a few frames, players agree to a median of 0.29 m). The
rule fixed beforehand was not met: the co-visible ball p90 was 5.94 m (limit 5) and the ball coverage
rose only 3.9 points (limit 5). Report: `docs/report_v3.md`; code: `src/broadcast.py`, `src/align.py`,
`src/fusion.py`.

## 5d. Calibration on other games (v4), adopted

Rule fixed beforehand, using only the external games: adopt the new possession configuration if it
beats v1 in all three held-out games with and without artificial ball gaps (it did: spellF1 0.810 vs
0.752, 0.794 vs 0.755, 0.756 vs 0.736 with the true ball; 0.660 vs 0.626, 0.679 vs 0.650, 0.607 vs
0.605 with gaps), and adopt the stopped-ball model if it beats R5 on AUC and balanced accuracy in all
three held-out games (it did: AUC 0.917, 0.950, 0.950 vs 0.867, 0.898, 0.926). BAR-ATH plausibility
without ground truth: D 41.5% of the graded frames, 353 possessions against 339 for v1. Details,
including each component alone (v4b, v4c): `docs/report_v4.md`.

## 6. Reproduction

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use bin/activate elsewhere
pip install -r requirements.txt
# unpack the data bundle into data/ (see data/README.md), then:
python -m src.run --out "submissions/branca analytics.csv"
python evaluation/validate_submission.py "submissions/branca analytics.csv"
```

The command is deterministic: two runs give byte-identical CSVs (checked by `tests/test_pipeline.py`).
The tracking is cached in `data/cache/` after the first run. The stopped-ball model is in
`models/dead_model.json` (committed); it can be regenerated from the public sample games:
`git clone https://github.com/metrica-sports/sample-data data_external/sample-data` and
`python scripts/train_dead_model.py`. `python -m src.run --version v1` reproduces the earlier safety
submission. Tests: `pip install -r requirements-dev.txt && python -m pytest`.
