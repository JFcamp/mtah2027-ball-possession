# branca analytics: ball possession, Phase 1 (More than a Hack 2027)

**Submission file:** `submissions/branca analytics.csv`, produced by pipeline **v4** with one command.
SHA-256 `1caa369c9258f02ca7337e75b5256b300205a476e1cab666e29187f9326947be`.
**Repository:** https://github.com/JFcamp/mtah2027-ball-possession (public, no match data inside).

Summary. For every frame of the tactical video we output H, A, D or X. A dead-ball mask (six rules,
one of them a small model of the players) splits the match into live spells; inside a live spell a
control-and-hysteresis state machine decides who owns the ball. The whole-match output has 353
possessions of 10.5 s on average, 33.8% H, 24.7% A and 41.5% D of the 158,857 graded frames. The
public 5-minute window, reported only, gives macroF1 0.748 and spellF1 0.489 (bootstrap 5-95%:
0.34 to 0.63 with 17 possessions, so it cannot rank anything).

## 1. Approach and why

The ranking metric counts whole possessions, so the pipeline separates two questions that fail in
different ways.

1. **Is the ball alive?** Stoppages are long and well anchored (ball out of the pitch, Smart Tagging
   set pieces and goals, kick-offs), so rules locate them and their restarts. Where the ball is not
   tracked for 5 s or more, a logistic model that sees only the players (speeds, share of players
   walking or running, movement and compactness of each team, over 1 s and 5 s windows) decides if
   play is stopped.
2. **Who has the ball in a live spell?** A team controls the ball when its nearest player is within
   1.0 m and at least 0.3 m closer than the nearest opponent. Ownership changes after 6 frames of
   control by the other team, or after two separate touches; touches, deflections and aerial duels
   alone never change it. If the ball is not tracked the owner is kept; a change inside a gap is
   placed at the phase change of the Smart Tagging if there is one, else in the middle of the gap.

Rules give structure and match the task definition (`CHALLENGE.md`); the only learned component is
the stopped-ball model, trained on other matches. Everything is deterministic (no random numbers).

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (tracking, `.txt` + `.xml`) | ball and 46 player slots in metres; team of each slot from the metadata |
| tactical Smart Tagging | anchors of set pieces and goals (start, team); phase tags as a weak possession cue |
| tactical homography, both videos | not used |
| broadcast ATD and homography | tried (v3), dropped: see section 6 |
| Metrica sample-data games 1-3 | other matches: validation and training of the stopped-ball model (section 4) |

The ATD `speed` field is not used: it was checked against positions (same unit, median ratio 0.99)
and positions give a smoothing we control.

## 3. How the public ground truth was used

(i) To check that the pipeline is aligned and well formed. (ii) One error analysis, read to
understand how cases are labelled (`docs/error_analysis_v1.md`, on v1). It showed that most missed
dead-ball frames have the ball visible and moving (67% with nobody controlling it, 23% with a team
controlling it) and that possession switches are mostly displaced by 1-5 s. That analysis motivated
**which mechanisms were put into the calibration grid** (control radius, margin, minimum control,
two touches, back-dating, minimum possession; a model for stopped play). (iii) **No threshold,
configuration or version was chosen with it.** Every choice was made on the external games with
decision rules fixed beforehand; the public score is only reported.

## 4. Training data

- **Source:** the three public sample games of Metrica Sports,
  https://github.com/metrica-sports/sample-data (Sample_Game_1, 2, 3). Other matches, not BAR-ATH.
- **Who labelled:** Metrica Sports (corrected tracking and event data). We labelled nothing by hand;
  our labels are derived by code (`src/external.py`): *stopped* = the corrected tracking has no ball
  inside a period; *H/A* = team of the latest ball event (pass, shot, recovery, carry, set piece) of the
  live spell, held until the opponent has a ball event of its own.
- **How much:** 3 games, 429,920 graded frames at 25 fps (about 4.8 hours), 37.4% stopped.
- **What for:** (a) train the stopped-ball model on the three games (regularised logistic regression
  in numpy, 20 player features, balanced classes, 25 Newton steps); weights in `models/dead_model.json`;
  (b) select the possession configuration (324 configurations; chosen on two games, tested on the third,
  with the true ball and with 40% of the live-play ball frames erased in long gaps, E2 gap
  distribution); (c) compare the model with the pace rule it replaces, leave-one-game-out.
- **Degraded copies (v5 attempt, not in the submitted file).** To test robustness, the external games were also
  degraded towards BAR-ATH statistics measured without labels (`src/degrade.py`, `docs/domain_gap.md`): smooth
  position errors of 0.17 m (players) and 0.47 m (ball), the dropout pattern of the tracked players
  (mean 20.5, sd 3.3), and the ball erased in 10.6% (the share of v4's H/A frames without a ball) or 40% of the live
  frames. Wrong-ball episodes were not simulated (they cannot be estimated reliably). The model features of the clean
  games were already close to BAR-ATH (mean absolute standardised difference 0.116; 0.108 after degrading).
- **Definition differences from `CHALLENGE.md`:** (1) possession here is "last ball event" (a touch),
  not controlled possession; a recovery lost at once counts as a change. (2) The external tracking is
  corrected, BAR-ATH's is automatic: noisier positions and speeds. (3) In the external games the dead
  ball starts when the corrected tracking drops the ball, often after it left the pitch (the ball is
  still present 1-3 s after a BALL OUT event in 33-54% of the cases), whereas BAR-ATH's ground truth
  marks dead ball from the tracking. (4) No Smart Tagging there, so the logic ran without restart
  teams. (5) Pitch 105 x 68 m instead of 105.3 x 68.
- **Checks of the labels** (`docs/external_data.md`): the ball is present in 100% of the frames of
  passes, shots and carries; absent in 95-98% of the 1-3 s before a restart; the official baseline logic
  scores spellF1 0.09-0.13 (raw) and 0.41-0.46 (smooth50) on them, the same order as on BAR-ATH.

## 5. Rules and thresholds of v4

Dead ball: R1 from the start of each half to the kick-off restart; R2 ball more than 0.5 m outside the
lines until it is back inside and moving; R3 set pieces: the tag start lies in the middle of the
stoppage, so D runs from the last live frame before it to the restart; R4 goal to the kick-off restart;
R5 ball untracked for 5 s or more: D where the model says P(stopped) >= 0.5 (v1: players' median speed
< 1.4 m/s); R6 ball still for 2 s. Dead runs shorter than 2 s are dropped. Origin: **P** physical,
**F** football reasoning, **D** tracking diagnostic of BAR-ATH (no labels), **X** external calibration.

| parameter | value | origin |
|---|---|---|
| ball: impossible jump | > 35 m/s between samples | P (fastest kicks 36-38 m/s) |
| ball: minimum run kept | 5 samples (0.2 s) | P |
| ball: interpolated gap | <= 25 frames (1 s) | F (a pass lasts 0.5-1.5 s) |
| ball: smoothing | 5 frames (0.2 s) | P (tracking noise vs. kick duration) |
| player speed: central difference, cap | +/-2 frames, 12 m/s | P |
| activity smoothing, minimum tracked | 1 s, 6 players | F |
| ball out margin | 0.5 m | P (radius 0.11 m + tracking noise) |
| restart: motion, duration | > 2 m/s for 0.5 s (onset only) | P (walking <= 2 m/s, kicks > 5 m/s) |
| restart search, fallback | 8 s, 3.5 s after the tag start | F, D (kick-off tag measured) |
| R3 look-back cap | 8 s | D (ball leaves ~3 s before the tag start) |
| live frame (R3) | ball > 3 m/s inside, or player pace >= 2 m/s | P, F |
| R2 maximum stoppage | 40 s | F (throw-in, corner, goal kick: 10-40 s) |
| R4 window around the goal tag | -2 s to +15 s | D |
| R5 minimum ball gap | 5 s | F |
| model decision, ridge, Newton steps | P >= 0.5, 1.0, 25 | X (balanced classes; fixed budget) |
| features: walking, running pace | < 1.4 m/s, > 4 m/s | P |
| features: windows, acceleration span | 1 s and 5 s, +/-5 frames | F |
| R6: still ball, duration | < 0.5 m/s for 2 s | F |
| minimum dead run | 2 s | F |
| control radius | 1.0 m | X (grid 1.0, 1.5, 2.0) |
| margin over the opponent | 0.3 m | X (grid 0, 0.3, 0.6; ~tracking noise) |
| minimum control to switch | 6 frames | X (grid 6, 12, 25) |
| two touches | on, within 3 s | X (grid on/off), F |
| back-dating to the first touch | off | X (grid on/off) |
| minimum possession | 25 frames (1 s) | X (grid 25, 38, 75) |
| control gap merged | 5 frames (0.2 s) | P (occlusion, jitter) |

## 6. Attempts discarded

- **v2, learned live/stopped model on algorithmic pseudo-labels of BAR-ATH** (stopped: ball out, still
  ball, set piece to restart; live: ball moving under control; no human label) with an HMM. Rule fixed beforehand: replace v1 only if the model beats the pace
  rule on the held-out half in both directions. It lost in one (AUC 0.898 vs 0.909, balanced accuracy
  0.829 vs 0.833). The idea returned in v4 with real labels from other games.
- **v3, broadcast ball fusion.** Broadcast frames were aligned to tactical frames by player positions only
  (91.1% of the broadcast frames with a valid homography; offsets constant to a few frames, 4014 and 26920;
  players agree to a median of 0.29 m). Rule: median ball distance <= 2 m and p90 <= 5 m, and ball
  coverage +5 points. Median 0.79 m, but p90 5.94 m, and coverage rose only from 59.3% to 63.2% (+3.9),
  because the broadcast tracker loses the ball where the tactical one does.

- **v5, calibration under degradation (E6), v4 kept.** Four pre-registered checks on the external games. (B) A
  possession configuration chosen on two games lost to v4 on the third (game 1: 0.695 vs 0.700 with 10.6% ball loss), so
  the v4 configuration stays; the radius 1.0 m was the best of 0.75-2.5 m in all three selections. (C) A gradient-boosting
  model on 65 features passed (degraded AUC 0.919 / 0.955 / 0.957 against 0.913 / 0.946 / 0.934 for the v4 model) but the
  end-to-end check on the complete pipeline failed on two held-out games (differences of 0.0004 in DEG_BAR and 0.011 in
  DEG_40), so the v4 model and decoder stay. (D) Applying the v4 model to every frame failed the transfer test on
  algorithmic weak labels of BAR-ATH (mean hit rate over the stoppage sources 0.67 < 0.70; it misses the half openings,
  0.43). The unused gradient-boosting model reached AUC 0.973 and mean hit 0.757 on the same labels. Decisions:
  `docs/decision_v5.md`; code: `src/lab.py`, `src/features_v5.py`, `src/models_v5.py`, `src/decode_v5.py`.

## 7. Results

External games, true dead-ball mask (spellF1, game 1 / 2 / 3):

| | true ball | 40% of live ball erased |
|---|---|---|
| v1 configuration | 0.752 / 0.755 / 0.736 | 0.626 / 0.650 / 0.605 |
| selected on the other two games | 0.810 / 0.794 / 0.756 | 0.660 / 0.679 / 0.607 |

Stopped-ball model vs. the pace rule, leave-one-game-out, AUC: 0.917 / 0.950 / 0.950 against
0.867 / 0.898 / 0.926; balanced accuracy 0.860 / 0.893 / 0.877 against 0.806 / 0.816 / 0.854.

BAR-ATH, whole match (no ground truth): v1 339 possessions, 40.7% D; v4 353 possessions, 41.5% D; the
two differ on 9.07% of the graded frames. Public window: v1 macroF1 0.751, spellF1 0.432; v4 macroF1
0.748, spellF1 0.489 (reported only).

## 8. Limitations

- The ball is missing in 41% of the graded frames, mostly in gaps over 5 s. There the owner is kept and
  stopped play is inferred from the players only (AUC 0.92-0.95 on other games). With 40% of the live
  ball erased the external spellF1 drops by about 0.10-0.13.
- **One test of the possession selection passed by a minimal margin:** game 3 with erased ball, 0.6073
  against 0.6047 for v1 (the other five comparisons gained 0.02-0.06).
- **Values at the edge of the grid:** control radius 1.0 m, minimum control 6 frames and minimum
  possession 25 frames are the lowest values tried; the grid was not extended, so the optimum may lie
  beyond.
- **Transfer risk:** the radius of 1.0 m and the model were validated on corrected tracking; BAR-ATH is
  automatic, so a 1.0 m radius may be too tight there. A simulated degradation (smooth errors of 0.17 m, player
  dropouts, 10.6% or 40% of the live ball erased) did not favour a larger radius: 1.0 m was chosen in all three
  cross-game selections. The degradation itself rests on the disagreement between the two BAR-ATH cameras and cannot
  be validated against the true error of the tracker.
- **Known weakness of the stopped-ball model:** on algorithmic weak labels of BAR-ATH the v4 model finds only 43% of the
  frames before the opening kick-off of each half and 67% on average over four stoppage sources (v5 analysis, `docs/decision_v5.md`).
  A better model existed (see section 6) but did not pass the pre-registered end-to-end check.
- The external labels are event-based, the BAR-ATH ones tracking-based (section 4); the model's stopped
  label (ball absent) also differs from BAR-ATH's dead ball.
- Stoppage starts rely on the Smart Tagging, itself automatic; the restart is found from the ball only
  when it is seen. The Smart Tagging team of corners agrees with the first controller in 3 of 10 cases.
- With 17 possessions the public window cannot separate versions.

## 9. Reproduction

```bash
python -m venv .venv && .venv/Scripts/activate        # Python 3.12 (tested 3.12.3)
pip install -r requirements.txt                        # numpy==2.5.4, pandas==3.1.1
# unpack the data bundle into data/ (or set MTAH_DATA_DIR), see data/README.md, then:
python -m src.run --out "submissions/branca analytics.csv"
python evaluation/validate_submission.py "submissions/branca analytics.csv"
```

Verified on a clean clone of the public repository, with only the match data supplied: the output has
the SHA-256 above and equals the committed file byte for byte (also without the tracking cache).
`models/dead_model.json` (SHA-256
`93587ee2b9798ed77e90437125fc817072b4828d6ed9a41c7da3e16c93f888ff`) is regenerated identically by `python scripts/train_dead_model.py` after cloning the sample
data into `data_external/sample-data`; the pipeline does not need that folder. Tests:
`pip install -r requirements-dev.txt && python -m pytest`. Reports: `docs/report_v1.md` to `report_v4.md`.

## 10. Use of AI assistants

AI assistants were used for two purposes: fixing bugs in the code and supporting the team's strategic
decisions about which path to take. In line with the competition rules, no videos, datasets,
confidential information or other restricted materials provided for the competition, nor any extracts
of them, were uploaded, disclosed or otherwise made available to any AI assistant. The team checked
the results against the competition rules and is responsible for the submission.
