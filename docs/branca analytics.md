# branca analytics: ball possession, Phase 1

Status: describes the pipeline v1 (the submitted file at this point). Public 5-minute window,
reported only and never used to choose anything: macroF1 0.751, spellF1 0.432 (bootstrap 5-95%:
0.28 to 0.57, 17 possessions, so it is not distinguishable from the best baseline). Whole match:
339 possessions, mean 11.1 s, 33.9% H / 25.4% A / 40.7% D of the graded frames.

## 1. Approach and why

The label of a frame is decided in two independent steps, because the metric ranks on possessions
(`spellF1`) and the two questions have different failure modes.

1. **Is the ball alive?** A dead-ball mask built from six rules (half boundaries, ball out of the
   field, Smart Tagging set pieces, goals, long ball gaps while players walk, a ball that stays
   still). It is mostly structural: stoppages are long, so they are easy to get right once the
   moment of restart is located, and they split the match into live segments.
2. **Who has the ball in the live segments?** A control-and-hysteresis state machine. A team
   controls the ball when its nearest player is within 1.5 m and clearly closer than the nearest
   opponent. Possession changes only after 0.5 s of control by the other team or two separate
   touches, and the change is back-dated to the first control frame. When the ball is not seen,
   the owner is kept.

Everything is rule-based and unsupervised. A rule system was chosen over a learned model because
there is no labelled data we may use (see section 3), the rules are the definition of the task
(`CHALLENGE.md`), and every threshold can be justified in football or physical terms.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (`*_tactical_atd.txt`, `.xml`) | ball and 46 player slots in metres; team of each slot from the metadata |
| tactical Smart Tagging | set-piece and goal anchors (start, team), phase tags as a weak possession cue |
| tactical homography | not used in v1 |
| tactical / broadcast video | not used (not available locally in v1) |
| broadcast ATD / homography | tried in v3 (ball fusion) and not kept: aligned to the tactical timeline by player positions only (91% of the valid broadcast frames), but it adds only 3.9 points of ball coverage |

Dropped and why: the ATD `speed` field is not used for the activity measure (it was checked against
positions: same unit, ratio 0.99, but positions give a smoothing we control).

## 3. Training data

**None.** No labels of any kind, and no learning. The public 5-minute ground truth is used only to
report a score after the pipeline is fixed; no threshold was chosen or changed by looking at it.

## 4. Possession rules implemented (v1)

- **Dead ball (`src/dead_ball.py`)**: R1 start of each half until the kick-off; R2 ball more than
  0.5 m outside the lines until play restarts (ball back inside and moving above 2 m/s for 0.5 s);
  R3 set pieces: the tag start sits in the middle of the stoppage, so D runs from the last live
  frame before it (ball moving, or players jogging) to the restart; R4 goals: from the goal to the
  restart of the kick-off; R5 ball unseen for at least 5 s while the players walk (median speed
  below 1.4 m/s); R6 ball still for at least 2 s. D runs shorter than 2 s are dropped.
- **Possession (`src/possession.py`)**: control radius 1.5 m with 0.3 m margin over the opponent;
  switch after 0.5 s of control or two touches within 3 s; back-dating to the first control frame;
  ball unseen: keep the owner, and place a change that happens inside a gap at the phase change if
  there is one, else in the middle of the gap; after a restart the taker is the Team of the set
  piece, or the first team to control the ball when it is N/A (throw-ins, first kick-off).
- **Loose balls, challenges, restarts**: a touch, a deflection or an aerial duel does not change
  the owner; a possession shorter than 1.5 s is absorbed by its neighbour
  (`src/postprocess.py`).

Full list of thresholds and their justification: module docstrings of `src/ball.py`,
`src/features.py`, `src/dead_ball.py`, `src/possession.py`, `src/postprocess.py`.

## 5. Limitations (known after v1)

- The ball is missing in 41% of graded frames, mostly in gaps longer than 5 s. Inside them the owner
  is kept and dead ball is inferred only from the players' pace (R5), which is coarse.
- The stoppage start is inferred from an anchor that is itself automatic (Smart Tagging), and the
  restart is located from the ball only when it is seen. Restart times can be off by seconds.
- The Smart Tagging `Team` of corners is not confirmed by the tracking (the first controller after a
  corner is the tagged team in 3 of 10 cases); it is used as given.
- Next version (v2) replaces the pace rule R5 and the fixed look-back caps with a model of live /
  stopped play trained on algorithmic pseudo-labels; see the later sections once it is decided.

## 5b. v2 attempted and discarded (E4)

v2 replaced the pace rule R5 and the fixed look-back / stoppage caps by a learned live/stopped model:
algorithmic pseudo-labels (stopped: ball out for 1 s, ball still for 2 s, set-piece tag start to
restart, goal to kick-off, opening of each half; live: ball inside, moving, someone controlling it),
a numpy logistic regression on player-only features (speed percentiles, fraction walking/running,
team centroid speed, compactness, acceleration, number tracked, over 1 s and 5 s windows) and a
two-state Viterbi HMM with football-duration transitions (stopped 25 s, live 40 s on average).

The decision rule was fixed before looking at any public score: v2 replaces v1 only if the model
beats R5 on the pseudo-labels of the held-out half in both directions (AUC and balanced accuracy)
and the whole-match statistics stay plausible. Result: AUC 0.898 vs 0.909 and balanced accuracy
0.829 vs 0.833 when training on the 1st half and testing on the 2nd (R5 won); 0.920 vs 0.882 and
0.845 vs 0.787 in the other direction (model won). The first direction failed, so v1 stays the
submitted version. The whole-match statistics of v2 were plausible (39.9% D, 301 possessions, no
long unexplained stoppage). v2 is kept in the repository (`--version v2`, `submissions/v2.csv`)
and its report is `docs/report_v2.md`. The public window was reported only and played no part in
the decision.

## 5c. Error analysis and broadcast ball fusion (v3), discarded

`docs/error_analysis_v1.md` compares v1 with the public 5 minutes (read only to understand the cases,
no threshold changed). Main findings: the missed dead-ball frames are mostly frames where the ball is
visible and moving but nobody controls it (67%) or a team controls it (23%), and only 8% are frames
without a ball; possession switches are missed or shifted mostly because the new team controlled the
ball for at least 0.5 s earlier or later than the predicted switch, or only touched it briefly.

v3 completes the tactical ball with the broadcast ball. The broadcast frames are aligned to the
tactical ones without the ball: for blocks of 250 frames the offset that makes the same-team player
positions coincide (matching within 2.5 m) is searched around a coarse global estimate, and only
blocks that match strongly and lie at the dominant offset of their half are kept (replays are left
out). The offsets are constant to within a few frames (4014 in the first half, 26920 in the
second); players agree to a median of 0.29 m. The fused ball goes through the same cleaning as
before and the rest of v1 is unchanged.

The decision rule was fixed before looking at any v3 score. (a) Co-visible ball distance: median
0.79 m (limit 2 m) but p90 5.94 m (limit 5 m), fail. (b) Ball coverage on graded frames rose from
59.3% to 63.2% (+3.9 points, limit 5), fail, because the broadcast tracker usually loses the ball
where the tactical one does. (c) The whole-match statistics were plausible. v3 therefore did not
replace v1. Reports: `docs/report_v3.md`; code: `src/broadcast.py`, `src/align.py`,
`src/fusion.py`; `python -m src.run --version v3`.

## 6. Reproduction

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use bin/activate elsewhere
pip install -r requirements.txt
# unpack the data bundle into data/ (see data/README.md), then:
python -m src.run --out "submissions/branca analytics.csv"
python evaluation/validate_submission.py "submissions/branca analytics.csv"
```

The command is deterministic: two runs give byte-identical CSVs (checked by
`tests/test_pipeline.py`). The tracking is cached in `data/cache/` after the first run. Tests:
`pip install -r requirements-dev.txt && python -m pytest`.
