# branca analytics: ball possession, Phase 1

Status: skeleton plus what v1 (E3) decided. Sections marked TODO are filled in later stages.

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
| broadcast ATD / homography | not used: not synchronised with the tactical timeline |

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
- TODO: evaluate and reduce these on the full match without labels (consistency checks only).

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
