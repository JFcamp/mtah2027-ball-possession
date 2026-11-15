"""High-confidence live / stopped pseudo-labels, produced by algorithms only.

No human label is involved and the public ground truth is not read. The labels come from the ball
and from the Smart Tagging structure; they are used to train a model that sees only the players
(src/features_players.py), so the model can then say live / stopped where the ball is unseen.

STOPPED (code 1)
  ball_out    the ball is outside the field (0.5 m margin) for at least 1 s: a blip of the tracker
              at the line is not a stoppage
  still       the ball inside the field below 0.5 m/s for at least 2 s (same rule as R6)
  set_piece   from the tag start of a throw-in / corner / goal kick / free kick to the detected
              restart (the tag start is inside the stoppage, E2 diagnostics)
  goal        from the goal to the restart of the following kick-off (R4)
  half_start  from the first graded frame of a half to the restart of the opening kick-off (R1)
LIVE (code 0)
  the ball is inside the field, seen, moving at >= 1 m/s, and some team has a control episode in
  progress (nearest player within 1.5 m and clearly nearer than the opponent): someone is playing it.
  1 m/s is twice the "still" threshold and walking pace; a carried or passed ball is above it.
Everything else is unlabelled (-1). Where a stopped source and the live rule overlap (for example
the ball carried back to the centre circle after a goal), the stopped source wins: it comes from the
structure of the match, the live rule only from a few seconds of ball motion.
"""

from typing import Dict, Tuple

import numpy as np

from . import dead_ball as db
from . import features as feat
from . import io as sio
from . import possession as pos

UNLABELLED, LIVE, STOPPED = -1, 0, 1
OUT_MIN = 25          # frames (1 s)
LIVE_MIN_SPEED = 1.0  # m/s


def build(track, features: feat.Features, intervals, dead_v1: db.DeadBall) -> Tuple[np.ndarray, Dict[str, int]]:
    n = sio.N_FRAMES
    graded = sio.graded_mask()
    present = np.isfinite(track.xy[:, 0])
    out = features.ball_out
    speed = track.speed
    moving = db.sustained_motion(present, out, speed)

    sources = {}
    # ball out for >= 1 s
    m = np.zeros(n, dtype=bool)
    for a, b in db.runs(out & graded):
        if b - a >= OUT_MIN:
            m[a:b] = True
    sources["ball_out"] = m
    # ball still for >= 2 s inside the field
    sources["still"] = dead_v1.by_rule["R6"] & ~out
    # set pieces: tag start -> detected restart
    m = np.zeros(n, dtype=bool)
    for iv in intervals:
        if iv.code == "SET PIECES" and "Kick Off" not in (iv.tags.get("Type") or ""):
            m[iv.start_frame:db.restart_after(iv.start_frame, moving)] = True
    sources["set_piece"] = m
    sources["goal"] = dead_v1.by_rule["R4"]
    sources["half_start"] = dead_v1.by_rule["R1"]

    stopped = np.zeros(n, dtype=bool)
    for mask in sources.values():
        stopped |= mask
    stopped &= graded

    control = pos.control_teams(features, present)
    in_episode = np.zeros(n, dtype=bool)
    for _, first, last in sio.half_blocks():
        for _, s, e in pos.episodes(control, first, last + 1):
            in_episode[s:e + 1] = True
    live = (present & ~out & np.isfinite(speed) & (speed >= LIVE_MIN_SPEED) & in_episode & graded)

    labels = np.full(n, UNLABELLED, dtype=np.int8)
    labels[live] = LIVE
    labels[stopped] = STOPPED
    counts = {name: int((mask & graded).sum()) for name, mask in sources.items()}
    counts["stopped (union)"] = int(stopped.sum())
    counts["live"] = int(live.sum())
    counts["live overridden by stopped"] = int((live & stopped).sum())
    counts["unlabelled graded"] = int((graded & (labels == UNLABELLED)).sum())
    return labels, counts
