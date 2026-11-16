"""Algorithmic weak labels of BAR-ATH used ONLY to check that the stopped-ball model transfers.

No human label and no player feature is involved: the ball, the rules R1-R4 and the Smart Tagging.

WEAK STOPPED, by source: R1 (opening of each half), R2 (ball outside the field), the core of R3 (from the
tag start of a set piece to the detected restart), R4 (goal to kick-off).
WEAK LIVE: ball seen, inside the field, above 3 m/s, inside a running-play phase of the Smart Tagging
(BUILD UP, PROGRESSION, ATTACKING TRANSITION, DEFENSIVE TRANSITION, LONG BALL), more than 5 s away from every
weak-stopped frame and from every SET PIECES / GOALS interval.
"""

from typing import Dict, Tuple

import numpy as np

from . import dead_ball as db
from . import io as sio

RUNNING_PHASES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "DEFENSIVE TRANSITION", "LONG BALL")
FAR = 125          # 5 s
LIVE_SPEED = 3.0   # m/s


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    csum = np.concatenate(([0], np.cumsum(mask)))
    idx = np.arange(len(mask))
    lo, hi = np.maximum(0, idx - radius), np.minimum(len(mask), idx + radius + 1)
    return (csum[hi] - csum[lo]) > 0


def build(track, features, intervals) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """(stopped sources, weak live mask, weak live inside BUILD UP)."""
    n = sio.N_FRAMES
    graded = sio.graded_mask()
    rules = db.build(track, features, intervals, r5_stopped=np.zeros(n, dtype=bool))
    present = np.isfinite(track.xy[:, 0])
    moving = db.sustained_motion(present, features.ball_out, track.speed)
    core = np.zeros(n, dtype=bool)
    for iv in intervals:
        if iv.code == "SET PIECES" and "Kick Off" not in (iv.tags.get("Type") or ""):
            core[iv.start_frame:db.restart_after(iv.start_frame, moving)] = True
    sources = {"R1 half opening": rules.by_rule["R1"] & graded,
               "R2 ball outside": rules.by_rule["R2"] & graded,
               "R3 set-piece core": core & graded,
               "R4 goal to kick-off": rules.by_rule["R4"] & graded}
    stopped = np.zeros(n, dtype=bool)
    for m in sources.values():
        stopped |= m

    protected = np.zeros(n, dtype=bool)
    phase = np.zeros(n, dtype=bool)
    build_up = np.zeros(n, dtype=bool)
    for iv in intervals:
        sl = slice(max(iv.start_frame, 0), min(iv.end_frame, n))
        if iv.code in ("SET PIECES", "GOALS"):
            protected[sl] = True
        elif iv.code in RUNNING_PHASES:
            phase[sl] = True
            if iv.code == "BUILD UP":
                build_up[sl] = True
    far = ~_dilate(stopped, FAR) & ~_dilate(protected, FAR)
    live = (present & ~features.ball_out & np.isfinite(track.speed) & (track.speed > LIVE_SPEED)
            & phase & far & graded)
    return sources, live, live & build_up
