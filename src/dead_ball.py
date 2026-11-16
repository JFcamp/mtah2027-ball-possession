"""Dead-ball mask: frames in which the ball is out of play (label D).

Six rules, each producing its own mask so the report can say how many frames came from where:

  R1 half boundaries   from the first graded frame of a half to the Kick Off (Start) restart
  R2 ball out          from the frame the ball crosses a line until play restarts
  R3 set pieces        the tag start is an anchor in the MIDDLE of the stoppage (E2 diagnostics):
                       D starts at the last live frame before it, ends at the restart
  R4 goals             from the goal until the restart of the Kick Off (After Goal)
  R5 long ball gaps    ball unseen for >= 5 s while the players walk
  R6 stopped ball      ball inside the field below 0.5 m/s for >= 2 s

Nothing here looks at labels. Every threshold is a physical or football quantity (comments).

Thresholds
----------
V_RESTART = 2.0 m/s, MOVE_MIN = 12 frames (0.5 s)
    Play has restarted when the ball is inside the field and moves above 2 m/s for half a second.
    A person walking with, placing or picking up the ball moves at <= ~1.5-2 m/s; a kick or throw
    is well above 5 m/s. Sustained for 0.5 s so a one-frame tracking jump is not a restart.
RESTART_SEARCH = 200 frames (8 s), RESTART_FALLBACK = 88 frames (3.5 s)
    Restarts are looked for in the 8 s after the tag start (a placed ball is put in play within a
    few seconds). If the ball is not seen moving, the E2 diagnostics measured the real kick-off
    3.5 s after its tag start (the tag is a 5 s window around the restart); that is the fallback.
CAP_BACK = 200 frames (8 s)
    Furthest R3 looks back from the tag start for the last live frame. The diagnostics put the
    ball leaving the pitch ~3 s before the start; a ball rolling out and being fetched can take
    several more seconds, so 8 s.
V_LIVE = 3.0 m/s, A_LIVE = 2.0 m/s
    A frame before the stoppage is live if the ball is inside the field and above 3 m/s (faster
    than any walk or a ball being carried by hand) or the players' median speed is a jog
    (>= 2 m/s, the usual walk/jog boundary).
MAX_STOP = 1000 frames (40 s)
    Longest R2 stoppage when nothing bounds it. Throw-ins, corners and goal kicks last 10-40 s;
    longer ones (injuries, substitutions) are rare and caught by R5 when the ball is unseen.
GOAL_BACK = 50 frames (2 s), GOAL_FWD = 375 frames (15 s)
    Window around the GOALS tag start in which the ball crossing a line marks the goal. The
    diagnostics show the ball unseen from the goal start on and out of the pitch right after.
GAP_MIN = 125 frames (5 s), A_WALK = 1.4 m/s, ACT_WIN = 125 frames (5 s)
    R5: gaps of the ball of at least 5 s in which the 5 s mean of the players' median speed is
    below the preferred walking speed (1.4 m/s). The brief suggested ~2 m/s, but that is the
    median of live play on this activity measure (the median speed of ALL tracked players,
    goalkeepers and walkers included): the tracking shows stoppages near 1.0 m/s and fast live
    play near 2.0 m/s, so 2 m/s would call half of live play dead.
STILL_SPEED = 0.5 m/s, STILL_MIN = 50 frames (2 s)
    R6: a ball in play that has not moved for 2 s is not in play (kick-off, free kick or corner
    placement, ball held by the goalkeeper). Dribbling never leaves the ball still that long.
MIN_DEAD = 50 frames (2 s)
    A real stoppage is not shorter than that; shorter D runs are removed.
"""

from typing import Dict, List, NamedTuple, Optional

import numpy as np

from . import features as feat
from . import io as sio

V_RESTART = 2.0
MOVE_MIN = 12
RESTART_SEARCH = 200
RESTART_FALLBACK = 88
CAP_BACK = 200
V_LIVE = 3.0
A_LIVE = 2.0
MAX_STOP = 1000
GOAL_BACK = 50
GOAL_FWD = 375
GAP_MIN = 125
A_WALK = 1.4
ACT_WIN = 125
STILL_SPEED = 0.5
STILL_MIN = 50
MIN_DEAD = 50

RULES = ("R1", "R2", "R3", "R4", "R5", "R6")
PRIORITY = ("R1", "R4", "R3", "R2", "R6", "R5")  # who is credited when rules overlap


class Restart(NamedTuple):
    frame: int             # first live frame after the stoppage
    team: Optional[int]    # team taking the restart, None when unknown (throw-ins, kick-offs)
    kind: str              # tag type, for the report
    tagged_team: Optional[int] = None


class DeadBall(NamedTuple):
    dead: np.ndarray                 # [N] bool, graded frames that are D
    by_rule: Dict[str, np.ndarray]   # [N] bool per rule, after the minimum-duration filter
    credited: Dict[str, int]         # frames credited to each rule (+ "overlap-free" total)
    restarts: List[Restart]


def runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def sustained_motion(present, ball_out, speed) -> np.ndarray:
    ok = present & ~ball_out & np.isfinite(speed) & (speed >= V_RESTART)
    out = np.zeros(len(ok), dtype=bool)
    for a, b in runs(ok):
        if b - a >= MOVE_MIN:
            out[a:b] = True
    return out


def restart_after(anchor: int, moving: np.ndarray) -> int:
    """First frame at or after `anchor` where sustained motion STARTS, else anchor + 3.5 s.

    Only the onset of a movement counts. A ball already moving at the anchor belongs to the
    play (or a tracking artefact) before the restart, not to the restart itself: in the second
    half the ball is tracked at ~10 m/s across the pitch through the whole kick-off tag, ahead
    of the real kick-off.
    """
    onset = moving & ~np.concatenate(([False], moving[:-1]))
    hit = np.flatnonzero(onset[anchor:anchor + RESTART_SEARCH])
    return int(anchor + hit[0]) if hit.size else anchor + RESTART_FALLBACK


def _side(tag, sides):
    return sides.get(tag)


def build(track, features: feat.Features, intervals, r5_stopped=None) -> DeadBall:
    """Dead-ball mask. `r5_stopped` ([N] bool, optional) replaces the pace test of rule R5: inside a
    ball gap of at least 5 s a frame is dead where it is True (v4 passes a model's decision)."""
    n = sio.N_FRAMES
    graded = sio.graded_mask()
    blocks = sio.half_blocks()
    sides = feat.club_sides()

    present = np.isfinite(track.xy[:, 0])
    out = features.ball_out
    speed = track.speed
    moving = sustained_motion(present, out, speed)

    masks = {r: np.zeros(n, dtype=bool) for r in RULES}
    restarts: List[Restart] = []
    anchors = []  # (tag start, restart frame) of every anchored stoppage, for R2

    # ---- R1: half boundaries -------------------------------------------------------------------
    kickoffs = [i for i in intervals if i.code == "SET PIECES" and i.tags.get("Type") == "Kick Off (Start)"]
    half_name = {1: "1st Half", 2: "2nd Half"}
    for half, first, _ in blocks:
        for k in kickoffs:
            if k.tags.get("Half") == half_name[half]:
                r = restart_after(k.start_frame, moving)
                masks["R1"][first:max(first, r)] = True
                restarts.append(Restart(r, None, "Kick Off (Start)"))
                anchors.append((k.start_frame, r))

    # ---- R3: set pieces ------------------------------------------------------------------------
    live = (present & ~out & np.isfinite(speed) & (speed >= V_LIVE)) | (
        np.isfinite(features.activity) & (features.activity >= A_LIVE))
    for iv in intervals:
        if iv.code != "SET PIECES" or "Kick Off" in (iv.tags.get("Type") or ""):
            continue
        a = iv.start_frame
        r = restart_after(a, moving)
        lo = max(a - CAP_BACK, 0)
        recent = np.flatnonzero(live[lo:a])
        start = lo + int(recent[-1]) + 1 if recent.size else lo
        masks["R3"][start:r] = True
        team = _side(iv.tags.get("Team"), sides)
        restarts.append(Restart(r, team, iv.tags.get("Type"), team))
        anchors.append((a, r))

    # ---- R4: goals -----------------------------------------------------------------------------
    goals = sorted([i for i in intervals if i.code == "GOALS"], key=lambda i: i.start_frame)
    kicks = sorted([i for i in intervals if i.code == "SET PIECES"
                    and i.tags.get("Type") == "Kick Off (After Goal)"], key=lambda i: i.start_frame)
    for g in goals:
        k = next((k for k in kicks if k.start_frame >= g.start_frame
                  and k.tags.get("Half") == g.tags.get("Half")), None)
        lo, hi = max(g.start_frame - GOAL_BACK, 0), g.start_frame + GOAL_FWD
        crossing = np.flatnonzero(out[lo:hi])
        if crossing.size:
            t_goal = lo + int(crossing[0])
        else:
            # Ball never seen leaving: the goal is where the ball disappears for good.
            t_goal = g.start_frame
        if k is None:
            masks["R4"][t_goal:g.end_frame] = True
            continue
        r = restart_after(k.start_frame, moving)
        masks["R4"][t_goal:r] = True
        scorer = _side(g.tags.get("Team"), sides)
        taker = None if scorer is None else 1 - scorer  # the conceding team kicks off
        restarts.append(Restart(r, taker, "Kick Off (After Goal)", None))
        anchors.append((k.start_frame, r))

    # ---- R2: ball out of the field ---------------------------------------------------------------
    t = 0
    out_frames = np.flatnonzero(out & graded)
    covered_until = -1
    for s in out_frames:
        if s <= covered_until:
            continue
        after = np.flatnonzero(moving[s:s + MAX_STOP])
        e_det = s + int(after[0]) if after.size else s + MAX_STOP
        # A tagged restart close to this exit bounds the stoppage (the ball is often unseen for
        # a long time after it restarts, which would otherwise stretch the stoppage).
        bound = [r for a, r in anchors if r > s and a - CAP_BACK <= s + MAX_STOP and a >= s - CAP_BACK
                 and r < e_det]
        e = min(bound) if bound else e_det
        masks["R2"][s:e] = True
        covered_until = e

    # ---- R5: long ball gaps while players walk -------------------------------------------------
    act = feat.rolling_nanmean(features.activity, ACT_WIN)
    for _, first, last in blocks:
        seg = ~present[first:last + 1]
        for a, b in runs(seg):
            if b - a >= GAP_MIN:
                lo, hi = first + a, first + b
                if r5_stopped is None:
                    low = np.isfinite(act[lo:hi]) & (act[lo:hi] < A_WALK)
                else:
                    low = r5_stopped[lo:hi]
                masks["R5"][lo:hi] = low

    # ---- R6: ball stopped inside the field -----------------------------------------------------
    still = present & ~out & np.isfinite(speed) & (speed < STILL_SPEED)
    for a, b in runs(still):
        if b - a >= STILL_MIN:
            masks["R6"][a:b] = True

    # ---- union, minimum duration, credit -------------------------------------------------------
    for r in RULES:
        masks[r] &= graded
    union = np.zeros(n, dtype=bool)
    for r in RULES:
        union |= masks[r]
    final = np.zeros(n, dtype=bool)
    for a, b in runs(union):
        if b - a >= MIN_DEAD:
            final[a:b] = True
    by_rule = {r: masks[r] & final for r in RULES}

    credited = {r: 0 for r in RULES}
    taken = np.zeros(n, dtype=bool)
    for r in PRIORITY:
        credit = by_rule[r] & ~taken
        credited[r] = int(credit.sum())
        taken |= credit
    credited["removed (< 2 s)"] = int((union & ~final).sum())
    return DeadBall(final, by_rule, credited, sorted(restarts))
