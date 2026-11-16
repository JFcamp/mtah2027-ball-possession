"""Possession (H / A) on the live frames, from who controls the ball.

For every frame the ball is visible and one team's nearest player is close to it and clearly
closer than the opponent's nearest player, that team *controls* the ball. Runs of control are
"episodes". The owner only changes when the other team shows control that is more than a touch.
Loose ball, deflection, aerial duel and dispute never change the owner by themselves.

Thresholds
----------
R_CTRL = 1.5 m
    A player with the ball is within a metre of it (a dribble touch pushes it ~1-1.5 m, a foot
    is ~0.3 m from the ball when it plays it) and the tracked positions carry a few decimetres of
    noise. The E2 diagnostics measured the nearest player at a median of 1.8-2.0 m over all ball
    frames, of which most are passes in flight, so a radius below that median separates "has the
    ball" from "the ball is nearby".
CONTEST_MARGIN = 0.3 m
    The nearest player must be this much closer than the nearest opponent, about the tracking
    noise. Closer than that is a duel and nobody controls the ball.
MERGE_GAP = 5 frames (0.2 s)
    Control by the same team interrupted for <= 0.2 s (an occluded or jittery frame, a touch
    that rolls the ball slightly away) is one episode.
SWITCH_MIN = 12 frames (0.5 s)
    The other team takes possession after controlling the ball for 0.5 s, the "few seconds"
    proxy of CHALLENGE.md scaled to the shortest control that is still a deliberate action, or
    with two separate touches within TOUCH_WINDOW and no touch by the old owner in between
    (an interception followed by a pass).
TOUCH_WINDOW = 75 frames (3 s)
    Two touches by the same team within 3 s with no opponent touch in between are one action.
"""

from typing import Dict, List, NamedTuple, Optional

import numpy as np

from . import dead_ball as db
from . import features as feat
from . import io as sio

R_CTRL = 1.5
CONTEST_MARGIN = 0.3
MERGE_GAP = 5
SWITCH_MIN = 12
TOUCH_WINDOW = 75
TWO_TOUCH = True   # switch also on two separate touches (v1 behaviour)
BACKDATE = True    # back-date a switch to the first control frame (v1 behaviour)

H, A, D, X = 0, 1, 2, 3  # label codes


class SegmentInfo(NamedTuple):
    start: int
    end: int
    restart_kind: Optional[str]
    tagged_team: Optional[int]
    first_controller: Optional[int]
    n_switches: int
    n_gap_switches: int
    fallback: str  # "", "phase", "previous" or "none"


def control_teams(features: feat.Features, present: np.ndarray) -> np.ndarray:
    """[N] int8: team controlling the ball (0 / 1) or -1."""
    d = features.nearest_dist
    ok = present & np.isfinite(d).all(axis=1)
    near = np.where(d[:, 0] <= d[:, 1], 0, 1)
    d_near = np.minimum(d[:, 0], d[:, 1])
    d_far = np.maximum(d[:, 0], d[:, 1])
    control = np.full(len(d), -1, dtype=np.int8)
    sel = ok & (d_near <= R_CTRL) & (d_far - d_near >= CONTEST_MARGIN)
    control[sel] = near[sel]
    return control


def episodes(control: np.ndarray, lo: int, hi: int):
    """[(team, first_frame, last_frame)] of control runs inside [lo, hi)."""
    eps = []
    cur = None
    for t in range(lo, hi):
        k = control[t]
        if k < 0:
            if cur is not None and t - cur[2] > MERGE_GAP:
                eps.append(tuple(cur))
                cur = None
            continue
        if cur is not None and cur[0] == k:
            cur[2] = t
        else:
            if cur is not None:
                eps.append(tuple(cur))
            cur = [int(k), t, t]
    if cur is not None:
        eps.append(tuple(cur))
    return eps


def place_switch(first_control: int, new_team: int, lo: int, present: np.ndarray,
                 phase_team: np.ndarray, floor: int):
    """Frame at which possession passes to `new_team`, and whether it fell inside a ball gap.

    With the ball visible right up to the first control frame the switch is that frame
    (back-dated to the start of the new team's control). If the ball was unseen just before it,
    the change happened somewhere in the gap: put it where the phase tag flips to the new team,
    otherwise in the middle of the gap, which minimises the worst-case error.
    """
    seen = np.flatnonzero(present[max(lo, floor):first_control])
    last_seen = (max(lo, floor) + int(seen[-1])) if seen.size else max(lo, floor) - 1
    if first_control - 1 <= last_seen:
        return max(first_control, floor + 1), False
    start = max(last_seen + 1, floor + 1)
    for t in range(start, first_control + 1):
        if phase_team[t] == new_team and (t == 0 or phase_team[t - 1] != new_team):
            return t, True
    return max((last_seen + first_control + 1) // 2, floor + 1), True


def run_segment(lo, hi, control, present, phase_team, init_owner):
    """(owner [hi-lo] int8 (-1 unknown), first_controller, n_switches, n_gap_switches)."""
    eps = episodes(control, lo, hi)
    owner_arr = np.full(hi - lo, -1, dtype=np.int8)
    owner = init_owner
    first_controller = eps[0][0] if eps else None
    switches = []  # (frame, team)
    pending = None
    n_gap = 0
    for team, s, e in eps:
        if owner is None:
            owner = team
            continue
        if team == owner:
            pending = None
            continue
        if TWO_TOUCH and pending is not None and pending[0] == team and s - pending[2] <= TOUCH_WINDOW:
            seq_start, confirmed, confirm_at = pending[1], True, s
        else:
            seq_start, confirmed, confirm_at = s, (e - s + 1) >= SWITCH_MIN, s + SWITCH_MIN - 1
        if not confirmed:
            pending = (team, s, e)
            continue
        floor = switches[-1][0] if switches else lo
        # BACKDATE: the switch starts at the first control frame of the new team's sequence; when
        # off, at the moment the evidence was complete (second touch, or 0.5 s of control).
        frame, in_gap = place_switch(seq_start if BACKDATE else confirm_at, team, lo, present,
                                     phase_team, floor)
        switches.append((frame, team))
        n_gap += int(in_gap)
        owner = team
        pending = None

    first_owner = init_owner if init_owner is not None else (eps[0][0] if eps else None)
    if first_owner is not None:
        owner_arr[:] = first_owner
        for frame, team in switches:
            owner_arr[frame - lo:] = team
    return owner_arr, first_controller, len(switches), n_gap


def assign(dead_ball: db.DeadBall, features: feat.Features, track):
    """(labels [N] with H/A/D/X codes, [SegmentInfo])."""
    n = sio.N_FRAMES
    graded = sio.graded_mask()
    present = np.isfinite(track.xy[:, 0])
    control = control_teams(features, present)

    labels = np.full(n, X, dtype=np.int8)
    labels[graded & dead_ball.dead] = D
    live_runs = db.runs(graded & ~dead_ball.dead)
    dead_runs = {b: a for a, b in db.runs(dead_ball.dead & graded)}  # end -> start

    infos: List[SegmentInfo] = []
    previous_owner: Optional[int] = None
    for lo, hi in live_runs:
        # Restart that opened this segment, if any: the latest one inside the stoppage before it.
        kind, tagged, init = None, None, None
        d_start = dead_runs.get(lo)
        if d_start is not None:
            cands = [r for r in dead_ball.restarts if d_start <= r.frame <= lo + 25]
            if cands:
                r = cands[-1]
                kind, tagged, init = r.kind, r.tagged_team, r.team

        owner_arr, first_ctrl, n_sw, n_gap = run_segment(
            lo, hi, control, present, features.phase_team, init)
        fallback = ""
        if (owner_arr < 0).all():
            votes = features.phase_team[lo:hi]
            h, a = int((votes == sio.HOME).sum()), int((votes == sio.AWAY).sum())
            if h != a:
                owner_arr[:] = sio.HOME if h > a else sio.AWAY
                fallback = "phase"
            elif previous_owner is not None:
                owner_arr[:] = previous_owner
                fallback = "previous"
            else:
                fallback = "none"
        labels[lo:hi] = np.where(owner_arr >= 0, owner_arr, D)
        if (owner_arr >= 0).any():
            previous_owner = int(owner_arr[owner_arr >= 0][-1])
        infos.append(SegmentInfo(lo, hi, kind, tagged, first_ctrl, n_sw, n_gap, fallback))
    return labels, infos
