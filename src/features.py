"""Per-frame features from the tracking, the cleaned ball and the Smart Tagging.

Every array is indexed by video frame (see src/io.py). Nothing here reads a label or the public
ground truth. The only data-dependent step (reading the `Team` tag of each Smart Tagging code) is
checked against the tracking itself, see `phase_semantics`.

Constants and why
-----------------
PLAYER_SPEED_MAX = 12 m/s
    Faster than any footballer can run (sprint record ~12.4 m/s). A tracklet that "moves" faster
    has switched identity or jumped, so that sample is dropped.
SPEED_HALF_WIN = 2 frames
    Player speed is a central difference over +/-2 frames (0.16 s). A one-frame difference of
    positions that carry a few decimetres of noise is dominated by noise (0.3 m / 0.04 s = 7 m/s).
    The ATD `speed` field was checked against this: same unit (m/s), median ratio 0.99.
ACTIVITY_WIN = 25 frames (1 s)
    Smoothing of the per-frame median speed: long enough to remove tracking flicker, short enough
    to follow the change of pace after a whistle.
MIN_TRACKED = 6
    A median over fewer than 6 tracked players is not representative of the 22 on the pitch.
BALL_OUT_MARGIN = 0.5 m
    The ball is out when it has wholly crossed the line (centre more than a ball radius, 0.11 m,
    beyond it) and the tracked ball position carries a few decimetres of noise, so 0.5 m keeps
    balls rolling along the line inside the field and still marks real exits within a frame or
    two (the ball travels ~0.2-1 m per frame).
"""

import warnings
from typing import Dict, NamedTuple, Tuple

import numpy as np

from . import ball as ballmod
from . import io as sio

FPS = sio.FPS
LENGTH, WIDTH = sio.PITCH
PLAYER_SPEED_MAX = 12.0
SPEED_HALF_WIN = 2
ACTIVITY_WIN = 25
MIN_TRACKED = 6
BALL_OUT_MARGIN = 0.5

NO_TEAM = -1


class Features(NamedTuple):
    activity: np.ndarray       # [N] m/s, smoothed median speed of tracked players
    ball_out: np.ndarray       # [N] bool
    nearest_dist: np.ndarray   # [N, 2] metres from the ball to the closest player of home / away
    nearest_team: np.ndarray   # [N] int8, 0 home / 1 away / -1 unknown
    phase_team: np.ndarray     # [N] int8, team in possession according to the phases, -1 unknown


def club_sides() -> Dict[str, int]:
    """{"Barcelona": 0, "Atletic Club": 1}: the first team of the ATD metadata is the home side."""
    import games
    from baseline import team_names
    meta = games.require(["atd_meta"], sio.GAME)[0]
    home, away = team_names(meta)
    return {home: sio.HOME, away: sio.AWAY}


def player_speeds(players_xy: np.ndarray) -> np.ndarray:
    """[N, 46] m/s from positions (central difference), NaN where unavailable or implausible."""
    h = SPEED_HALF_WIN
    n = len(players_xy)
    speed = np.full(players_xy.shape[:2], np.nan)
    diff = players_xy[2 * h:] - players_xy[:n - 2 * h]
    speed[h:n - h] = np.hypot(diff[..., 0], diff[..., 1]) * FPS / (2 * h)
    speed[speed > PLAYER_SPEED_MAX] = np.nan
    return speed


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that ignores NaN (NaN where the whole window is NaN)."""
    ok = np.isfinite(values)
    csum = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(values))
    lo = np.maximum(0, idx - window // 2)
    hi = np.minimum(len(values), idx + window // 2 + 1)
    cnt = ccnt[hi] - ccnt[lo]
    out = np.full(len(values), np.nan)
    good = cnt > 0
    out[good] = (csum[hi] - csum[lo])[good] / cnt[good]
    return out


def activity_from(players_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(activity [N], per-slot speeds [N, 46]). Activity = 1 s smoothed median player speed."""
    speed = player_speeds(players_xy)
    tracked = np.isfinite(speed).sum(axis=1)
    median = np.full(len(speed), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for lo in range(0, len(speed), 20000):
            median[lo:lo + 20000] = np.nanmedian(speed[lo:lo + 20000], axis=1)
    median[tracked < MIN_TRACKED] = np.nan
    return rolling_nanmean(median, ACTIVITY_WIN), speed


def ball_out_mask(xy: np.ndarray, margin: float = BALL_OUT_MARGIN) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    with np.errstate(invalid="ignore"):
        out = (x < -margin) | (x > LENGTH + margin) | (y < -margin) | (y > WIDTH + margin)
    return out & np.isfinite(x)


def nearest_players(players_xy: np.ndarray, ball_xy: np.ndarray, team_of_slot: np.ndarray):
    """(dist [N,2], nearest_team [N]): distance from the ball to the closest player of each team."""
    n = len(ball_xy)
    dist = np.full((n, 2), np.nan)
    for lo in range(0, n, 20000):
        hi = min(lo + 20000, n)
        diff = players_xy[lo:hi] - ball_xy[lo:hi, None, :]
        d = np.hypot(diff[..., 0], diff[..., 1])
        d = np.where(np.isfinite(d), d, np.inf)
        for side in (sio.HOME, sio.AWAY):
            m = d[:, team_of_slot == side].min(axis=1)
            dist[lo:hi, side] = np.where(np.isfinite(m), m, np.nan)
    nearest = np.full(n, NO_TEAM, dtype=np.int8)
    ok = np.isfinite(dist).all(axis=1)
    nearest[ok] = np.where(dist[ok, 0] <= dist[ok, 1], sio.HOME, sio.AWAY)
    return dist, nearest


def phase_arrays(intervals, sides: Dict[str, int]) -> Dict[str, np.ndarray]:
    """{code: [N] int8} with the raw Team tag of each phase code (NO_TEAM where none)."""
    out: Dict[str, np.ndarray] = {}
    for iv in intervals:
        if iv.code in ("SET PIECES", "GOALS"):
            continue
        side = sides.get(iv.tags.get("Team"), NO_TEAM)
        if side == NO_TEAM:
            continue
        arr = out.setdefault(iv.code, np.full(sio.N_FRAMES, NO_TEAM, dtype=np.int8))
        arr[max(iv.start_frame, 0):min(iv.end_frame, sio.N_FRAMES)] = side
    return out


def phase_semantics(raw: Dict[str, np.ndarray], nearest_team: np.ndarray, graded: np.ndarray):
    """Per code: agreement of the Team tag with the nearest team, and the reading to use.

    Football prior from the code names: "DEFENDING IN ..." and "DEFENSIVE TRANSITION" are tagged
    with the team that defends (the possessor is the opponent); every other phase is tagged with
    the team on the ball. The
    prior is then checked against the tracking, with no labels: on graded frames with the ball
    visible and the code active, "owner" must agree with the nearest team more than half of the
    time and "defender" less than half. A code that contradicts its prior is ignored.
    Returns {code: (n_frames, agreement, reading)}, reading in {"owner", "defender", "ignore"}.
    """
    table = {}
    for code, arr in sorted(raw.items()):
        sel = graded & (arr != NO_TEAM) & (nearest_team != NO_TEAM)
        n = int(sel.sum())
        agree = float((arr[sel] == nearest_team[sel]).mean()) if n else float("nan")
        prior = "defender" if code.startswith("DEFENDING") or code == "DEFENSIVE TRANSITION" else "owner"
        consistent = n > 0 and ((agree > 0.5) == (prior == "owner"))
        table[code] = (n, agree, prior if consistent else "ignore")
    return table


def combine_phases(raw: Dict[str, np.ndarray], table) -> np.ndarray:
    """phase_team [N]: majority vote of the usable codes, each read as owner or defender."""
    votes = np.zeros((sio.N_FRAMES, 2), dtype=np.int16)
    for code, arr in raw.items():
        reading = table[code][2]
        if reading == "ignore":
            continue
        owner = arr if reading == "owner" else np.where(arr == NO_TEAM, NO_TEAM, 1 - arr)
        for side in (sio.HOME, sio.AWAY):
            votes[:, side] += (owner == side)
    team = np.full(sio.N_FRAMES, NO_TEAM, dtype=np.int8)
    team[votes[:, 0] > votes[:, 1]] = sio.HOME
    team[votes[:, 1] > votes[:, 0]] = sio.AWAY
    return team


def compute_features(tracking, track: ballmod.BallTrack, intervals) -> Tuple[Features, dict]:
    """All per-frame features, plus the phase table (for the report)."""
    activity, _ = activity_from(tracking.players_xy)
    out = ball_out_mask(track.xy)
    dist, nearest = nearest_players(tracking.players_xy, track.xy, tracking.team_of_slot)
    raw = phase_arrays(intervals, club_sides())
    table = phase_semantics(raw, nearest, sio.graded_mask())
    phase = combine_phases(raw, table)
    return Features(activity, out, dist, nearest, phase), table
