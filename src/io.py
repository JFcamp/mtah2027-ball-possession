"""Reading the match data: tracking (ATD), Smart Tagging and the graded-frame mask.

Everything here is indexed by *video frame* (0 ... N-1), the same index as a submission row:
video_frame = ATD frame_id - 1. Positions are in metres on the pitch of games.GAMES["BAR-ATH"].

The raw ATD parse is a ~200 MB text file and takes about a minute, so the arrays are cached in
data/cache/tracking.npz (git-ignored with the rest of data/). The cache is only reused if it was
built from a file of the same size, and the parse is deterministic, so a cold run and a cached run
return identical arrays.
"""

import os
import sys
from typing import Dict, List, NamedTuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import games  # noqa: E402
import load_smart_tagging as _starter_tagging  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  (reused so the team mapping is the official one)

GAME = games.GAME
N_FRAMES = games.source_frames("tactical", GAME)
FPS = games.GAMES[GAME]["fps"]
PITCH = games.GAMES[GAME]["pitch"]  # (length, width) in metres
HOME, AWAY = 0, 1

CACHE_DIR = os.path.join(games.DATA_DIR, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "tracking.npz")


class Tracking(NamedTuple):
    ball_xy: np.ndarray        # [N, 2] metres, NaN where the ball is not observed
    players_xy: np.ndarray     # [N, 46, 2] metres, NaN where the slot has no observation
    players_speed: np.ndarray  # [N, 46] as reported by the ATD (m/s), NaN where missing
    team_of_slot: np.ndarray   # [46] 0 = home, 1 = away


class Interval(NamedTuple):
    start_frame: int  # video frame
    end_frame: int    # video frame, first frame after the phase
    code: str
    tags: Dict[str, str]


def _parse_atd(raw_path: str, team_of_slot: np.ndarray) -> Tracking:
    n_slots = len(team_of_slot)
    length, width = PITCH
    scale = np.array([length, width])

    ball = np.full((N_FRAMES, 2), np.nan)
    players = np.full((N_FRAMES, n_slots, 2), np.nan)
    speed = np.full((N_FRAMES, n_slots), np.nan)
    seen = np.zeros(N_FRAMES, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if not 0 <= frame < N_FRAMES:
                raise ValueError("frame_id {} outside 1..{}".format(frame + 1, N_FRAMES))

            group = np.fromstring(parts[1].replace(";", ",").strip(), sep=",")
            if group.size != 3 * n_slots:
                raise ValueError("frame {}: expected {} values for players, got {}".format(
                    frame + 1, 3 * n_slots, group.size))
            group = group.reshape(n_slots, 3)
            players[frame] = group[:, :2] * scale
            speed[frame] = group[:, 2]

            b = np.fromstring(parts[2].strip(), sep=",")
            if b.size < 2:
                raise ValueError("frame {}: bad ball field".format(frame + 1))
            ball[frame] = b[:2] * scale
            seen[frame] = True

    if not seen.all():
        raise ValueError("{} frames missing from the ATD".format(int((~seen).sum())))
    # A ball needs both coordinates; a half-observed one is treated as not observed.
    ball[~np.isfinite(ball).all(axis=1)] = np.nan
    return Tracking(ball, players, speed, team_of_slot)


def load_tracking(use_cache: bool = True) -> Tracking:
    """Tracking arrays for the whole video; see the module docstring for the conventions."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], GAME)
    raw_size = os.path.getsize(raw_path)

    if use_cache and os.path.isfile(CACHE_PATH):
        with np.load(CACHE_PATH) as cache:
            if int(cache["raw_size"]) == raw_size:
                return Tracking(cache["ball_xy"], cache["players_xy"],
                                cache["players_speed"], cache["team_of_slot"])

    tracking = _parse_atd(raw_path, team_of_each_track(meta_path))
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(CACHE_PATH, ball_xy=tracking.ball_xy,
                            players_xy=tracking.players_xy,
                            players_speed=tracking.players_speed,
                            team_of_slot=tracking.team_of_slot, raw_size=raw_size)
    return tracking


def load_smart_tagging() -> List[Interval]:
    """Every Smart Tagging instance as (start_frame, end_frame, code, tags), in file order."""
    return [Interval(e["start_frame"], e["end_frame"], e["code"], e["tags"])
            for e in _starter_tagging.load(GAME)]


def graded_mask() -> np.ndarray:
    """Boolean [N], True on the frames of the two halves (the only frames that are scored)."""
    mask = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in games.half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def half_blocks() -> List[tuple]:
    """[(half, first_video_frame, last_video_frame)]."""
    return games.half_video_blocks(GAME)
