"""Broadcast recording: tracking (ATD) and homography validity, on the broadcast's own timeline.

Same file format as the tactical tracking (see src/io.py): positions normalised 0-1 in the file,
returned here in metres on the same pitch; the ball and 51 player slots; the first team of the
metadata is the home side. 164,488 frames at 25 fps. The broadcast timeline is not the tactical one
(cuts, replays, no halftime), see src/align.py.

The homography maps the pitch to the image. Where `valid` is 0 the camera shows no usable view of
the pitch (replay graphics, close-ups, crowd), so the tracking of those frames is not trusted.
"""

import os

import numpy as np
import pandas as pd

from . import io as sio

import games  # noqa: E402  (src/io.py put the repository root on sys.path)

N_FRAMES = games.source_frames("broadcast", sio.GAME)
CACHE_PATH = os.path.join(sio.CACHE_DIR, "broadcast.npz")


def load_tracking(use_cache: bool = True) -> sio.Tracking:
    """Broadcast ball [N,2], players [N,51,2] (metres), speeds [N,51] and team of each slot."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], sio.GAME, source="broadcast")
    raw_size = os.path.getsize(raw_path)
    if use_cache and os.path.isfile(CACHE_PATH):
        with np.load(CACHE_PATH) as cache:
            if int(cache["raw_size"]) == raw_size:
                return sio.Tracking(cache["ball_xy"], cache["players_xy"],
                                    cache["players_speed"], cache["team_of_slot"])
    tracking = sio._parse_atd(raw_path, sio.team_of_each_track(meta_path), N_FRAMES)
    if use_cache:
        os.makedirs(sio.CACHE_DIR, exist_ok=True)
        np.savez_compressed(CACHE_PATH, ball_xy=tracking.ball_xy,
                            players_xy=tracking.players_xy,
                            players_speed=tracking.players_speed,
                            team_of_slot=tracking.team_of_slot, raw_size=raw_size)
    return tracking


def load_valid() -> np.ndarray:
    """[N] bool: the broadcast frame has a usable homography (the pitch is in view)."""
    path = games.require(["homography"], sio.GAME, source="broadcast")[0]
    frame = pd.read_csv(path, usecols=["frame", "valid"])
    valid = np.zeros(N_FRAMES, dtype=bool)
    valid[frame["frame"].to_numpy()] = frame["valid"].to_numpy() == 1
    return valid
