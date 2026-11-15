"""Player-only features: nothing here uses the ball, the Smart Tagging, a frame index or an
absolute position of the pitch (compactness is a spread, not a place).

Per frame, base quantities from the 46 track slots:
  speed percentiles (p25, median, p75, p90) of the tracked players
  fraction of players walking (< 1.4 m/s, preferred walking speed), running (> 4 m/s) 
  centroid speed: magnitude of the mean velocity of a team's tracked players (how fast the whole
      team moves as a block), averaged over the two teams
  compactness: mean distance of the players to their team centroid, and the spread along the pitch
      length, averaged over the two teams
  mean absolute acceleration of the players (speed change over +/-5 frames)
  number of tracked players
Each base quantity is then averaged over a centred 1 s and a centred 5 s window (NaN ignored),
giving 2 x 12 features. Windows are centred because the pipeline is offline.
"""

import warnings
from typing import List, Tuple

import numpy as np

from . import features as feat
from . import io as sio

FPS = sio.FPS
WALK = 1.4   # m/s, preferred walking speed
RUN = 4.0    # m/s, running pace (jog is ~2-3.5, running above)
MIN_TEAM = 3  # players needed for a team statistic
MIN_TRACKED = feat.MIN_TRACKED
WINDOWS = (25, 125)  # 1 s and 5 s
ACC_HALF = 5


def _team_stats(vel, pos_xy, team_of_slot):
    n = len(vel)
    cspeed = np.full((n, 2), np.nan)
    spread = np.full((n, 2), np.nan)
    depth = np.full((n, 2), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for side in (sio.HOME, sio.AWAY):
            sel = team_of_slot == side
            v = vel[:, sel]
            ok = np.isfinite(v).all(axis=2)
            mean_v = np.nanmean(np.where(ok[..., None], v, np.nan), axis=1)
            cs = np.hypot(mean_v[:, 0], mean_v[:, 1])
            cs[ok.sum(axis=1) < MIN_TEAM] = np.nan
            cspeed[:, side] = cs

            p = pos_xy[:, sel]
            okp = np.isfinite(p).all(axis=2)
            centroid = np.nanmean(np.where(okp[..., None], p, np.nan), axis=1)
            dist = np.hypot(p[..., 0] - centroid[:, None, 0], p[..., 1] - centroid[:, None, 1])
            sp = np.nanmean(dist, axis=1)
            dx = np.nanstd(np.where(okp, p[..., 0], np.nan), axis=1)
            few = okp.sum(axis=1) < MIN_TEAM
            sp[few] = np.nan
            dx[few] = np.nan
            spread[:, side] = sp
            depth[:, side] = dx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmean(cspeed, axis=1), np.nanmean(spread, axis=1), np.nanmean(depth, axis=1))


def base_features(players_xy: np.ndarray, team_of_slot: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    n = len(players_xy)
    speed = feat.player_speeds(players_xy)
    h = feat.SPEED_HALF_WIN
    vel = np.full(players_xy.shape, np.nan)
    vel[h:n - h] = (players_xy[2 * h:] - players_xy[:n - 2 * h]) * FPS / (2 * h)
    vel[~np.isfinite(speed)] = np.nan

    names = ["speed_p25", "speed_median", "speed_p75", "speed_p90", "frac_walk", "frac_run",
             "centroid_speed", "spread", "depth_std", "mean_abs_acc", "n_tracked"]
    cols = np.full((n, len(names)), np.nan)
    tracked = np.isfinite(speed).sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for lo in range(0, n, 20000):
            chunk = speed[lo:lo + 20000]
            q = np.nanpercentile(chunk, [25, 50, 75, 90], axis=1)
            cols[lo:lo + 20000, 0:4] = q.T
            ok = np.isfinite(chunk)
            cnt = np.maximum(ok.sum(axis=1), 1)
            cols[lo:lo + 20000, 4] = (ok & (chunk < WALK)).sum(axis=1) / cnt
            cols[lo:lo + 20000, 5] = (ok & (chunk > RUN)).sum(axis=1) / cnt

        cols[:, 6], cols[:, 7], cols[:, 8] = _team_stats(vel, players_xy, team_of_slot)

        acc = np.full(speed.shape, np.nan)
        acc[ACC_HALF:n - ACC_HALF] = (speed[2 * ACC_HALF:] - speed[:n - 2 * ACC_HALF]) * FPS / (2 * ACC_HALF)
        for lo in range(0, n, 20000):
            cols[lo:lo + 20000, 9] = np.nanmean(np.abs(acc[lo:lo + 20000]), axis=1)
    cols[:, 10] = tracked
    cols[tracked < MIN_TRACKED, :10] = np.nan
    return cols, names


def compute(players_xy: np.ndarray, team_of_slot: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """(X [N, 22], names): windowed player features, NaN where undefined."""
    base, names = base_features(players_xy, team_of_slot)
    feats, out_names = [], []
    for window, tag in zip(WINDOWS, ("1s", "5s")):
        for j, name in enumerate(names):
            feats.append(feat.rolling_nanmean(base[:, j], window))
            out_names.append("{}_{}".format(name, tag))
    return np.stack(feats, axis=1), out_names
