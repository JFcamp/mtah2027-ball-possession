"""Player-only features for the v5 stopped-ball model (no ball, no frame index, no absolute counts).

The 10 base quantities of src/features_players.py (speed percentiles, shares walking / running,
centroid speed, spread, depth, acceleration) plus, per frame:

  coh_team, coh_all   motion coherence: |mean velocity vector| / (mean speed + 0.1 m/s), per team (mean of
                      the two) and over all players. High when everybody moves the same way (a block
                      pressing or retreating together), low when people mill about.
  frac_still          share of tracked players below 0.5 m/s
  speed_std           sd of the players' speeds
  line_still          share of players standing (< 1 m/s) within 1.5 m of a touchline or beyond it (throw-in)
  corner              share of players within 3 m of a corner flag (corner kick)
  box_max, box_min    share of the tracked players inside the larger / smaller of the two penalty areas
  gk_only             1 if some penalty area holds exactly one player (the keeper alone: goal kick)
  halfness_min        smallest over the teams of the share of that team in its more populated half (1 = every
                      player of both teams in one half each: kick-off layout)
  opposite            1 if the two team centroids lie on opposite sides of the halfway line

All are shares of the TRACKED players (never counts), so a missing player does not move them much. Each is
averaged over centred windows of 1 s, 5 s and 10 s (NaN ignored). Two time-contrast features compare the
1 s activity now with 5 s before and with 5 s after (a stoppage is a drop in pace, a restart a rise).
"""

import warnings
from typing import List, Tuple

import numpy as np

from . import features as feat
from . import features_players as fp
from . import io as sio

FPS = sio.FPS
LENGTH, WIDTH = 105.0, 68.0       # pitch used for the geometry; 105.3 vs 105 is irrelevant at this scale
WINDOWS = (25, 125, 250)
WTAG = ("1s", "5s", "10s")
BOX_X, BOX_HALF_Y = 16.5, 20.16
LINE_MARGIN, STILL, CORNER_R = 1.5, 1.0, 3.0
QUIET = 0.5
SHIFT = 125


def _team_coherence(vel, speed, team_of_slot):
    out = []
    for side in (sio.HOME, sio.AWAY):
        sel = team_of_slot == side
        v, s = vel[:, sel], speed[:, sel]
        ok = np.isfinite(s)
        mv = np.nanmean(np.where(ok[..., None], v, np.nan), axis=1)
        ms = np.nanmean(np.where(ok, s, np.nan), axis=1)
        c = np.hypot(mv[:, 0], mv[:, 1]) / (ms + 0.1)
        c[ok.sum(axis=1) < fp.MIN_TEAM] = np.nan
        out.append(c)
    return np.nanmean(np.stack(out, axis=1), axis=1)


def _geometry(p, speed, team_of_slot):
    """Per-frame shares computed on one chunk. p [n,P,2] metres (the pitch is 105 x 68 here)."""
    ok = np.isfinite(p[..., 0])
    cnt = np.maximum(ok.sum(axis=1), 1)
    x, y = p[..., 0], p[..., 1]
    near_line = ok & ((y < LINE_MARGIN) | (y > WIDTH - LINE_MARGIN))
    still = np.isfinite(speed) & (speed < STILL)
    line_still = (near_line & still).sum(axis=1) / cnt
    corner = np.zeros(len(p))
    for cx in (0.0, LENGTH):
        for cy in (0.0, WIDTH):
            corner += (ok & (np.hypot(x - cx, y - cy) < CORNER_R)).sum(axis=1)
    corner /= cnt
    in_left = ok & (x < BOX_X) & (np.abs(y - WIDTH / 2) < BOX_HALF_Y)
    in_right = ok & (x > LENGTH - BOX_X) & (np.abs(y - WIDTH / 2) < BOX_HALF_Y)
    bl, br = in_left.sum(axis=1), in_right.sum(axis=1)
    box_max, box_min = np.maximum(bl, br) / cnt, np.minimum(bl, br) / cnt
    gk_only = ((bl == 1) | (br == 1)).astype(float)
    halves, centroids = [], []
    for side in (sio.HOME, sio.AWAY):
        sel = team_of_slot == side
        o = ok[:, sel]
        c = np.maximum(o.sum(axis=1), 1)
        left = ((x[:, sel] < LENGTH / 2) & o).sum(axis=1) / c
        halves.append(np.maximum(left, 1 - left))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centroids.append(np.nanmean(np.where(o, x[:, sel], np.nan), axis=1))
    halfness_min = np.minimum(halves[0], halves[1])
    opposite = ((centroids[0] - LENGTH / 2) * (centroids[1] - LENGTH / 2) < 0).astype(float)
    return line_still, corner, box_max, box_min, gk_only, halfness_min, opposite


def base_features(players_xy: np.ndarray, team_of_slot: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    n = len(players_xy)
    old, old_names = fp.base_features(players_xy, team_of_slot)
    old = old[:, :10]
    old_names = old_names[:10]
    speed = feat.player_speeds(players_xy)
    h = feat.SPEED_HALF_WIN
    vel = np.full(players_xy.shape, np.nan)
    vel[h:n - h] = (players_xy[2 * h:] - players_xy[:n - 2 * h]) * FPS / (2 * h)
    vel[~np.isfinite(speed)] = np.nan

    names = ["coh_team", "coh_all", "frac_still", "speed_std", "line_still", "corner", "box_max",
             "box_min", "gk_only", "halfness_min", "opposite"]
    cols = np.full((n, len(names)), np.nan)
    tracked = np.isfinite(speed).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cols[:, 0] = _team_coherence(vel, speed, team_of_slot)
        for lo in range(0, n, 20000):
            sl = slice(lo, lo + 20000)
            s, v = speed[sl], vel[sl]
            okc = np.isfinite(s)
            cols[sl, 1] = (np.hypot(*np.nanmean(np.where(okc[..., None], v, np.nan), axis=1).T)
                           / (np.nanmean(s, axis=1) + 0.1))
            cols[sl, 2] = (okc & (s < QUIET)).sum(axis=1) / np.maximum(okc.sum(axis=1), 1)
            cols[sl, 3] = np.nanstd(s, axis=1)
            geo = _geometry(players_xy[sl], s, team_of_slot)
            for k, g in enumerate(geo):
                cols[sl, 4 + k] = g
    cols[tracked < feat.MIN_TRACKED] = np.nan
    return np.concatenate([old, cols], axis=1), old_names + names


def compute(players_xy: np.ndarray, team_of_slot: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """(X [N, 65] float32, names)."""
    base, base_names = base_features(players_xy, team_of_slot)
    feats, names = [], []
    for w, tag in zip(WINDOWS, WTAG):
        for j, name in enumerate(base_names):
            feats.append(feat.rolling_nanmean(base[:, j], w))
            names.append("{}_{}".format(name, tag))
    act = feat.rolling_nanmean(base[:, base_names.index("speed_median")], 25)
    before = np.full(len(act), np.nan)
    after = np.full(len(act), np.nan)
    before[SHIFT:] = act[SHIFT:] - act[:-SHIFT]
    after[:-SHIFT] = act[:-SHIFT] - act[SHIFT:]
    feats += [before, after]
    names += ["act_minus_before_5s", "act_minus_after_5s"]
    return np.stack(feats, axis=1).astype(np.float32), names


def l0_columns(names: List[str]) -> List[int]:
    """Columns of the v4 model: the 10 old bases at 1 s and 5 s (src/features_players.py without n_tracked)."""
    old = ["speed_p25", "speed_median", "speed_p75", "speed_p90", "frac_walk", "frac_run",
           "centroid_speed", "spread", "depth_std", "mean_abs_acc"]
    return [names.index("{}_{}".format(b, t)) for t in ("1s", "5s") for b in old]


def short_columns(names: List[str]) -> List[int]:
    """Features of the edge-refinement model: everything at 1 s plus the two time contrasts."""
    return [i for i, n in enumerate(names) if n.endswith("_1s") or n.startswith("act_minus")]
