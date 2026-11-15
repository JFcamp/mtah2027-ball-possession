"""Align broadcast frames to tactical frames WITHOUT the ball.

Both recordings are trackings of the same match in the same pitch coordinates (metres), so a
broadcast frame whose homography is valid shows the same 20 or so players, in the same places, as
the tactical frame of the same instant. The alignment looks for the tactical frame whose players
(same team) lie nearest to the broadcast players; the ball is never used here (it is kept for the
independent validation in docs/report_v3.md).

Live broadcast runs at a constant offset from the tactical clock (the TV feed is continuous in real
time between its cuts); a replay shows an earlier moment, so its offset differs and it is rejected.
A frame's offset is estimated from its block of BLOCK frames, summing the matching cost over
SAMPLES evenly spaced frames of the block, which pins the offset to the frame even where the
players are almost still.

    tactical_frame = broadcast_frame + offset

Constants and why
-----------------
MATCH_RADIUS = 2.5 m
    Two trackers see the same player; their positions differ by tracking and calibration error of
    about a metre. Within 2.5 m is "the same player". The chance a random frame puts a given
    player within 2.5 m of a same-team player is ~3-5%, so a true match (~90-100% matched) is
    unmistakable.
COST_CAP = 5 m   per-player distance is capped so one tracking error cannot dominate the cost.
MIN_PLAYERS = 10   fewer visible players than that is not a reliable signature.
ACCEPT_FRAC = 0.75   a block is aligned only if >= 75% of its players match at the chosen offset.
TOL_LIVE = 60 frames (2.4 s)   a block offset farther than that from the half's dominant offset
    belongs to a replay (or another cut) and is left unaligned.
"""

import os
from typing import NamedTuple

import numpy as np

from . import broadcast as bc
from . import io as sio

MATCH_RADIUS = 2.5
COST_CAP = 5.0
MIN_PLAYERS = 10
ACCEPT_FRAC = 0.75
TOL_LIVE = 60
BLOCK = 250
MIN_BLOCK = 50
SAMPLES = 12
COARSE_STEP = 5      # tactical stride in the coarse global search (positions move < 1 m in 5 frames)
COARSE_EVERY = 500   # broadcast stride of the coarse samples
REFINE_HALF_RANGE = 150
NO_ALIGNMENT = np.iinfo(np.int32).min

CACHE_PATH = os.path.join(sio.CACHE_DIR, "align.npz")


class Alignment(NamedTuple):
    offset: np.ndarray     # [Nb] int32, tactical = broadcast + offset; NO_ALIGNMENT where none
    frac: np.ndarray       # [Nb] float32, matched fraction at the block offset (NaN if unknown)
    dominant: dict         # {1: offset, 2: offset}


def _team_mask(team_b, team_t):
    return team_b[:, None] == team_t[None, :]


def block_costs(pb, tb, tactical, team_t, frames, offsets):
    """Cost and matched fraction of broadcast `frames` against tactical frames frame+offset.

    Returns (cost [K, M], frac [K, M]) with NaN where the tactical frame is out of range.
    """
    n_t = len(tactical)
    mask = _team_mask(tb, team_t)                                  # [51, 46]
    cost = np.full((len(frames), len(offsets)), np.nan)
    frac = np.full((len(frames), len(offsets)), np.nan)
    for k, f in enumerate(frames):
        p = pb[f]
        ok_p = np.isfinite(p[:, 0])
        if ok_p.sum() < 3:
            continue
        t = f + offsets
        ok_t = (t >= 0) & (t < n_t)
        tt = tactical[np.clip(t, 0, n_t - 1)]                      # [M, 46, 2]
        q = p[ok_p]
        d = np.hypot(q[None, :, None, 0] - tt[:, None, :, 0],
                     q[None, :, None, 1] - tt[:, None, :, 1])      # [M, n, 46]
        d = np.where(mask[None, ok_p, :] & np.isfinite(d), d, np.inf)
        nearest = d.min(axis=2)                                    # [M, n]
        cost[k] = np.where(ok_t, np.minimum(nearest, COST_CAP).mean(axis=1), np.nan)
        frac[k] = np.where(ok_t, (nearest < MATCH_RADIUS).mean(axis=1), np.nan)
    return cost, frac


def strong_frames(players, valid):
    """Broadcast frames with a valid homography and enough visible players."""
    n = np.isfinite(players[..., 0]).sum(axis=1)
    return np.flatnonzero(valid & (n >= MIN_PLAYERS))


def coarse_offsets(broadcast, valid, tactical, team_t, graded):
    """Global search on sparse samples: (sample frames, best offset, best fraction, margin)."""
    frames = strong_frames(broadcast.players_xy, valid)
    frames = frames[::COARSE_EVERY]
    cand_t = np.flatnonzero(graded)[::COARSE_STEP]
    out = []
    for f in frames:
        p = broadcast.players_xy[f]
        ok = np.isfinite(p[:, 0])
        mask = _team_mask(broadcast.team_of_slot[ok], team_t)
        best_fr = np.zeros(len(cand_t))
        for lo in range(0, len(cand_t), 4000):
            tt = tactical[cand_t[lo:lo + 4000]]
            q = p[ok]
            d = np.hypot(q[None, :, None, 0] - tt[:, None, :, 0],
                         q[None, :, None, 1] - tt[:, None, :, 1])
            d = np.where(mask[None] & np.isfinite(d), d, np.inf).min(axis=2)
            best_fr[lo:lo + 4000] = (d < MATCH_RADIUS).mean(axis=1)
        k = int(np.argmax(best_fr))
        far = np.abs(cand_t - cand_t[k]) > 150
        second = float(best_fr[far].max()) if far.any() else 0.0
        out.append((f, int(cand_t[k] - f), float(best_fr[k]), float(best_fr[k] - second)))
    return np.array(out, dtype=float)


def dominant_offsets(coarse):
    """Per half, the most common offset among confident coarse samples (bins of 2 * TOL_LIVE)."""
    f, off, fr, margin = coarse.T
    good = (fr >= 0.85) & (margin >= 0.2)
    offs = off[good]
    dom = {}
    # the broadcast has the first half early and the second half late: split at the biggest jump
    order = np.argsort(f[good])
    offs_sorted = offs[order]
    jumps = np.abs(np.diff(offs_sorted))
    split = int(np.argmax(jumps)) + 1
    for half, part in ((1, offs_sorted[:split]), (2, offs_sorted[split:])):
        vals, counts = np.unique(np.round(part / TOL_LIVE).astype(int), return_counts=True)
        mode = vals[np.argmax(counts)] * TOL_LIVE
        dom[half] = int(np.median(part[np.abs(part - mode) <= TOL_LIVE]))
    return dom


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[:: 2].tolist(), edges[1::2].tolist()))


def align(broadcast=None, valid=None, tactical=None, use_cache=True) -> Alignment:
    """Per-frame offset of the broadcast onto the tactical timeline (see the module docstring)."""
    if use_cache and os.path.isfile(CACHE_PATH):
        with np.load(CACHE_PATH) as cache:
            return Alignment(cache["offset"], cache["frac"],
                             {1: int(cache["dom1"]), 2: int(cache["dom2"])})
    broadcast = broadcast if broadcast is not None else bc.load_tracking()
    valid = valid if valid is not None else bc.load_valid()
    tactical = tactical if tactical is not None else sio.load_tracking()
    graded = sio.graded_mask()
    team_t = tactical.team_of_slot

    coarse = coarse_offsets(broadcast, valid, tactical.players_xy, team_t, graded)
    dominant = dominant_offsets(coarse)
    confident = coarse[(coarse[:, 2] >= 0.85) & (coarse[:, 3] >= 0.2)]
    near = {h: confident[np.abs(confident[:, 1] - dominant[h]) <= TOL_LIVE, 0] for h in (1, 2)}
    split = int((near[1].max() + near[2].min()) // 2)

    n_b = len(valid)
    offset = np.full(n_b, NO_ALIGNMENT, dtype=np.int32)
    frac = np.full(n_b, np.nan, dtype=np.float32)
    strong = np.zeros(n_b, dtype=bool)
    strong[strong_frames(broadcast.players_xy, valid)] = True

    for lo, hi in _runs(valid):
        for a in range(lo, hi, BLOCK):
            b = min(a + BLOCK, hi)
            if hi - b < MIN_BLOCK:   # do not leave a tiny tail block
                b = hi
            if b - a < MIN_BLOCK:
                continue
            idx = np.flatnonzero(strong[a:b]) + a
            if len(idx) < 4:
                continue
            pick = idx[np.unique(np.linspace(0, len(idx) - 1, SAMPLES).astype(int))]
            dom = dominant[1] if a < split else dominant[2]
            offsets = np.arange(dom - REFINE_HALF_RANGE, dom + REFINE_HALF_RANGE + 1)
            cost, fr = block_costs(broadcast.players_xy, broadcast.team_of_slot, tactical.players_xy,
                                   team_t, pick, offsets)
            total = np.nanmean(cost, axis=0)
            k = int(np.nanargmin(total))
            mean_frac = float(np.nanmean(fr[:, k]))
            if mean_frac >= ACCEPT_FRAC and abs(offsets[k] - dom) <= TOL_LIVE:
                offset[a:b] = offsets[k]
                frac[a:b] = mean_frac
    result = Alignment(offset, frac, dominant)
    if use_cache:
        os.makedirs(sio.CACHE_DIR, exist_ok=True)
        np.savez_compressed(CACHE_PATH, offset=offset, frac=frac, dom1=dominant[1],
                            dom2=dominant[2])
    return result
