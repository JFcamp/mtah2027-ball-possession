"""Experiments on the external games: possession logic with the TRUE dead-ball mask, artificial ball
gaps, configuration sweep, cross-game selection.

Nothing here touches BAR-ATH. The public sample games are used with labels from the provider.
"""

import contextlib
import itertools
import zlib
from typing import Dict, List

import numpy as np

from . import dead_ball as db
from . import external as ext
from . import possession as pos
from . import postprocess as post

FPS = 25

# The v1 configuration, exactly as in the pipeline.
V1 = {"r_ctrl": 1.5, "margin": 0.3, "min_control": 12, "two_touch": True, "backdate": True,
      "min_poss": 38}

# Sweep (a factorial grid; the first value of each list is not special).
GRID = {
    "r_ctrl": [1.0, 1.5, 2.0],
    "margin": [0.0, 0.3, 0.6],
    "min_control": [6, 12, 25],
    "two_touch": [True, False],
    "backdate": [True, False],
    "min_poss": [25, 38, 75],
}
KEYS = tuple(GRID)


def configs() -> List[dict]:
    return [dict(zip(KEYS, values)) for values in itertools.product(*GRID.values())]


def distance_from_v1(cfg: dict) -> int:
    return sum(cfg[k] != V1[k] for k in KEYS)


@contextlib.contextmanager
def configured(cfg: dict):
    names = [("R_CTRL", "r_ctrl"), ("CONTEST_MARGIN", "margin"), ("SWITCH_MIN", "min_control"),
             ("TWO_TOUCH", "two_touch"), ("BACKDATE", "backdate")]
    saved = {n: getattr(pos, n) for n, _ in names}
    for n, k in names:
        setattr(pos, n, cfg[k])
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(pos, n, v)


@contextlib.contextmanager
def cached_episodes():
    """Memoise pos.episodes: the control runs do not depend on the switch parameters."""
    original = pos.episodes
    memo: Dict[tuple, list] = {}

    def wrapper(control, lo, hi):
        key = (zlib.crc32(control.tobytes()), lo, hi, pos.MERGE_GAP)
        if key not in memo:
            memo[key] = original(control, lo, hi)
        return memo[key]

    pos.episodes = wrapper
    try:
        yield
    finally:
        pos.episodes = original


def erase_ball(game: ext.Game, live: np.ndarray, seed: int, fraction: float = 0.40) -> np.ndarray:
    """Ball track with gaps erased inside live play, following the E2 gap distribution.

    E2 (BAR-ATH, graded frames): the ball is missing in long gaps only; of the missing frames ~14%
    are in gaps of 5-10 s and ~86% in gaps over 10 s (mean of those ~29 s). Gaps are drawn that way
    (5-10 s uniform; 10-60 s log-uniform) at random live positions until `fraction` of the live
    frames is erased. Seeded, so the result is deterministic.
    """
    rng = np.random.RandomState(seed)
    ball = game.ball_xy.copy()
    live_idx = np.flatnonzero(live)
    erased = np.zeros(len(ball), dtype=bool)
    target = fraction * len(live_idx)
    count = 0
    while count < target:
        if rng.uniform() < 0.14:
            length = rng.uniform(5.0, 10.0)
        else:
            length = float(np.exp(rng.uniform(np.log(10.0), np.log(60.0))))
        start = live_idx[rng.randint(len(live_idx))]
        stop = start + int(length * FPS)
        erased[start:stop] = True
        count = int((erased & live).sum())
    ball[erased] = np.nan
    return ball


def run_possession(game: ext.Game, dead_mask: np.ndarray, track, features, cfg: dict) -> np.ndarray:
    """H/A/D/X codes: possession.assign + postprocess with the given dead-ball mask and config."""
    graded = ext.graded_mask(game)
    with ext.as_pipeline_game(game), configured(cfg):
        dead = db.DeadBall(dead_mask & graded, {}, {}, [])
        pre, _ = pos.assign(dead, features, track)
        return post.finalize(pre, graded, cfg["min_poss"])


def cross_game_selection(configs, results, games, v1_index):
    """Choose on two games, test on the third. Returns per held-out game: chosen index and scores.

    Objective on the training games: mean spellF1 over games and over the two ball conditions;
    ties are broken by fewest differences from v1, then by grid order. Both are deterministic.
    """
    conds = ("full", "erased")
    dist = [distance_from_v1(c) for c in configs]

    def best(train):
        obj = np.mean([[np.array(results[g][c])[:, 0] for c in conds] for g in train], axis=(0, 1))
        order = sorted(range(len(configs)), key=lambda i: (-round(obj[i], 9), dist[i], i))
        return order[0], obj

    out = {}
    for test in games:
        train = [g for g in games if g != test]
        idx, obj = best(train)
        out[test] = {"chosen": idx, "train_objective": float(obj[idx]),
                     "v1_train_objective": float(obj[v1_index]),
                     "test": {c: float(results[test][c][idx][0]) for c in conds},
                     "v1_test": {c: float(results[test][c][v1_index][0]) for c in conds}}
    all_idx, obj = best(list(games))
    return out, all_idx, float(obj[all_idx]), float(obj[v1_index])
