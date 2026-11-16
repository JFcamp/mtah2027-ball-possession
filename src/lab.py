"""Shared pieces of the v5 experiments on the external games (conditions, inputs, selection).

Conditions (src/degrade.py, docs/domain_gap.md):
  CLEAN    the corrected games as they are
  DEG_BAR  degraded tracking + ball erased in 10.6% of the live frames (BAR-ATH level)
  DEG_40   degraded tracking + ball erased in 40% of the live frames
The degraded conditions use two seeds (0, 1); their scores are averaged.
"""

import itertools
from typing import Dict, List

import numpy as np

from . import calibration as cal
from . import degrade
from . import external as ext

CONDITIONS = ("CLEAN", "DEG_BAR", "DEG_40")
SEEDS = (0, 1)
V4_CONFIG = {"r_ctrl": 1.0, "margin": 0.3, "min_control": 6, "two_touch": True, "backdate": False,
             "min_poss": 25}

GRID_V5 = {
    "r_ctrl": [0.75, 1.0, 1.5, 2.0, 2.5],
    "margin": [0.0, 0.3],
    "min_control": [3, 6, 12, 25],
    "two_touch": [True, False],
    "backdate": [True, False],
    "min_poss": [12, 25, 38, 75],
}
KEYS = tuple(GRID_V5)


def configs_v5() -> List[dict]:
    return [dict(zip(KEYS, v)) for v in itertools.product(*GRID_V5.values())]


def changes_from_v4(cfg: dict) -> int:
    return sum(cfg[k] != V4_CONFIG[k] for k in KEYS)


def degraded_games(game: ext.Game) -> Dict[int, ext.Game]:
    return {s: degrade.degrade_game(game, s) for s in SEEDS}


def pipeline_inputs_by_condition(game: ext.Game, gt: np.ndarray, degraded: Dict[int, ext.Game] = None):
    """{condition: [(track, features, game_used)]} for the possession pipeline."""
    live = gt < 2
    degraded = degraded or degraded_games(game)
    out = {"CLEAN": [(*ext.pipeline_inputs(game), game)], "DEG_BAR": [], "DEG_40": []}
    for s, D in degraded.items():
        for cond, frac in (("DEG_BAR", degrade.LIVE_ERASE_BAR), ("DEG_40", 0.40)):
            ball = cal.erase_ball(D, live, s, frac)
            out[cond].append((*ext.pipeline_inputs(D, ball), D))
    return out


def spell_by_condition(game: ext.Game, gt: np.ndarray, dead: np.ndarray, inputs, cfg: dict) -> dict:
    """spellF1 of the possession logic with a dead-ball mask, averaged over seeds, per condition."""
    scores = {}
    for cond, items in inputs.items():
        vals = [ext.score(gt, cal.run_possession(g, dead, track, feats, cfg))["spell_f1"]
                for track, feats, g in items]
        scores[cond] = float(np.mean(vals))
    return scores
