"""Synthetic checks of the broadcast alignment cost and the ball fusion. No match data needed."""

import numpy as np

from src import align as al
from src import fusion
from src import io as sio


def test_block_costs_find_the_true_offset():
    rng = np.random.RandomState(0)
    n_t = 400
    team = np.array([0] * 11 + [1] * 11)
    start = rng.uniform(5, 100, size=(22, 2))
    steps = rng.normal(0, 0.15, size=(n_t, 22, 2)).cumsum(axis=0)
    tactical = start[None] + steps
    true_offset = 37
    broadcast = np.full((300, 22, 2), np.nan)
    broadcast[:] = tactical[true_offset:true_offset + 300] + rng.normal(0, 0.3, size=(300, 22, 2))
    frames = np.arange(0, 300, 25)
    offsets = np.arange(0, 80)
    cost, frac = al.block_costs(broadcast, team, tactical, team, frames, offsets)
    assert offsets[int(np.nanargmin(np.nanmean(cost, axis=0)))] == true_offset
    assert np.nanmean(frac[:, true_offset]) > 0.9


def test_fusion_fills_only_missing_tactical_ball():
    n = sio.N_FRAMES
    tactical = np.full((n, 2), np.nan)
    tactical[100] = [10.0, 10.0]                       # tactical has a ball here
    broadcast = np.full((1000, 2), np.nan)
    broadcast[50] = [99.0, 99.0]                       # maps to tactical frame 100 (kept tactical)
    broadcast[60] = [20.0, 20.0]                       # maps to tactical frame 110 (filled)
    offset = np.full(1000, al.NO_ALIGNMENT, dtype=np.int32)
    offset[40:70] = 50
    alignment = al.Alignment(offset, np.zeros(1000, dtype=np.float32), {1: 50, 2: 50})
    fused = fusion.fuse_ball(tactical, broadcast, alignment)
    assert np.allclose(fused.ball_xy[100], [10.0, 10.0]) and not fused.from_broadcast[100]
    assert np.allclose(fused.ball_xy[110], [20.0, 20.0]) and fused.from_broadcast[110]
    assert fused.from_broadcast.sum() == 1
