"""Checks of the v5 building blocks on synthetic data (no match data needed, except where noted)."""

import numpy as np
import pytest

from src import decode_v5 as dec5
from src import degrade
from src import external as ext
from src import features_players as fp
from src import features_v5 as f5
from src import models_v5 as m5


def _players(n=1500, p=6, seed=0):
    rng = np.random.RandomState(seed)
    start = rng.uniform(10, 90, size=(1, p, 2))
    return start + rng.normal(0, 0.05, size=(n, p, 2)).cumsum(axis=0), np.array([0, 0, 0, 1, 1, 1])


def test_v5_features_contain_the_v4_features_unchanged():
    players, teams = _players()
    new, names = f5.compute(players, teams)
    old, old_names = fp.compute(players, teams)
    keep = [i for i, n in enumerate(old_names) if not n.startswith("n_tracked")]
    cols = f5.l0_columns(names)
    np.testing.assert_allclose(new[:, cols], old[:, keep].astype(np.float32), rtol=1e-4, atol=1e-5,
                               equal_nan=True)


def test_v5_features_are_shares_not_counts_and_ignore_the_frame_index():
    players, teams = _players()
    x, names = f5.compute(players, teams)
    assert not any("tracked" in n or "index" in n for n in names)
    shifted, _ = f5.compute(players[300:], teams)   # a different frame range gives the same features
    np.testing.assert_allclose(x[600:900], shifted[300:600], rtol=1e-4, atol=1e-5, equal_nan=True)


def test_decode_threshold_closing_minimum_and_refinement():
    p = np.zeros(1000)
    p[200:400] = 0.9
    p[420:600] = 0.9         # hole of 20 frames
    p[700:720] = 0.9         # a 20-frame run
    blocks = [(1, 0, 999)]
    plain = dec5.decode(p, blocks, 0.5, 50, 0, False)
    assert plain[200:400].all() and not plain[400:420].any() and not plain[700:720].any()
    closed = dec5.decode(p, blocks, 0.5, 50, 25, False)
    assert closed[200:600].all()
    short = np.where(np.arange(1000) >= 180, 0.9, 0.1)    # the short model says: stopped from frame 180
    refined = dec5.decode(p, blocks, 0.5, 50, 25, True, short)
    assert refined[185:200].all() and not refined[:170].any()


def test_degrade_is_deterministic_and_removes_players_only_where_present():
    players, teams = _players(n=3000, p=6)
    game = ext.Game("syn", np.full((3000, 2), 50.0), players, teams, np.ones(3000, dtype=np.int8),
                    None)
    a, b = degrade.degrade_game(game, 3), degrade.degrade_game(game, 3)
    np.testing.assert_array_equal(a.players_xy, b.players_xy)
    c = degrade.degrade_game(game, 4)
    assert not np.array_equal(np.nan_to_num(a.players_xy), np.nan_to_num(c.players_xy))
    assert np.isfinite(a.ball_xy).all()                       # the ball is moved, never removed here
    moved = np.nanmedian(np.abs(a.players_xy - players))
    assert 0.02 < moved < 0.5                                 # a decimetre-scale error


def test_gradient_boosting_export_matches_scikit_learn():
    sk = pytest.importorskip("sklearn")
    rng = np.random.RandomState(0)
    x = rng.normal(size=(3000, 6))
    y = (x[:, 0] + 0.5 * x[:, 1] ** 2 + rng.normal(0, 0.3, 3000) > 0.5).astype(int)
    model = m5.train("G1", x, y, list(range(6)), 1)
    from sklearn.ensemble import GradientBoostingClassifier
    weight = np.where(y == 1, len(y) / (2.0 * (y == 1).sum()), len(y) / (2.0 * (y == 0).sum()))
    clf = GradientBoostingClassifier(**m5.GB_PARAMS).fit(x, y, sample_weight=weight)
    np.testing.assert_allclose(m5.predict(model, x), clf.predict_proba(x)[:, 1], atol=1e-9)
    # NaN is replaced by the training mean
    xn = x.copy()
    xn[:5, 0] = np.nan
    filled = xn.copy()
    filled[:5, 0] = np.array(model["mean"])[0]
    np.testing.assert_allclose(m5.predict(model, xn), clf.predict_proba(filled)[:, 1], atol=1e-9)
