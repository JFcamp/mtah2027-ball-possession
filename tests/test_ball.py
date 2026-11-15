"""Synthetic checks of the ball cleaning. No match data needed."""

import numpy as np

from src import ball


def _track(n=400, speed=10.0):
    t = np.arange(n) / ball.FPS
    return np.stack([10.0 + speed * t, np.full(n, 30.0)], axis=1)


def test_isolated_outlier_is_removed_and_neighbours_kept():
    xy = _track()
    xy[100] = [90.0, 5.0]
    out = ball.remove_outliers(xy)
    assert out[100] and out.sum() == 1


def test_fast_but_continuous_ball_is_kept():
    xy = _track(speed=30.0)  # 30 m/s < V_MAX
    assert ball.remove_outliers(xy).sum() == 0


def test_short_gap_is_filled_and_long_gap_is_not():
    xy = _track()
    xy[50:60] = np.nan   # 10 frames, filled
    xy[200:260] = np.nan  # 60 frames, left alone
    filled, mask = ball.interpolate_short_gaps(xy)
    assert mask[50:60].all() and not mask[200:260].any()
    np.testing.assert_allclose(filled[50:60], _track()[50:60], atol=1e-9)
    assert np.isnan(filled[200:260]).all()


def test_kinematics_recover_constant_speed_and_zero_acceleration():
    result = ball.clean_ball(_track(speed=12.0))
    assert np.allclose(result.speed[10:-10], 12.0, atol=1e-6)
    assert np.allclose(result.acc[10:-10], 0.0, atol=1e-6)
    assert not result.interpolated.any() and not result.outlier.any()


def test_gap_does_not_leak_into_derivatives():
    xy = _track()
    xy[100:200] = np.nan
    result = ball.clean_ball(xy)
    assert np.isnan(result.speed[100:200]).all()
    assert np.isfinite(result.speed[:100]).all()
