"""Checks that the parse of the tactical ATD lines up with the video, the official baseline and the
Smart Tagging. They need the match data in data/ and skip if it is not there."""

import itertools
import os

import numpy as np
import pandas as pd
import pytest

from src import io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_RAW = os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")


@pytest.fixture(scope="module")
def tracking():
    try:
        return sio.load_tracking()
    except IOError as error:
        pytest.skip(str(error))


def _line(path, index):
    with open(path) as handle:
        return next(itertools.islice(handle, index, index + 1))


def test_shapes(tracking):
    assert sio.N_FRAMES == 199319
    assert tracking.ball_xy.shape == (199319, 2)
    assert tracking.players_xy.shape == (199319, 46, 2)
    assert tracking.players_speed.shape == (199319, 46)
    assert tracking.team_of_slot.shape == (46,)
    assert set(np.unique(tracking.team_of_slot)) == {0, 1}


@pytest.mark.parametrize("line_index", [0, 11184, 77776, 199318])
def test_video_frame_is_frame_id_minus_one(tracking, line_index):
    """Re-read single lines straight from the file and compare with the array row frame_id - 1."""
    from games import require
    raw = require(["atd_raw"])[0]
    parts = _line(raw, line_index).split(":")
    frame_id = int(parts[0])
    assert frame_id == line_index + 1  # the file is contiguous and starts at 1
    row = frame_id - 1

    length, width = sio.PITCH
    ball = [float(v) for v in parts[2].split(",")[:2]]
    expected_ball = np.array(ball) * np.array([length, width])
    np.testing.assert_array_equal(tracking.ball_xy[row], expected_ball)

    first_slot = [float(v) for v in parts[1].split(";")[0].split(",")]
    expected = np.array(first_slot[:2]) * np.array([length, width])
    np.testing.assert_array_equal(tracking.players_xy[row, 0], expected)
    np.testing.assert_array_equal(tracking.players_speed[row, 0], first_slot[2])


def test_reproduces_baseline_raw(tracking):
    """Nearest-player-to-the-ball on our arrays must give exactly the official baseline_raw.

    Checked on every frame, in chunks. The baseline says D where the ball is missing, otherwise
    the team of the closest player (first slot wins a tie).
    """
    official = pd.read_csv(BASELINE_RAW)["label"].to_numpy()
    assert len(official) == sio.N_FRAMES

    ball, players, team = tracking.ball_xy, tracking.players_xy, tracking.team_of_slot
    ours = np.full(sio.N_FRAMES, "D", dtype="<U1")
    for lo in range(0, sio.N_FRAMES, 20000):
        hi = min(lo + 20000, sio.N_FRAMES)
        visible = np.isfinite(ball[lo:hi, 0])
        diff = players[lo:hi] - ball[lo:hi, None, :]
        dist = diff[..., 0] ** 2 + diff[..., 1] ** 2
        dist[np.isnan(dist)] = np.inf
        nearest = np.argmin(dist, axis=1)
        has_player = np.isfinite(dist[np.arange(hi - lo), nearest])
        pick = np.where(team[nearest] == sio.HOME, "H", "A")
        chunk = np.full(hi - lo, "D", dtype="<U1")
        ok = visible & has_player
        chunk[ok] = pick[ok]
        ours[lo:hi] = chunk

    mismatch = np.flatnonzero(ours != official)
    assert mismatch.size == 0, "first mismatches at frames {}".format(mismatch[:10])


def test_kickoffs_open_each_half():
    """Both 'Kick Off (Start)' tags fall inside their half block of games.half_video_blocks."""
    intervals = sio.load_smart_tagging()
    kickoffs = [i for i in intervals if i.code == "SET PIECES"
                and i.tags.get("Type") == "Kick Off (Start)"]
    assert len(kickoffs) == 2

    blocks = {half: (first, last) for half, first, last in sio.half_blocks()}
    by_half = {i.tags["Half"]: i for i in kickoffs}
    assert set(by_half) == {"1st Half", "2nd Half"}

    for name, half in (("1st Half", 1), ("2nd Half", 2)):
        first, last = blocks[half]
        interval = by_half[name]
        # The tag is a ~5 s window around the restart, so it may straddle the block edge.
        assert interval.end_frame >= first and interval.start_frame <= last, name
        # It sits at the start of its half: within the first 5% of the block.
        assert abs(interval.start_frame - first) < 0.05 * (last - first), name

    # First half: the graded block opens inside the tagged kick-off window.
    first_half = by_half["1st Half"]
    assert first_half.start_frame <= blocks[1][0] <= first_half.end_frame
