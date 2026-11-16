"""End-to-end checks of the submission: format, structure and determinism. Need data/."""

import hashlib

import numpy as np
import pytest

from src import io as sio
from src import possession as pos
from src import postprocess as post
from src import run as runner


@pytest.fixture(scope="module")
def result():
    try:
        return runner.build()
    except IOError as error:
        pytest.skip(str(error))


def test_labels_structure(result):
    labels, graded = result.labels, sio.graded_mask()
    assert labels.shape == (sio.N_FRAMES,)
    assert (labels[~graded] == pos.X).all()                 # X outside the two halves
    assert np.isin(labels[graded], [pos.H, pos.A, pos.D]).all()


def test_no_short_possessions(result):
    labels = result.labels
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts, stops = np.concatenate(([0], edges)), np.concatenate((edges, [len(labels)]))
    for a, b in zip(starts, stops):
        if labels[a] in (pos.H, pos.A):
            assert b - a >= runner.V4_POSSESSION["min_poss"]


def test_same_input_same_bytes(result, tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    runner.write_csv(str(first), result.labels)
    runner.write_csv(str(second), runner.build().labels)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert digest(first) == digest(second)
    assert first.read_text().splitlines()[0] == "frame,label"
    assert len(first.read_text().splitlines()) == sio.N_FRAMES + 1


def test_v1_still_reproduces_the_safety_submission(tmp_path):
    path = tmp_path / "v1.csv"
    runner.write_csv(str(path), runner.build("v1").labels)
    assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == (
        "27B67FAE3936C2D42E611E25E448D9AEF7759FD7A4B889CE472CA7B4C1EEFB4F")


def test_v2_is_deterministic_and_well_formed(tmp_path):
    first, second = runner.build('v2'), runner.build('v2')
    assert np.array_equal(first.labels, second.labels)
    graded = sio.graded_mask()
    assert (first.labels[~graded] == pos.X).all()
    assert np.isin(first.labels[graded], [pos.H, pos.A, pos.D]).all()
