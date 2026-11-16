"""Clean-up of the label sequence: no possession shorter than a real one, no flicker.

MIN_POSSESSION = 38 frames (1.5 s)
    A possession is a team in control of the ball. Teams do not hold the ball for less than about
    1.5 s in a way that matters: a first-time pass chain moves it every 1-2 s, while a touch
    without control (deflection, clearance, aerial challenge) lasts a fraction of that. Runs
    shorter than this are absorbed by the neighbouring possession; a run with no possession
    neighbour (stoppage on both sides) is itself a stoppage, not a possession.
"""

import numpy as np

from . import possession as pos

MIN_POSSESSION = 38


def _runs(labels: np.ndarray):
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(labels)]))
    return list(zip(starts.tolist(), stops.tolist()))


def merge_short_possessions(labels: np.ndarray, min_len: int = MIN_POSSESSION) -> np.ndarray:
    """Repeatedly absorb the shortest H/A run below `min_len`; ties go to the earliest."""
    labels = labels.copy()
    while True:
        runs = _runs(labels)
        short = [(b - a, a, i) for i, (a, b) in enumerate(runs)
                 if labels[a] in (pos.H, pos.A) and b - a < min_len]
        if not short:
            return labels
        _, a, i = min(short)
        b = runs[i][1]
        left = labels[runs[i - 1][0]] if i > 0 else pos.X
        right = labels[runs[i + 1][0]] if i + 1 < len(runs) else pos.X
        left_len = runs[i - 1][1] - runs[i - 1][0] if i > 0 else 0
        right_len = runs[i + 1][1] - runs[i + 1][0] if i + 1 < len(runs) else 0
        candidates = [(length, lab) for length, lab in ((left_len, left), (right_len, right))
                      if lab in (pos.H, pos.A)]
        labels[a:b] = max(candidates)[1] if candidates else pos.D


def finalize(labels: np.ndarray, graded: np.ndarray, min_len=None) -> np.ndarray:
    """Merge short possessions inside each graded block and force X outside the blocks."""
    min_len = MIN_POSSESSION if min_len is None else min_len
    out = np.full(len(labels), pos.X, dtype=np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], graded.astype(np.int8), [0]))))
    for lo, hi in zip(edges[::2], edges[1::2]):
        out[lo:hi] = merge_short_possessions(labels[lo:hi], min_len)
    out[~graded] = pos.X
    return out
