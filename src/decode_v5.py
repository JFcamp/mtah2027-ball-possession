"""Decoder of the stopped-ball probability: threshold, hole closing, minimum stoppage, edge refinement.

    probabilities P(stopped) per frame  ->  boolean stopped mask

Steps, per half (graded frames only):
  1. stopped where P >= threshold
  2. close the live holes shorter than `close_gap` frames that sit between two stopped runs
  3. drop the stopped runs shorter than `min_stop` frames
  4. (optional) edge refinement: each run's start and end are moved by at most REFINE_FRAMES (3 s) to the
     place that maximises the summed log-likelihood ratio of a short-window model (1 s features), i.e. the
     boundary where the short-window evidence flips between live and stopped.
"""

from typing import Optional

import numpy as np

REFINE_FRAMES = 75
LLR_CLIP = 5.0


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def decode(p: np.ndarray, blocks, threshold: float, min_stop: int, close_gap: int, refine: bool,
           p_short: Optional[np.ndarray] = None) -> np.ndarray:
    n = len(p)
    out = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        stop = np.where(np.isfinite(p[sl]), p[sl] >= threshold, False)
        if close_gap > 0:
            runs = _runs(stop)
            for (a0, b0), (a1, b1) in zip(runs[:-1], runs[1:]):
                if a1 - b0 <= close_gap:
                    stop[b0:a1] = True
        keep = np.zeros_like(stop)
        for a, b in _runs(stop):
            if b - a >= min_stop:
                keep[a:b] = True
        if refine and p_short is not None:
            ps = np.clip(np.where(np.isfinite(p_short[sl]), p_short[sl], 0.5), 1e-4, 1 - 1e-4)
            llr = np.clip(np.log(ps / (1 - ps)), -LLR_CLIP, LLR_CLIP)
            cum = np.concatenate(([0.0], np.cumsum(llr)))   # cum[t] = sum of llr[:t]
            length = len(keep)
            refined = np.zeros_like(keep)
            floor = 0
            for a, b in _runs(keep):
                lo, hi = max(floor, a - REFINE_FRAMES), a + REFINE_FRAMES
                a2 = lo + int(np.argmin(cum[lo:min(hi, length) + 1]))       # min S(a') maximises S(a) - S(a')
                lo_b, hi_b = max(a2 + 1, b - REFINE_FRAMES), min(b + REFINE_FRAMES, length)
                if hi_b < lo_b:
                    b2 = b
                else:
                    b2 = lo_b + int(np.argmax(cum[lo_b:hi_b + 1]))
                refined[a2:b2] = True
                floor = b2
            keep = refined
        out[sl] = keep
    return out
