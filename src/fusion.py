"""Ball fusion: fill tactical frames that have no ball with the aligned broadcast ball.

The tactical ball is never replaced where it exists; the broadcast ball is used only where the
tactical tracker has none and the broadcast frame is aligned (src/align.py). The fused track then
goes through the same cleaning as before (src/ball.py: impossible-jump removal, short-gap
interpolation, kinematics) with no threshold changed. `from_broadcast` marks the filled frames.
"""

from typing import NamedTuple

import numpy as np

from . import align as al
from . import io as sio


class Fused(NamedTuple):
    ball_xy: np.ndarray         # [N, 2] tactical ball, completed with the broadcast ball
    from_broadcast: np.ndarray  # [N] bool
    source_frame: np.ndarray    # [N] int32, broadcast frame used (-1 where none)


def tactical_frame_of_broadcast(alignment: al.Alignment):
    """(broadcast frames, tactical frames) of every aligned broadcast frame inside the video."""
    b = np.flatnonzero(alignment.offset != al.NO_ALIGNMENT)
    t = b + alignment.offset[b].astype(np.int64)
    ok = (t >= 0) & (t < sio.N_FRAMES)
    return b[ok], t[ok]


def fuse_ball(tactical_ball: np.ndarray, broadcast_ball: np.ndarray,
              alignment: al.Alignment) -> Fused:
    b, t = tactical_frame_of_broadcast(alignment)
    # one broadcast frame per tactical frame: keep the first (lowest broadcast frame)
    t, first = np.unique(t, return_index=True)
    b = b[first]
    has_b = np.isfinite(broadcast_ball[b]).all(axis=1)
    b, t = b[has_b], t[has_b]
    fill = ~np.isfinite(tactical_ball[t]).all(axis=1)
    b, t = b[fill], t[fill]

    fused = tactical_ball.copy()
    fused[t] = broadcast_ball[b]
    from_b = np.zeros(sio.N_FRAMES, dtype=bool)
    from_b[t] = True
    source = np.full(sio.N_FRAMES, -1, dtype=np.int32)
    source[t] = b
    return Fused(fused, from_b, source)
