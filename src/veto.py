"""Live-play veto by pass chains (v5, only if it passes its criterion).

A chain is 3 or more consecutive control episodes of the SAME team with the ball inside the field, at
positions at least 5 m apart, and at most 4 s between consecutive episodes: a team passing the ball
around is in live play. The veto covers the chain from its first to its last episode and removes those
frames from the model's stopped mask M (never from the rules R1-R4, R6).
"""

import numpy as np

from . import calibration as cal
from . import io as sio
from . import possession as pos

MIN_EPISODES = 3
MIN_DISTANCE = 5.0   # m between the ball at consecutive episodes
MAX_GAP = 100        # frames (4 s)


def chain_veto(track, features, config: dict) -> np.ndarray:
    n = sio.N_FRAMES
    present = np.isfinite(track.xy[:, 0])
    with cal.configured(config):
        control = pos.control_teams(features, present)
    veto = np.zeros(n, dtype=bool)
    for _, first, last in sio.half_blocks():
        eps = pos.episodes(control, first, last + 1)
        # position of the ball at the middle of each episode, if it is inside the field
        mids = []
        for team, s, e in eps:
            m = (s + e) // 2
            ok = present[m] and not features.ball_out[m]
            mids.append((team, s, e, track.xy[m] if ok else None))
        chain = [0] if mids else []
        chains = []
        for i in range(1, len(mids)):
            a, b = mids[i - 1], mids[i]
            linked = (a[0] == b[0] and a[3] is not None and b[3] is not None
                      and b[1] - a[2] <= MAX_GAP and np.hypot(*(a[3] - b[3])) >= MIN_DISTANCE)
            if linked:
                chain.append(i)
            else:
                chains.append(chain)
                chain = [i]
        chains.append(chain)
        for c in chains:
            if len(c) >= MIN_EPISODES:
                veto[mids[c[0]][1]:mids[c[-1]][2] + 1] = True
    return veto
