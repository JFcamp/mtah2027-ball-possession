"""Two-state HMM (live / stopped) decoded with Viterbi, per half, on graded frames only.

Emissions: the model's P(stopped) for frames without a pseudo-label; the pseudo-labels (stopped or
live) are hard evidence that pins the state of their frame. R1 and R4 are inside the stopped
pseudo-labels, so they stay hard evidence. This replaces the pace rule R5 and the fixed look-back
and stoppage caps of v1.

Transition probabilities come from football durations, not from data fitting:
  stopped -> live  1 / (25 s * 25 fps) = 0.0016 per frame  (expected stoppage 25 s)
      A typical stoppage lasts 10-30 s: a throw-in 10-15 s, a goal kick, corner or free kick
      20-35 s, and the rarer injuries, substitutions and goals are much longer.
  live -> stopped  1 / (40 s * 25 fps) = 0.0010 per frame  (expected rolling spell 40 s)
      A match has on the order of 100 stoppages in ~55-60 minutes of live play, so uninterrupted
      spells average 30-45 s.
These imply ~38% of the time stopped, the usual share of dead time in a professional match.

Evidence tempering: a frame's evidence is not independent of the next one (features average over
1 s and 5 s, and the tracking is 25 fps), so the per-frame log-probabilities are multiplied by
TEMPER = 1/25, as if one independent observation per second. Without it the HMM would follow every
fluctuation of the model and ignore the transition prior.
"""

import numpy as np

FPS = 25
EXPECTED_STOP_S = 25.0
EXPECTED_LIVE_S = 40.0
P_STOP_TO_LIVE = 1.0 / (EXPECTED_STOP_S * FPS)
P_LIVE_TO_STOP = 1.0 / (EXPECTED_LIVE_S * FPS)
TEMPER = 1.0 / FPS
EPS = 1e-6

LIVE, STOPPED = 0, 1


def viterbi(p_stop: np.ndarray, hard: np.ndarray, expected_stop_s=None, expected_live_s=None,
            temper=None) -> np.ndarray:
    """Most likely state path.

    p_stop: [T] model P(stopped), NaN = no evidence. hard: [T] int, -1 free, 0 live, 1 stopped.
    Returns [T] bool, True where stopped.
    """
    es = EXPECTED_STOP_S if expected_stop_s is None else expected_stop_s
    el = EXPECTED_LIVE_S if expected_live_s is None else expected_live_s
    tp = TEMPER if temper is None else temper
    p_stop_to_live, p_live_to_stop = 1.0 / (es * FPS), 1.0 / (el * FPS)
    t_len = len(p_stop)
    p = np.clip(np.where(np.isfinite(p_stop), p_stop, 0.5), EPS, 1 - EPS)
    emit = np.stack([np.log(1 - p), np.log(p)], axis=1) * tp
    emit[hard == LIVE, STOPPED] = -np.inf
    emit[hard == STOPPED, LIVE] = -np.inf

    stay = {LIVE: np.log(1 - p_live_to_stop), STOPPED: np.log(1 - p_stop_to_live)}
    move = {LIVE: np.log(p_live_to_stop), STOPPED: np.log(p_stop_to_live)}  # from the key state
    # log transition matrix [from, to]
    trans = np.array([[stay[LIVE], move[LIVE]], [move[STOPPED], stay[STOPPED]]])

    score = np.empty((t_len, 2))
    back = np.zeros((t_len, 2), dtype=np.int8)
    # Start: the opening of a half is a stoppage (kick-off); the prior is the stationary one.
    start = np.log(np.array([el, es]) / (el + es))
    score[0] = start + emit[0]
    for t in range(1, t_len):
        cand = score[t - 1][:, None] + trans          # [from, to]
        back[t] = np.argmax(cand, axis=0)
        score[t] = cand[back[t], [0, 1]] + emit[t]
    path = np.empty(t_len, dtype=np.int8)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path == STOPPED
