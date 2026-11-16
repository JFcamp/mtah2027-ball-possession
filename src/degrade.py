"""Degrade the (corrected) external games towards the statistics of the BAR-ATH automatic tracking.

Every constant below comes from a measurement WITHOUT labels, see docs/domain_gap.md:

PLAYER_SIGMA = 0.17 m, BALL_SIGMA = 0.47 m
    The two BAR-ATH trackers (tactical and broadcast, aligned in v3) disagree on a player by a median
    of 0.29 m (nearest same-team neighbour) and on the ball by a median of 0.79 m (central part;
    co-visible frames). For two independent trackers with the same error s per axis, the median radial
    disagreement is 1.177 * sqrt(2) * s, so s ~ 0.17 m for players and ~ 0.47 m for the ball. These are
    upper bounds on the error of one tracker if part of the disagreement is common to both.
PLAYER_SMOOTH = 25 frames, BALL_SMOOTH = 10 frames
    The errors are slow. The ball disagreement has autocorrelation 0.995 at 1 frame, 0.90 at 5, 0.28
    at 1 s, and the second differences of the BAR-ATH players are as small as those of the corrected
    games (p50 1.1 mm in both, docs/domain_gap.md): the players' error must not add frame-to-frame
    jitter, so it is Gaussian-smoothed white noise with a 1 s kernel (a first attempt with a 10-frame
    kernel doubled the players' second differences to 1.8-2.1 mm, above BAR-ATH). The ball's BAR-ATH
    second differences ARE larger than the corrected games' (p50 4.8 mm vs 1.0 mm), which a 10-frame
    kernel on 0.47 m reproduces.
Dropouts (BURST_*, SOLO_*)
    Tracked players per frame in BAR-ATH: mean 20.5, std 3.3, p1 5, p5 14, p25 21, median 22. Two
    processes reproduce that: each present player is lost independently (stationary share 2%, mean
    5 s), and bursts (2 per minute, mean 2 s) take away a random 5-90% of the players at once.
    Fitted by a grid on those percentiles; achieved values are in docs/domain_gap.md.
LIVE_ERASE_BAR = 0.106
    Share of the frames labelled H/A by v4 in which the ball is absent (10.6%, 10.5% / 10.7% by half).
Not simulated: wrong-ball episodes. The cross-camera disagreement above 5 m (10% of the co-visible
frames) cannot be attributed to a tracker, and balls in the air project differently in two cameras,
so part of it is not an error. The attributable jumps are 0.3% of the observed ball frames in
BAR-ATH and are removed by src/ball.py in both domains.
"""

import numpy as np

from . import external as ext

PLAYER_SIGMA = 0.17
BALL_SIGMA = 0.47
PLAYER_SMOOTH = 25
BALL_SMOOTH = 10
BURST_RATE_PER_MIN = 2.0
BURST_MEAN_S = 2.0
BURST_MAX_FRACTION = 0.9
SOLO_SHARE = 0.02
SOLO_MEAN_S = 5.0
LIVE_ERASE_BAR = 0.106
FPS = 25


def _smooth_noise(n: int, series: int, sigma: float, smooth: int, rng) -> np.ndarray:
    """[n, series] zero-mean Gaussian noise, sd `sigma`, slowly varying (kernel sd `smooth` frames)."""
    half = 3 * smooth
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / smooth) ** 2)
    kernel /= np.sqrt((kernel ** 2).sum())
    white = rng.normal(size=(n + 2 * half, series))
    out = np.empty((n, series))
    for k in range(series):
        out[:, k] = np.convolve(white[:, k], kernel, mode="valid")
    return out * sigma


def _dropouts(present: np.ndarray, rng) -> np.ndarray:
    """[N, P] bool, True where a present player is lost."""
    n, p = present.shape
    p_on = 1.0 / (SOLO_MEAN_S * FPS)
    p_off = p_on * SOLO_SHARE / (1 - SOLO_SHARE)
    state = rng.uniform(size=p) < SOLO_SHARE
    u = rng.uniform(size=(n, p))
    lost = np.zeros((n, p), dtype=bool)
    for t in range(n):
        state = np.where(state, ~(u[t] < p_on), u[t] < p_off)
        lost[t] = state
    for _ in range(int(BURST_RATE_PER_MIN * n / FPS / 60)):
        start = rng.randint(n)
        length = int(rng.exponential(BURST_MEAN_S * FPS)) + 1
        fraction = rng.uniform(0.05, BURST_MAX_FRACTION)
        who = rng.uniform(size=p) < fraction
        lost[start:start + length] |= who[None, :]
    return lost & present


def degrade_game(game: ext.Game, seed: int) -> ext.Game:
    """The game with position errors on players and ball, and player dropouts (deterministic)."""
    rng = np.random.RandomState(seed)
    n, p = game.players_xy.shape[:2]
    noise_p = _smooth_noise(n, 2 * p, PLAYER_SIGMA, PLAYER_SMOOTH, rng).reshape(n, p, 2)
    noise_b = _smooth_noise(n, 2, BALL_SIGMA, BALL_SMOOTH, rng)
    players = game.players_xy + noise_p
    lost = _dropouts(np.isfinite(game.players_xy[..., 0]), rng)
    players[lost] = np.nan
    ball = game.ball_xy + noise_b
    return game._replace(players_xy=players, ball_xy=ball)
