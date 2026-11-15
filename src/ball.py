"""Ball cleaning: impossible-jump removal, short-gap interpolation, velocity and acceleration.

Input is the raw ball track in metres, [N, 2] with NaN where the tracker did not see the ball.
Nothing here looks at labels, tagging or any specific frame; every threshold is a physical
quantity or a duration.

Thresholds
----------
V_MAX = 35 m/s
    Speed above which a ball displacement between two observations is treated as impossible.
    Hard shots and long clearances are 25-35 m/s; the fastest measured kicks are about 36-38 m/s
    (130 km/h). A step above 35 m/s between visible samples is therefore a mis-detection (another
    object, a line mark, a boundary-crossing track) rather than ball flight. The speed is measured
    as distance / elapsed time between the two samples, so a long gap tolerates a long jump
    and a one-frame step tolerates at most 1.4 m.

MIN_RUN = 5 samples (0.2 s)
    Samples are split into runs wherever two consecutive visible samples violate V_MAX. A run
    shorter than MIN_RUN that is cut off from its neighbours by such a violation is dropped: a
    "ball" that appears for under 0.2 s and is physically unreachable from both sides is a
    spurious detection. Real ball segments are far longer than 0.2 s, so this removes isolated
    outliers (which create two violations, one in and one out) without removing a ball that is
    genuinely fast, because a fast but real ball produces long consistent runs.

MAX_GAP_S = 1.0 s (25 frames)
    Longest run of missing frames that is linearly interpolated. During open play the ball is
    touched, and so changes direction, roughly every 1-3 s, and a pass takes 0.5-1.5 s; straight
    line motion is a fair model on one second of a pass or a carry and a poor one beyond it,
    where a deflection, bounce or touch is likely to have happened inside the gap. Longer gaps
    stay NaN, so downstream code sees "no ball" instead of an invented trajectory. A gap is also
    left alone if the straight line between its ends would exceed V_MAX.

SMOOTH_WIN = 5 frames (0.2 s)
    Moving-average width for positions (and for velocity before differencing into acceleration).
    Tracking noise is a few decimetres, which differentiated over one frame (0.04 s) is several
    m/s; averaging five frames cuts that by about a factor of two while leaving a pass (several
    metres per second changing over >0.5 s) intact. Kept small because a kick is a one-frame
    velocity change and wide smoothing would smear it.
"""

from typing import NamedTuple

import numpy as np

FPS = 25
V_MAX = 35.0
MIN_RUN = 5
MAX_GAP_FRAMES = 25  # 1.0 s at 25 fps
SMOOTH_WIN = 5


class BallTrack(NamedTuple):
    xy: np.ndarray            # [N, 2] cleaned position (observed-kept + interpolated), NaN elsewhere
    observed: np.ndarray      # [N] bool, ball present in the raw tracking (original)
    outlier: np.ndarray       # [N] bool, observed but removed as physically impossible
    interpolated: np.ndarray  # [N] bool, filled by short-gap interpolation
    vel: np.ndarray           # [N, 2] m/s, NaN where xy is NaN or the run is too short
    speed: np.ndarray         # [N] m/s
    acc: np.ndarray           # [N] m/s^2 (magnitude of the acceleration vector)


def remove_outliers(xy: np.ndarray, v_max: float = V_MAX, min_run: int = MIN_RUN,
                    fps: int = FPS) -> np.ndarray:
    """Boolean [N], True on observed samples removed as impossible jumps."""
    observed = np.isfinite(xy).all(axis=1)
    idx = np.flatnonzero(observed)
    outlier = np.zeros(len(xy), dtype=bool)
    if idx.size < 2:
        return outlier

    dt = np.maximum(np.diff(idx), 1) / fps
    step = np.hypot(*np.diff(xy[idx], axis=0).T)
    violation = step / dt > v_max
    run_id = np.concatenate(([0], np.cumsum(violation)))
    run_len = np.bincount(run_id)
    if run_len.size > 1:  # no violation anywhere -> a single run, nothing to drop
        outlier[idx[run_len[run_id] < min_run]] = True
    return outlier


def interpolate_short_gaps(xy: np.ndarray, max_gap: int = MAX_GAP_FRAMES, v_max: float = V_MAX,
                           fps: int = FPS):
    """Linear interpolation across gaps of at most `max_gap` missing frames.

    Returns (xy_filled, interpolated_mask). Leading and trailing gaps are never filled.
    """
    out = xy.copy()
    filled = np.zeros(len(xy), dtype=bool)
    idx = np.flatnonzero(np.isfinite(xy).all(axis=1))
    for a, b in zip(idx[:-1], idx[1:]):
        missing = b - a - 1
        if missing == 0 or missing > max_gap:
            continue
        if np.hypot(*(xy[b] - xy[a])) / ((b - a) / fps) > v_max:
            continue
        t = (np.arange(a + 1, b) - a) / (b - a)
        out[a + 1:b] = xy[a] + t[:, None] * (xy[b] - xy[a])
        filled[a + 1:b] = True
    return out, filled


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid")
                     for k in range(values.shape[1])], axis=1)


def kinematics(xy: np.ndarray, fps: int = FPS, window: int = SMOOTH_WIN):
    """(vel [N,2], speed [N], acc [N]) from a track whose gaps are NaN.

    Computed separately on every contiguous run of valid positions, so a gap never leaks into a
    derivative. Runs shorter than 3 frames have no defined derivative and stay NaN.
    """
    n = len(xy)
    vel = np.full((n, 2), np.nan)
    acc = np.full(n, np.nan)
    valid = np.isfinite(xy).all(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.view(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start < 3:
            continue
        smooth = _moving_average(xy[start:stop], window)
        v = np.gradient(smooth, 1.0 / fps, axis=0)
        a = np.gradient(_moving_average(v, window), 1.0 / fps, axis=0)
        vel[start:stop] = v
        acc[start:stop] = np.hypot(a[:, 0], a[:, 1])
    speed = np.hypot(vel[:, 0], vel[:, 1])
    return vel, speed, acc


def clean_ball(ball_xy: np.ndarray) -> BallTrack:
    """Full pipeline: outliers out, short gaps in, then velocity and acceleration."""
    observed = np.isfinite(ball_xy).all(axis=1)
    outlier = remove_outliers(ball_xy)
    kept = ball_xy.copy()
    kept[outlier] = np.nan
    xy, interpolated = interpolate_short_gaps(kept)
    vel, speed, acc = kinematics(xy)
    return BallTrack(xy, observed, outlier, interpolated, vel, speed, acc)
