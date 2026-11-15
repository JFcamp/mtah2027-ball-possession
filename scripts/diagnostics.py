"""E2 diagnostics -> docs/diagnostics_e2.md. Aggregate statistics only, on graded frames only.

    python scripts/diagnostics.py

Uses no ground truth. Smart Tagging is used only to describe what the ball does around tagged
events. Nothing per-frame is exported.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import ball as ballmod  # noqa: E402
from src import io as sio  # noqa: E402

FPS = sio.FPS
LENGTH, WIDTH = sio.PITCH
STILL = 0.5  # m/s, "ball nearly still"
OUT_PATH = os.path.join(ROOT, "docs", "diagnostics_e2.md")


def fmt(value, digits=2):
    return "-" if value is None or not np.isfinite(value) else "{:.{}f}".format(value, digits)


def pct(part, whole):
    return 100.0 * part / whole if whole else np.nan


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def percentiles(values, qs):
    values = values[np.isfinite(values)]
    return [np.percentile(values, q) for q in qs] if values.size else [np.nan] * len(qs)


def runs(mask):
    """(start, stop) pairs of consecutive True values."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def out_of_bounds(xy, margin):
    x, y = xy[:, 0], xy[:, 1]
    return (x < -margin) | (x > LENGTH + margin) | (y < -margin) | (y > WIDTH + margin)


GAP_BINS = [("< 0.5 s", 0, 0.5), ("0.5 - 1 s", 0.5, 1), ("1 - 2 s", 1, 2), ("2 - 5 s", 2, 5),
            ("5 - 10 s", 5, 10), ("> 10 s", 10, np.inf)]


def gap_section(lines, blocks, raw_present, clean_present):
    lines += ["## a) Gaps without the ball", "",
              "A gap is a maximal run of frames with no ball inside one half. *raw* is the tracking "
              "as delivered; *clean* is after impossible-jump removal and <= 1 s interpolation "
              "(`src/ball.py`). Percentages are of the graded frames of that half.", ""]
    for half, first, last in blocks:
        n = last - first + 1
        lines.append("### Half {} ({} graded frames)".format(half, n))
        lines.append("")
        stats = {}
        for name, present in (("raw", raw_present), ("clean", clean_present)):
            seg = present[first:last + 1]
            gaps = [(b - a) / FPS for a, b in runs(~seg)]
            stats[name] = gaps
            lines.append("- ball present, {}: {:.1f}% of frames ({} gaps)".format(
                name, pct(seg.sum(), n), len(gaps)))
        lines.append("")
        rows = []
        for label, lo, hi in GAP_BINS:
            row = [label]
            for name in ("raw", "clean"):
                g = np.array(stats[name])
                sel = g[(g >= lo) & (g < hi)] if g.size else g
                row += [len(sel), fmt(pct(sel.sum() * FPS, n), 1)]
            rows.append(row)
        lines += table(["gap length", "raw: n gaps", "raw: % frames", "clean: n gaps",
                        "clean: % frames"], rows)
    return lines


def bounds_section(lines, blocks, raw_present, kept):
    lines += ["## b) Ball outside the pitch", "",
              "Pitch = {:.1f} x {:.1f} m. A margin of m means more than m metres beyond a "
              "touchline or goal line. Percentage of frames in which the ball is present. *raw* "
              "includes the frames later removed as impossible jumps; *kept* excludes them."
              .format(LENGTH, WIDTH), ""]
    rows = []
    for half, first, last in blocks:
        sl = slice(first, last + 1)
        for name, present in (("raw", raw_present), ("kept", kept.present)):
            xy = (ARRAYS["ball_xy"] if name == "raw" else kept.xy)[sl][present[sl]]
            row = [half, name, len(xy)]
            for margin in (0.0, 0.5, 1.0):
                row.append(fmt(pct(out_of_bounds(xy, margin).sum(), len(xy)), 1))
            rows.append(row)
    lines += table(["half", "ball", "frames with ball", "outside (0 m)", "outside > 0.5 m",
                    "outside > 1 m"], rows)
    return lines


def speed_section(lines, blocks, track):
    lines += ["## c) Ball speed", "",
              "Cleaned ball (observed + interpolated), speed from the smoothed track, m/s. "
              "'Still' means < {} m/s. A still stretch is a run of consecutive frames below that "
              "speed (a missing ball ends it).".format(STILL), ""]
    qs = [5, 25, 50, 75, 90, 95, 99, 99.9]
    rows, still_rows = [], []
    for half, first, last in blocks:
        speed = track.speed[first:last + 1]
        finite = speed[np.isfinite(speed)]
        rows.append([half, len(finite)] + [fmt(v, 1) for v in percentiles(speed, qs)]
                    + [fmt(finite.max(), 1), fmt(pct((finite < STILL).sum(), len(finite)), 1)])

        still = np.isfinite(speed) & (speed < STILL)
        durations = np.array([(b - a) / FPS for a, b in runs(still)])
        if durations.size:
            q = np.percentile(durations, [50, 75, 90, 99])
            still_rows.append([half, len(durations), fmt(durations.sum() * FPS / len(speed) * 100, 1)]
                              + [fmt(v, 1) for v in q] + [fmt(durations.max(), 1),
                                                          int((durations >= 1).sum()),
                                                          int((durations >= 5).sum())])
    lines += table(["half", "frames with speed"] + ["p{}".format(q) for q in qs]
                   + ["max", "% still"], rows)
    lines += ["Still stretches (duration in seconds):", ""]
    lines += table(["half", "n stretches", "% of half's frames", "median", "p75", "p90", "p99",
                    "max", "n >= 1 s", "n >= 5 s"], still_rows)
    return lines


def nearest_distances(players, ball_xy, rows):
    """Distance from the ball to the nearest tracked player, for the given frames."""
    team = ARRAYS["team"]
    out = {"all": [], "home": [], "away": []}
    for lo in range(0, len(rows), 20000):
        idx = rows[lo:lo + 20000]
        diff = players[idx] - ball_xy[idx][:, None, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        for name, sel in (("all", slice(None)), ("home", team == sio.HOME),
                          ("away", team == sio.AWAY)):
            d = dist[:, sel]
            with np.errstate(all="ignore"):
                out[name].append(np.where(np.isfinite(d).any(axis=1),
                                          np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1),
                                          np.nan))
    return {k: np.concatenate(v) for k, v in out.items()}


def nearest_section(lines, blocks, tracking, kept):
    lines += ["## d) Distance from the ball to the nearest player", "",
              "Frames with an observed (kept) ball; metres. 'home' / 'away' is the nearest player "
              "of that team, 'all' the nearest of either.", ""]
    qs = [5, 25, 50, 75, 95, 99]
    rows = []
    for half, first, last in blocks:
        frames = np.arange(first, last + 1)
        frames = frames[kept.observed_kept[first:last + 1]]
        d = nearest_distances(tracking.players_xy, tracking.ball_xy, frames)
        for name in ("all", "home", "away"):
            rows.append([half, name, int(np.isfinite(d[name]).sum())]
                        + [fmt(v, 1) for v in percentiles(d[name], qs)])
    lines += table(["half", "team", "frames"] + ["p{}".format(q) for q in qs], rows)
    return lines


def players_section(lines, blocks, tracking):
    lines += ["## e) Tracked players per frame", "",
              "Count of the 46 track slots that have a position, per graded frame.", ""]
    present = np.isfinite(tracking.players_xy[..., 0])
    qs = [1, 5, 25, 50, 75, 95, 99]
    rows = []
    for half, first, last in blocks:
        sl = slice(first, last + 1)
        for name, sel in (("both", slice(None)), ("home", tracking.team_of_slot == sio.HOME),
                          ("away", tracking.team_of_slot == sio.AWAY)):
            count = present[sl][:, sel].sum(axis=1).astype(float)
            rows.append([half, name, fmt(count.mean(), 2)] + [fmt(v, 0) for v in
                                                              percentiles(count, qs)])
    lines += table(["half", "team", "mean"] + ["p{}".format(q) for q in qs], rows)
    return lines


# ---- set pieces and goals -------------------------------------------------------------------

def window_stats(track, frames):
    """Ball behaviour pooled over `frames` (graded only)."""
    frames = frames[GRADED[frames]]
    n = len(frames)
    if n == 0:
        return [0, "-", "-", "-", "-"]
    vis = track.observed[frames].mean() * 100
    speed = track.speed[frames]
    finite = speed[np.isfinite(speed)]
    still = pct((finite < STILL).sum(), len(finite)) if finite.size else np.nan
    kept = track.observed[frames] & ~track.outlier[frames]
    oob = pct(out_of_bounds(track.xy[frames][kept], 0.0).sum(), kept.sum()) if kept.any() else np.nan
    med = np.median(finite) if finite.size else np.nan
    return [n, fmt(vis, 0), fmt(still, 0), fmt(oob, 0), fmt(med, 1)]


WINDOW_HEAD = ["window (s from anchor)", "frames", "% ball seen", "% still", "% outside",
               "median speed"]


def window_rows(track, anchors, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        frames = np.concatenate([np.arange(a + int(round(lo * FPS)), a + int(round(hi * FPS)))
                                 for a in anchors]) if len(anchors) else np.array([], int)
        frames = frames[(frames >= 0) & (frames < sio.N_FRAMES)]
        rows.append(["{:+.1f} .. {:+.1f}".format(lo, hi)] + window_stats(track, frames))
    return rows


def setpiece_section(lines, track, intervals):
    sp = [i for i in intervals if i.code == "SET PIECES"]
    types = sorted({i.tags.get("Type") for i in sp})
    lines += ["## f) Smart Tagging set pieces", "",
              "Interval durations, then what the ball does around the interval *start* "
              "(10 s before to 5 s after), pooled over all instances of a type, graded frames "
              "only. '% still' is the share of frames with speed < {} m/s; '% outside' is of "
              "frames with the ball seen; speed is the cleaned ball's median in m/s."
              .format(STILL), ""]
    rows = []
    for t in types:
        d = np.array([(i.end_frame - i.start_frame) / FPS for i in sp if i.tags.get("Type") == t])
        rows.append([t, len(d), fmt(d.min(), 1), fmt(np.median(d), 1), fmt(d.max(), 1)])
    lines += table(["Type", "n", "min (s)", "median (s)", "max (s)"], rows)

    edges = [-10, -7.5, -5, -2.5, 0, 2.5, 5]
    for t in types:
        anchors = [i.start_frame for i in sp if i.tags.get("Type") == t]
        lines.append("**{}** (n = {})".format(t, len(anchors)))
        lines.append("")
        lines += table(WINDOW_HEAD, window_rows(track, anchors, edges))

    lines += ["**All set pieces except kick-offs, 1 s bins** (n = {})".format(
        sum(1 for i in sp if "Kick Off" not in i.tags.get("Type", ""))), ""]
    anchors = [i.start_frame for i in sp if "Kick Off" not in i.tags.get("Type", "")]
    lines += table(WINDOW_HEAD, window_rows(track, anchors, list(range(-10, 6))))
    return lines


def goals_section(lines, track, intervals):
    goals = sorted([i for i in intervals if i.code == "GOALS"], key=lambda i: i.start_frame)
    kicks = sorted([i for i in intervals if i.code == "SET PIECES"
                    and i.tags.get("Type") == "Kick Off (After Goal)"],
                   key=lambda i: i.start_frame)
    pairs = []
    for g in goals:
        nxt = next((k for k in kicks if k.start_frame >= g.start_frame
                    and k.tags.get("Half") == g.tags.get("Half")), None)
        pairs.append((g, nxt))

    lines += ["## g) Smart Tagging goals up to the next Kick Off (After Goal)", "",
              "Each GOALS interval is paired with the first Kick Off (After Goal) of the same "
              "half that starts after it.", ""]
    rows = []
    for g, k in pairs:
        rows.append([g.tags.get("Half"), g.tags.get("Team"), g.start_frame, g.end_frame,
                     fmt((g.end_frame - g.start_frame) / FPS, 1),
                     k.start_frame if k else "-",
                     fmt((k.start_frame - g.start_frame) / FPS, 1) if k else "-",
                     fmt((g.end_frame - k.start_frame) / FPS, 1) if k else "-"])
    lines += table(["half", "team", "goal start", "goal end", "goal length (s)",
                    "kick-off start", "goal start -> kick-off (s)", "goal end - kick-off start (s)"],
                   rows)

    edges = [-10, -7.5, -5, -2.5, 0, 2.5, 5]
    lines += ["**Anchored at the goal start** (n = {})".format(len(pairs)), ""]
    lines += table(WINDOW_HEAD, window_rows(track, [g.start_frame for g, _ in pairs], edges))
    lines += ["**Anchored at the Kick Off (After Goal) start** (n = {})".format(
        sum(1 for _, k in pairs if k)), ""]
    lines += table(WINDOW_HEAD, window_rows(track, [k.start_frame for _, k in pairs if k], edges))

    # Between the two: the stretch goal start -> kick-off start, in 10 s steps, pooled.
    span = []
    for g, k in pairs:
        if k:
            span.append(np.arange(g.start_frame, k.start_frame))
    lines += ["**Whole stretch from goal start to kick-off start, by time since the goal start** "
              "(10 s bins, pooled)", ""]
    rows = []
    for lo in range(0, 160, 10):
        frames = np.concatenate([s[(s - s[0] >= lo * FPS) & (s - s[0] < (lo + 10) * FPS)]
                                 for s in span])
        rows.append(["{} .. {}".format(lo, lo + 10)] + window_stats(track, frames))
    lines += table(["s since goal start", "frames", "% ball seen", "% still", "% outside",
                    "median speed"], rows)
    return lines


# ---- main -------------------------------------------------------------------------------------

ARRAYS = {}
GRADED = None


class _Kept:
    pass


def main():
    global GRADED
    tracking = sio.load_tracking()
    track = ballmod.clean_ball(tracking.ball_xy)
    GRADED = sio.graded_mask()
    blocks = sio.half_blocks()
    ARRAYS.update(ball_xy=tracking.ball_xy, team=tracking.team_of_slot)

    present_clean = np.isfinite(track.xy[:, 0])
    kept = _Kept()
    kept.xy = track.xy.copy()
    kept.xy[track.interpolated] = np.nan  # observed and not an outlier
    kept.xy[track.outlier] = np.nan
    kept.present = np.isfinite(kept.xy[:, 0])
    kept.observed_kept = kept.present

    graded_n = int(GRADED.sum())
    lines = ["# E2 diagnostics", "",
             "Generated by `scripts/diagnostics.py`. Graded frames only ({} frames, 25 fps); "
             "aggregate statistics only. No ground truth was used.".format(graded_n), "",
             "Ball cleaning summary (graded frames): observed {} ({:.1f}%), removed as impossible "
             "jumps {}, interpolated {}, ball available after cleaning {} ({:.1f}%). "
             "Thresholds: V_MAX = {} m/s, MIN_RUN = {} samples, MAX_GAP = {} frames, "
             "SMOOTH_WIN = {} frames.".format(
                 int((track.observed & GRADED).sum()),
                 pct((track.observed & GRADED).sum(), graded_n),
                 int((track.outlier & GRADED).sum()), int((track.interpolated & GRADED).sum()),
                 int((present_clean & GRADED).sum()), pct((present_clean & GRADED).sum(), graded_n),
                 ballmod.V_MAX, ballmod.MIN_RUN, ballmod.MAX_GAP_FRAMES, ballmod.SMOOTH_WIN), ""]

    lines = gap_section(lines, blocks, track.observed, present_clean)
    lines = bounds_section(lines, blocks, track.observed, _Bounds(kept, track))
    lines = speed_section(lines, blocks, track)
    lines = nearest_section(lines, blocks, tracking, kept)
    lines = players_section(lines, blocks, tracking)

    intervals = sio.load_smart_tagging()
    lines = setpiece_section(lines, track, intervals)
    lines = goals_section(lines, track, intervals)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    print("wrote", os.path.relpath(OUT_PATH, ROOT))


class _Bounds:
    """Adapter so bounds_section can read .present and .xy of the kept (non-interpolated) ball."""

    def __init__(self, kept, track):
        self.present, self.xy = kept.present, kept.xy


if __name__ == "__main__":
    main()
