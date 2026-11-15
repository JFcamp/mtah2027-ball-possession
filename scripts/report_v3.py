"""docs/report_v3.md: broadcast alignment validation (no ground truth), ball coverage, v1 vs v3 on
the whole match, the decision criterion and the public window (reported only).

    python scripts/report_v3.py

Decision criterion, fixed before any public score of v3 was seen. v3 becomes final iff
 (a) on frames where both cameras see the ball, the median distance between the two balls <= 2 m
     and the p90 <= 5 m, and
 (b) ball coverage on graded frames rises by at least 5 percentage points, and
 (c) whole-match statistics stay plausible: D between 30% and 50% of graded frames and at most
     1.5x the v1 number of possessions.
"""

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import align as al  # noqa: E402
from src import broadcast as bcast  # noqa: E402
from src import fusion  # noqa: E402
from src import io as sio  # noqa: E402
from src import possession as pos  # noqa: E402
from src import run as runner  # noqa: E402

FPS = sio.FPS


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def f(x, d=1):
    return "{:.{}f}".format(x, d)


def possessions(labels):
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts, stops = np.concatenate(([0], edges)), np.concatenate((edges, [len(labels)]))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if labels[a] in (pos.H, pos.A)]


def runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def pct(x, q):
    return np.percentile(x, q) if len(x) else float("nan")


def main():
    tactical = sio.load_tracking()
    broadcast = bcast.load_tracking()
    valid = bcast.load_valid()
    alignment = al.align(broadcast, valid, tactical)
    graded = sio.graded_mask()
    L = ["# Report v3", "",
         "v3 = the v1 pipeline with no threshold changed, fed with a ball track completed by the "
         "broadcast ball where the tactical tracker has none. Sections 1-3 use no ground truth; the "
         "public window at the end is only reported.", ""]

    # ---- 1. alignment --------------------------------------------------------------------------
    ok = alignment.offset != al.NO_ALIGNMENT
    b_idx, t_idx = fusion.tactical_frame_of_broadcast(alignment)
    L += ["## 1. Alignment of the broadcast onto the tactical timeline", "",
          "Matching of same-team player positions only (the ball is not used). Tactical frame = "
          "broadcast frame + offset.", ""]
    n_b = len(valid)
    L += table(["", "broadcast frames", "% of all", "% of valid homography"],
               [["total", n_b, "100.0", "-"], ["valid homography", int(valid.sum()),
                                                f(100 * valid.mean()), "100.0"],
                ["aligned", int(ok.sum()), f(100 * ok.mean()), f(100 * ok.sum() / valid.sum())]])
    rows = []
    for half in (1, 2):
        o = alignment.offset[ok]
        o = o[o < 15000] if half == 1 else o[o >= 15000]
        rows.append([half, len(o), alignment.dominant[half], int(np.median(o)),
                     "{}..{}".format(int(o.min()), int(o.max())),
                     "{:.1f}%".format(100 * np.mean(np.abs(o - np.median(o)) <= 2))])
    L += table(["half", "aligned frames", "coarse offset", "median refined offset", "range",
                "within 2 frames of median"], rows)
    L += ["The refined offset is constant to within a few frames in each half; the second half is "
          "~22,900 frames later (halftime and the start of the broadcast).", ""]

    # player agreement on a sample of aligned frames (every 10th)
    sample = np.arange(0, len(b_idx), 10)
    dists = []
    n_fr = []
    for b, t in zip(b_idx[sample], t_idx[sample]):
        p = broadcast.players_xy[b]
        okp = np.isfinite(p[:, 0])
        if okp.sum() < al.MIN_PLAYERS:
            continue
        q = tactical.players_xy[t]
        d = np.hypot(p[okp, None, 0] - q[None, :, 0], p[okp, None, 1] - q[None, :, 1])
        same = broadcast.team_of_slot[okp][:, None] == tactical.team_of_slot[None, :]
        d = np.where(same & np.isfinite(d), d, np.inf)
        nn = d.min(axis=1)
        dists.append(nn[np.isfinite(nn)])
        n_fr.append(1)
    dists = np.concatenate(dists)
    L += ["Players: for every 10th aligned frame with >= {} visible broadcast players, distance "
          "from each broadcast player to the nearest same-team tactical player ({} frames, {} "
          "players): median {} m, p90 {} m, p99 {} m.".format(
              al.MIN_PLAYERS, len(n_fr), len(dists), f(pct(dists, 50), 2), f(pct(dists, 90), 2),
              f(pct(dists, 99), 2)), ""]

    tb = tactical.ball_xy[t_idx]
    bb = broadcast.ball_xy[b_idx]
    co = np.isfinite(tb).all(axis=1) & np.isfinite(bb).all(axis=1)
    dball = np.hypot(*(tb[co] - bb[co]).T)
    dball_g = dball[graded[t_idx[co]]]
    L += ["Ball: frames where both cameras see the ball ({} aligned frames; {} on graded frames).".format(
        int(co.sum()), len(dball_g)), ""]
    L += table(["distance between the two ball positions (m)", "p50", "p75", "p90", "p95", "p99"],
               [["co-visible frames", f(pct(dball, 50), 2), f(pct(dball, 75), 2), f(pct(dball, 90), 2),
                 f(pct(dball, 95), 2), f(pct(dball, 99), 2)],
                ["graded only", f(pct(dball_g, 50), 2), f(pct(dball_g, 75), 2), f(pct(dball_g, 90), 2),
                 f(pct(dball_g, 95), 2), f(pct(dball_g, 99), 2)]])
    crit_a = pct(dball, 50) <= 2.0 and pct(dball, 90) <= 5.0

    # ---- 2. coverage ---------------------------------------------------------------------------
    v1 = runner.build("v1")
    v3 = runner.build("v3")
    fu = v3.extra["fused"]
    seen_t = np.isfinite(tactical.ball_xy[:, 0])
    seen_f = np.isfinite(fu.ball_xy[:, 0])
    clean1 = np.isfinite(v1.track.xy[:, 0])
    clean3 = np.isfinite(v3.track.xy[:, 0])
    L += ["## 2. Ball coverage on graded frames", ""]
    rows = []
    for name, mask_ in (("both halves", graded),) + tuple(
            ("half {}".format(h), (np.arange(sio.N_FRAMES) >= a) & (np.arange(sio.N_FRAMES) <= z))
            for h, a, z in sio.half_blocks()):
        n = int(mask_.sum())
        rows.append([name, f(100 * (seen_t & mask_).sum() / n), f(100 * (seen_f & mask_).sum() / n),
                     f(100 * (clean1 & mask_).sum() / n), f(100 * (clean3 & mask_).sum() / n),
                     int((fu.from_broadcast & mask_).sum()),
                     f(100 * ((clean3 & mask_).sum() - (clean1 & mask_).sum()) / n, 2)])
    L += table(["frames", "% raw tactical", "% raw fused", "% cleaned v1", "% cleaned v3",
                "frames from broadcast", "gain after cleaning (pp)"], rows)
    gain_clean = 100 * ((clean3 & graded).sum() - (clean1 & graded).sum()) / graded.sum()
    gain_raw = 100 * ((seen_f & graded).sum() - (seen_t & graded).sum()) / graded.sum()
    crit_b = gain_clean >= 5.0 and gain_raw >= 5.0
    miss_t = graded & ~seen_t
    L += ["The tactical ball is missing on {} graded frames; the aligned broadcast has a ball on {} "
          "of them ({}%). Where one tracker loses the ball the other usually does too.".format(
              int(miss_t.sum()), int((fu.from_broadcast & graded).sum()),
              f(100 * (fu.from_broadcast & graded).sum() / miss_t.sum())), ""]

    # ---- 3. whole match ------------------------------------------------------------------------
    L += ["## 3. Whole match: v1 and v3", ""]
    rows = []
    for half, first, last in sio.half_blocks():
        for name, res in (("v1", v1), ("v3", v3)):
            seg = res.labels[first:last + 1]
            poss = possessions(seg)
            dur = [(b - a) / FPS for a, b in poss]
            stops = [(b - a) / FPS for a, b in runs(seg == pos.D)]
            rows.append([half, name, len(poss), f(np.mean(dur)), f(100 * (seg == pos.H).mean()),
                         f(100 * (seg == pos.A).mean()), f(100 * (seg == pos.D).mean()), len(stops),
                         f(np.mean(stops)), f(np.median(stops)), f(max(stops))])
    L += table(["half", "version", "possessions", "mean poss. (s)", "% H", "% A", "% D",
                "stoppages", "mean (s)", "median (s)", "max (s)"], rows)
    n_poss = {n: sum(len(possessions(r.labels[a:z + 1])) for _, a, z in sio.half_blocks())
              for n, r in (("v1", v1), ("v3", v3))}
    d_share = float((v3.labels[graded] == pos.D).mean())
    crit_c = 0.30 <= d_share <= 0.50 and n_poss["v3"] <= 1.5 * n_poss["v1"]
    diff = graded & (v1.labels != v3.labels)
    L += ["v1 and v3 labels differ on {} graded frames ({}%).".format(
        int(diff.sum()), f(100 * diff.sum() / graded.sum(), 2)), ""]

    # ---- decision ------------------------------------------------------------------------------
    L += ["## Decision criterion (fixed beforehand)", "",
          "- (a) co-visible ball distance median <= 2 m and p90 <= 5 m: median {} m, p90 {} m: **{}**".format(
              f(pct(dball, 50), 2), f(pct(dball, 90), 2), "PASS" if crit_a else "FAIL"),
          "- (b) coverage gain >= 5 pp: +{} pp raw, +{} pp after cleaning: **{}**".format(
              f(gain_raw, 2), f(gain_clean, 2), "PASS" if crit_b else "FAIL"),
          "- (c) D share {}% (30-50%), possessions v3 {} vs v1 {} (limit {:.0f}): **{}**".format(
              f(100 * d_share), n_poss["v3"], n_poss["v1"], 1.5 * n_poss["v1"],
              "PASS" if crit_c else "FAIL"),
          "- Result: **{}**".format("v3 becomes the final version" if crit_a and crit_b and crit_c
                                   else "v1 stays the final version"), ""]

    # ---- public window -------------------------------------------------------------------------
    results_dir = os.path.join(ROOT, "evaluation", "results")
    for name, res in (("v1", v1), ("v3", v3)):
        runner.write_csv(os.path.join(results_dir, name + ".csv"), res.labels)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                           "--bootstrap", "1000"], cwd=ROOT, capture_output=True, text=True)
    text = proc.stdout.splitlines()
    board = [l for l in text if l.startswith(("rank", "----")) or
             (l.split()[:1] and l.split()[0].isdigit() and len(l.split()) > 8)]
    L += ["## Public 5-minute window (reported only, not used for the decision)", "",
          "17 possessions: differences below ~0.3 spellF1 are noise.", "", "```"] + board + ["```", ""]
    for name in ("v1", "v3"):
        start = text.index(name)
        end = next(i for i in range(start + 1, len(text)) if text[i] and not text[i].startswith(" "))
        L += ["```"] + text[start:end] + ["```", ""]

    out = os.path.join(ROOT, "docs", "report_v3.md")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", os.path.relpath(out, ROOT), "| a", crit_a, "b", crit_b, "c", crit_c)


if __name__ == "__main__":
    main()
