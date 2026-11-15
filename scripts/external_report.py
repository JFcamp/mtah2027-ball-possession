"""docs/external_data.md: the three public Metrica sample games as a validation set.

    python scripts/external_report.py

Labels come only from the provider's data (see src/external.py). This script describes them,
checks their coherence with the provider's events, and scores the official baseline logic on them.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from src import external as ext  # noqa: E402

FPS = 25
STOP_EVENTS = ("BALL OUT", "FAULT RECEIVED", "SET PIECE", "CARD")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def f(x, d=1):
    return "{:.{}f}".format(x, d)


def baseline_labels(game, window):
    """starter/baseline.py: nearest player to the ball, majority vote of +/- window frames; D without ball."""
    import baseline
    n = len(game.ball_xy)
    codes = np.full(n, baseline.NOBODY, dtype=np.int8)
    ok = np.flatnonzero(np.isfinite(game.ball_xy[:, 0]))
    for lo in range(0, len(ok), 20000):
        idx = ok[lo:lo + 20000]
        d = np.hypot(game.players_xy[idx, :, 0] - game.ball_xy[idx, None, 0],
                     game.players_xy[idx, :, 1] - game.ball_xy[idx, None, 1])
        d = np.where(np.isfinite(d), d, np.inf)
        k = np.argmin(d, axis=1)
        codes[idx] = np.where(np.isfinite(d[np.arange(len(idx)), k]), game.team_of_slot[k], baseline.NOBODY)
    lab = baseline.smooth(codes, window) if window else baseline.smooth(codes, 0)
    out = np.where(lab == "H", 0, np.where(lab == "A", 1, 2)).astype(np.int8)
    out[~ext.graded_mask(game)] = ext.X
    return out


def main():
    L = ["# External validation data: Metrica sample games", "",
         "Source: https://github.com/metrica-sports/sample-data (Sample_Game_1, 2, 3), downloaded to "
         "`data_external/` (git-ignored, not redistributed). Tracking and events were produced and "
         "corrected by Metrica Sports; nothing was labelled by hand by us. None of this data is from "
         "BAR-ATH and none of the BAR-ATH data is used to choose anything in this stage.", ""]
    L += ["## Label rules", "",
          "- **D**: the ball is absent from the corrected tracking, inside a period (the tracking of "
          "these games is corrected: the ball exists exactly while it is in play).",
          "- **H / A**: the team of the latest ball event of the live spell. Ball events: PASS, SHOT, "
          "RECOVERY, SET PIECE and (game 3 only) CARRY. BALL LOST, CHALLENGE, FAULT RECEIVED, CARD and "
          "BALL OUT never change the owner, so the owner changes only when the opponent has a ball event "
          "of its own. Frames of a spell before its first event take that event's team.",
          "- **X**: outside the periods, and the first frame of the second period (so the official "
          "scorer does not join the halves).", ""]

    games = [(g, ext.load_game(g)) for g in ext.GAMES]
    rows = []
    for g, G in games:
        lab = ext.labels(G)
        graded = ext.graded_mask(G)
        poss, dur, stops = [], [], []
        for _, a, z in ext.half_blocks(G):
            seg = lab[a:z + 1]
            for lbl in (0, 1):
                pass
            edges = np.flatnonzero(np.diff(seg)) + 1
            starts, stops_ = np.concatenate(([0], edges)), np.concatenate((edges, [len(seg)]))
            for s, e in zip(starts, stops_):
                if seg[s] in (0, 1):
                    poss.append((e - s) / FPS)
                elif seg[s] == 2:
                    stops.append((e - s) / FPS)
        long_stops = [s for s in stops if s >= 2]
        n = int(graded.sum())
        rows.append([G.name, n, len(poss), f(np.mean(poss)), f(100 * (lab == 0).sum() / n),
                     f(100 * (lab == 1).sum() / n), f(100 * (lab == 2).sum() / n), len(stops),
                     len(long_stops), f(np.mean(long_stops)), f(np.median(long_stops))])
    L += ["## Per game", ""]
    L += table(["game", "graded frames", "possessions", "mean duration (s)", "% H", "% A", "% D",
                "D runs", "D runs >= 2 s", "mean (s)", "median (s)"], rows)

    # ---- coherence with events -----------------------------------------------------------------
    L += ["## Coherence of D (ball absent) with the provider's events", "",
          "If the ball is absent exactly while it is dead, ball events should find the ball present, "
          "and stoppage events should be followed by an absent ball.", ""]
    rows = []
    for g, G in games:
        present = np.isfinite(G.ball_xy[:, 0])
        n = len(present)
        ev = G.events
        live = ev[ev["type"].isin(("PASS", "SHOT", "CARRY")) & (ev["end"] > ev["start"])]
        in_event = []
        for s, e in zip(live["start"], live["end"]):
            if e - s < 400:
                in_event.append(present[s:e + 1].mean())
        bo = ev[ev["type"] == "BALL OUT"]["start"].to_numpy()
        after = [(~present[s + 25:min(s + 75, n)]).mean() for s in bo if s + 75 < n]
        sp = ev[(ev["type"] == "SET PIECE") & (ev["subtype"].str.split("+").str[0] != "KICK OFF")]["start"].to_numpy()
        before = [(~present[s - 75:s - 25]).mean() for s in sp if s > 75]
        resume = [present[s + 10:s + 50].mean() for s in sp if s + 50 < n]
        fr = ev[ev["type"] == "FAULT RECEIVED"]["start"].to_numpy()
        fr_after = [(~present[s + 1:s + 51]).mean() for s in fr if s + 51 < n]
        stop_starts = ev[ev["type"].isin(STOP_EVENTS)]["start"].to_numpy()
        d_runs = [(a, b) for a, b in ext.runs(~present & ext.graded_mask(G)) if b - a >= 50]
        near = [any(a - 150 <= s <= b + 150 for s in stop_starts) for a, b in d_runs]
        rows.append([G.name,
                     "{} ({} events)".format(f(100 * np.mean(in_event)), len(in_event)),
                     "{} ({})".format(f(100 * np.mean(after)), len(after)),
                     "{} ({})".format(f(100 * np.mean(before)), len(before)),
                     "{} ({})".format(f(100 * np.mean(resume)), len(resume)),
                     "{} ({})".format(f(100 * np.mean(fr_after)), len(fr_after)),
                     "{} of {}".format(int(np.sum(near)), len(d_runs))])
    L += table(["game", "% frames with ball during PASS/SHOT/CARRY", "% ball absent 1-3 s after BALL OUT",
                "% ball absent 1-3 s before a restart (corner, free kick, throw-in...)",
                "% ball present 0.4-2 s after the restart", "% ball absent 0-2 s after FAULT RECEIVED",
                "D runs >= 2 s with a stoppage event within 6 s"], rows)

    # ---- baseline with the official metrics ---------------------------------------------------
    L += ["## Official metrics on these labels (starter baseline logic)", "",
          "Nearest player to the ball, majority vote of +/- w frames, D where the ball is absent "
          "(`starter/baseline.py` functions), scored with `evaluation/possession_metrics.py`. On BAR-ATH "
          "the public window gave spellF1 0.04-0.42 and macroF1 0.58-0.62 for the same four windows.", ""]
    rows = []
    for g, G in games:
        gt = ext.labels(G)
        for w in (0, 25, 50, 125):
            s = ext.score(gt, baseline_labels(G, w))
            rows.append([G.name, "smooth{}".format(w) if w else "raw", f(s["macro_f1"], 4),
                         f(s["spell_f1"], 4), s["n_pred"], s["n_gt"], f(s["dur"])])
    L += table(["game", "baseline", "macroF1", "spellF1", "predicted possessions",
                "true possessions", "mean duration (s)"], rows)

    L += ["## Differences from the CHALLENGE.md definitions", "",
          "- BAR-ATH labels come from corrected **tracking** (D where the corrected tracking marks the ball "
          "dead; otherwise the team the tracking names as owner). Here D is the same idea (ball absent in the "
          "corrected tracking) but H/A comes from **events**: the team of the latest ball event. A change of "
          "possession here is the opponent's first ball event (a pass, shot, recovery or carry), which is a "
          "touch, not necessarily the 'controlled possession' of CHALLENGE.md; a recovery that is "
          "immediately lost counts as a change.",
          "- In BAR-ATH the ground truth names an owner during loose-ball phases through the tracking; here "
          "the owner is held until the opponent has an event, the same stay-with-the-last-team idea.",
          "- Event timestamps are the touch times; the tracking ball can reappear a few frames before or "
          "after them, so spell boundaries here carry an error of a few frames.",
          "- Pitch: 105 x 68 m (BAR-ATH: 105.3 x 68). Tracking is corrected and has no per-player speed, so "
          "player speeds are computed from positions (as in the pipeline). Corrected tracking is smoother "
          "than the automatic BAR-ATH tracking: speed-based thresholds may not transfer identically.",
          "- No Smart Tagging exists for these games: no Team tag at restarts, no goal/set-piece anchors.", ""]

    path = os.path.join(ROOT, "docs", "external_data.md")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
