"""docs/report_v1.md: whole-match statistics (no ground truth) and the public-window score.

    python scripts/report_v1.py [--csv submissions/v1.csv]

The public window is only reported. Nothing in the pipeline reads it.
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import dead_ball as db  # noqa: E402
from src import io as sio  # noqa: E402
from src import possession as pos  # noqa: E402
from src import run as runner  # noqa: E402

FPS = sio.FPS
SIDE = {0: "home", 1: "away", None: "-"}


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def possessions(labels):
    """[(label, start, stop)] of maximal H / A runs."""
    out = []
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(labels)]))
    for a, b in zip(starts, stops):
        if labels[a] in (pos.H, pos.A):
            out.append((int(labels[a]), int(a), int(b)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(ROOT, "submissions", "v1.csv"))
    args = parser.parse_args()

    result = runner.build()
    labels = result.labels
    blocks = sio.half_blocks()
    L = ["# Report v1", "",
         "Whole-match statistics use no ground truth. The public-window score at the end is only "
         "reported; no threshold was adjusted after seeing it.", ""]

    # ---- whole match ---------------------------------------------------------------------------
    L += ["## Whole match, graded frames", ""]
    rows = []
    for half, first, last in blocks:
        seg = labels[first:last + 1]
        poss = possessions(seg)
        dur = [(b - a) / FPS for _, a, b in poss]
        n = len(seg)
        rows.append([half, n, len(poss), "{:.1f}".format(np.mean(dur)),
                     "{:.1f}".format(100 * (seg == pos.H).mean()),
                     "{:.1f}".format(100 * (seg == pos.A).mean()),
                     "{:.1f}".format(100 * (seg == pos.D).mean())])
    seg_all = np.concatenate([labels[a:z + 1] for _, a, z in blocks])
    poss_all = [p for _, a, z in blocks for p in possessions(labels[a:z + 1])]
    dur_all = [(b - a) / FPS for _, a, b in poss_all]
    rows.append(["both", len(seg_all), len(poss_all), "{:.1f}".format(np.mean(dur_all)),
                 "{:.1f}".format(100 * (seg_all == pos.H).mean()),
                 "{:.1f}".format(100 * (seg_all == pos.A).mean()),
                 "{:.1f}".format(100 * (seg_all == pos.D).mean())])
    L += table(["half", "graded frames", "possessions", "mean duration (s)", "% H", "% A",
                "% D"], rows)
    L += ["Possession = maximal run of H or of A; D separates possessions. Before "
          "post-processing there were {} possession runs; short ones are merged "
          "(`src/postprocess.py`).".format(
              sum(len(possessions(result.pre_merge[a:z + 1])) for _, a, z in blocks)), ""]

    # ---- dead ball by rule ---------------------------------------------------------------------
    L += ["## Dead-ball frames by rule", "",
          "A frame covered by several rules is credited to the first of "
          "R1, R4, R3, R2, R6, R5 (listed in that order of priority). 'raw' counts every frame a "
          "rule marks, overlaps included. Both after the 2 s minimum-duration filter.", ""]
    rows = []
    for half, first, last in blocks:
        taken = np.zeros(sio.N_FRAMES, dtype=bool)
        credit = {}
        for r in db.PRIORITY:
            m = result.dead.by_rule[r].copy()
            m[:first] = False
            m[last + 1:] = False
            credit[r] = int((m & ~taken).sum())
            taken |= m
        n = last - first + 1
        raw = {r: int(result.dead.by_rule[r][first:last + 1].sum()) for r in db.RULES}
        rows.append([half] + ["{} ({:.1f}%) / raw {}".format(credit[r], 100 * credit[r] / n, raw[r])
                              for r in db.RULES]
                    + [int(result.dead.dead[first:last + 1].sum())])
    L += table(["half"] + list(db.RULES) + ["total D"], rows)
    L += ["Frames removed by the minimum duration (2 s), whole match: {}.".format(
        result.dead.credited["removed (< 2 s)"]), ""]

    # ---- phase semantics -----------------------------------------------------------------------
    L += ["## phase_team: reading of the Smart Tagging Team tag", "",
          "Agreement = share of graded frames with the ball visible, inside the phase, where the "
          "nearest team to the ball is the one named in the Team tag. No labels involved. "
          "'Prior' comes from the code name; it is used only if the measured agreement lies on the "
          "same side of 0.5 as the prior predicts.", ""]
    rows = [[code, n, "{:.3f}".format(a), reading]
            for code, (n, a, reading) in result.phase_table.items()]
    L += table(["code", "frames", "agreement with nearest team", "reading used"], rows)
    g = sio.graded_mask()
    sel = g & (result.features.phase_team >= 0) & (result.features.nearest_team >= 0)
    L += ["Combined `phase_team` covers {:.1f}% of graded frames and agrees with the nearest team "
          "on {:.1f}% of the graded frames where both exist.".format(
              100 * (g & (result.features.phase_team >= 0)).sum() / g.sum(),
              100 * (result.features.phase_team[sel] == result.features.nearest_team[sel]).mean()),
          ""]

    # ---- restart team check --------------------------------------------------------------------
    L += ["## Restart team: tag vs first controller", "",
          "For restarts whose team is known (Team tag of the set piece; for a Kick Off (After "
          "Goal), the team that did not score), is the first team to control the ball after "
          "the restart the same? Throw-ins and kick-offs at the start are N/A in the tags and "
          "are handled by the first controller.", ""]
    stats = {}
    for s in result.segments:
        if s.restart_kind is None:
            continue
        key = s.restart_kind
        d = stats.setdefault(key, {"n": 0, "known": 0, "agree": 0, "ctrl": 0})
        d["n"] += 1
        if s.first_controller is not None:
            d["ctrl"] += 1
        taker = next((r.team for r in result.dead.restarts
                      if r.kind == key and abs(r.frame - s.start) <= 100), None)
        if taker is not None and s.first_controller is not None:
            d["known"] += 1
            d["agree"] += int(taker == s.first_controller)
    rows = [[k, v["n"], v["ctrl"], v["known"], v["agree"]] for k, v in sorted(stats.items())]
    L += table(["restart type", "segments", "with a controller", "team known and controller seen",
                "first controller = taker"], rows)

    # ---- possession segments -------------------------------------------------------------------
    n_seg = len(result.segments)
    L += ["## Live segments", "",
          "{} live segments; {} owner switches in total, {} of them placed inside a ball gap "
          "(phase change or middle of the gap); segments with no control episode: {} "
          "(phase fallback {}, previous owner {}, nothing {}).".format(
              n_seg, sum(s.n_switches for s in result.segments),
              sum(s.n_gap_switches for s in result.segments),
              sum(1 for s in result.segments if s.fallback),
              sum(1 for s in result.segments if s.fallback == "phase"),
              sum(1 for s in result.segments if s.fallback == "previous"),
              sum(1 for s in result.segments if s.fallback == "none")), ""]

    # ---- public window -------------------------------------------------------------------------
    results_dir = os.path.join(ROOT, "evaluation", "results")
    name = os.path.splitext(os.path.basename(args.csv))[0]
    shutil.copyfile(args.csv, os.path.join(results_dir, name + ".csv"))
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                           "--bootstrap", "1000"], cwd=ROOT, capture_output=True, text=True)
    text = proc.stdout.splitlines()
    board = [l for l in text if l.startswith(("rank", "----")) or
             (l.split()[:1] and l.split()[0].isdigit() and len(l.split()) > 8)]
    start = text.index(name)
    end = next(i for i in range(start + 1, len(text)) if text[i] and not text[i].startswith(" "))
    L += ["## Public 5-minute window (reported, not tuned on)", "",
          "Frames 11185-18684, 17 possessions. Produced by `python evaluation/run_evaluation.py "
          "--bootstrap 1000` after copying the CSV to `evaluation/results/{}.csv`. With 17 "
          "possessions the bootstrap interval is wide: differences below ~0.3 spellF1 are "
          "noise.".format(name), "", "```"] + board + ["```", "", "```"] + text[start:end] + ["```", ""]

    out = os.path.join(ROOT, "docs", "report_v1.md")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
