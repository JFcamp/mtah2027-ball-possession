"""docs/report_v2.md: cross-validation of the live/stopped model on pseudo-labels, v1 vs v2 on the
whole match, the decision criterion, and the public window (reported only).

    python scripts/report_v2.py

Decision criterion, fixed before looking at any public score. v2 becomes final iff
 (a) in BOTH directions (train 1st half -> test 2nd, train 2nd -> test 1st) the model beats the
     v1 pace rule R5 (activity 5 s mean < 1.4 m/s) on the pseudo-labelled frames of the unseen half,
     on AUC and on balanced accuracy, and
 (b) the whole-match statistics stay plausible: D between 30% and 50% of graded frames, no
     stoppage longer than 180 s that is not a goal or the opening of a half, and a number of
     possessions at most 1.5 x the v1 count.
"""

import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import dead_ball as db  # noqa: E402
from src import dead_hmm as hmm  # noqa: E402
from src import dead_model as dm  # noqa: E402
from src import features as feat  # noqa: E402
from src import io as sio  # noqa: E402
from src import possession as pos  # noqa: E402
from src import run as runner  # noqa: E402

FPS = sio.FPS
LETTER = {pos.H: "H", pos.A: "A", pos.D: "D"}


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def possessions(labels):
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts, stops = np.concatenate(([0], edges)), np.concatenate((edges, [len(labels)]))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if labels[a] in (pos.H, pos.A)]


def stoppages(dead_slice):
    return [(a, b) for a, b in db.runs(dead_slice)]


def f(x, d=1):
    return "{:.{}f}".format(x, d)


def main():
    v1 = runner.build("v1")
    v2 = runner.build("v2")
    e = v2.extra
    graded = sio.graded_mask()
    blocks = sio.half_blocks()
    L = ["# Report v2", "",
         "Live/stopped model trained on algorithmic pseudo-labels (no human labels, no ground "
         "truth), decoded with a two-state HMM, replacing rule R5 and the look-back/stoppage caps "
         "of v1. Everything below the cross-validation uses no ground truth; the public window at "
         "the end is only reported.", ""]

    # ---- pseudo-labels -------------------------------------------------------------------------
    c = e["pseudo_counts"]
    L += ["## Pseudo-labels (graded frames)", ""]
    L += table(["source", "frames"], [[k, v] for k, v in c.items()])

    # ---- cross-validation ----------------------------------------------------------------------
    L += ["## Cross-validation on pseudo-labels, no ground truth", "",
          "Train on one half, test on the pseudo-labelled frames of the other. The rule is v1's R5 "
          "applied as a stopped/live classifier everywhere (5 s mean of the median player speed "
          "< 1.4 m/s = stopped), on exactly the same test frames (those where that mean exists). "
          "Model threshold 0.5 (class-balanced training). Accuracy depends on the class mix, so "
          "balanced accuracy and AUC are the comparison.", ""]
    pseudo, x = e["pseudo"], e["x"]
    act5 = feat.rolling_nanmean(v2.features.activity, db.ACT_WIN)
    rows, wins = [], []
    halves = {h: (a, z) for h, a, z in blocks}
    for train_h, test_h in ((1, 2), (2, 1)):
        ta, tz = halves[train_h]
        sa, sz = halves[test_h]
        tr_idx = np.arange(ta, tz + 1)
        tr_idx = tr_idx[pseudo[tr_idx] >= 0]
        te_idx = np.arange(sa, sz + 1)
        te_idx = te_idx[(pseudo[te_idx] >= 0) & np.isfinite(act5[te_idx])]
        model = dm.fit(x[tr_idx], pseudo[tr_idx].astype(int))
        prob = dm.predict_proba(model, x[te_idx])
        y = pseudo[te_idx].astype(int)
        m = dm.scores(prob >= 0.5, y)
        m_auc = dm.auc(prob, y)
        r = dm.scores(act5[te_idx] < db.A_WALK, y)
        r_auc = dm.auc(-act5[te_idx], y)
        for name, s, a in (("model", m, m_auc), ("rule R5", r, r_auc)):
            rows.append(["train {} -> test {}".format(train_h, test_h), name, len(te_idx),
                         int((y == 1).sum()), f(s["accuracy"], 3), f(s["balanced_accuracy"], 3),
                         f(a, 3)])
        wins.append(m_auc > r_auc and m["balanced_accuracy"] > r["balanced_accuracy"])
    L += table(["direction", "classifier", "test frames", "of which stopped", "accuracy",
                "balanced accuracy", "AUC"], rows)
    crit_a = all(wins)

    # ---- whole match v1 x v2 -------------------------------------------------------------------
    L += ["## Whole match: v1 and v2", ""]
    rows = []
    for half, first, last in blocks:
        for name, res in (("v1", v1), ("v2", v2)):
            seg = res.labels[first:last + 1]
            poss = possessions(seg)
            dur = [(b - a) / FPS for a, b in poss]
            stops = [(b - a) / FPS for a, b in stoppages(seg == pos.D)]
            rows.append([half, name, len(poss), f(np.mean(dur)),
                         f(100 * (seg == pos.H).mean()), f(100 * (seg == pos.A).mean()),
                         f(100 * (seg == pos.D).mean()), len(stops), f(np.mean(stops)),
                         f(np.median(stops)), f(np.percentile(stops, 90)), f(max(stops))])
    L += table(["half", "version", "possessions", "mean poss. (s)", "% H", "% A", "% D",
                "stoppages", "mean (s)", "median (s)", "p90 (s)", "max (s)"], rows)

    n_poss = {name: sum(len(possessions(res.labels[a:z + 1])) for _, a, z in blocks)
              for name, res in (("v1", v1), ("v2", v2))}
    d_share = float((v2.labels[graded] == pos.D).mean())
    guard = db.build  # noqa: F841  (keeps the import used)
    structural = v1.dead.by_rule["R1"] | v1.dead.by_rule["R4"]
    long_stops = []
    for _, a, z in blocks:
        for s, t in stoppages(v2.labels[a:z + 1] == pos.D):
            if (t - s) / FPS > 180 and not structural[a + s:a + t].any():
                long_stops.append(((t - s) / FPS, a + s))
    crit_b = (0.30 <= d_share <= 0.50 and not long_stops and n_poss["v2"] <= 1.5 * n_poss["v1"])

    # ---- disagreement --------------------------------------------------------------------------
    L += ["## Where v1 and v2 disagree", ""]
    present = np.isfinite(v2.track.xy[:, 0])
    a, b = v1.labels, v2.labels
    rows = []
    for name, sel in (("all graded frames", graded), ("ball seen", graded & present),
                      ("ball not seen", graded & ~present)):
        n = int(sel.sum())
        diff = sel & (a != b)
        both_live = (a < 2) & (b < 2)
        rows.append([name, n, f(100 * diff.sum() / n, 2),
                     f(100 * (diff & (a == pos.D) & (b < 2)).sum() / n, 2),
                     f(100 * (diff & (a < 2) & (b == pos.D)).sum() / n, 2),
                     f(100 * (diff & both_live).sum() / n, 2)])
    L += table(["frames", "count", "% differ", "% v1 D, v2 live", "% v1 live, v2 D",
                "% live in both, other team"], rows)

    # ---- decision ------------------------------------------------------------------------------
    L += ["## Decision criterion (fixed beforehand)", "",
          "- (a) model beats R5 on AUC and balanced accuracy in both directions: **{}**".format(
              "PASS" if crit_a else "FAIL"),
          "- (b) plausible whole match: D share {:.1f}% (30-50%), stoppages > 180 s outside goals "
          "and half openings: {}, possessions v2 {} vs v1 {} (limit 1.5x = {:.0f}): **{}**".format(
              100 * d_share, len(long_stops), n_poss["v2"], n_poss["v1"], 1.5 * n_poss["v1"],
              "PASS" if crit_b else "FAIL"),
          "- Result: **{}**".format("v2 becomes the final version" if crit_a and crit_b
                                   else "v1 stays the final version"), ""]

    # ---- model, HMM ----------------------------------------------------------------------------
    L += ["## Model and HMM", "", "Standardised logistic weights of the model trained on both halves "
          "(positive = more likely stopped):", ""]
    w = e["model"].weights
    L += table(["feature", "weight"], [[n, f(v, 2)] for n, v in zip(e["names"], w[:-1])] +
               [["intercept", f(w[-1], 2)]])
    L += table(["transition", "per frame", "expected duration"],
               [["stopped -> live", "{:.5f}".format(hmm.P_STOP_TO_LIVE), "{:.0f} s".format(hmm.EXPECTED_STOP_S)],
                ["live -> stopped", "{:.5f}".format(hmm.P_LIVE_TO_STOP), "{:.0f} s".format(hmm.EXPECTED_LIVE_S)]])
    L += ["Evidence tempering {:.3f} (one independent observation per second).".format(hmm.TEMPER), ""]

    # ---- public window -------------------------------------------------------------------------
    results_dir = os.path.join(ROOT, "evaluation", "results")
    for name in ("v1", "v2"):
        runner.write_csv(os.path.join(results_dir, name + ".csv"), (v1 if name == "v1" else v2).labels)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                           "--bootstrap", "1000"], cwd=ROOT, capture_output=True, text=True)
    text = proc.stdout.splitlines()
    board = [l for l in text if l.startswith(("rank", "----")) or
             (l.split()[:1] and l.split()[0].isdigit() and len(l.split()) > 8)]
    L += ["## Public 5-minute window (reported only, not used for the decision)", "",
          "17 possessions: differences below ~0.3 spellF1 are noise.", "", "```"] + board + ["```", ""]
    for name in ("v1", "v2"):
        start = text.index(name)
        end = next(i for i in range(start + 1, len(text)) if text[i] and not text[i].startswith(" "))
        L += ["```"] + text[start:end] + ["```", ""]

    out = os.path.join(ROOT, "docs", "report_v2.md")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", os.path.relpath(out, ROOT), "| (a)", crit_a, "(b)", crit_b)


if __name__ == "__main__":
    main()
