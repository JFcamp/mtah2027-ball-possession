"""docs/report_v4.md: results of Parts B and C on the external games, the decision, v1 vs v4 on
BAR-ATH (no ground truth), and the public window (reported only).

    python scripts/report_v4.py     (needs data_external/cache/sweep.json and dead_model.json)

Decision criterion, fixed before looking at any of these numbers (uses only the external games):
  B  adopt the new possession configuration if, in all three held-out games and both ball
     conditions (true ball, erased gaps), it beats the v1 configuration on spellF1;
  C  adopt the stopped-ball model if, in all three held-out games, it beats the pace rule R5 on AUC
     and on balanced accuracy;
  v4 = v1 + each component that passed; final plausibility on BAR-ATH without ground truth:
  D between 30% and 50% of graded frames and at most 1.5x the v1 number of possessions.
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import calibration as cal  # noqa: E402
from src import io as sio  # noqa: E402
from src import possession as pos  # noqa: E402
from src import run as runner  # noqa: E402

CACHE = os.path.join(ROOT, "data_external", "cache")
FPS = 25


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def f(x, d=1):
    return "{:.{}f}".format(x, d)


def short(cfg):
    return "r={} m={} min={} 2touch={} backdate={} poss>={}".format(
        cfg["r_ctrl"], cfg["margin"], cfg["min_control"], "on" if cfg["two_touch"] else "off",
        "on" if cfg["backdate"] else "off", cfg["min_poss"])


def runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def possessions(labels):
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts, stops = np.concatenate(([0], edges)), np.concatenate((edges, [len(labels)]))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if labels[a] in (pos.H, pos.A)]


def main():
    sweep = json.load(open(os.path.join(CACHE, "sweep.json")))
    cfgs, res = sweep["configs"], sweep["results"]
    games = list(res)
    dmodel = json.load(open(os.path.join(CACHE, "dead_model.json")))
    v1_idx = [i for i, c in enumerate(cfgs) if c == cal.V1][0]
    L = ["# Report v4", "",
         "Calibration on the three public Metrica sample games (other matches, labels from the data "
         "provider; see `docs/external_data.md`). Nothing from BAR-ATH, not even the public 5 minutes, "
         "was used to choose or tune anything. The public window is only reported at the end.", ""]

    # ---- Part B --------------------------------------------------------------------------------
    out_b, all_idx, all_obj, v1_obj = cal.cross_game_selection(cfgs, res, games, v1_idx)
    L += ["## Part B: possession logic (true dead-ball mask)", "",
          "The dead-ball mask is the true one, so only the H/A logic is measured. 'erased' removes "
          "40% of the live-play ball frames in long gaps (5-10 s: 14% of the erased frames, 10-60 s: "
          "86%, the E2 distribution), mean of 2 seeds. Grid: {} configurations (control radius "
          "{}, margin {}, minimum control to switch {} frames, two touches on/off, back-dating on/off, "
          "minimum possession {} frames). Restart teams are not given to the logic (no Smart Tagging "
          "here): after a stoppage the first team to control the ball owns it.".format(
              len(cfgs), cal.GRID["r_ctrl"], cal.GRID["margin"], cal.GRID["min_control"],
              cal.GRID["min_poss"]), "",
          "**v1 configuration** ({}):".format(short(cal.V1)), ""]
    L += table(["game", "spellF1 true ball", "spellF1 erased", "macroF1 true ball", "macroF1 erased"],
               [[g, f(res[g]["full"][v1_idx][0], 4), f(res[g]["erased"][v1_idx][0], 4),
                 f(res[g]["full"][v1_idx][1], 4), f(res[g]["erased"][v1_idx][1], 4)] for g in games])
    L += ["**Cross-game selection** (configuration chosen on the two training games by the mean spellF1 "
          "over games and ball conditions, tie-break fewest changes from v1; scored on the third):", ""]
    rows, passed_b = [], True
    for t, o in out_b.items():
        for cond in ("full", "erased"):
            ok = o["test"][cond] > o["v1_test"][cond]
            passed_b &= ok
        rows.append([t, short(cfgs[o["chosen"]]),
                     "{} vs {}".format(f(o["test"]["full"], 4), f(o["v1_test"]["full"], 4)),
                     "{} vs {}".format(f(o["test"]["erased"], 4), f(o["v1_test"]["erased"], 4)),
                     "yes" if all(o["test"][c] > o["v1_test"][c] for c in ("full", "erased")) else "no"])
    L += table(["held-out game", "chosen on the other two", "spellF1 true ball (chosen vs v1)",
                "spellF1 erased (chosen vs v1)", "beats v1 in both"], rows)
    best = cfgs[all_idx]
    L += ["**Configuration adopted if B passes** (best on all three games, objective {} vs {} for v1): "
          "{}".format(f(all_obj, 4), f(v1_obj, 4), short(best)), ""]
    L += table(["game", "v1 true", "adopted true", "v1 erased", "adopted erased"],
               [[g, f(res[g]["full"][v1_idx][0], 4), f(res[g]["full"][all_idx][0], 4),
                 f(res[g]["erased"][v1_idx][0], 4), f(res[g]["erased"][all_idx][0], 4)] for g in games])
    # marginal effect of each parameter (mean over the grid, over games and conditions)
    obj = np.mean([[np.array(res[g][c])[:, 0] for c in ("full", "erased")] for g in games], axis=(0, 1))
    L += ["Mean spellF1 over the grid for each parameter value (all games, both conditions):", ""]
    rows = []
    for key in cal.KEYS:
        for value in cal.GRID[key]:
            sel = [i for i, c in enumerate(cfgs) if c[key] == value]
            rows.append([key, value, f(float(np.mean(obj[sel])), 4), f(float(np.max(obj[sel])), 4)])
    L += table(["parameter", "value", "mean spellF1", "best spellF1"], rows)
    L += ["The adopted values of the control radius, the minimum control and the minimum possession "
          "sit at the lower edge of the grid; the grid was not extended.", ""]

    # ---- Part C --------------------------------------------------------------------------------
    L += ["## Part C: stopped ball without the ball", "",
          "Logistic model (numpy, balanced classes) on player-only features without the number of "
          "tracked players, trained with the REAL dead-ball labels of two games and tested on the third. "
          "The comparison is the v1 pace rule R5 (5 s mean of the median player speed < 1.4 m/s) on the "
          "same frames.", ""]
    rows, passed_c = [], True
    for t, c in dmodel["cv"].items():
        ok = (c["model"]["auc"] > c["rule_R5"]["auc"] and
              c["model"]["balanced_accuracy"] > c["rule_R5"]["balanced_accuracy"])
        passed_c &= ok
        for name, k in (("model", "model"), ("rule R5", "rule_R5")):
            rows.append([t, name, c["frames"], c["stopped"], f(c[k]["accuracy"], 3),
                         f(c[k]["balanced_accuracy"], 3), f(c[k]["auc"], 3)])
    L += table(["test game", "classifier", "frames", "stopped", "accuracy", "balanced accuracy", "AUC"], rows)

    dec = dmodel["decoder_cv"]
    L += ["**Decoder calibration** by the spellF1 of the complete pipeline (decoded dead mask + possession "
          "logic with the Part B configuration, true ball and erased), nested cross-game validation: "
          "{} decoders (HMM transition durations {} s stopped x {} s live, evidence tempering "
          "{}, or a plain threshold, each with a minimum stoppage of {} frames).".format(
              dmodel["n_decoders"] * 3, "10/25/50", "20/40/80", "1/10, 1/25, 1/100", "25/50/100"), ""]
    rows = []
    for t, d in dec.items():
        ch = d["chosen"]
        name = "threshold 0.5" if ch["decoder"][0] == "threshold" else "HMM stop {} s, live {} s, tempering {}".format(
            ch["decoder"][1], ch["decoder"][2], f(ch["decoder"][3], 3))
        rows.append([t, "{}, min {} f".format(name, ch["min_dead"]),
                     "{} / {}".format(f(d["test"]["full"], 3), f(d["test"]["erased"], 3)),
                     "{} / {}".format(f(d["test_default_hmm"]["full"], 3), f(d["test_default_hmm"]["erased"], 3)),
                     "{} / {}".format(f(d["test_rule_R5"]["full"], 3), f(d["test_rule_R5"]["erased"], 3))])
    L += table(["held-out game", "decoder chosen on the other two", "test spellF1 (true / erased)",
                "v2-style HMM (25 s, 40 s, 1/25)", "R5 pace rule"], rows)
    fd = dmodel["final_decoder"]
    L += ["Decoder chosen on all three games: **{}** with a minimum stoppage of {} frames (objective {}). "
          "The choice among decoders was not stable across folds (a plain threshold in one, HMMs in the "
          "others), and the differences between them are small next to the gap to R5; v4 uses the "
          "simplest.".format("threshold P(stopped) >= 0.5" if fd["decoder"][0] == "threshold" else fd["decoder"],
                             fd["min_dead"], f(fd["objective"], 4)), ""]

    # ---- BAR-ATH -------------------------------------------------------------------------------
    graded = sio.graded_mask()
    versions = {}
    for name in ("v1", "v4b", "v4c", "v4"):
        versions[name] = runner.build(name)
    L += ["## BAR-ATH: v1 and v4 (no ground truth)", "",
          "v4b = v1 + the new possession configuration only; v4c = v1 + the model in place of R5 only; "
          "v4 = both.", ""]
    rows = []
    for half, first, last in sio.half_blocks():
        for name, res_ in versions.items():
            seg = res_.labels[first:last + 1]
            poss = possessions(seg)
            stops = [(b - a) / FPS for a, b in runs(seg == pos.D)]
            rows.append([half, name, len(poss), f(np.mean([(b - a) / FPS for a, b in poss])),
                         f(100 * (seg == pos.H).mean()), f(100 * (seg == pos.A).mean()),
                         f(100 * (seg == pos.D).mean()), len(stops), f(np.mean(stops)),
                         f(np.median(stops)), f(max(stops))])
    L += table(["half", "version", "possessions", "mean poss. (s)", "% H", "% A", "% D", "stoppages",
                "mean (s)", "median (s)", "max (s)"], rows)
    n_poss = {n: sum(len(possessions(r.labels[a:z + 1])) for _, a, z in sio.half_blocks())
              for n, r in versions.items()}
    d_share = {n: float((r.labels[graded] == pos.D).mean()) for n, r in versions.items()}
    diff = graded & (versions["v1"].labels != versions["v4"].labels)
    L += ["Whole match possessions: " + ", ".join("{} {}".format(n, n_poss[n]) for n in versions) +
          "; D share: " + ", ".join("{} {}%".format(n, f(100 * d_share[n])) for n in versions) +
          ". v1 and v4 differ on {}% of the graded frames.".format(f(100 * diff.sum() / graded.sum(), 2)), ""]
    plausible = {n: (0.30 <= d_share[n] <= 0.50 and n_poss[n] <= 1.5 * n_poss["v1"]) for n in versions}

    # ---- decision ------------------------------------------------------------------------------
    adopted = []
    if passed_b:
        adopted.append("B (possession configuration)")
    if passed_c:
        adopted.append("C (stopped-ball model)")
    final = "v4" if adopted and plausible["v4"] else "v1"
    L += ["## Decision (criterion fixed beforehand)", "",
          "- Part B (beats v1 in 3 of 3 held-out games, true and erased ball): **{}**".format("PASS" if passed_b else "FAIL"),
          "- Part C (beats R5 on AUC and balanced accuracy in 3 of 3 held-out games): **{}**".format("PASS" if passed_c else "FAIL"),
          "- v4 = v1 + {}".format(" + ".join(adopted) if adopted else "nothing"),
          "- BAR-ATH plausibility of v4 (D {}% in 30-50%, possessions {} vs {} for v1, limit {:.0f}): **{}**".format(
              f(100 * d_share["v4"]), n_poss["v4"], n_poss["v1"], 1.5 * n_poss["v1"],
              "PASS" if plausible["v4"] else "FAIL"),
          "- Result: **{} is the final version**".format(final), ""]

    # ---- public window -------------------------------------------------------------------------
    results_dir = os.path.join(ROOT, "evaluation", "results")
    for name, res_ in versions.items():
        runner.write_csv(os.path.join(results_dir, name + ".csv"), res_.labels)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                           "--bootstrap", "1000"], cwd=ROOT, capture_output=True, text=True)
    text = proc.stdout.splitlines()
    board = [l for l in text if l.startswith(("rank", "----")) or
             (l.split()[:1] and l.split()[0].isdigit() and len(l.split()) > 8)]
    L += ["## Public 5-minute window (reported only, not used for any choice)", "",
          "17 possessions: differences below ~0.3 spellF1 are noise.", "", "```"] + board + ["```", ""]
    for name in ("v1", "v4"):
        start = text.index(name)
        end = next(i for i in range(start + 1, len(text)) if text[i] and not text[i].startswith(" "))
        L += ["```"] + text[start:end] + ["```", ""]

    out = os.path.join(ROOT, "docs", "report_v4.md")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", os.path.relpath(out, ROOT), "| B", passed_b, "C", passed_c, "plausible", plausible, "final", final)


if __name__ == "__main__":
    main()
