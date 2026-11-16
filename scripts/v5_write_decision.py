"""docs/decision_v5.md from data_external/cache/v5_*.json. No public ground truth anywhere in it."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_external", "cache")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def f(x, d=4):
    return "{:.{}f}".format(x, d)


def short(cfg):
    return "r {} m {} min {} 2touch {} backdate {} poss {}".format(
        cfg["r_ctrl"], cfg["margin"], cfg["min_control"], "on" if cfg["two_touch"] else "off",
        "on" if cfg["backdate"] else "off", cfg["min_poss"])


def dec(d):
    return "P>={} min {} close {} refine {}".format(d["threshold"], d["min_stop"], d["close_gap"],
                                                      "on" if d["refine"] else "off")


def yn(x):
    return "PASS" if x else "FAIL"


def main():
    D = json.load(open(os.path.join(CACHE, "v5_decision.json")))
    b, c = D["B"], D["C"]
    cm = json.load(open(os.path.join(CACHE, "v5_model.json")))["results"]
    L = ["# Decision v5", "",
         "Written BEFORE the public ground truth was looked at; every number below comes from the external "
         "games or from algorithmic statistics of BAR-ATH (rules R1-R6, Smart Tagging). The public window "
         "appears only in `docs/report_v5.md`, after this file was committed.", ""]

    L += ["## Part A: degradation", "",
          "See `docs/domain_gap.md`. The degraded external games reproduce the measured BAR-ATH statistics "
          "(tracked players 20.8-21.0 mean vs 20.5, second differences of players 1.3-1.6 mm vs 1.1, ball "
          "3.6-3.9 mm vs 4.8, ball erased in 10.6% of the live frames). The model features were already close "
          "(mean absolute standardised difference 0.116 for the clean games, 0.108 after degrading), so the "
          "degradation works as a robustness stress test more than as a correction of a large gap. Wrong-ball "
          "episodes are not simulated.", ""]

    L += ["## Part B: possession configuration (true dead-ball mask)", "",
          "v4 configuration: {}. Spell F1 on the test game, chosen on the other two:".format(short(
              {"r_ctrl": 1.0, "margin": 0.3, "min_control": 6, "two_touch": True, "backdate": False, "min_poss": 25})), ""]
    rows = []
    for t, o in b["folds"].items():
        rows.append([t, short(o["chosen"])] + [
            "{} vs {}".format(f(o["test"][k]), f(o["v4_test"][k])) for k in ("DEG_BAR", "DEG_40", "CLEAN")]
            + [yn(o["pass"])])
    L += table(["test game", "chosen on the other two", "DEG_BAR (chosen vs v4)", "DEG_40", "CLEAN (reported)", "beats v4 in both"], rows)
    L += ["**Criterion B: {}.** Best configuration on the three games: {} (objective {} vs {} for v4).".format(
        yn(b["passed"]), short(b["final_config"]), f(b["final_objective"]), f(b["v4_objective"])), ""]

    L += ["## Part C: stopped-ball model (leave-one-game-out)", ""]
    rows = []
    for t, r in cm.items():
        for k in ("L0", "L1", "G1"):
            rows.append([t, k, f(r[k]["deg"]["auc"]), f(r[k]["deg"]["bal"]), f(r[k]["clean"]["auc"]), f(r[k]["clean"]["bal"])])
    L += table(["test game", "model", "AUC degraded", "bal. acc. degraded", "AUC clean", "bal. acc. clean"], rows)
    L += ["Mean degraded AUC: {}. Passed: {}. **Criterion C: {}; chosen model: {}.**".format(
        ", ".join("{} {}".format(k, f(v)) for k, v in c["mean_deg_auc"].items()),
        ", ".join("{} {}".format(k, yn(v)) for k, v in c["passed"].items()),
        yn(any(c["passed"].values())), c["chosen"]), ""]

    L += ["## Part C2: decoder and end-to-end check", "",
          "Chosen on the two training games (nested), scored on the third with the complete pipeline; v4-like = L0, "
          "P>=0.5, minimum 50 frames, v4 configuration.", ""]
    rows = []
    for t, o in D["E2E"]["folds"].items():
        rows.append([t, dec(o["chosen"]), "{} vs {}".format(f(o["new"]["DEG_BAR"]), f(o["v4_like"]["DEG_BAR"])),
                     "{} vs {}".format(f(o["new"]["DEG_40"]), f(o["v4_like"]["DEG_40"])), yn(o["pass"])])
    L += table(["test game", "decoder chosen on the other two", "DEG_BAR (new vs v4-like)", "DEG_40", "beats v4-like"], rows)
    L += ["**Criterion E2E: {}.** Decoder chosen on the three games: {}.".format(
        yn(D["E2E"]["passed"]), dec(D["E2E"]["decoder"])), ""]

    L += ["## Part D: model on every graded frame of BAR-ATH", ""]
    t1 = D["D"]["T1"]
    L += ["Model used: {}. Algorithmic weak labels: {} weak-stopped frames, {} weak-live frames ({} inside BUILD UP).".format(
        D["D"]["model_used"], t1["weak_stop_frames"], t1["weak_live_frames"], t1["weak_buildup_frames"]), ""]
    rows = [["AUC on the weak labels", f(t1["auc"], 3), ">= 0.85", yn(t1["auc"] >= 0.85)],
            ["mean hit rate over the weak-stopped sources", f(t1["mean_hit"], 3), ">= 0.70", yn(t1["mean_hit"] >= 0.70)],
            ["false stopped in weak live", f(t1["false_stop_live"], 3), "<= 0.20", yn(t1["false_stop_live"] <= 0.20)],
            ["false stopped in BUILD UP (reported)", f(t1["false_stop_buildup"], 3), "-", "-"]]
    L += table(["T1 transfer", "value", "limit", ""], rows)
    L += table(["weak-stopped source", "hit rate"], [[k, f(v, 3)] for k, v in t1["source_hit"].items()])
    t1l = D["D"]["T1_L0"]
    L += ["The v4 model (L0) under the same test: AUC {}, mean hit {}, false stopped {} ({}).".format(
        f(t1l["auc"], 3), f(t1l["mean_hit"], 3), f(t1l["false_stop_live"], 3), yn(t1l["pass"])), ""]
    sa = D["D"]["stats_all_frames"]
    if sa:
        L += ["Applied to all frames: D {}% ({}; limits 35-55%), {} stoppages of 2 s or more ({}; limits 80-170), "
              "{} possessions (limits {}-{}).".format(f(100 * sa["d_share"], 1), yn(D["D"]["T2"]), sa["stoppages_2s"],
                                                      yn(80 <= sa["stoppages_2s"] <= 170), sa["possessions"],
                                                      int(0.6 * 353), int(1.5 * 353)), ""]
    L += ["**Criterion D: {}** (T1 {}, T2 {}, T3 {}).".format(yn(D["D"]["pass"]), yn(t1["pass"]), yn(D["D"]["T2"]), yn(D["D"]["T3"])), ""]

    L += ["## Part V: pass-chain veto", ""]
    v = D["V"]
    if v["evaluated"]:
        L += ["Veto fires on {}% of the weak-stopped frames with the ball inside the field (limit 2%) and covers {}% "
              "of the weak-live frames (minimum 30%). **Criterion V: {}.**".format(
                  f(100 * v["false_trigger_on_weak_stopped"], 2), f(100 * v["coverage_of_weak_live"], 1), yn(v["pass"])), ""]
    else:
        L += ["Not evaluated (it only applies if D passed).", ""]

    o = D["options"]
    L += ["## Composition of v5", "",
          "- possession configuration: {}".format(short(o["possession"])),
          "- model: {} (`{}`), decoder: {}".format(D["D"]["model_used"], o["model"], dec(o["decoder"])),
          "- model applied on: {}".format("every graded frame (R5 removed)" if o["all_frames"] else "ball gaps of 5 s or more (as v4)"),
          "- pass-chain veto: {}".format("yes" if o["veto"] else "no"),
          "- **v5 differs from v4: {}**".format("yes" if D["changed"] else "no (v4 stays final)"), ""]
    s4, s5 = D["v4_stats"], D["v5_stats"]
    L += ["## BAR-ATH without ground truth", ""]
    L += table(["", "D share %", "possessions", "mean possession (s)", "stoppages >= 2 s", "mean stoppage (s)"],
               [[n, f(100 * s["d_share"], 1), s["possessions"], f(s["mean_possession_s"], 1), s["stoppages_2s"],
                 f(s["mean_stoppage_s"], 1)] for n, s in (("v4", s4), ("v5", s5))])
    path = os.path.join(ROOT, "docs", "decision_v5.md")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", path)


if __name__ == "__main__":
    main()
