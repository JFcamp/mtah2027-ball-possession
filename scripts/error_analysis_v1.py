"""docs/error_analysis_v1.md: where v1 disagrees with the public 5-minute ground truth, and why.

    python scripts/error_analysis_v1.py

Only a report. It reads the public ground truth to understand how cases are labelled and changes
nothing in the pipeline (the possession switch log is obtained by wrapping `place_switch` in memory).
No threshold is proposed from it.
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import dead_ball as db  # noqa: E402
from src import io as sio  # noqa: E402
from src import possession as pos  # noqa: E402
from src import run as runner  # noqa: E402

FPS = sio.FPS
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
NAME = {0: "H", 1: "A", 2: "D", 3: "X"}
TAU = 25            # a predicted switch within +/-1 s of the true one counts as matched
NEAR = 250          # +/-10 s: a switch this close is "late/early", farther is "missing"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def fmt(x, d=1):
    return "{:.{}f}".format(x, d)


def direct_switches(labels, lo, hi):
    """[(frame, old, new)]: frames in (lo, hi] where H changes to A or A to H directly."""
    out = []
    for t in range(lo + 1, hi + 1):
        a, b = labels[t - 1], labels[t]
        if a in (0, 1) and b in (0, 1) and a != b:
            out.append((t, int(a), int(b)))
    return out


def main():
    gt_letters = pd.read_csv(GT_PATH)["label"].to_numpy()
    gt = np.array([CODE[x] for x in gt_letters], dtype=np.int8)
    window = np.flatnonzero(gt < 3)
    lo, hi = int(window[0]), int(window[-1])

    # Run v1 while logging every confirmed possession switch (place_switch is called once per switch).
    log = []
    original = pos.place_switch

    def logged(first_control, new_team, seg_lo, present, phase_team, floor):
        frame, in_gap = original(first_control, new_team, seg_lo, present, phase_team, floor)
        log.append({"frame": frame, "in_gap": in_gap, "seq_start": first_control, "team": new_team})
        return frame, in_gap

    pos.place_switch = logged
    try:
        v1 = runner.build("v1")
    finally:
        pos.place_switch = original
    pred = v1.labels
    present = np.isfinite(v1.track.xy[:, 0])
    out = v1.features.ball_out
    speed = v1.track.speed
    control = pos.control_teams(v1.features, present)
    eps = []
    for _, a, z in sio.half_blocks():
        eps += pos.episodes(control, a, z + 1)
    ep_team = np.full(sio.N_FRAMES, -1, dtype=np.int8)
    for team, s, e in eps:
        ep_team[s:e + 1] = team
    ep_by_start = {s: (team, e - s + 1) for team, s, e in eps}
    sp = [(i.start_frame, i.tags.get("Type")) for i in sio.load_smart_tagging() if i.code == "SET PIECES"]

    L = ["# Error analysis of v1 on the public window", "",
         "Frames {}-{} ({} graded frames, {} s). Prediction = `submissions/v1.csv` pipeline. The public "
         "ground truth is read only to understand how cases are labelled; nothing in the pipeline "
         "changed and no threshold value is proposed here.".format(lo, hi, len(window), fmt(len(window) / FPS)), ""]

    # ===== 1. GT dead-ball stretches ============================================================
    gt_dead = db.runs((gt == 2))
    gt_dead = [(a, b) for a, b in gt_dead if a >= lo and b <= hi + 1]
    L += ["## 1. Dead-ball stretches of the ground truth", "",
          "{} stretches. 'covered' is the share of the stretch that v1 also calls D.".format(len(gt_dead)), ""]
    rows, miss_frames, stretch_cat = [], Counter(), Counter()
    credit_hit = Counter()
    for a, b in gt_dead:
        n = b - a
        hit = pred[a:b] == 2
        cov = hit.mean()
        # which rule covers the part we got right (credit order of the pipeline)
        taken = np.zeros(n, dtype=bool)
        for r in db.PRIORITY:
            m = v1.dead.by_rule[r][a:b] & hit & ~taken
            credit_hit[r] += int(m.sum())
            taken |= m
        # predicted D run overlapping this stretch the most
        runs_pred = [(s, e) for s, e in db.runs(pred == 2) if e > a and s < b]
        best = max(runs_pred, key=lambda r: min(r[1], b) - max(r[0], a)) if runs_pred else None
        d_start = (best[0] - a) if best else None
        d_end = (best[1] - b) if best else None
        # remainder
        rem = np.flatnonzero(~hit) + a
        cats = Counter()
        for t in rem:
            if not present[t]:
                cats["ball unseen (owner kept)"] += 1
            elif out[t]:
                cats["ball seen, outside the field"] += 1
            elif np.isfinite(speed[t]) and speed[t] < 0.5:
                cats["ball seen, still (<0.5 m/s) but not long enough for R6"] += 1
            elif ep_team[t] >= 0:
                cats["ball seen, moving, a team controls it"] += 1
            else:
                cats["ball seen, moving, nobody controls it"] += 1
        miss_frames.update(cats)
        top = max(cats, key=cats.get) if cats else "-"
        stretch_cat["fully covered (>= 90%)" if cov >= 0.9 else
                    ("partly covered (30-90%)" if cov >= 0.3 else "mostly missed (< 30%)")] += 1
        near = min(sp, key=lambda s: abs(s[0] - a))
        rows.append(["{}-{}".format(a, b - 1), fmt(n / FPS), fmt(100 * cov, 0),
                     "-" if d_start is None else "{:+d}".format(d_start),
                     "-" if d_end is None else "{:+d}".format(d_end),
                     "{} ({:+d} f)".format(near[1], near[0] - a) if abs(near[0] - a) <= 25 * 60 else "none within 60 s",
                     top])
    L += table(["stretch (frames)", "length (s)", "covered %", "pred D start - GT start (f)",
                "pred D end - GT end (f)", "nearest set-piece tag (start - GT start)", "main reason for the missed part"], rows)
    L += ["Remainder of the stretches (frames that the GT calls D and v1 does not):", ""]
    tot_miss = sum(miss_frames.values())
    L += table(["what the tracking shows at those frames", "frames", "%"],
               [[k, v, fmt(100 * v / tot_miss)] for k, v in miss_frames.most_common()])
    L += ["Frames both call D, credited to the first rule (R1, R4, R3, R2, R6, R5) that marks them:", ""]
    L += table(["rule", "frames"], [[r, credit_hit[r]] for r in db.PRIORITY])
    false_d = (pred == 2) & (gt < 2)
    false_d &= (np.arange(sio.N_FRAMES) >= lo) & (np.arange(sio.N_FRAMES) <= hi)
    taken = np.zeros(sio.N_FRAMES, dtype=bool)
    fd = Counter()
    for r in db.PRIORITY:
        m = v1.dead.by_rule[r] & false_d & ~taken
        fd[r] = int(m.sum())
        taken |= m
    L += ["v1 calls D where the ground truth says H or A on {} frames, by rule: {}.".format(
        int(false_d.sum()), ", ".join("{} {}".format(r, fd[r]) for r in db.PRIORITY if fd[r])), ""]

    # ===== 2. GT possession switches ============================================================
    gt_sw = direct_switches(gt, lo, hi)
    pred_sw = direct_switches(pred, lo - 1, hi)
    L += ["## 2. Possession switches H<->A of the ground truth", "",
          "{} direct switches in the ground truth, {} direct switches predicted in the same window. "
          "Delay = predicted frame - true frame (negative = v1 early).".format(len(gt_sw), len(pred_sw)), ""]
    rows, cat2 = [], Counter()
    for g, old, new in gt_sw:
        cand = sorted(pred_sw, key=lambda s: abs(s[0] - g))
        near_pred = cand[0] if cand else None
        delay = near_pred[0] - g if near_pred else None
        wlo, whi = g - 50, g + 100
        vis = float(present[wlo:whi].mean())
        new_eps = [(s, ln) for t, (tm, ln) in ep_by_start.items() for s in [t]
                   if wlo <= s <= whi and tm == new]
        first_new = min(new_eps)[0] - g if new_eps else None
        if near_pred is not None and abs(delay) <= TAU:
            c, why = "matched within 1 s", ""
        elif near_pred is not None and abs(delay) <= NEAR:
            c, why = "switch predicted {} by {} s".format("late" if delay > 0 else "early", fmt(abs(delay) / FPS)), ""
        else:
            c = "no switch predicted within 10 s"
            why = ""
        rc = ""
        if c != "matched within 1 s":
            if pred[g] == 2 or pred[g - 1] == 2:
                rc = "v1 has a dead ball at that moment"
                why = "v1 has a dead ball at that moment"
            elif vis < 0.5:
                rc = "ball unseen around the switch"
                why = "ball unseen around it ({}% of frames seen in -2..+4 s)".format(fmt(100 * vis, 0))
            elif not new_eps:
                rc = "ball seen, new team never within control distance"
                why = "ball seen but no control episode by the new team in -2..+4 s"
            elif max(ln for _, ln in new_eps) < pos.SWITCH_MIN:
                rc = "new team only touched it briefly (hysteresis)"
                why = "new team touched it briefly ({} f at most), under the 0.5 s hysteresis".format(max(ln for _, ln in new_eps))
            else:
                rc = "new team controlled it >= 0.5 s: timing / placement of the switch"
                why = "new team had a control episode >= 0.5 s (first at {:+d} f); owner already new or switch placed elsewhere".format(first_new)
        cls = ("matched within 1 s" if c.startswith("matched") else
               "predicted 1-10 s away" if c.startswith("switch") else "none predicted within 10 s")
        cat2[(cls, rc or "-")] += 1
        rows.append([g, "{}->{}".format(NAME[old], NAME[new]),
                     "-" if near_pred is None else near_pred[0],
                     "-" if delay is None else "{:+d}".format(delay), c, why])
    L += table(["GT switch frame", "GT", "nearest predicted switch", "delay (f)", "result", "what the tracking shows"], rows)

    # ===== 3. predicted switches without a match ================================================
    L += ["## 3. Predicted switches with no ground-truth switch within 1 s", ""]
    logged_by_frame = {d["frame"]: d for d in log}
    rows, cat3 = [], Counter()
    for p, old, new in pred_sw:
        if any(abs(p - g) <= TAU for g, _, _ in gt_sw):
            continue
        entry = next((logged_by_frame[f] for f in (p, p - 1, p + 1) if f in logged_by_frame), None)
        if entry is None:
            trigger = "created by post-processing (short possession merged)"
        else:
            tm_len = ep_by_start.get(entry["seq_start"], (None, 0))[1]
            if entry["in_gap"]:
                trigger = "switch placed inside a ball gap (phase change or middle of the gap)"
            elif tm_len >= pos.SWITCH_MIN:
                trigger = "new team controlled the ball for >= 0.5 s ({} f)".format(tm_len)
            else:
                trigger = "two touches / short touch followed by control (first touch {} f)".format(tm_len)
        g_here = NAME[int(gt[p])] if lo <= p <= hi else "?"
        context = ("GT already has {} (v1 late/early by more than 1 s)".format(NAME[new]) if gt[p] == new
                   else ("GT has a dead ball" if gt[p] == 2 else "GT keeps {}".format(g_here)))
        key = trigger.split(" (")[0] if trigger.startswith(("new team", "two touches")) else trigger
        cat3[(trigger.split(" (")[0], context.split(" (")[0])] += 1
        rows.append([p, "{}->{}".format(NAME[old], NAME[new]), trigger, context])
    L += table(["frame", "pred", "what triggered it", "ground truth at that frame"], rows)

    # ===== 4. summary ===========================================================================
    L += ["## 4. Summary by category", "", "**Ground-truth dead-ball stretches ({})**".format(len(gt_dead)), ""]
    L += table(["category", "stretches"], [[k, v] for k, v in stretch_cat.most_common()])
    L += ["**Frames of the missed dead ball by what the tracking shows**", ""]
    L += table(["category", "frames", "%"], [[k, v, fmt(100 * v / tot_miss)] for k, v in miss_frames.most_common()])
    L += ["**Ground-truth possession switches ({})**".format(len(gt_sw)), ""]
    L += table(["result", "what the tracking shows", "switches"], [[k[0], k[1], v] for k, v in sorted(cat2.items())])
    L += ["**Predicted switches without a match ({})**".format(sum(cat3.values())), ""]
    L += table(["trigger", "ground truth at that frame", "switches"],
               [[k[0], k[1], v] for k, v in cat3.most_common()])

    path = os.path.join(ROOT, "docs", "error_analysis_v1.md")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
