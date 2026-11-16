"""docs/report_v5.md: tables of Parts A-V, BAR-ATH v4 vs v5 (no ground truth), then the public window.

    python scripts/report_v5.py     (run only AFTER docs/decision_v5.md is committed)
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

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


def runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def possessions(labels):
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts, stops = np.concatenate(([0], edges)), np.concatenate((edges, [len(labels)]))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if labels[a] in (pos.H, pos.A)]


def main():
    D = json.load(open(os.path.join(CACHE, "v5_decision.json")))
    v4 = runner.build("v4")
    v5 = runner.build("v5") if D["changed"] else v4
    graded = sio.graded_mask()
    L = ["# Report v5", "",
         "Decisions: `docs/decision_v5.md` (committed before this report). Calibration data: the three external "
         "games, degraded to BAR-ATH statistics (`docs/domain_gap.md`). The public window is at the end and was "
         "not used for any choice.", "",
         "The tables of Parts A to V are in `docs/decision_v5.md` and `docs/domain_gap.md`; summary:", ""]
    L += table(["criterion", "result"],
               [["B (possession configuration, DEG_BAR and DEG_40, 3 of 3)", "PASS" if D["B"]["passed"] else "FAIL"],
                ["C (model, degraded AUC and balanced accuracy, 3 of 3)", "PASS: " + D["C"]["chosen"] if any(
                    json.load(open(os.path.join(CACHE, "v5_model.json")))["passed"].values()) else "FAIL: L0 stays"],
                ["E2E (new pipeline beats the v4-like one, 3 of 3)", "PASS" if D["E2E"]["passed"] else "FAIL"],
                ["D (T1 transfer, T2 plausibility, T3 structure)", "PASS" if D["D"]["pass"] else "FAIL"],
                ["V (pass-chain veto)", ("PASS" if D["V"].get("pass") else "FAIL") if D["V"]["evaluated"] else "not evaluated"]])

    L += ["## BAR-ATH: v4 and v5 (no ground truth)", ""]
    rows = []
    for half, first, last in sio.half_blocks():
        for name, res in (("v4", v4), ("v5", v5)):
            seg = res.labels[first:last + 1]
            poss = possessions(seg)
            stops = [(b - a) / FPS for a, b in runs(seg == pos.D) if (b - a) >= 50]
            rows.append([half, name, len(poss), f(np.mean([(b - a) / FPS for a, b in poss])),
                         f(100 * (seg == pos.H).mean()), f(100 * (seg == pos.A).mean()),
                         f(100 * (seg == pos.D).mean()), len(stops), f(np.mean(stops)), f(np.median(stops)),
                         f(max(stops))])
    L += table(["half", "version", "possessions", "mean poss. (s)", "% H", "% A", "% D", "stoppages >= 2 s",
                "mean (s)", "median (s)", "max (s)"], rows)
    present = np.isfinite(v4.track.xy[:, 0])
    diff = graded & (v4.labels != v5.labels)
    rows = []
    for name, sel in (("all graded frames", graded), ("ball seen", graded & present), ("ball not seen", graded & ~present)):
        n = int(sel.sum())
        rows.append([name, n, f(100 * (diff & sel).sum() / n, 2),
                     f(100 * (sel & (v4.labels == 2) & (v5.labels < 2)).sum() / n, 2),
                     f(100 * (sel & (v4.labels < 2) & (v5.labels == 2)).sum() / n, 2),
                     f(100 * (diff & sel & (v4.labels < 2) & (v5.labels < 2)).sum() / n, 2)])
    L += ["Frames that change between v4 and v5:", ""]
    L += table(["frames", "count", "% change", "% v4 D, v5 live", "% v4 live, v5 D", "% live in both, other team"], rows)

    results_dir = os.path.join(ROOT, "evaluation", "results")
    for name, res in (("v4", v4), ("v5", v5)):
        runner.write_csv(os.path.join(results_dir, name + ".csv"), res.labels)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                           "--bootstrap", "1000"], cwd=ROOT, capture_output=True, text=True)
    text = proc.stdout.splitlines()
    board = [l for l in text if l.startswith(("rank", "----")) or
             (l.split()[:1] and l.split()[0].isdigit() and len(l.split()) > 8)]
    L += ["## Public 5-minute window (reported only)", "",
          "17 possessions: differences below ~0.3 spellF1 are noise.", "", "```"] + board + ["```", ""]
    for name in ("v4", "v5"):
        start = text.index(name)
        end = next(i for i in range(start + 1, len(text)) if text[i] and not text[i].startswith(" "))
        L += ["```"] + text[start:end] + ["```", ""]
    path = os.path.join(ROOT, "docs", "report_v5.md")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", path)


if __name__ == "__main__":
    main()
