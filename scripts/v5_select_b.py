"""Part B selection: choose on two games, test on the third (from data_external/cache/v5_sweep.json).

    python scripts/v5_select_b.py

Objective on the training games: mean spellF1 over DEG_BAR and DEG_40 (each already a mean of two seeds).
Tie-break: fewest changes from the v4 configuration, then grid order. Criterion B: the chosen configuration
beats the v4 one on the test game in DEG_BAR and in DEG_40, in all three combinations. The final
configuration is the best on the three games. CLEAN is reported, not used. Writes v5_b.json.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import lab  # noqa: E402

CACHE = os.path.join(ROOT, "data_external", "cache")
TRAIN_CONDS = ("DEG_BAR", "DEG_40")


def main():
    data = json.load(open(os.path.join(CACHE, "v5_sweep.json")))
    cfgs, res = data["configs"], data["results"]
    games = list(res)
    v4 = [i for i, c in enumerate(cfgs) if c == lab.V4_CONFIG][0]
    dist = [lab.changes_from_v4(c) for c in cfgs]

    def best(train):
        obj = np.mean([[np.array(res[g][c])[:, 0] for c in TRAIN_CONDS] for g in train], axis=(0, 1))
        order = sorted(range(len(cfgs)), key=lambda i: (-round(obj[i], 9), dist[i], i))
        return order[0], obj

    folds, passed = {}, True
    for test in games:
        train = [g for g in games if g != test]
        i, obj = best(train)
        test_scores = {c: res[test][c][i][0] for c in lab.CONDITIONS}
        v4_scores = {c: res[test][c][v4][0] for c in lab.CONDITIONS}
        ok = all(test_scores[c] > v4_scores[c] for c in TRAIN_CONDS)
        passed &= ok
        folds[test] = {"chosen": cfgs[i], "train_objective": float(obj[i]), "v4_train_objective": float(obj[v4]),
                       "test": test_scores, "v4_test": v4_scores, "pass": bool(ok)}
        print(test, cfgs[i], test_scores, v4_scores, ok)
    i, obj = best(games)
    final = {"final_config": cfgs[i], "final_objective": float(obj[i]), "v4_objective": float(obj[v4]),
             "per_game": {g: {c: [res[g][c][i][0], res[g][c][v4][0]] for c in lab.CONDITIONS} for g in games},
             "passed": bool(passed), "folds": folds}
    # marginal effect of each parameter value (mean over the grid, games, DEG conditions)
    mean_all = np.mean([[np.array(res[g][c])[:, 0] for c in TRAIN_CONDS] for g in games], axis=(0, 1))
    final["marginals"] = {k: {str(v): float(np.mean(mean_all[[j for j, c in enumerate(cfgs) if c[k] == v]]))
                              for v in lab.GRID_V5[k]} for k in lab.KEYS}
    with open(os.path.join(CACHE, "v5_b.json"), "w") as handle:
        json.dump(final, handle, indent=1)
    print("final", cfgs[i], "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
