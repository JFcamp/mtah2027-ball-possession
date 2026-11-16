"""Part C: better stopped-ball model (players only), leave-one-game-out on the external games.

    python scripts/v5_model.py

L0 = the v4 model (clean data, 20 features); L1 = logistic on the 65 v5 features; G1 = gradient boosting.
L1 and G1 are trained on clean + two degraded copies of the training games, L0 as in v4 (clean only).
Scores: AUC and balanced accuracy on the DEGRADED test game (mean of the two seeds) and on the CLEAN one.
Criterion (fixed beforehand): a candidate passes if it beats L0 on AUC and balanced accuracy on the degraded
test game in all three combinations, losing no more than 0.005 AUC on the clean test game; among the
candidates that pass, the highest mean degraded AUC wins. The final model is trained on the three games.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import dead_model as dm  # noqa: E402
from src import external as ext  # noqa: E402
from src import features_v5 as f5  # noqa: E402
from src import models_v5 as m5  # noqa: E402
import v5_features as v5f  # noqa: E402

OUT = os.path.join(ROOT, "data_external", "cache", "v5_model.json")
VARIANTS = ("clean", "deg0", "deg1")
STRIDE = {"L0": 2, "L1": 2, "G1": 5}


def load_games():
    games = {}
    names = None
    for g in ext.GAMES:
        G = ext.load_game(g)
        gm = ext.graded_mask(G)
        y = (ext.labels(G) == ext.D).astype(int)
        games[G.name] = {"graded": gm, "y": y[gm],
                         "x": {v: v5f.load_ext(G.name, v)[gm] for v in VARIANTS}}
    _, names = f5.compute(np.zeros((300, 2, 2)), np.array([0, 1]))
    return games, names


def training_set(games, train_names, kind):
    variants = ("clean",) if kind == "L0" else VARIANTS
    xs = [games[n]["x"][v] for n in train_names for v in variants]
    ys = [games[n]["y"] for n in train_names for _ in variants]
    return np.concatenate(xs), np.concatenate(ys)


def columns_for(kind, names):
    return f5.l0_columns(names) if kind == "L0" else list(range(len(names)))


def evaluate(model, game):
    out = {}
    y = game["y"]
    res = {}
    for v in VARIANTS:
        p = m5.predict(model, game["x"][v])
        res[v] = (dm.auc(p, y), dm.scores(p >= 0.5, y)["balanced_accuracy"])
    out["clean"] = {"auc": res["clean"][0], "bal": res["clean"][1]}
    out["deg"] = {"auc": float(np.mean([res[v][0] for v in ("deg0", "deg1")])),
                  "bal": float(np.mean([res[v][1] for v in ("deg0", "deg1")]))}
    return out


def main():
    games, names = load_games()
    gnames = list(games)
    results = {}
    for test in gnames:
        train = [n for n in gnames if n != test]
        results[test] = {}
        for kind in ("L0", "L1", "G1"):
            x, y = training_set(games, train, kind)
            model = m5.train(kind, x, y, columns_for(kind, names), STRIDE[kind])
            results[test][kind] = evaluate(model, games[test])
            print(test, kind, results[test][kind], flush=True)
    passed = {}
    for kind in ("L1", "G1"):
        ok = True
        for t in gnames:
            r, b = results[t][kind], results[t]["L0"]
            ok &= (r["deg"]["auc"] > b["deg"]["auc"] and r["deg"]["bal"] > b["deg"]["bal"]
                   and r["clean"]["auc"] >= b["clean"]["auc"] - 0.005)
        passed[kind] = bool(ok)
    mean_deg = {k: float(np.mean([results[t][k]["deg"]["auc"] for t in gnames])) for k in ("L0", "L1", "G1")}
    winners = [k for k in ("L1", "G1") if passed[k]]
    chosen = max(winners, key=lambda k: mean_deg[k]) if winners else "L0"
    # final models on the three games
    for kind in ("L0", "L1", "G1"):
        x, y = training_set(games, gnames, kind)
        model = m5.train(kind, x, y, columns_for(kind, names), STRIDE[kind])
        m5.save(model, os.path.join(ROOT, "data_external", "cache", "final_{}.json".format(kind)))
    with open(OUT, "w") as handle:
        json.dump({"results": results, "passed": passed, "mean_deg_auc": mean_deg, "chosen": chosen,
                   "names": names}, handle, indent=1)
    print("passed", passed, "chosen", chosen, mean_deg)


if __name__ == "__main__":
    main()
