"""Part C: stopped-ball model trained with the REAL dead-ball labels of the external games.

    python scripts/dead_model_external.py

Writes data_external/cache/dead_model.json (git-ignored) with
  - item 8: leave-one-game-out AUC / balanced accuracy of the logistic model (player features, no
    n_tracked) against the v1 pace rule R5 (5 s mean of the median player speed < 1.4 m/s), on the
    same frames;
  - item 9: decoder calibration (HMM transition durations, evidence tempering, minimum stoppage
    length, or a plain threshold) by the spellF1 of the complete pipeline (decoded dead mask +
    possession logic) with nested cross-game validation: to test on game t, the model and the
    decoder are chosen using only the other two games.
"""

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import calibration as cal  # noqa: E402
from src import dead_hmm as hmm  # noqa: E402
from src import dead_model as dm  # noqa: E402
from src import external as ext  # noqa: E402
from src import features as feat  # noqa: E402
from src import features_players as fp  # noqa: E402

CACHE = os.path.join(ROOT, "data_external", "cache")
OUT = os.path.join(CACHE, "dead_model.json")
SEEDS = (0, 1)
HMM_STOP = (10.0, 25.0, 50.0)
HMM_LIVE = (20.0, 40.0, 80.0)
HMM_TEMPER = (1 / 10, 1 / 25, 1 / 100)
MIN_DEAD = (25, 50, 100)
R5_THRESHOLD = 1.4
ACT_WIN = 125


def player_matrix(game):
    """Windowed player features without n_tracked (cached on disk)."""
    path = os.path.join(CACHE, game.name + "_playerfeat.npy")
    if os.path.isfile(path):
        return np.load(path)
    x, names = fp.compute(game.players_xy, game.team_of_slot)
    keep = [i for i, n in enumerate(names) if not n.startswith("n_tracked")]
    x = x[:, keep]
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, x)
    return x


def filter_min(stop, graded, min_len):
    out = np.zeros(len(stop), dtype=bool)
    for a, b in ext.runs(stop & graded):
        if b - a >= min_len:
            out[a:b] = True
    return out


def main():
    games = {}
    for g in ext.GAMES:
        G = ext.load_game(g)
        gt = ext.labels(G)
        graded = ext.graded_mask(G)
        games[G.name] = {"game": G, "gt": gt, "graded": graded, "x": player_matrix(G),
                         "y": (gt == ext.D).astype(int),
                         "act5": feat.rolling_nanmean(feat.activity_from(G.players_xy)[0], ACT_WIN)}
    names = list(games)

    # ---- item 8: leave-one-game-out classification -----------------------------------------------
    cv = {}
    for test in names:
        train = [n for n in names if n != test]
        tr_rows = np.concatenate([np.flatnonzero(games[n]["graded"]) for n in train])
        xtr = np.concatenate([games[n]["x"][games[n]["graded"]] for n in train])
        ytr = np.concatenate([games[n]["y"][games[n]["graded"]] for n in train])
        model = dm.fit(xtr, ytr)
        g = games[test]
        rows = np.flatnonzero(g["graded"] & np.isfinite(g["act5"]))
        y = g["y"][rows]
        p = dm.predict_proba(model, g["x"][rows])
        m = dm.scores(p >= 0.5, y)
        r = dm.scores(g["act5"][rows] < R5_THRESHOLD, y)
        cv[test] = {"train": train, "frames": int(len(rows)), "stopped": int(y.sum()),
                    "model": {**m, "auc": dm.auc(p, y)},
                    "rule_R5": {**r, "auc": dm.auc(-g["act5"][rows], y)}}
        print("cv", test, cv[test]["model"], cv[test]["rule_R5"], flush=True)

    # ---- possession config used downstream (Part B result) --------------------------------------
    sweep = json.load(open(os.path.join(CACHE, "sweep.json")))
    cfgs, res = sweep["configs"], sweep["results"]
    v1_idx = [i for i, c in enumerate(cfgs) if c == cal.V1][0]
    out_b, all_idx, _, _ = cal.cross_game_selection(cfgs, res, names, v1_idx)
    passed_b = all(o["test"][c] > o["v1_test"][c] for o in out_b.values() for c in ("full", "erased"))
    cfg = cfgs[all_idx] if passed_b else cal.V1

    # ---- inputs of the pipeline per game and ball condition -----------------------------------
    inputs = {}
    for n in names:
        G, gt = games[n]["game"], games[n]["gt"]
        live = gt < 2
        inputs[n] = {"full": [ext.pipeline_inputs(G)]}
        inputs[n]["erased"] = [ext.pipeline_inputs(G, cal.erase_ball(G, live, s)) for s in SEEDS]

    def pipeline_spell(n, dead):
        """Mean spellF1 over the ball conditions: (full, erased)."""
        G, gt = games[n]["game"], games[n]["gt"]
        scores = {}
        for cond, items in inputs[n].items():
            vals = []
            for track, f in items:
                pred = cal.run_possession(G, dead, track, f, cfg)
                vals.append(ext.score(gt, pred)["spell_f1"])
            scores[cond] = float(np.mean(vals))
        return scores

    # probabilities: p[(game, tuple of training games)]
    probs = {}
    def prob(n, train):
        key = (n, tuple(train))
        if key not in probs:
            xtr = np.concatenate([games[t]["x"][games[t]["graded"]] for t in train])
            ytr = np.concatenate([games[t]["y"][games[t]["graded"]] for t in train])
            probs[key] = dm.predict_proba(dm.fit(xtr, ytr), games[n]["x"])
        return probs[key]

    decoders = [("hmm", es, el, tp) for es, el, tp in itertools.product(HMM_STOP, HMM_LIVE, HMM_TEMPER)]
    decoders.append(("threshold", None, None, None))
    stop_cache = {}

    def stop_path(n, train, dec):
        key = (n, tuple(train), dec)
        if key not in stop_cache:
            p = prob(n, train)
            G = games[n]["game"]
            if dec[0] == "threshold":
                stop = p >= 0.5
            else:
                stop = np.zeros(len(p), dtype=bool)
                for _, a, z in ext.half_blocks(G):
                    hard = np.full(z - a + 1, -1, dtype=np.int8)
                    stop[a:z + 1] = hmm.viterbi(p[a:z + 1], hard, dec[1], dec[2], dec[3])
            stop_cache[key] = stop
        return stop_cache[key]

    def spell_of(n, train, dec, md):
        dead = filter_min(stop_path(n, train, dec), games[n]["graded"], md)
        return pipeline_spell(n, dead)

    def r5_spell(n):
        dead = filter_min(games[n]["act5"] < R5_THRESHOLD, games[n]["graded"], 50)
        return pipeline_spell(n, dead)

    cands = [(dec, md) for dec in decoders for md in MIN_DEAD]
    decoder_cv = {}
    for test in names:
        a, b = [n for n in names if n != test]
        objective = []
        for dec, md in cands:
            va = spell_of(a, (b,), dec, md)
            vb = spell_of(b, (a,), dec, md)
            objective.append(np.mean([va["full"], va["erased"], vb["full"], vb["erased"]]))
        order = sorted(range(len(cands)), key=lambda i: (-round(objective[i], 9), i))
        best = order[0]
        dec, md = cands[best]
        test_scores = spell_of(test, (a, b), dec, md)
        default = spell_of(test, (a, b), ("hmm", 25.0, 40.0, 1 / 25), 50)   # the v2 HMM settings
        decoder_cv[test] = {"chosen": {"decoder": list(dec), "min_dead": md},
                            "train_objective": float(objective[best]),
                            "test": test_scores, "test_default_hmm": default,
                            "test_rule_R5": r5_spell(test)}
        print("decoder", test, decoder_cv[test], flush=True)

    # final decoder: chosen on all three games with out-of-fold-style probabilities (each game scored
    # with a model trained on the other two)
    objective = []
    for dec, md in cands:
        vals = []
        for n in names:
            train = tuple(m for m in names if m != n)
            v = spell_of(n, train, dec, md)
            vals += [v["full"], v["erased"]]
        objective.append(np.mean(vals))
    order = sorted(range(len(cands)), key=lambda i: (-round(objective[i], 9), i))
    final = cands[order[0]]
    result = {"cv": cv, "possession_config": cfg, "possession_config_adopted": passed_b,
              "decoder_cv": decoder_cv,
              "final_decoder": {"decoder": list(final[0]), "min_dead": final[1],
                                "objective": float(objective[order[0]])},
              "n_decoders": len(cands)}
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=1)
    print("final decoder", result["final_decoder"])
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    with cal.cached_episodes():
        main()
