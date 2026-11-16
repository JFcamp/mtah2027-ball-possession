"""Part C2: decoder selection (nested cross-game) and the end-to-end check E2E.

    python scripts/v5_decoder.py

Reads data_external/cache/v5_model.json (Part C: chosen model kind) and v5_b.json (Part B: possession
configuration). Writes v5_decoder.json.

For each held-out game t (others a, b): the decoder is chosen using ONLY a and b, with the model that
scores game a trained on b alone and vice versa (nested); then the model trained on a+b and that decoder
are scored on t. Objective: mean spellF1 of the complete pipeline (stopped mask from the model +
possession logic) over DEG_BAR and DEG_40 (two seeds each). Tie-break: smaller index in the grid.
E2E: the new pipeline must beat the v4-like pipeline (L0 trained on a+b, threshold 0.5, minimum 50 frames,
v4 configuration) on t in DEG_BAR and DEG_40, in all three combinations.
"""

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import calibration as cal  # noqa: E402
from src import decode_v5 as dec5  # noqa: E402
from src import external as ext  # noqa: E402
from src import features_v5 as f5  # noqa: E402
from src import lab  # noqa: E402
from src import models_v5 as m5  # noqa: E402
import v5_features as v5f  # noqa: E402
import v5_model as v5m  # noqa: E402

CACHE = os.path.join(ROOT, "data_external", "cache")
THRESHOLDS = (0.4, 0.5, 0.6)
MIN_STOP = (25, 50, 75, 100)
CLOSE = (0, 25, 50)
REFINE = (False, True)
DEG_VARIANTS = ("deg0", "deg1")
DECODERS = [dict(threshold=t, min_stop=m, close_gap=c, refine=r)
            for t, m, c, r in itertools.product(THRESHOLDS, MIN_STOP, CLOSE, REFINE)]
V4_DECODER = dict(threshold=0.5, min_stop=50, close_gap=0, refine=False)


def main():
    model_info = json.load(open(os.path.join(CACHE, "v5_model.json")))
    kind = model_info["chosen"]
    b_info = json.load(open(os.path.join(CACHE, "v5_b.json")))
    cfg = b_info["final_config"] if b_info["passed"] else lab.V4_CONFIG

    games, names = v5m.load_games()
    gnames = list(games)
    full = {n: {v: v5f.load_ext(n, v) for v in v5m.VARIANTS} for n in gnames}   # all frames
    G = {g: ext.load_game(g) for g in ext.GAMES}
    G = {x.name: x for x in G.values()}
    gt = {n: ext.labels(G[n]) for n in gnames}
    inputs = {n: lab.pipeline_inputs_by_condition(G[n], gt[n]) for n in gnames}
    blocks = {n: ext.half_blocks(G[n]) for n in gnames}
    graded = {n: ext.graded_mask(G[n]) for n in gnames}
    short_cols = f5.short_columns(names)

    cache = {}

    def models(train, k):
        key = (tuple(train), k)
        if key not in cache:
            x, y = v5m.training_set(games, list(train), k)
            stride = v5m.STRIDE[k] * (2 if (k == "G1" and len(train) == 1) else 1)   # nested single-game fits: cheaper
            main_model = m5.train(k, x, y, v5m.columns_for(k, names), stride)
            short = None
            if k != "L0":
                xs, ys = v5m.training_set(games, list(train), "L1")
                short = m5.train("L1", xs, ys, short_cols, 2)
            cache[key] = (main_model, short)
        return cache[key]

    def probs(n, train, k):
        main_model, short = models(train, k)
        p = {v: m5.predict(main_model, full[n][v]) for v in DEG_VARIANTS}
        ps = {v: (m5.predict(short, full[n][v]) if short else None) for v in DEG_VARIANTS}
        return p, ps

    def spell(n, p, ps, dec, config):
        """DEG_BAR and DEG_40 spellF1 of the complete pipeline for game n (mean over the two seeds)."""
        out = {}
        for cond in ("DEG_BAR", "DEG_40"):
            vals = []
            for s, v in enumerate(DEG_VARIANTS):
                dead = dec5.decode(p[v], blocks[n], dec["threshold"], dec["min_stop"], dec["close_gap"],
                                   dec["refine"], ps[v])
                track, feats, used = inputs[n][cond][s]
                vals.append(ext.score(gt[n], cal.run_possession(used, dead, track, feats, config))["spell_f1"])
            out[cond] = float(np.mean(vals))
        return out

    result = {"model_kind": kind, "config": cfg, "folds": {}}
    with cal.cached_episodes():
        for test in gnames:
            a, b = [n for n in gnames if n != test]
            obj = []
            for dec in DECODERS:
                vals = []
                for n, other in ((a, b), (b, a)):
                    p, ps = probs(n, (other,), kind)
                    s = spell(n, p, ps, dec, cfg)
                    vals += [s["DEG_BAR"], s["DEG_40"]]
                obj.append(float(np.mean(vals)))
            order = sorted(range(len(DECODERS)), key=lambda i: (-round(obj[i], 9), i))
            best = DECODERS[order[0]]
            p, ps = probs(test, (a, b), kind)
            new = spell(test, p, ps, best, cfg)
            p0, ps0 = probs(test, (a, b), "L0")
            old = spell(test, p0, ps0, V4_DECODER, lab.V4_CONFIG)
            result["folds"][test] = {"chosen": best, "train_objective": obj[order[0]], "new": new,
                                     "v4_like": old, "pass": all(new[c] > old[c] for c in new)}
            print(test, result["folds"][test], flush=True)

        # final decoder on all three games, each scored by a model trained on the other two
        obj = []
        for dec in DECODERS:
            vals = []
            for n in gnames:
                others = tuple(m for m in gnames if m != n)
                p, ps = probs(n, others, kind)
                s = spell(n, p, ps, dec, cfg)
                vals += [s["DEG_BAR"], s["DEG_40"]]
            obj.append(float(np.mean(vals)))
        order = sorted(range(len(DECODERS)), key=lambda i: (-round(obj[i], 9), i))
        result["final_decoder"] = DECODERS[order[0]]
        result["final_objective"] = obj[order[0]]
        result["e2e_pass"] = all(f["pass"] for f in result["folds"].values())
    xs, ys = v5m.training_set(games, gnames, "L1")
    m5.save(m5.train("L1", xs, ys, short_cols, 2), os.path.join(CACHE, "final_short.json"))
    with open(os.path.join(CACHE, "v5_decoder.json"), "w") as handle:
        json.dump(result, handle, indent=1)
    print("final", result["final_decoder"], "E2E", result["e2e_pass"])


if __name__ == "__main__":
    main()
