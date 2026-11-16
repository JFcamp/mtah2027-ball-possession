"""Part B: possession-logic sweep on the external games with the TRUE dead-ball mask.

    python scripts/possession_sweep.py

Writes data_external/cache/sweep.json (git-ignored): spellF1 and macroF1 of every configuration, per
game, with the true ball ('full') and with ball gaps erased in live play ('erased', mean of two seeds).
Selection is by cross-game validation (choose on two games, test on the third), see src/calibration.py.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import calibration as cal  # noqa: E402
from src import external as ext  # noqa: E402

SEEDS = (0, 1)
OUT = os.path.join(ROOT, "data_external", "cache", "sweep.json")


def main():
    configs = cal.configs()
    results = {}  # game -> cond -> list over configs of [spell, macro]
    for g in ext.GAMES:
        game = ext.load_game(g)
        gt = ext.labels(game)
        dead = gt == ext.D
        live = gt < 2
        conditions = {"full": [None]}
        conditions["erased"] = list(SEEDS)
        results[game.name] = {}
        for cond, seeds in conditions.items():
            per_seed = []
            for seed in seeds:
                ball = None if seed is None else cal.erase_ball(game, live, seed)
                track, feats = ext.pipeline_inputs(game, ball)
                scores = []
                with cal.cached_episodes():
                    for cfg in configs:
                        pred = cal.run_possession(game, dead, track, feats, cfg)
                        s = ext.score(gt, pred)
                        scores.append([s["spell_f1"], s["macro_f1"], s["n_pred"]])
                per_seed.append(scores)
            results[game.name][cond] = np.mean(np.array(per_seed), axis=0).tolist()
            print(game.name, cond, "done", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump({"configs": configs, "results": results}, handle)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
