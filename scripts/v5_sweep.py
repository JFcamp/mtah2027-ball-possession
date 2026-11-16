"""Part B: possession configuration under degradation, with the TRUE dead-ball mask.

    python scripts/v5_sweep.py

Writes data_external/cache/v5_sweep.json: spellF1 (and macroF1) of every configuration of the enlarged
grid, per game and condition (CLEAN, DEG_BAR, DEG_40; degraded conditions averaged over 2 seeds).
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import calibration as cal  # noqa: E402
from src import external as ext  # noqa: E402
from src import lab  # noqa: E402

OUT = os.path.join(ROOT, "data_external", "cache", "v5_sweep.json")


def main():
    configs = lab.configs_v5()
    results = {}
    for g in ext.GAMES:
        game = ext.load_game(g)
        gt = ext.labels(game)
        dead = gt == ext.D
        inputs = lab.pipeline_inputs_by_condition(game, gt)
        results[game.name] = {c: [] for c in lab.CONDITIONS}
        with cal.cached_episodes():
            for cfg in configs:
                for cond, items in inputs.items():
                    vals = []
                    for track, feats, used in items:
                        s = ext.score(gt, cal.run_possession(used, dead, track, feats, cfg))
                        vals.append([s["spell_f1"], s["macro_f1"]])
                    results[game.name][cond].append(np.mean(vals, axis=0).tolist())
        print(game.name, "done", flush=True)
    with open(OUT, "w") as handle:
        json.dump({"configs": configs, "results": results}, handle)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
