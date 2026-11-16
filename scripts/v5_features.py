"""Cache the v5 player features of the external games (clean and two degraded copies) and of BAR-ATH.

    python scripts/v5_features.py            # external games
    python scripts/v5_features.py --bar      # BAR-ATH graded tracking too (data/cache)
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import external as ext  # noqa: E402
from src import features_v5 as f5  # noqa: E402
from src import io as sio  # noqa: E402
from src import lab  # noqa: E402

EXT_CACHE = os.path.join(ROOT, "data_external", "cache")
BAR_PATH = os.path.join(ROOT, "data", "cache", "bar_v5feat.npy")


def ext_path(name, variant):
    return os.path.join(EXT_CACHE, "v5feat_{}_{}.npy".format(name, variant))


def load_ext(name, variant):
    return np.load(ext_path(name, variant))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar", action="store_true")
    args = parser.parse_args()
    if args.bar:
        tracking = sio.load_tracking()
        x, names = f5.compute(tracking.players_xy, tracking.team_of_slot)
        np.save(BAR_PATH, x)
        print("BAR", x.shape, flush=True)
        return
    for g in ext.GAMES:
        game = ext.load_game(g)
        variants = {"clean": game}
        variants.update({"deg{}".format(s): d for s, d in lab.degraded_games(game).items()})
        for variant, G in variants.items():
            if os.path.isfile(ext_path(game.name, variant)):
                continue
            x, names = f5.compute(G.players_xy, G.team_of_slot)
            np.save(ext_path(game.name, variant), x)
            print(game.name, variant, x.shape, flush=True)


if __name__ == "__main__":
    main()
