"""Train the stopped-ball model on the three external Metrica games and save it to models/.

    python scripts/train_dead_model.py

Training data: Sample_Game_1, 2 and 3 of https://github.com/metrica-sports/sample-data (corrected
tracking and ball-dead state by Metrica; not BAR-ATH). Label: stopped where the corrected tracking
has no ball inside a period. Features: src/features_players.py without the number of tracked
players. Deterministic (src/dead_model.py has no randomness), so the file can be regenerated.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import dead_model as dm  # noqa: E402
from src import external as ext  # noqa: E402
import dead_model_external as dme  # noqa: E402

OUT = os.path.join(ROOT, "models", "dead_model.json")


def main():
    xs, ys = [], []
    for g in ext.GAMES:
        game = ext.load_game(g)
        graded = ext.graded_mask(game)
        xs.append(dme.player_matrix(game)[graded])
        ys.append((ext.labels(game)[graded] == ext.D).astype(int))
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = dm.fit(x, y)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dm.save(model, OUT, {"train": "Metrica sample-data games 1-3", "frames": int(len(y)),
                         "stopped_share": float(y.mean()), "features": "src/features_players.py minus n_tracked"})
    print("wrote", os.path.relpath(OUT, ROOT), "frames", len(y), "stopped", float(y.mean()))


if __name__ == "__main__":
    main()
