"""One command from the match data to the submission CSV.

    python -m src.run --out submissions/v1.csv
    python -m src.run --out "submissions/branca analytics.csv"

Deterministic: no randomness, no threads, no wall-clock; the same data gives the same bytes.
Reads only the tactical tracking (ATD) and the Smart Tagging from data/ (or MTAH_DATA_DIR).
"""

import argparse
import os
from typing import NamedTuple

import numpy as np

from . import ball as ballmod
from . import dead_ball as db
from . import dead_hmm as hmm
from . import dead_model as dmodel
from . import features as feat
from . import features_players as fplayers
from . import io as sio
from . import possession as pos
from . import postprocess as post
from . import pseudo_labels as pl

LETTERS = np.array(["H", "A", "D", "X"])


class Pipeline(NamedTuple):
    labels: np.ndarray            # [N] int8 final codes (H/A/D/X)
    pre_merge: np.ndarray         # [N] int8 codes before post-processing
    track: ballmod.BallTrack
    features: feat.Features
    phase_table: dict
    dead: db.DeadBall
    segments: list
    extra: dict = {}


def build(version: str = "v1", use_cache: bool = True) -> Pipeline:
    """v1: dead-ball rules R1-R6. v2: R5 and the look-back / stoppage caps are replaced by a model
    of live / stopped play (player features, trained on pseudo-labels) decoded with an HMM."""
    tracking = sio.load_tracking(use_cache=use_cache)
    intervals = sio.load_smart_tagging()
    track = ballmod.clean_ball(tracking.ball_xy)
    features, table = feat.compute_features(tracking, track, intervals)
    dead = db.build(track, features, intervals)
    extra = {"dead_v1": dead}
    if version == "v2":
        dead, extra = _dead_v2(tracking, track, features, intervals, dead)
    elif version != "v1":
        raise ValueError("unknown version {!r}".format(version))
    pre, segments = pos.assign(dead, features, track)
    labels = post.finalize(pre, sio.graded_mask())
    return Pipeline(labels, pre, track, features, table, dead, segments, extra)


def _dead_v2(tracking, track, features, intervals, dead_v1):
    graded = sio.graded_mask()
    pseudo, counts = pl.build(track, features, intervals, dead_v1)
    x, names = fplayers.compute(tracking.players_xy, tracking.team_of_slot)
    train = pseudo >= 0
    model = dmodel.fit(x[train], pseudo[train].astype(int))
    p_stop = dmodel.predict_proba(model, x)
    stopped = np.zeros(sio.N_FRAMES, dtype=bool)
    for _, first, last in sio.half_blocks():
        sl = slice(first, last + 1)
        stopped[sl] = hmm.viterbi(p_stop[sl], pseudo[sl])
    final = np.zeros(sio.N_FRAMES, dtype=bool)
    for a, b in db.runs(stopped & graded):
        if b - a >= db.MIN_DEAD:
            final[a:b] = True
    dead = db.DeadBall(final, dead_v1.by_rule, {}, dead_v1.restarts)
    extra = {"dead_v1": dead_v1, "pseudo": pseudo, "pseudo_counts": counts, "model": model,
             "p_stop": p_stop, "x": x, "names": names, "viterbi": stopped}
    return dead, extra


def write_csv(path: str, labels: np.ndarray) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.write("".join("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[labels])))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="CSV to write, e.g. submissions/v1.csv")
    parser.add_argument("--version", default="v1", choices=("v1", "v2"), help="pipeline generation")
    args = parser.parse_args(argv)
    result = build(args.version)
    write_csv(args.out, result.labels)
    counts = {str(c): int((result.labels == i).sum()) for i, c in enumerate(LETTERS)}
    print("wrote {}  {}".format(args.out, counts))


if __name__ == "__main__":
    main()
