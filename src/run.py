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
from . import features as feat
from . import io as sio
from . import possession as pos
from . import postprocess as post

LETTERS = np.array(["H", "A", "D", "X"])


class Pipeline(NamedTuple):
    labels: np.ndarray            # [N] int8 final codes (H/A/D/X)
    pre_merge: np.ndarray         # [N] int8 codes before post-processing
    track: ballmod.BallTrack
    features: feat.Features
    phase_table: dict
    dead: db.DeadBall
    segments: list


def build(use_cache: bool = True) -> Pipeline:
    tracking = sio.load_tracking(use_cache=use_cache)
    intervals = sio.load_smart_tagging()
    track = ballmod.clean_ball(tracking.ball_xy)
    features, table = feat.compute_features(tracking, track, intervals)
    dead = db.build(track, features, intervals)
    pre, segments = pos.assign(dead, features, track)
    labels = post.finalize(pre, sio.graded_mask())
    return Pipeline(labels, pre, track, features, table, dead, segments)


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
    args = parser.parse_args(argv)
    result = build()
    write_csv(args.out, result.labels)
    counts = {str(c): int((result.labels == i).sum()) for i, c in enumerate(LETTERS)}
    print("wrote {}  {}".format(args.out, counts))


if __name__ == "__main__":
    main()
