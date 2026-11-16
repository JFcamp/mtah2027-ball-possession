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
from . import calibration
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


def build(version: str = "v4", use_cache: bool = True) -> Pipeline:
    """v1: dead-ball rules R1-R6. v2: R5 and the look-back / stoppage caps are replaced by a model
    of live / stopped play (player features, trained on pseudo-labels) decoded with an HMM."""
    if version == "v5":
        opts = load_v5_options()
        if opts is not None:
            return build_v5(opts, use_cache)
        version = "v4"     # no component of v5 passed its criterion: v5 is v4 (see docs/decision_v5.md)
    tracking = sio.load_tracking(use_cache=use_cache)
    fused = None
    if version == "v3":
        # v1 pipeline unchanged, with the tactical ball completed by the aligned broadcast ball.
        from . import align as alignmod
        from . import broadcast as bcast
        from . import fusion
        broadcast = bcast.load_tracking(use_cache=use_cache)
        fused = fusion.fuse_ball(tracking.ball_xy, broadcast.ball_xy,
                                 alignmod.align(broadcast, None, tracking, use_cache=use_cache))
        tracking = tracking._replace(ball_xy=fused.ball_xy)
    intervals = sio.load_smart_tagging()
    track = ballmod.clean_ball(tracking.ball_xy)
    features, table = feat.compute_features(tracking, track, intervals)
    cfg = None
    r5 = None
    if version in ("v4", "v4b", "v4c"):
        # v4 = v1 + (a) the possession configuration validated on the external games and
        #           (b) a stopped-ball model, trained on them, in place of the pace test of R5.
        # v4b / v4c apply only (a) / only (b).
        if version in ("v4", "v4c"):
            r5 = _model_stopped(tracking)
        if version in ("v4", "v4b"):
            cfg = V4_POSSESSION
    dead = db.build(track, features, intervals, r5)
    extra = {"dead_v1": dead, "fused": fused}
    if version == "v2":
        dead, extra = _dead_v2(tracking, track, features, intervals, dead)
    elif version not in ("v1", "v3", "v4", "v4b", "v4c"):
        raise ValueError("unknown version {!r}".format(version))
    if cfg is None:
        pre, segments = pos.assign(dead, features, track)
        labels = post.finalize(pre, sio.graded_mask())
    else:
        with calibration.configured(cfg):
            pre, segments = pos.assign(dead, features, track)
        labels = post.finalize(pre, sio.graded_mask(), cfg["min_poss"])
    return Pipeline(labels, pre, track, features, table, dead, segments, extra)


V5_PATH = os.path.join(sio.ROOT, "models", "v5_config.json")


def load_v5_options() -> dict:
    """What v5 is made of (written by scripts/v5_decide.py from the decision criteria); None if nothing passed."""
    import json
    if not os.path.isfile(V5_PATH):
        return None
    with open(V5_PATH) as handle:
        return json.load(handle)


def v5_features(tracking, use_cache=True):
    from . import features_v5
    path = os.path.join(sio.CACHE_DIR, "bar_v5feat.npy")
    if use_cache and os.path.isfile(path):
        x = np.load(path)
        if x.shape[0] == sio.N_FRAMES:
            return x
    x, _ = features_v5.compute(tracking.players_xy, tracking.team_of_slot)
    if use_cache:
        os.makedirs(sio.CACHE_DIR, exist_ok=True)
        np.save(path, x)
    return x


def build_v5(opts: dict, use_cache: bool = True) -> Pipeline:
    """v5: v4 + (as decided) a better stopped-ball model on EVERY graded frame, a decoder, a possession
    configuration robust to noise and a pass-chain veto. `opts`: model (path), decoder, possession,
    all_frames, veto, short_model (path or None)."""
    from . import decode_v5, models_v5, veto as vetomod
    tracking = sio.load_tracking(use_cache=use_cache)
    intervals = sio.load_smart_tagging()
    track = ballmod.clean_ball(tracking.ball_xy)
    features, table = feat.compute_features(tracking, track, intervals)
    cfg = opts["possession"]
    x = v5_features(tracking, use_cache)
    model = models_v5.load(os.path.join(sio.ROOT, opts["model"]))
    short = models_v5.load(os.path.join(sio.ROOT, opts["short_model"])) if opts.get("short_model") else None
    d = opts["decoder"]
    p = models_v5.predict(model, x)
    ps = models_v5.predict(short, x) if short is not None else None
    stopped = decode_v5.decode(p, sio.half_blocks(), d["threshold"], d["min_stop"], d["close_gap"],
                               d["refine"], ps)
    if opts.get("veto"):
        stopped &= ~vetomod.chain_veto(track, features, cfg)
    if opts.get("all_frames", True):
        dead = db.build(track, features, intervals, r5_stopped=np.zeros(sio.N_FRAMES, dtype=bool),
                        extra_dead=stopped)
    else:
        dead = db.build(track, features, intervals, r5_stopped=stopped)
    with calibration.configured(cfg):
        pre, segments = pos.assign(dead, features, track)
    labels = post.finalize(pre, sio.graded_mask(), cfg["min_poss"])
    return Pipeline(labels, pre, track, features, table, dead, segments, {"stopped": stopped, "p": p})


# Possession configuration chosen on the external games (scripts/possession_sweep.py, selection by
# cross-game validation, docs/report_v4.md): control radius 1.0 m, margin 0.3 m, switch after 6 frames of
# control or two touches, switch placed when the evidence is complete (no back-dating), minimum
# possession 25 frames.
V4_POSSESSION = {"r_ctrl": 1.0, "margin": 0.3, "min_control": 6, "two_touch": True,
                 "backdate": False, "min_poss": 25}
MODEL_PATH = os.path.join(sio.ROOT, "models", "dead_model.json")
MODEL_THRESHOLD = 0.5   # P(stopped) >= 0.5: the model is trained with balanced classes


def _model_stopped(tracking) -> np.ndarray:
    """[N] bool: the model (src/dead_model.py, models/dead_model.json) says stopped, from the
    players only. Trained on the external games; features as in src/features_players.py, without
    the number of tracked players."""
    x, names = fplayers.compute(tracking.players_xy, tracking.team_of_slot)
    keep = [i for i, n in enumerate(names) if not n.startswith("n_tracked")]
    p = dmodel.predict_proba(dmodel.load(MODEL_PATH), x[:, keep])
    return p >= MODEL_THRESHOLD





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
    parser.add_argument("--version", default="v4", choices=("v1", "v2", "v3", "v4", "v4b", "v4c", "v5"), help="pipeline generation")
    args = parser.parse_args(argv)
    result = build(args.version)
    write_csv(args.out, result.labels)
    counts = {str(c): int((result.labels == i).sum()) for i, c in enumerate(LETTERS)}
    print("wrote {}  {}".format(args.out, counts))


if __name__ == "__main__":
    main()
