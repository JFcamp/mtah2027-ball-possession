"""Metrica public sample games (https://github.com/metrica-sports/sample-data) as a validation set.

Allowed by the rules (other games, labels produced by the data provider) and never mixed with the
BAR-ATH match. Files go in data_external/sample-data/ (git-ignored).

Each game is converted to the structures of the pipeline: 25 fps, positions in metres on a
105 x 68 m pitch (the sample data states these dimensions; BAR-ATH uses 105.3 x 68), frame index
0-based (the files count from 1), `team_of_slot` 0 = home / 1 = away.

Labels, from the provider's data only:
  D  the ball is absent from the corrected tracking (dead ball) inside a period
  H/A  the team of the latest ball event (PASS, SHOT, RECOVERY, CARRY, SET PIECE) of the live
       spell; losses, challenges, fouls, cards and "ball out" do not change the owner, so the
       owner changes only when the opponent has a ball event of its own.
  X  outside the two periods and on the first frame of the second period (so that the scorer
     never joins the two halves into one run).
"""

import contextlib
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, NamedTuple

import numpy as np
import pandas as pd

from . import ball as ballmod
from . import features as feat
from . import io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "data_external", "sample-data", "data")
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
BALL_EVENTS = ("PASS", "SHOT", "RECOVERY", "CARRY", "SET PIECE")
H, A, D, X = 0, 1, 2, 3


class Game(NamedTuple):
    name: str
    ball_xy: np.ndarray       # [N, 2] metres, NaN = dead ball (corrected tracking)
    players_xy: np.ndarray    # [N, P, 2] metres
    team_of_slot: np.ndarray  # [P]
    period: np.ndarray        # [N] 0 none, 1, 2
    events: pd.DataFrame      # team (0/1), type, subtype, period, start, end (0-based frames)


def _read_csv_game(g: int) -> Game:
    base = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(g), "Sample_Game_{}_".format(g))
    sides = {}
    for side in ("Home", "Away"):
        sides[side] = pd.read_csv(base + "RawTrackingData_{}_Team.csv".format(side), skiprows=2)
    home, away = sides["Home"], sides["Away"]
    frames = home["Frame"].to_numpy()
    assert (frames == np.arange(1, len(frames) + 1)).all() and (frames == away["Frame"].to_numpy()).all()
    scale = np.array(PITCH)
    ball = home.iloc[:, -2:].to_numpy(float) * scale
    players = []
    team = []
    for side, frame in ((0, home), (1, away)):
        vals = frame.iloc[:, 3:-2].to_numpy(float)
        p = vals.reshape(len(frame), -1, 2) * scale
        players.append(p)
        team += [side] * p.shape[1]
    ev = pd.read_csv(base + "RawEventsData.csv")
    events = pd.DataFrame({
        "team": np.where(ev["Team"] == "Home", 0, 1),
        "type": ev["Type"], "subtype": ev["Subtype"].fillna(""),
        "period": ev["Period"], "start": ev["Start Frame"] - 1, "end": ev["End Frame"] - 1})
    return Game("Sample_Game_{}".format(g), ball, np.concatenate(players, axis=1),
                np.array(team, dtype=np.int8), home["Period"].to_numpy(dtype=np.int8), events)


def _read_epts_game(g: int) -> Game:
    folder = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(g))
    base = os.path.join(folder, "Sample_Game_{}_".format(g))
    meta = ET.parse(base + "metadata.xml").getroot()
    params = {p.findtext("Name"): p.findtext("Value") for p in meta.iter("ProviderParameter")}
    first_end, second_start, second_end = (int(params["first_half_end"]), int(params["second_half_start"]),
                                           int(params["second_half_end"]))
    with open(base + "tracking.txt") as handle:
        lines = handle.readlines()
    n = len(lines)
    players = np.full((n, 22, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for i, line in enumerate(lines):
        parts = line.split(":")
        players[i] = np.fromstring(parts[1].replace(";", ","), sep=",").reshape(22, 2)
        ball[i] = np.fromstring(parts[2], sep=",")
    scale = np.array(PITCH)
    period = np.zeros(n, dtype=np.int8)
    period[:first_end] = 1
    period[second_start - 1:second_end] = 2
    with open(base + "events.json") as handle:
        data = json.load(handle)["data"]
    rows = []
    for e in data:
        st = e.get("subtypes")
        st = "" if st is None else (st["name"] if isinstance(st, dict) else "+".join(x["name"] for x in st))
        rows.append((0 if e["team"]["id"] == "FIFATMA" else 1, e["type"]["name"], st, e["period"],
                     e["start"]["frame"] - 1, e["end"]["frame"] - 1))
    events = pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])
    # slots 0-10 are team A (home), 11-21 team B in every data format specification of the file
    return Game("Sample_Game_{}".format(g), ball * scale, players * scale,
                np.array([0] * 11 + [1] * 11, dtype=np.int8), period, events)


def load_game(g: int) -> Game:
    return _read_csv_game(g) if g in (1, 2) else _read_epts_game(g)


def graded_mask(game: Game) -> np.ndarray:
    """Frames inside a period, except the first frame of the second period."""
    mask = game.period > 0
    mask[np.flatnonzero(game.period == 2)[0]] = False
    return mask


def half_blocks(game: Game):
    out = []
    for p in (1, 2):
        idx = np.flatnonzero((game.period == p) & graded_mask(game))
        out.append((p, int(idx[0]), int(idx[-1])))
    return out


def runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def labels(game: Game) -> np.ndarray:
    """[N] int8 codes H/A/D/X from the provider's data (see the module docstring)."""
    n = len(game.ball_xy)
    graded = graded_mask(game)
    present = np.isfinite(game.ball_xy[:, 0])
    out = np.full(n, X, dtype=np.int8)
    out[graded & ~present] = D
    ev = game.events[game.events["type"].isin(BALL_EVENTS)].sort_values("start", kind="stable")
    starts, teams = ev["start"].to_numpy(), ev["team"].to_numpy()
    for lo, hi in runs(present & graded):
        sel = np.flatnonzero((starts >= lo - 10) & (starts < hi))
        if len(sel):
            e_frames, e_team = starts[sel], teams[sel]
            owner = np.full(hi - lo, e_team[0], dtype=np.int8)
            for f, t in zip(e_frames, e_team):
                owner[max(f - lo, 0):] = t
        else:
            before = np.flatnonzero(starts < lo)
            if not len(before):
                out[lo:hi] = D
                continue
            owner = np.full(hi - lo, teams[before[-1]], dtype=np.int8)
        out[lo:hi] = owner
    return out


@contextlib.contextmanager
def as_pipeline_game(game: Game):
    """Make src.io describe `game` (frame count, graded frames, halves) while the block runs.

    The pipeline modules read sio.N_FRAMES, sio.graded_mask() and sio.half_blocks(); pointing them at
    another game lets the same unmodified code run on it.
    """
    saved = (sio.N_FRAMES, sio.graded_mask, sio.half_blocks)
    sio.N_FRAMES = len(game.ball_xy)
    sio.graded_mask = lambda: graded_mask(game)
    sio.half_blocks = lambda: half_blocks(game)
    try:
        yield
    finally:
        sio.N_FRAMES, sio.graded_mask, sio.half_blocks = saved


def pipeline_inputs(game: Game, ball_xy=None):
    """(BallTrack, Features) for the pipeline, with no Smart Tagging (phase_team unknown).

    `ball_xy` overrides the ball (for the artificial-gap tests); the tracking itself is untouched.
    """
    ball = game.ball_xy if ball_xy is None else ball_xy
    with as_pipeline_game(game):
        track = ballmod.clean_ball(ball)
        activity, _ = feat.activity_from(game.players_xy)
        out = feat.ball_out_mask(track.xy)
        dist, nearest = feat.nearest_players(game.players_xy, track.xy, game.team_of_slot)
        phase = np.full(len(ball), feat.NO_TEAM, dtype=np.int8)
    return track, feat.Features(activity, out, dist, nearest, phase)


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Official metrics of evaluation/possession_metrics.py: spellF1, macroF1, #possessions."""
    import sys
    path = os.path.join(ROOT, "evaluation")
    if path not in sys.path:
        sys.path.insert(0, path)
    import possession_metrics as pm
    result = pm.evaluate(gt.astype(np.int8), pred.astype(np.int8))
    return {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "n_pred": result["n_pred_spells"], "n_gt": result["n_gt_spells"],
            "dur": result["mean_spell_duration_s"]}
