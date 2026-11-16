"""docs/domain_gap.md: BAR-ATH (automatic tracking) vs the external games (corrected), without labels.

    python scripts/domain_gap.py

A.1 statistics on both domains, A.4 check that the simulated degradation (src/degrade.py) brings the
external games closer to BAR-ATH in the distribution of the model's own features.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import ball as ballmod  # noqa: E402
from src import degrade  # noqa: E402
from src import external as ext  # noqa: E402
from src import features_players as fp  # noqa: E402
from src import io as sio  # noqa: E402
from src import run as runner  # noqa: E402

SEEDS = (0, 1)
BAR_FEATURES = os.path.join(ROOT, "data", "cache", "bar_playerfeat.npy")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def f(x, d=3):
    return "{:.{}f}".format(x, d)


def d2_percentiles(xy):
    d = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    ok = np.isfinite(d).all(axis=-1)
    a = np.abs(d[ok]).ravel()
    return np.percentile(a, [50, 90, 99]) * 1000  # mm


def count_stats(players, mask):
    n = np.isfinite(players[mask][..., 0]).sum(axis=1)
    return [f(n.mean(), 1), f(n.std(), 1)] + [int(v) for v in np.percentile(n, [1, 5, 25, 50])]


def player_features(players, teams, keep_names=True):
    x, names = fp.compute(players, teams)
    keep = [i for i, n in enumerate(names) if not n.startswith("n_tracked")]
    return x[:, keep], [names[i] for i in keep]


def main():
    tracking = sio.load_tracking()
    graded = sio.graded_mask()
    games = {g: ext.load_game(g) for g in ext.GAMES}
    L = ["# Domain gap: BAR-ATH vs the external games, and the simulated degradation", "",
         "No labels are used on either side. BAR-ATH numbers are on the graded frames. Constants of "
         "`src/degrade.py` are justified in its docstring.", ""]

    # ---- A.1 ------------------------------------------------------------------------------------
    L += ["## A.1 Measurements", "", "**High-frequency noise**: |second difference| of positions, in mm "
          "(p50 / p90 / p99). The automatic BAR-ATH tracking is not noisier than the corrected games "
          "for players; only the ball is, by a few millimetres.", ""]
    rows = [["BAR-ATH", "players"] + [f(v, 1) for v in d2_percentiles(tracking.players_xy[graded][:80000])],
            ["BAR-ATH", "ball"] + [f(v, 1) for v in d2_percentiles(tracking.ball_xy[graded])]]
    for g, G in games.items():
        gm = ext.graded_mask(G)
        rows.append([G.name, "players"] + [f(v, 1) for v in d2_percentiles(G.players_xy[gm][:80000])])
        rows.append([G.name, "ball"] + [f(v, 1) for v in d2_percentiles(G.ball_xy[gm])])
    L += table(["data", "object", "p50", "p90", "p99"], rows)

    L += ["**Tracked players per frame** (graded frames): mean, sd, p1, p5, p25, median.", ""]
    rows = [["BAR-ATH"] + count_stats(tracking.players_xy, graded)]
    for g, G in games.items():
        rows.append([G.name] + count_stats(G.players_xy, ext.graded_mask(G)))
    L += table(["data", "mean", "sd", "p1", "p5", "p25", "p50"], rows)
    L += ["Track fragments: BAR-ATH has 46 anonymous slots (tracklets): 1,219 fragments in the graded "
          "frames, median 35.5 s, p10 2.6 s. The external games have one track per player (44-50 "
          "fragments of 24-48 minutes). No feature uses identities, so this is recorded, not simulated.", ""]

    obs = np.isfinite(tracking.ball_xy[:, 0])
    out_rate = (ballmod.remove_outliers(tracking.ball_xy) & graded).sum() / (obs & graded).sum()
    rows = [["BAR-ATH", f(100 * out_rate, 2)]]
    for g, G in games.items():
        gm = ext.graded_mask(G)
        o = np.isfinite(G.ball_xy[:, 0])
        rows.append([G.name, f(100 * (ballmod.remove_outliers(G.ball_xy) & gm).sum() / (o & gm).sum(), 2)])
    L += ["**Impossible ball jumps removed by `ball.py`** (% of observed ball frames):", ""]
    L += table(["data", "% removed"], rows)
    L += ["**Disagreement between the two BAR-ATH cameras** (v3 alignment, `docs/report_v3.md`): players, "
          "nearest same-team neighbour, median 0.29 m, p90 1.16 m; ball, co-visible frames, median 0.79 m, "
          "p75 2.15 m, p90 5.94 m, p95 10.8 m. Autocorrelation of the ball disagreement (central part): "
          "0.995 at 1 frame, 0.90 at 5, 0.28 at 25, 0.05 at 75. Wrong-ball episodes are **not simulated**: "
          "the disagreement above 5 m cannot be attributed to a tracker (balls in the air project "
          "differently in each camera) and the attributable jumps are only 0.3%.", ""]

    v4 = runner.build("v4")
    lab = v4.labels
    present = np.isfinite(v4.track.xy[:, 0])
    live = (lab < 2) & graded
    rows = [["both halves", f(100 * (~present & live).sum() / live.sum(), 1)]]
    for h, a, z in sio.half_blocks():
        m = np.zeros(len(lab), bool)
        m[a:z + 1] = True
        rows.append(["half {}".format(h), f(100 * (~present & live & m).sum() / (live & m).sum(), 1)])
    L += ["**Ball absent in the frames v4 labels H/A** (algorithmic estimate of the ball loss during live "
          "play, so DEG_BAR erases {:.1f}% of the live frames; DEG_40 erases 40%):".format(
              100 * degrade.LIVE_ERASE_BAR), ""]
    L += table(["frames", "% ball absent"], rows)

    # ---- degraded games ---------------------------------------------------------------------------
    L += ["## A.2 Degradation parameters (`src/degrade.py`)", "",
          "| component | value | source |", "|---|---|---|",
          "| player position error | sd {} m per axis, smooth (kernel sd {} frames) | camera disagreement, second differences |".format(
              degrade.PLAYER_SIGMA, degrade.PLAYER_SMOOTH),
          "| ball position error | sd {} m per axis, smooth (kernel sd {} frames) | camera disagreement, second differences |".format(degrade.BALL_SIGMA, degrade.BALL_SMOOTH),
          "| player dropouts | solo {}% (mean {} s) + bursts {}/min (mean {} s, up to {}% of players) | tracked-count distribution |".format(
              int(100 * degrade.SOLO_SHARE), degrade.SOLO_MEAN_S, degrade.BURST_RATE_PER_MIN,
              degrade.BURST_MEAN_S, int(100 * degrade.BURST_MAX_FRACTION)),
          "| ball erased, DEG_BAR / DEG_40 | {}% / 40% of the live frames | v4 labels / E4d |".format(
              f(100 * degrade.LIVE_ERASE_BAR, 1)),
          "| wrong-ball episodes | not simulated | cannot be estimated reliably |", ""]
    rows = []
    degraded = {(g, s): degrade.degrade_game(G, s) for g, G in games.items() for s in SEEDS}
    for (g, s), D in degraded.items():
        gm = ext.graded_mask(D)
        rows.append(["{} seed {}".format(D.name, s)] + count_stats(D.players_xy, gm) +
                    [f(d2_percentiles(D.players_xy[gm][:80000])[0], 1), f(d2_percentiles(D.ball_xy[gm])[0], 1)])
    L += ["Tracked players after degradation (target BAR-ATH: 20.5, 3.3, 5, 14, 21, 22):", ""]
    L += table(["game", "mean", "sd", "p1", "p5", "p25", "p50", "players 2nd diff p50 (mm)", "ball 2nd diff p50 (mm)"], rows)

    # ---- A.4 ----------------------------------------------------------------------------------------
    if os.path.isfile(BAR_FEATURES):
        xb = np.load(BAR_FEATURES)
    else:
        xb, _ = player_features(tracking.players_xy, tracking.team_of_slot)
        np.save(BAR_FEATURES, xb)
    xb = xb[graded]
    xs, names = [], None
    clean, deg = [], []
    for g, G in games.items():
        gm = ext.graded_mask(G)
        x, names = player_features(G.players_xy, G.team_of_slot)
        clean.append(x[gm])
    for (g, s), D in degraded.items():
        x, _ = player_features(D.players_xy, D.team_of_slot)
        deg.append(x[ext.graded_mask(D)])
    clean = np.concatenate(clean)
    deg = np.concatenate(deg)

    def smd(a, b):
        ma, mb = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
        va, vb = np.nanvar(a, axis=0), np.nanvar(b, axis=0)
        return (ma - mb) / np.sqrt((va + vb) / 2)

    before, after = smd(xb, clean), smd(xb, deg)
    rows = [[n, f(b, 2), f(a, 2)] for n, b, a in zip(names, before, after)]
    rows.append(["**mean abs SMD**", f(np.abs(before).mean(), 2), f(np.abs(after).mean(), 2)])
    L += ["## A.4 Standardised difference of the model features (BAR-ATH minus external)", "",
          "SMD = difference of means over the pooled standard deviation, graded frames; 'before' uses the "
          "clean external games, 'after' the degraded ones (both seeds pooled). The features are those of "
          "the v4 model (20 player features). Part of the gap is the matches themselves (style, pitch "
          "use), which no degradation should remove.", ""]
    L += table(["feature", "SMD before", "SMD after"], rows)
    L += ["Features closer after the degradation: {} of {}.".format(
        int((np.abs(after) < np.abs(before)).sum()), len(names)), ""]

    path = os.path.join(ROOT, "docs", "domain_gap.md")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(L))
    print("wrote", os.path.relpath(path, ROOT),
          "mean|SMD| before {:.3f} after {:.3f}".format(np.abs(before).mean(), np.abs(after).mean()))


if __name__ == "__main__":
    main()
