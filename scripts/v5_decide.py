"""Part 6.1: apply the criteria and write docs/decision_v5.md and models/v5_config.json.

    python scripts/v5_decide.py

Reads the results of Parts B, C and C2 (data_external/cache/v5_*.json), computes Part D (transfer to BAR-ATH
by algorithmic weak labels, plausibility, structure) and Part V (pass-chain veto), then writes the
composition of v5. It never reads the public ground truth.
"""

import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import ball as ballmod  # noqa: E402
from src import dead_model as dm  # noqa: E402
from src import decode_v5 as dec5  # noqa: E402
from src import features as feat  # noqa: E402
from src import features_v5 as f5  # noqa: E402
from src import io as sio  # noqa: E402
from src import lab  # noqa: E402
from src import models_v5 as m5  # noqa: E402
from src import possession as pos  # noqa: E402
from src import run as runner  # noqa: E402
from src import veto as vetomod  # noqa: E402
from src import weak_labels as wl  # noqa: E402

CACHE = os.path.join(ROOT, "data_external", "cache")
MODELS = os.path.join(ROOT, "models")
FPS = 25
V4_DECODER = dict(threshold=0.5, min_stop=50, close_gap=0, refine=False)


def runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def possessions(labels):
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts, stops = np.concatenate(([0], edges)), np.concatenate((edges, [len(labels)]))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if labels[a] in (0, 1)]


def stats(result):
    labels = result.labels
    graded = sio.graded_mask()
    poss = [p for _, a, z in sio.half_blocks() for p in possessions(labels[a:z + 1])]
    stops = [(b - a) for _, a, z in sio.half_blocks() for a, b in runs(labels[a:z + 1] == 2)]
    long_stops = [s for s in stops if s >= 50]
    return {"d_share": float((labels[graded] == 2).mean()), "possessions": len(poss),
            "mean_possession_s": float(np.mean([(b - a) / FPS for a, b in poss])),
            "stoppages_2s": len(long_stops), "mean_stoppage_s": float(np.mean(long_stops) / FPS)}


def v4_model_as_l0(path):
    """The v4 model (models/dead_model.json) in the v5 model format, bit for bit."""
    v4 = dm.load(os.path.join(MODELS, "dead_model.json"))
    names = f5.compute(np.zeros((300, 2, 2)), np.array([0, 1]))[1]
    return {"kind": "L0", "columns": f5.l0_columns(names), "mean": v4.mean.tolist(),
            "scale": v4.scale.tolist(), "weights": v4.weights.tolist()}


def main():
    b = json.load(open(os.path.join(CACHE, "v5_b.json")))
    c = json.load(open(os.path.join(CACHE, "v5_model.json")))
    d = json.load(open(os.path.join(CACHE, "v5_decoder.json")))

    # ---- composition from B, C and the end-to-end check ------------------------------------------
    use_b, kind, e2e = b["passed"], c["chosen"], d["e2e_pass"]
    cfg = b["final_config"] if use_b else lab.V4_CONFIG
    decoder = d["final_decoder"]
    if not e2e:
        cfg, decoder, kind, use_b = lab.V4_CONFIG, V4_DECODER, "L0", False
    model_dict = (m5.load(os.path.join(CACHE, "final_{}.json".format(kind))) if e2e
                  else v4_model_as_l0(None))
    short_dict = None
    if e2e and decoder["refine"]:
        short_dict = m5.load(os.path.join(CACHE, "final_short.json"))

    # ---- Part D: transfer to BAR-ATH, algorithmic weak labels -----------------------------------
    tracking = sio.load_tracking()
    intervals = sio.load_smart_tagging()
    track = ballmod.clean_ball(tracking.ball_xy)
    features, _ = feat.compute_features(tracking, track, intervals)
    graded = sio.graded_mask()
    x = runner.v5_features(tracking)
    sources, weak_live, weak_buildup = wl.build(track, features, intervals)
    weak_stop = np.zeros(sio.N_FRAMES, dtype=bool)
    for m in sources.values():
        weak_stop |= m

    def t1(model, decoder_):
        p = m5.predict(model, x)
        short = m5.predict(short_dict, x) if (short_dict is not None and decoder_["refine"]) else None
        stopped = dec5.decode(p, sio.half_blocks(), decoder_["threshold"], decoder_["min_stop"],
                              decoder_["close_gap"], decoder_["refine"], short)
        sel = (weak_stop | weak_live) & graded
        auc = dm.auc(p[sel], weak_stop[sel].astype(int))
        hits = {k: float(stopped[m].mean()) for k, m in sources.items()}
        res = {"auc": auc, "source_hit": hits, "mean_hit": float(np.mean(list(hits.values()))),
               "false_stop_live": float(stopped[weak_live].mean()),
               "false_stop_buildup": float(stopped[weak_buildup].mean()) if weak_buildup.any() else float("nan"),
               "weak_stop_frames": int((weak_stop & graded).sum()), "weak_live_frames": int(weak_live.sum()),
               "weak_buildup_frames": int(weak_buildup.sum())}
        res["pass"] = bool(auc >= 0.85 and res["mean_hit"] >= 0.70 and res["false_stop_live"] <= 0.20)
        return res, stopped

    t1_new, stopped_new = t1(model_dict, decoder)
    candidate_models = [("new" if kind != "L0" or e2e else "L0", model_dict, decoder, t1_new)]
    t1_l0, _ = t1(v4_model_as_l0(None), V4_DECODER)
    use_model, use_decoder, used_short = model_dict, decoder, short_dict
    use_name = kind
    if not t1_new["pass"] and t1_l0["pass"]:
        use_model, use_decoder, used_short, use_name = v4_model_as_l0(None), V4_DECODER, None, "L0 (v4 model)"
        t1_used = t1_l0
    else:
        t1_used = t1_new

    options = {"model": "models/dead_model_v5.json", "short_model": "models/dead_model_v5_short.json" if used_short else None,
               "decoder": use_decoder, "possession": cfg, "all_frames": True, "veto": False}
    m5.save(use_model, os.path.join(MODELS, "dead_model_v5.json"))
    if used_short:
        m5.save(used_short, os.path.join(MODELS, "dead_model_v5_short.json"))

    d_ok = t1_used["pass"]
    result_all = runner.build_v5(options) if d_ok else None
    st_all = stats(result_all) if d_ok else None
    t2 = bool(d_ok and 0.35 <= st_all["d_share"] <= 0.55)
    t3 = bool(d_ok and 80 <= st_all["stoppages_2s"] <= 170 and 0.6 * 353 <= st_all["possessions"] <= 1.5 * 353)
    d_pass = bool(d_ok and t2 and t3)
    options["all_frames"] = d_pass

    # ---- Part V ---------------------------------------------------------------------------------
    veto_info = {"evaluated": False}
    use_veto = False
    if d_pass:
        veto = vetomod.chain_veto(track, features, cfg)
        inside = np.isfinite(track.xy[:, 0]) & ~features.ball_out
        stop_inside = weak_stop & inside & graded
        trig = float(veto[stop_inside].mean()) if stop_inside.any() else float("nan")
        cover = float(veto[weak_live].mean())
        use_veto = bool(trig <= 0.02 and cover >= 0.30)
        veto_info = {"evaluated": True, "false_trigger_on_weak_stopped": trig, "coverage_of_weak_live": cover,
                     "pass": use_veto}
    options["veto"] = use_veto

    # ---- final composition and statistics -------------------------------------------------------
    changed = bool(use_b or (e2e and (kind != "L0" or decoder != V4_DECODER)) or d_pass or use_veto)
    v4 = runner.build("v4")
    v5 = runner.build_v5(options) if changed else v4
    st_v4, st_v5 = stats(v4), stats(v5)
    out = {"B": {"passed": b["passed"], "final_config": b["final_config"], "folds": b["folds"],
                 "final_objective": b["final_objective"], "v4_objective": b["v4_objective"]},
           "C": {"passed": c["passed"], "chosen": c["chosen"], "mean_deg_auc": c["mean_deg_auc"]},
           "E2E": {"passed": e2e, "folds": d["folds"], "decoder": d["final_decoder"]},
           "D": {"T1": t1_used, "T1_L0": t1_l0, "T1_new": t1_new, "model_used": use_name, "T2": t2, "T3": t3,
                 "pass": d_pass, "stats_all_frames": st_all},
           "V": veto_info, "options": options, "changed": changed, "v4_stats": st_v4, "v5_stats": st_v5}
    with open(os.path.join(CACHE, "v5_decision.json"), "w") as handle:
        json.dump(out, handle, indent=1, default=float)
    if changed:
        with open(os.path.join(MODELS, "v5_config.json"), "w") as handle:
            json.dump(options, handle, indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("changed", "D", "V")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
