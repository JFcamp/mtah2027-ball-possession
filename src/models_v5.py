"""Stopped-ball models of v5, trained with scikit-learn or numpy, run with numpy only.

  L0  the v4 logistic model: 20 old features at 1 s and 5 s (src/dead_model.py)
  L1  logistic regression on the 65 v5 features
  G1  gradient boosting (scikit-learn GradientBoostingClassifier, fixed random_state, no early stopping),
      NaN features replaced by the training mean; the trees are exported to JSON and evaluated here with
      numpy, so scikit-learn is needed to train a model but not to run the pipeline.

Class-balanced weights everywhere, so P >= 0.5 is the natural decision threshold. Deterministic.
"""

import json
from typing import Dict

import numpy as np

from . import dead_model as dm

GB_PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0,
             "random_state": 0}


def train(kind: str, x: np.ndarray, y: np.ndarray, columns, stride: int = 1) -> Dict:
    """kind in {"L0", "L1", "G1"}; `columns` selects the features; rows are taken every `stride`."""
    xs, ys = x[::stride][:, columns], y[::stride]
    if kind in ("L0", "L1"):
        m = dm.fit(xs, ys)
        return {"kind": kind, "columns": list(map(int, columns)), "mean": m.mean.tolist(),
                "scale": m.scale.tolist(), "weights": m.weights.tolist()}
    from sklearn.ensemble import GradientBoostingClassifier
    mean = np.nanmean(xs, axis=0)
    filled = np.where(np.isfinite(xs), xs, mean)
    weight = np.where(ys == 1, len(ys) / (2.0 * (ys == 1).sum()), len(ys) / (2.0 * (ys == 0).sum()))
    clf = GradientBoostingClassifier(**GB_PARAMS)
    clf.fit(filled, ys, sample_weight=weight)
    trees = []
    for stage in clf.estimators_[:, 0]:
        t = stage.tree_
        trees.append({"left": t.children_left.tolist(), "right": t.children_right.tolist(),
                      "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                      "value": t.value[:, 0, 0].tolist()})
    init = float(clf._raw_predict_init(filled[:1])[0, 0])
    return {"kind": kind, "columns": list(map(int, columns)), "mean": mean.tolist(), "init": init,
            "learning_rate": GB_PARAMS["learning_rate"], "trees": trees}


def predict(model: Dict, x: np.ndarray) -> np.ndarray:
    """P(stopped) for every row of the FULL feature matrix `x` (the model picks its columns)."""
    xs = x[:, model["columns"]].astype(np.float64)
    if model["kind"] in ("L0", "L1"):
        m = dm.Model(np.array(model["mean"]), np.array(model["scale"]), np.array(model["weights"]))
        return dm.predict_proba(m, xs)
    xs = np.where(np.isfinite(xs), xs, np.array(model["mean"]))
    raw = np.full(len(xs), model["init"])
    rows = np.arange(len(xs))
    for tree in model["trees"]:
        left, right = np.array(tree["left"]), np.array(tree["right"])
        feature, threshold, value = np.array(tree["feature"]), np.array(tree["threshold"]), np.array(tree["value"])
        node = np.zeros(len(xs), dtype=np.int64)
        while True:
            internal = left[node] != -1
            if not internal.any():
                break
            go_left = xs[rows, np.maximum(feature[node], 0)] <= threshold[node]
            node = np.where(internal, np.where(go_left, left[node], right[node]), node)
        raw += model["learning_rate"] * value[node]
    return 1.0 / (1.0 + np.exp(-raw))


def save(model: Dict, path: str) -> None:
    with open(path, "w") as handle:
        json.dump(model, handle, sort_keys=True)


def load(path: str) -> Dict:
    with open(path) as handle:
        return json.load(handle)
