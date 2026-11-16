"""Logistic regression (numpy only) for P(stopped | player features), trained on pseudo-labels.

Deterministic: standardised features, class-balanced weights, ridge penalty, and a fixed number of
Newton (IRLS) iterations from a zero start. There is no random number anywhere.

Constants
---------
RIDGE = 1.0   penalty on standardised weights (the intercept is not penalised). With ~10^4-10^5
              training frames and 22 correlated features a light ridge keeps the solution stable;
              it is a regulariser, not a tuned value.
ITERATIONS = 25   Newton steps. The problem is convex and converges in under 10; 25 is a fixed
              budget so every run does the same arithmetic.
Class weights are balanced so that the output is a likelihood ratio between the two classes (equal
priors). The prior of being stopped is supplied by the HMM transition probabilities.
"""

from typing import NamedTuple

import numpy as np

RIDGE = 1.0
ITERATIONS = 25


class Model(NamedTuple):
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray  # [F + 1], last is the intercept


def _design(x, mean, scale):
    z = (x - mean) / scale
    z = np.where(np.isfinite(z), z, 0.0)  # missing feature = the training mean
    return np.concatenate([z, np.ones((len(z), 1))], axis=1)


def fit(x: np.ndarray, y: np.ndarray) -> Model:
    """y in {0, 1} (1 = stopped); rows with NaN features are kept (mean-imputed)."""
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[scale == 0] = 1.0
    d = _design(x, mean, scale)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    w = np.where(y == 1, len(y) / (2.0 * n_pos), len(y) / (2.0 * n_neg))
    penalty = np.eye(d.shape[1]) * RIDGE
    penalty[-1, -1] = 0.0
    beta = np.zeros(d.shape[1])
    for _ in range(ITERATIONS):
        p = 1.0 / (1.0 + np.exp(-(d @ beta)))
        grad = d.T @ (w * (p - y)) + penalty @ beta
        hess = (d * (w * p * (1 - p))[:, None]).T @ d + penalty
        beta = beta - np.linalg.solve(hess, grad)
    return Model(mean, scale, beta)


def predict_proba(model: Model, x: np.ndarray) -> np.ndarray:
    """P(stopped) for each row."""
    d = _design(x, model.mean, model.scale)
    return 1.0 / (1.0 + np.exp(-(d @ model.weights)))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties get the average rank)."""
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(len(s))
    edges = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(s)]))
    for a, b in zip(starts, stops):
        ranks[a:b] = (a + b + 1) / 2.0
    r = np.empty(len(s))
    r[order] = ranks
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def scores(pred_stopped: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy and balanced accuracy of a boolean prediction."""
    acc = float((pred_stopped == (y == 1)).mean())
    tpr = float(pred_stopped[y == 1].mean())
    tnr = float((~pred_stopped[y == 0]).mean())
    return {"accuracy": acc, "balanced_accuracy": (tpr + tnr) / 2.0}


def save(model: Model, path: str, meta: dict = None) -> None:
    import json
    payload = {"mean": model.mean.tolist(), "scale": model.scale.tolist(),
               "weights": model.weights.tolist(), "meta": meta or {}}
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)


def load(path: str) -> Model:
    import json
    with open(path) as handle:
        payload = json.load(handle)
    return Model(np.array(payload["mean"]), np.array(payload["scale"]), np.array(payload["weights"]))
