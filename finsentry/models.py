"""Unsupervised ML anomaly ensemble.

Combines Isolation Forest and Local Outlier Factor with a robust statistical
score (Mahalanobis distance) over standardized account features. Each model
votes a 0..1 anomaly score; the ensemble averages them. A lightweight
per-account explanation lists the features that deviate most from the
population, as an interpretability aid for analysts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def ml_anomaly(feats: pd.DataFrame, seed: int = 7) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return a frame with per-model and ensemble anomaly scores (0..1) plus
    an explanation of the top deviating features per account."""
    X = feats.to_numpy(dtype=float)
    n = X.shape[0]
    Xs = RobustScaler().fit_transform(X)
    Xs = np.nan_to_num(Xs)

    scores = pd.DataFrame(index=feats.index)

    # 1) Isolation Forest — higher score = more anomalous
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    iso.fit(Xs)
    scores["iso"] = _minmax(-iso.score_samples(Xs))

    # 2) Local Outlier Factor (novelty off, fit_predict) — higher = more anomalous
    n_neighbors = int(min(20, max(5, n - 1)))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit_predict(Xs)
    scores["lof"] = _minmax(-lof.negative_outlier_factor_)

    # 3) Robust Mahalanobis distance over a covariance estimate
    try:
        mcd = MinCovDet(random_state=seed).fit(Xs)
        maha = mcd.mahalanobis(Xs)
    except Exception:
        center = np.median(Xs, axis=0)
        maha = np.sum((Xs - center) ** 2, axis=1)
    scores["maha"] = _minmax(maha)

    scores["ml_score"] = scores[["iso", "lof", "maha"]].mean(axis=1)

    # ---- explanation: top deviating standardized features per account ----
    feat_names = list(feats.columns)
    z = np.abs(Xs)
    explanations: dict[str, list[str]] = {}
    order = np.argsort(-z, axis=1)
    for i, acct in enumerate(feats.index):
        top = [feat_names[j] for j in order[i, :3] if z[i, j] > 1.5]
        explanations[acct] = top
    return scores, explanations
