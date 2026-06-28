"""End-to-end detection pipeline orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .features import account_features
from .rules import apply_rules, RuleHit
from .models import ml_anomaly
from .graph import graph_signals
from .scoring import score_accounts


@dataclass
class PipelineResult:
    txns: pd.DataFrame
    features: pd.DataFrame
    results: pd.DataFrame                       # one row per account, ranked by risk
    rule_hits: dict[str, list[RuleHit]]
    cycles: list[list[str]]
    graph_sig: pd.DataFrame
    ml_scores: pd.DataFrame
    ml_explain: dict[str, list[str]] = field(default_factory=dict)

    @property
    def alerts(self) -> pd.DataFrame:
        """Accounts at MEDIUM risk or above — the analyst alert queue."""
        return self.results[self.results["risk_band"].isin(["MEDIUM", "HIGH", "CRITICAL"])]


def run_pipeline(txns: pd.DataFrame, weights: dict[str, float] | None = None, seed: int = 7) -> PipelineResult:
    feats = account_features(txns)
    graph_sig, cycles = graph_signals(txns)
    cycle_accounts = {a for cyc in cycles for a in cyc}
    rule_hits = apply_rules(feats, txns, cycle_accounts=cycle_accounts)
    ml_scores, ml_explain = ml_anomaly(feats, seed=seed)
    results = score_accounts(feats, rule_hits, ml_scores, graph_sig, ml_explain, weights)
    return PipelineResult(
        txns=txns,
        features=feats,
        results=results,
        rule_hits=rule_hits,
        cycles=cycles,
        graph_sig=graph_sig,
        ml_scores=ml_scores,
        ml_explain=ml_explain,
    )


def evaluate(results: pd.DataFrame, truth: dict[str, str], alert_band=("HIGH", "CRITICAL")) -> dict:
    """Precision/recall/F1 of the alert queue vs injected ground truth, plus ROC-AUC."""
    truth_accounts = set(truth)
    if not truth_accounts:
        return {}
    predicted = set(results[results["risk_band"].isin(alert_band)].index)
    universe = set(results.index)
    tp = len(predicted & truth_accounts)
    fp = len(predicted - truth_accounts)
    fn = len((truth_accounts & universe) - predicted)
    tn = len(universe - predicted - truth_accounts)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    auc = None
    try:
        from sklearn.metrics import roc_auc_score

        y = results.index.isin(truth_accounts).astype(int)
        if y.min() != y.max():
            auc = float(roc_auc_score(y, results["risk_score"].to_numpy()))
    except Exception:
        pass

    return dict(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        roc_auc=auc,
        n_truth=len(truth_accounts),
        n_alerts=len(predicted),
    )
