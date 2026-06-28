"""Composite risk scoring.

Blends the rule engine, the ML ensemble and the graph signals into a single
0..100 risk score per account, assigns a severity band, and produces the
reason codes that justify each alert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import RuleHit

DEFAULT_WEIGHTS = {"rules": 0.5, "ml": 0.3, "graph": 0.2}

BANDS = [
    (80, "CRITICAL"),
    (60, "HIGH"),
    (40, "MEDIUM"),
    (0, "LOW"),
]


def _band(score: float) -> str:
    for cutoff, name in BANDS:
        if score >= cutoff:
            return name
    return "LOW"


def score_accounts(
    feats: pd.DataFrame,
    rule_hits: dict[str, list[RuleHit]],
    ml_scores: pd.DataFrame,
    graph_sig: pd.DataFrame,
    ml_explain: dict[str, list[str]],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # rule contribution: saturating sum of fired-rule weights -> 0..1
    rule_strength = {}
    rule_codes = {}
    rule_reasons = {}
    for acct in feats.index:
        hits = rule_hits.get(acct, [])
        s = sum(h.weight for h in hits)
        rule_strength[acct] = float(1 - np.exp(-s))  # diminishing returns
        rule_codes[acct] = [h.code for h in hits]
        rule_reasons[acct] = [h.reason for h in hits]

    out = pd.DataFrame(index=feats.index)
    out["rule_score"] = pd.Series(rule_strength)
    out["ml_score"] = ml_scores["ml_score"].reindex(feats.index).fillna(0.0)
    out["graph_score"] = graph_sig["graph_score"].reindex(feats.index).fillna(0.0)

    out["risk_score"] = (
        100.0
        * (
            w["rules"] * out["rule_score"]
            + w["ml"] * out["ml_score"]
            + w["graph"] * out["graph_score"]
        )
    ).clip(0, 100)
    out["risk_band"] = out["risk_score"].apply(_band)

    out["rule_codes"] = pd.Series(rule_codes)
    out["n_rules"] = out["rule_codes"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # human-readable reason codes (rules first, then ml/graph notes)
    reasons = {}
    for acct in feats.index:
        r = list(rule_reasons.get(acct, []))
        if out.at[acct, "ml_score"] >= 0.6:
            drivers = ", ".join(ml_explain.get(acct, [])) or "multivariate outlier"
            r.append(f"ML ensemble flagged unusual behaviour (drivers: {drivers}).")
        if graph_sig.reindex(feats.index).fillna(0.0).at[acct, "in_cycle"] == 1:
            r.append("Account sits on a circular money flow (round-tripping / layering).")
        reasons[acct] = r
    out["reason_codes"] = pd.Series(reasons)

    # attach a few headline features for the case view
    for c in ["throughput", "net_flow", "total_txns", "cash_ratio", "pass_through_ratio"]:
        out[c] = feats[c]

    out = out.sort_values("risk_score", ascending=False)
    out.index.name = "account"
    return out
