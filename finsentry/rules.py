"""AML typology rule engine.

Each rule inspects the account feature matrix (and, where needed, the raw
ledger) and returns a fired/not-fired flag, a severity weight, and a
human-readable reason code — the same "reason for alert" a transaction
monitoring analyst documents on a case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .datagen import REPORTING_THRESHOLD


@dataclass
class RuleHit:
    code: str
    weight: float
    reason: str


def _structuring(feats: pd.Series, txns_a: pd.DataFrame) -> RuleHit | None:
    near = txns_a[txns_a["amount"].between(REPORTING_THRESHOLD * 0.8, REPORTING_THRESHOLD)]
    cash_near = near[near["channel"] == "CASH"]
    if len(cash_near) >= 3:
        return RuleHit(
            "STRUCTURING",
            1.0,
            f"{len(cash_near)} cash transactions just below the "
            f"{REPORTING_THRESHOLD:,.0f} reporting threshold (possible structuring).",
        )
    return None


def _smurfing(feats: pd.Series, txns_a: pd.DataFrame) -> RuleHit | None:
    if feats["distinct_in_counterparties"] >= 10 and feats["in_count"] >= 12:
        small = txns_a[(txns_a["dst"] == feats.name) & (txns_a["amount"] < 3000)]
        if len(small) >= 10:
            return RuleHit(
                "SMURFING",
                0.9,
                f"{int(feats['distinct_in_counterparties'])} distinct senders making many "
                f"small deposits into one account (smurfing / collector pattern).",
            )
    return None


def _rapid_movement(feats: pd.Series, txns_a: pd.DataFrame) -> RuleHit | None:
    if feats["pass_through_ratio"] >= 0.7 and feats["throughput"] >= 50_000:
        return RuleHit(
            "RAPID_MOVEMENT",
            0.95,
            f"{feats['pass_through_ratio']*100:.0f}% of inflow forwarded within 24h on "
            f"{feats['throughput']:,.0f} throughput (pass-through / layering).",
        )
    return None


def _high_velocity(feats: pd.Series, pop: pd.DataFrame) -> RuleHit | None:
    thr = pop["txns_per_active_day"].mean() + 2.5 * pop["txns_per_active_day"].std()
    if feats["txns_per_active_day"] > max(thr, 6):
        return RuleHit(
            "HIGH_VELOCITY",
            0.6,
            f"{feats['txns_per_active_day']:.1f} txns/active-day, far above the "
            f"population norm (velocity spike).",
        )
    return None


def _round_amounts(feats: pd.Series) -> RuleHit | None:
    if feats["round_amount_ratio"] >= 0.6 and feats["total_txns"] >= 4:
        return RuleHit(
            "ROUND_AMOUNTS",
            0.45,
            f"{feats['round_amount_ratio']*100:.0f}% of activity in round amounts "
            f"(uncommon for genuine commerce).",
        )
    return None


def _cash_intensive(feats: pd.Series) -> RuleHit | None:
    if feats["cash_ratio"] >= 0.6 and feats["throughput"] >= 30_000:
        return RuleHit(
            "CASH_INTENSIVE",
            0.5,
            f"{feats['cash_ratio']*100:.0f}% cash on {feats['throughput']:,.0f} throughput.",
        )
    return None


def _cross_border_high_risk(feats: pd.Series) -> RuleHit | None:
    if feats["high_risk_country_ratio"] >= 0.4 and feats["cross_border_ratio"] >= 0.4:
        return RuleHit(
            "CROSS_BORDER_HIGH_RISK",
            0.75,
            f"{feats['high_risk_country_ratio']*100:.0f}% of activity touches a "
            f"high-risk jurisdiction.",
        )
    return None


def _odd_hour(feats: pd.Series) -> RuleHit | None:
    if feats["night_ratio"] >= 0.5 and feats["total_txns"] >= 5:
        return RuleHit(
            "ODD_HOUR",
            0.3,
            f"{feats['night_ratio']*100:.0f}% of transactions occur outside business hours.",
        )
    return None


def _circular_flow(acct: str, cycle_accounts: set[str]) -> RuleHit | None:
    if acct in cycle_accounts:
        return RuleHit(
            "CIRCULAR_FLOW",
            0.8,
            "Account participates in a circular money flow between a small set of "
            "parties (round-tripping / layering).",
        )
    return None


def apply_rules(
    feats: pd.DataFrame,
    txns: pd.DataFrame,
    cycle_accounts: set[str] | None = None,
) -> dict[str, list[RuleHit]]:
    """Run every rule for every account. Returns account -> list[RuleHit]."""
    cycle_accounts = cycle_accounts or set()
    hits: dict[str, list[RuleHit]] = {}
    # pre-index ledger by participating account for the rules that need raw txns
    by_acct: dict[str, pd.DataFrame] = {}
    for acct in feats.index:
        mask = (txns["src"] == acct) | (txns["dst"] == acct)
        by_acct[acct] = txns[mask]

    for acct, row in feats.iterrows():
        row = row.copy()
        row.name = acct
        txns_a = by_acct[acct]
        acct_hits: list[RuleHit] = []
        for fn in (_structuring, _smurfing, _rapid_movement):
            h = fn(row, txns_a)
            if h:
                acct_hits.append(h)
        h = _high_velocity(row, feats)
        if h:
            acct_hits.append(h)
        for fn in (_round_amounts, _cash_intensive, _cross_border_high_risk, _odd_hour):
            h = fn(row)
            if h:
                acct_hits.append(h)
        h = _circular_flow(acct, cycle_accounts)
        if h:
            acct_hits.append(h)
        if acct_hits:
            hits[acct] = acct_hits
    return hits
