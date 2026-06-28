"""Account-level feature engineering.

Aggregates a transaction ledger into one row per account with behavioural
features used by both the ML ensemble and the rule engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datagen import REPORTING_THRESHOLD, HIGH_RISK_COUNTRIES

FEATURE_COLS = [
    "total_txns",
    "in_count",
    "out_count",
    "in_amount",
    "out_amount",
    "net_flow",
    "throughput",
    "mean_amount",
    "max_amount",
    "std_amount",
    "distinct_in_counterparties",
    "distinct_out_counterparties",
    "cash_ratio",
    "crypto_ratio",
    "cross_border_ratio",
    "high_risk_country_ratio",
    "round_amount_ratio",
    "sub_threshold_ratio",
    "night_ratio",
    "pass_through_ratio",
    "txns_per_active_day",
]


def _entity_country(txns: pd.DataFrame) -> dict:
    country = {}
    for _, r in txns.iterrows():
        country.setdefault(r["src"], r["src_country"])
        country.setdefault(r["dst"], r["dst_country"])
    return country


def account_features(txns: pd.DataFrame) -> pd.DataFrame:
    """Build the per-account feature matrix."""
    txns = txns.copy()
    txns["hour"] = txns["timestamp"].dt.hour
    txns["day"] = txns["timestamp"].dt.normalize()
    txns["is_cash"] = txns["channel"].eq("CASH")
    txns["is_crypto"] = txns["channel"].eq("CRYPTO")
    txns["is_round"] = (txns["amount"] % 1000 == 0) | (txns["amount"] % 500 == 0)
    txns["is_sub_threshold"] = txns["amount"].between(REPORTING_THRESHOLD * 0.8, REPORTING_THRESHOLD)
    txns["is_night"] = (txns["hour"] < 6) | (txns["hour"] >= 22)
    txns["cross_border"] = txns["src_country"] != txns["dst_country"]
    txns["touches_high_risk"] = txns["src_country"].isin(HIGH_RISK_COUNTRIES) | txns["dst_country"].isin(
        HIGH_RISK_COUNTRIES
    )

    accounts = pd.Index(pd.concat([txns["src"], txns["dst"]]).unique(), name="account")
    feats = pd.DataFrame(index=accounts)

    out = txns.groupby("src")
    inc = txns.groupby("dst")

    feats["in_count"] = inc.size()
    feats["out_count"] = out.size()
    feats["in_amount"] = inc["amount"].sum()
    feats["out_amount"] = out["amount"].sum()
    feats = feats.fillna(0.0)
    feats["total_txns"] = feats["in_count"] + feats["out_count"]
    feats["net_flow"] = feats["in_amount"] - feats["out_amount"]
    feats["throughput"] = feats["in_amount"] + feats["out_amount"]

    # amount stats across all txns the account participates in
    part = pd.concat([txns.assign(acct=txns["src"]), txns.assign(acct=txns["dst"])])
    g = part.groupby("acct")
    feats["mean_amount"] = g["amount"].mean()
    feats["max_amount"] = g["amount"].max()
    feats["std_amount"] = g["amount"].std().fillna(0.0)
    feats["distinct_in_counterparties"] = inc["src"].nunique()
    feats["distinct_out_counterparties"] = out["dst"].nunique()
    feats["cash_ratio"] = g["is_cash"].mean()
    feats["crypto_ratio"] = g["is_crypto"].mean()
    feats["cross_border_ratio"] = g["cross_border"].mean()
    feats["high_risk_country_ratio"] = g["touches_high_risk"].mean()
    feats["round_amount_ratio"] = g["is_round"].mean()
    feats["sub_threshold_ratio"] = g["is_sub_threshold"].mean()
    feats["night_ratio"] = g["is_night"].mean()

    active_days = g["day"].nunique()
    feats["txns_per_active_day"] = (feats["total_txns"] / active_days).replace([np.inf, -np.inf], 0)

    # pass-through: how quickly inflow leaves again (layering signal)
    feats["pass_through_ratio"] = _pass_through(txns, accounts)

    feats = feats.fillna(0.0)
    # guarantee column order / presence
    for c in FEATURE_COLS:
        if c not in feats.columns:
            feats[c] = 0.0
    return feats[FEATURE_COLS]


def _pass_through(txns: pd.DataFrame, accounts: pd.Index, window_hours: float = 24.0) -> pd.Series:
    """Fraction of an account's inflow that is sent out again within `window_hours`."""
    result = {}
    for acct in accounts:
        ins = txns[txns["dst"] == acct][["timestamp", "amount"]].sort_values("timestamp")
        outs = txns[txns["src"] == acct][["timestamp", "amount"]].sort_values("timestamp")
        if ins.empty or outs.empty:
            result[acct] = 0.0
            continue
        matched = 0.0
        out_idx = 0
        out_times = outs["timestamp"].to_numpy()
        out_amts = outs["amount"].to_numpy()
        used = np.zeros(len(outs), dtype=bool)
        for t_in, _amt_in in zip(ins["timestamp"].to_numpy(), ins["amount"].to_numpy()):
            for j in range(out_idx, len(outs)):
                if used[j]:
                    continue
                dt = (out_times[j] - t_in) / np.timedelta64(1, "h")
                if 0 <= dt <= window_hours:
                    matched += out_amts[j]
                    used[j] = True
                    break
        in_total = ins["amount"].sum()
        result[acct] = float(min(matched / in_total, 1.0)) if in_total else 0.0
    return pd.Series(result)
