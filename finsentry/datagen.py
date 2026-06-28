"""Synthetic transaction generator.

Produces a realistic population of accounts and transactions with a configurable
fraction of accounts engaged in known money-laundering typologies. Ground-truth
labels are attached so detection quality can be measured (precision/recall).

The generator is deterministic for a given seed so results are reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Cash-transaction reporting threshold the launderers try to stay under
# (analogous to the CTR threshold used in transaction monitoring).
REPORTING_THRESHOLD = 10_000.0

HIGH_RISK_COUNTRIES = ["KY", "PA", "CY", "AE", "RU", "NG"]
NORMAL_COUNTRIES = ["IN", "US", "GB", "DE", "SG", "AU", "CA"]
CHANNELS = ["WIRE", "ACH", "CARD", "CASH", "CRYPTO"]

TYPOLOGIES = [
    "structuring",
    "smurfing",
    "rapid_movement",
    "round_tripping",
    "mule_fanout",
]


def _acct_id(i: int) -> str:
    return f"ACC{i:05d}"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_dataset(
    n_accounts: int = 600,
    n_days: int = 45,
    fraud_rate: float = 0.06,
    seed: int = 7,
):
    """Generate a transaction dataset.

    Returns
    -------
    txns : pd.DataFrame
        Columns: txn_id, timestamp, src, dst, amount, channel,
        src_country, dst_country, typology (ground-truth, 'normal' for legit).
    truth : dict[str, str]
        Maps each suspicious account id to its injected typology.
    """
    rng = _rng(seed)
    accounts = [_acct_id(i) for i in range(n_accounts)]
    country = {a: rng.choice(NORMAL_COUNTRIES) for a in accounts}
    start = pd.Timestamp("2025-01-06 00:00:00")

    rows: list[dict] = []
    txn_counter = [0]

    def add(ts, src, dst, amount, channel, typ):
        txn_counter[0] += 1
        rows.append(
            dict(
                txn_id=f"T{txn_counter[0]:07d}",
                timestamp=ts,
                src=src,
                dst=dst,
                amount=round(float(amount), 2),
                channel=channel,
                src_country=country.get(src, rng.choice(NORMAL_COUNTRIES)),
                dst_country=country.get(dst, rng.choice(NORMAL_COUNTRIES)),
                typology=typ,
            )
        )

    def rand_ts(day_jitter=None):
        day = rng.integers(0, n_days) if day_jitter is None else day_jitter
        # business hours skew for legit activity
        hour = int(np.clip(rng.normal(13, 4), 0, 23))
        minute = int(rng.integers(0, 60))
        return start + pd.Timedelta(days=int(day), hours=hour, minutes=minute)

    # ---- 1. Background / legitimate activity for every account ----
    for a in accounts:
        n_tx = int(np.clip(rng.normal(14, 6), 2, 40))
        for _ in range(n_tx):
            other = rng.choice(accounts)
            if other == a:
                continue
            amount = float(np.round(rng.lognormal(mean=6.2, sigma=1.0), 2))
            amount = min(amount, 9_500)  # legit kept modest
            channel = rng.choice(CHANNELS, p=[0.34, 0.32, 0.20, 0.10, 0.04])
            direction_out = rng.random() < 0.5
            src, dst = (a, other) if direction_out else (other, a)
            add(rand_ts(), src, dst, amount, channel, "normal")

    # ---- 2. Suspicious accounts with injected typologies ----
    n_fraud = max(1, int(n_accounts * fraud_rate))
    suspicious = list(rng.choice(accounts, size=n_fraud, replace=False))
    truth: dict[str, str] = {}

    for a in suspicious:
        typ = rng.choice(TYPOLOGIES)
        truth[a] = typ
        base_day = int(rng.integers(0, max(1, n_days - 7)))

        if typ == "structuring":
            # several cash deposits just below the reporting threshold within days
            n = int(rng.integers(4, 9))
            for k in range(n):
                amt = REPORTING_THRESHOLD - rng.uniform(50, 1200)
                ts = start + pd.Timedelta(days=base_day + k % 5, hours=int(rng.integers(8, 20)))
                counter = rng.choice(accounts)
                add(ts, counter, a, amt, "CASH", typ)

        elif typ == "smurfing":
            # many small incoming transfers from many distinct accounts (collector)
            n = int(rng.integers(12, 25))
            for _ in range(n):
                amt = rng.uniform(500, 2_500)
                src = rng.choice(accounts)
                ts = start + pd.Timedelta(days=base_day + int(rng.integers(0, 6)), hours=int(rng.integers(0, 24)))
                add(ts, src, a, amt, rng.choice(["ACH", "WIRE"]), typ)

        elif typ == "rapid_movement":
            # large funds in, then almost immediately out to several accounts (pass-through)
            inflow = rng.uniform(40_000, 120_000)
            t0 = start + pd.Timedelta(days=base_day, hours=int(rng.integers(0, 23)))
            add(t0, rng.choice(accounts), a, inflow, "WIRE", typ)
            remaining = inflow
            for _ in range(int(rng.integers(3, 7))):
                amt = remaining * rng.uniform(0.1, 0.35)
                remaining -= amt
                dst = rng.choice(accounts)
                ts = t0 + pd.Timedelta(hours=float(rng.uniform(0.5, 8)))
                add(ts, a, dst, amt, rng.choice(["WIRE", "CRYPTO"]), typ)

        elif typ == "round_tripping":
            # circular flow A -> B -> C -> A in round amounts
            b, c = rng.choice(accounts, size=2, replace=False)
            amt = float(rng.choice([25_000, 50_000, 75_000, 100_000]))
            t0 = start + pd.Timedelta(days=base_day, hours=10)
            add(t0, a, b, amt, "WIRE", typ)
            add(t0 + pd.Timedelta(hours=6), b, c, amt * 0.99, "WIRE", typ)
            add(t0 + pd.Timedelta(hours=12), c, a, amt * 0.98, "WIRE", typ)
            truth[b] = truth.get(b, typ)
            truth[c] = truth.get(c, typ)

        elif typ == "mule_fanout":
            # one account distributes to many mules in cross-border wires
            country[a] = rng.choice(HIGH_RISK_COUNTRIES)
            n = int(rng.integers(10, 20))
            for _ in range(n):
                amt = rng.uniform(3_000, 9_000)
                dst = rng.choice(accounts)
                country[dst] = rng.choice(HIGH_RISK_COUNTRIES) if rng.random() < 0.5 else country[dst]
                ts = start + pd.Timedelta(days=base_day + int(rng.integers(0, 4)), hours=int(rng.integers(0, 24)))
                add(ts, a, dst, amt, rng.choice(["WIRE", "CRYPTO"]), typ)

    txns = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    return txns, truth


def load_or_generate(uploaded: pd.DataFrame | None = None, **kwargs):
    """Return a usable transaction frame, either from an upload or freshly generated."""
    if uploaded is not None and len(uploaded):
        df = uploaded.copy()
        if "typology" not in df.columns:
            df["typology"] = "unknown"
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, {}
    return generate_dataset(**kwargs)
