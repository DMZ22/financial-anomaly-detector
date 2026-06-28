"""Run the FinSentry pipeline and export web/data.json for the static dashboard.

The Streamlit app is the live, interactive tool; this exports a real snapshot of
the engine's output so the same insights can be served as a fast, always-on
static site (e.g. on Vercel).
"""
from __future__ import annotations

import json
import os

import numpy as np
import networkx as nx

from finsentry.datagen import generate_dataset
from finsentry.pipeline import run_pipeline, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web")
os.makedirs(OUT, exist_ok=True)


def main():
    txns, truth = generate_dataset(n_accounts=600, n_days=45, fraud_rate=0.06, seed=7)
    res = run_pipeline(txns)
    results = res.results
    alerts = res.alerts

    m_med = evaluate(results, truth, alert_band=("MEDIUM", "HIGH", "CRITICAL"))
    m_high = evaluate(results, truth, alert_band=("HIGH", "CRITICAL"))

    n_acc = int(txns[["src", "dst"]].stack().nunique())
    band_counts = results["risk_band"].value_counts().reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ).fillna(0).astype(int).to_dict()

    # risk-score histogram
    counts, edges = np.histogram(results["risk_score"], bins=40, range=(0, 100))
    hist = {"x": [round((edges[i] + edges[i + 1]) / 2, 1) for i in range(len(counts))],
            "y": counts.astype(int).tolist()}

    # typology counts across the alert queue
    typ = {}
    for lst in alerts["rule_codes"]:
        for c in (lst if isinstance(lst, list) else []):
            typ[c] = typ.get(c, 0) + 1

    # daily flagged vs normal value
    flagged = set(alerts.index)
    tx = txns.copy()
    tx["flag"] = tx["src"].isin(flagged) | tx["dst"].isin(flagged)
    tx["day"] = tx["timestamp"].dt.strftime("%Y-%m-%d")
    daily = (tx.groupby(["day", "flag"])["amount"].sum().unstack(fill_value=0))
    daily.columns = ["normal", "flagged"] if False in daily.columns and True in daily.columns else daily.columns
    daily_obj = {
        "day": list(daily.index),
        "flagged": [float(daily.loc[d].get(True, 0)) for d in daily.index],
        "normal": [float(daily.loc[d].get(False, 0)) for d in daily.index],
    }

    # ML model score distributions
    ml = res.ml_scores
    ml_dist = {k: [round(float(v), 4) for v in ml[k]] for k in ["iso", "lof", "maha"]}

    # ROC + confusion matrix vs ground truth
    roc = {"fpr": [0, 1], "tpr": [0, 1]}
    try:
        from sklearn.metrics import roc_curve
        y = results.index.isin(set(truth)).astype(int)
        fpr, tpr, _ = roc_curve(y, results["risk_score"])
        roc = {"fpr": [round(float(x), 4) for x in fpr], "tpr": [round(float(x), 4) for x in tpr]}
    except Exception:
        pass
    cm = [[m_med["tp"], m_med["fp"]], [m_med["fn"], m_med["tn"]]]

    # alert rows
    feats = res.features
    alert_rows = []
    txns_by_acct = {}
    for acct, row in alerts.iterrows():
        alert_rows.append({
            "id": acct,
            "score": round(float(row["risk_score"]), 1),
            "band": row["risk_band"],
            "rule": round(float(row["rule_score"]), 3),
            "ml": round(float(row["ml_score"]), 3),
            "graph": round(float(row["graph_score"]), 3),
            "reasons": row["reason_codes"] if isinstance(row["reason_codes"], list) else [],
            "codes": row["rule_codes"] if isinstance(row["rule_codes"], list) else [],
            "throughput": round(float(row["throughput"]), 0),
            "net_flow": round(float(row["net_flow"]), 0),
            "txns": int(feats.loc[acct, "total_txns"]),
            "cash_ratio": round(float(feats.loc[acct, "cash_ratio"]), 2),
            "pass_through": round(float(feats.loc[acct, "pass_through_ratio"]), 2),
            "cross_border": round(float(feats.loc[acct, "cross_border_ratio"]), 2),
            "distinct_in": int(feats.loc[acct, "distinct_in_counterparties"]),
        })
        ta = txns[(txns["src"] == acct) | (txns["dst"] == acct)].sort_values("timestamp")
        txns_by_acct[acct] = [{
            "t": r["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "src": r["src"], "dst": r["dst"], "amt": round(float(r["amount"]), 2),
            "ch": r["channel"], "dir": "out" if r["src"] == acct else "in",
            "sc": r["src_country"], "dc": r["dst_country"],
        } for _, r in ta.iterrows()]

    # money-flow network: top-15 risk accounts + neighbours (mirror the app)
    focus = list(results.head(15).index)
    nbrs = set(focus)
    for _, r in txns[txns["src"].isin(focus) | txns["dst"].isin(focus)].iterrows():
        nbrs.add(r["src"]); nbrs.add(r["dst"])
    nbrs = list(nbrs)[:120]
    sub = txns[txns["src"].isin(nbrs) & txns["dst"].isin(nbrs)]
    g = nx.DiGraph()
    for _, r in sub.groupby(["src", "dst"])["amount"].sum().reset_index().iterrows():
        g.add_edge(r["src"], r["dst"], amount=float(r["amount"]))
    network = {"nodes": [], "edges": []}
    if g.number_of_nodes():
        pos = nx.spring_layout(g, seed=7, k=0.6)
        for n in g.nodes():
            x, y = pos[n]
            score = float(results.loc[n, "risk_score"]) if n in results.index else 0.0
            band = results.loc[n, "risk_band"] if n in results.index else "LOW"
            network["nodes"].append({"id": n, "x": round(float(x), 4), "y": round(float(y), 4),
                                     "score": round(score, 1), "band": band})
        amax = max((d["amount"] for *_e, d in g.edges(data=True)), default=1)
        for u, v, d in g.edges(data=True):
            network["edges"].append({
                "x0": round(float(pos[u][0]), 4), "y0": round(float(pos[u][1]), 4),
                "x1": round(float(pos[v][0]), 4), "y1": round(float(pos[v][1]), 4),
                "w": round(0.5 + 3 * d["amount"] / amax, 2),
            })

    data = {
        "meta": {
            "n_txns": int(len(txns)), "n_accounts": n_acc,
            "n_alerts": int(len(alerts)),
            "n_high": int((results["risk_band"].isin(["HIGH", "CRITICAL"])).sum()),
            "flagged_value": round(float(alerts["throughput"].sum()), 0),
            "metrics": {
                "precision": round(m_med["precision"], 3), "recall": round(m_med["recall"], 3),
                "f1": round(m_med["f1"], 3), "roc_auc": round(m_med["roc_auc"], 3),
                "high_precision": round(m_high["precision"], 3), "high_recall": round(m_high["recall"], 3),
                "n_truth": m_med["n_truth"],
            },
        },
        "bands": band_counts, "hist": hist, "typologies": typ, "daily": daily_obj,
        "ml_dist": ml_dist, "roc": roc, "cm": cm,
        "alerts": alert_rows, "txns_by_acct": txns_by_acct,
        "network": network, "cycles": [list(c) for c in res.cycles[:25]],
    }

    path = os.path.join(OUT, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)")
    print(f"alerts={len(alert_rows)} nodes={len(network['nodes'])} edges={len(network['edges'])} cycles={len(data['cycles'])}")


if __name__ == "__main__":
    main()
