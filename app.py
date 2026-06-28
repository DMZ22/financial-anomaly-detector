"""FinSentry — Streamlit dashboard.

An interactive transaction-monitoring console: generate or upload a transaction
ledger, run the detection pipeline, triage the alert queue, drill into a case,
explore the money-flow network, and measure detection quality.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finsentry.datagen import generate_dataset, load_or_generate, REPORTING_THRESHOLD
from finsentry.pipeline import run_pipeline, evaluate

st.set_page_config(page_title="FinSentry — AML Anomaly Detection", page_icon="🛡️", layout="wide")

BAND_COLORS = {"CRITICAL": "#EF4444", "HIGH": "#F59E0B", "MEDIUM": "#FACC15", "LOW": "#3B82F6"}


# --------------------------------------------------------------------------- #
# Cached pipeline                                                             #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Running detection pipeline…")
def run(source, n_accounts, n_days, fraud_rate, seed, w_rules, w_ml, w_graph, file_bytes):
    if source == "Upload CSV" and file_bytes:
        df = pd.read_csv(io.BytesIO(file_bytes))
        txns, truth = load_or_generate(df)
    else:
        txns, truth = generate_dataset(
            n_accounts=n_accounts, n_days=n_days, fraud_rate=fraud_rate, seed=seed
        )
    total = w_rules + w_ml + w_graph or 1
    weights = {"rules": w_rules / total, "ml": w_ml / total, "graph": w_graph / total}
    res = run_pipeline(txns, weights=weights, seed=seed)
    return txns, truth, res


# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ FinSentry")
st.sidebar.caption("Financial transaction anomaly & AML detection")

source = st.sidebar.radio("Data source", ["Synthetic generator", "Upload CSV"], index=0)
file_bytes = None
n_accounts, n_days, fraud_rate, seed = 600, 45, 0.06, 7
if source == "Synthetic generator":
    n_accounts = st.sidebar.slider("Accounts", 200, 1500, 600, 100)
    n_days = st.sidebar.slider("Days of activity", 14, 90, 45, 1)
    fraud_rate = st.sidebar.slider("Suspicious account rate", 0.01, 0.15, 0.06, 0.01)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 7)
else:
    up = st.sidebar.file_uploader(
        "Transactions CSV", type=["csv"],
        help="Columns: txn_id, timestamp, src, dst, amount, channel, src_country, dst_country",
    )
    if up is not None:
        file_bytes = up.getvalue()
    st.sidebar.caption("No file? Switch back to the synthetic generator.")

st.sidebar.markdown("**Detection weights**")
w_rules = st.sidebar.slider("AML rules", 0.0, 1.0, 0.5, 0.05)
w_ml = st.sidebar.slider("ML ensemble", 0.0, 1.0, 0.3, 0.05)
w_graph = st.sidebar.slider("Network graph", 0.0, 1.0, 0.2, 0.05)

txns, truth, res = run(source, n_accounts, n_days, fraud_rate, seed, w_rules, w_ml, w_graph, file_bytes)
results = res.results
alerts = res.alerts

# --------------------------------------------------------------------------- #
# Header + KPIs                                                               #
# --------------------------------------------------------------------------- #
st.title("Financial Transaction Anomaly & AML Detection")
st.caption(
    "Unsupervised ML ensemble · AML typology rule engine · money-flow graph analysis → "
    "explainable risk score per account."
)

n_acc = pd.concat([txns["src"], txns["dst"]]).nunique()
high = results[results["risk_band"].isin(["HIGH", "CRITICAL"])]
flagged_value = float(alerts["throughput"].sum())

k = st.columns(5)
k[0].metric("Transactions", f"{len(txns):,}")
k[1].metric("Accounts", f"{n_acc:,}")
k[2].metric("Alerts (MEDIUM+)", f"{len(alerts):,}")
k[3].metric("High-risk (HIGH+)", f"{len(high):,}")
k[4].metric("Flagged throughput", f"${flagged_value/1e6:,.1f}M")

if truth:
    m = evaluate(results, truth, alert_band=("MEDIUM", "HIGH", "CRITICAL"))
    b = st.columns(4)
    b[0].metric("Recall (MEDIUM+)", f"{m['recall']*100:.0f}%")
    b[1].metric("Precision (MEDIUM+)", f"{m['precision']*100:.0f}%")
    b[2].metric("F1", f"{m['f1']:.2f}")
    b[3].metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m["roc_auc"] else "—")

tabs = st.tabs(
    ["📊 Overview", "🚨 Alert queue", "🔍 Case detail", "🕸️ Network", "🧠 Model insights", "✅ Evaluation"]
)

# --------------------------------------------------------------------------- #
# Overview                                                                    #
# --------------------------------------------------------------------------- #
with tabs[0]:
    c1, c2 = st.columns(2)
    fig = px.histogram(results, x="risk_score", nbins=40, title="Risk-score distribution")
    fig.add_vline(x=40, line_dash="dash", line_color="#FACC15")
    fig.add_vline(x=60, line_dash="dash", line_color="#F59E0B")
    fig.add_vline(x=80, line_dash="dash", line_color="#EF4444")
    c1.plotly_chart(fig, width="stretch")

    band_counts = (
        results["risk_band"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0)
    )
    fig2 = px.bar(
        band_counts, title="Accounts by risk band",
        color=band_counts.index, color_discrete_map=BAND_COLORS,
    )
    fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="accounts")
    c2.plotly_chart(fig2, width="stretch")

    # typology / rule-code frequency across alerts
    codes = [c for lst in alerts["rule_codes"] for c in (lst if isinstance(lst, list) else [])]
    if codes:
        cc = pd.Series(codes).value_counts()
        fig3 = px.bar(cc, title="AML typologies triggered (alert queue)")
        fig3.update_layout(showlegend=False, xaxis_title="", yaxis_title="alerts")
        st.plotly_chart(fig3, width="stretch")

    # flagged value over time
    flagged_accts = set(alerts.index)
    tx = txns.copy()
    tx["flagged"] = tx["src"].isin(flagged_accts) | tx["dst"].isin(flagged_accts)
    daily = (
        tx.assign(day=tx["timestamp"].dt.date)
        .groupby(["day", "flagged"])["amount"]
        .sum()
        .reset_index()
    )
    daily["flagged"] = daily["flagged"].map({True: "flagged", False: "normal"})
    fig4 = px.area(daily, x="day", y="amount", color="flagged", title="Daily value: flagged vs normal")
    st.plotly_chart(fig4, width="stretch")


# --------------------------------------------------------------------------- #
# Alert queue                                                                 #
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Alert queue — accounts at MEDIUM risk or above")
    band_filter = st.multiselect(
        "Filter bands", ["CRITICAL", "HIGH", "MEDIUM"], default=["CRITICAL", "HIGH", "MEDIUM"]
    )
    view = alerts[alerts["risk_band"].isin(band_filter)].copy()
    view["risk_score"] = view["risk_score"].round(1)
    view["top_reason"] = view["reason_codes"].apply(
        lambda r: r[0] if isinstance(r, list) and r else ""
    )
    show = view[["risk_score", "risk_band", "n_rules", "throughput", "net_flow", "top_reason"]].copy()
    show["throughput"] = show["throughput"].round(0)
    show["net_flow"] = show["net_flow"].round(0)
    st.dataframe(
        show.rename_axis("account").reset_index(),
        width="stretch", height=460, hide_index=True,
    )
    st.download_button(
        "⬇️ Download alert queue (CSV)",
        view.drop(columns=["reason_codes", "rule_codes"]).to_csv().encode(),
        file_name="finsentry_alerts.csv", mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Case detail                                                                 #
# --------------------------------------------------------------------------- #
with tabs[2]:
    if len(alerts) == 0:
        st.info("No alerts to investigate at the current settings.")
    else:
        acct = st.selectbox("Select an account to investigate", alerts.index.tolist())
        row = results.loc[acct]
        left, right = st.columns([1, 1.3])

        with left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(row["risk_score"]),
                    title={"text": f"{acct} — {row['risk_band']}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": BAND_COLORS.get(row["risk_band"], "#3B82F6")},
                        "steps": [
                            {"range": [0, 40], "color": "#1f2937"},
                            {"range": [40, 60], "color": "#3f3f1f"},
                            {"range": [60, 80], "color": "#4a2f17"},
                            {"range": [80, 100], "color": "#4a1717"},
                        ],
                    },
                )
            )
            gauge.update_layout(height=260, margin=dict(t=40, b=10))
            st.plotly_chart(gauge, width="stretch")
            st.markdown("**Score composition**")
            comp = pd.Series(
                {"AML rules": row["rule_score"], "ML ensemble": row["ml_score"], "Network": row["graph_score"]}
            )
            st.plotly_chart(
                px.bar(comp, range_y=[0, 1]).update_layout(
                    showlegend=False, height=220, xaxis_title="", yaxis_title="signal (0-1)"
                ),
                width="stretch",
            )

        with right:
            st.markdown("**Reason codes**")
            reasons = row["reason_codes"] if isinstance(row["reason_codes"], list) else []
            if reasons:
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.caption("No rule narrative; flagged primarily by the ML ensemble.")

            st.markdown("**Key features**")
            feat = res.features.loc[acct]
            st.dataframe(
                feat[
                    ["total_txns", "throughput", "net_flow", "cash_ratio",
                     "pass_through_ratio", "distinct_in_counterparties", "cross_border_ratio"]
                ].round(2).rename("value").to_frame(),
                width="stretch",
            )

        st.markdown("**Transaction timeline**")
        tx_a = txns[(txns["src"] == acct) | (txns["dst"] == acct)].copy()
        tx_a["direction"] = np.where(tx_a["src"] == acct, "outgoing", "incoming")
        figt = px.scatter(
            tx_a, x="timestamp", y="amount", color="direction", symbol="channel",
            hover_data=["src", "dst", "src_country", "dst_country"], title=f"Activity for {acct}",
        )
        figt.add_hline(y=REPORTING_THRESHOLD, line_dash="dot", line_color="#EF4444",
                       annotation_text="reporting threshold")
        st.plotly_chart(figt, width="stretch")

        # SAR/STR-style case export
        case = {
            "account": acct,
            "risk_score": round(float(row["risk_score"]), 1),
            "risk_band": row["risk_band"],
            "reason_codes": reasons,
            "throughput": round(float(row["throughput"]), 2),
            "net_flow": round(float(row["net_flow"]), 2),
            "transactions": len(tx_a),
        }
        st.download_button(
            "⬇️ Export case report (JSON)",
            json.dumps(case, indent=2).encode(),
            file_name=f"case_{acct}.json", mime="application/json",
        )


# --------------------------------------------------------------------------- #
# Network                                                                     #
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Money-flow network")
    st.caption("Top-risk accounts and their counterparties. Red nodes are alerts; "
               "thicker edges carry more value.")
    top_n = st.slider("Focus on top-N risk accounts", 5, 40, 15)
    focus = list(results.head(top_n).index)
    neighbours = set(focus)
    for _, r in txns[txns["src"].isin(focus) | txns["dst"].isin(focus)].iterrows():
        neighbours.add(r["src"]); neighbours.add(r["dst"])
    neighbours = list(neighbours)[:120]

    sub = txns[txns["src"].isin(neighbours) & txns["dst"].isin(neighbours)]
    g = nx.DiGraph()
    for _, r in sub.groupby(["src", "dst"])["amount"].sum().reset_index().iterrows():
        g.add_edge(r["src"], r["dst"], amount=float(r["amount"]))

    if g.number_of_nodes() == 0:
        st.info("Not enough linked activity to draw a network at these settings.")
    else:
        pos = nx.spring_layout(g, seed=7, k=0.6)
        edge_x, edge_y = [], []
        amax = max((d["amount"] for *_e, d in g.edges(data=True)), default=1)
        for u, v, d in g.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(width=0.6, color="#475569"), hoverinfo="none")
        node_x, node_y, color, size, text = [], [], [], [], []
        for n in g.nodes():
            x, y = pos[n]; node_x.append(x); node_y.append(y)
            band = results.loc[n, "risk_band"] if n in results.index else "LOW"
            score = float(results.loc[n, "risk_score"]) if n in results.index else 0.0
            color.append(BAND_COLORS.get(band, "#3B82F6"))
            size.append(10 + score / 6)
            text.append(f"{n}<br>risk {score:.0f} ({band})")
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers", hoverinfo="text", text=text,
            marker=dict(color=color, size=size, line=dict(width=0.5, color="#0E1117")),
        )
        fign = go.Figure([edge_trace, node_trace])
        fign.update_layout(showlegend=False, height=560,
                           xaxis=dict(visible=False), yaxis=dict(visible=False),
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fign, width="stretch")

    if res.cycles:
        st.markdown(f"**Circular money flows detected: {len(res.cycles)}**")
        st.dataframe(
            pd.DataFrame({"cycle": [" → ".join(c + [c[0]]) for c in res.cycles[:25]]}),
            width="stretch", hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Model insights                                                              #
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Ensemble model insights")
    ms = res.ml_scores.copy()
    long = ms[["iso", "lof", "maha"]].melt(var_name="model", value_name="score")
    st.plotly_chart(
        px.violin(long, x="model", y="score", box=True, title="Per-model anomaly-score distributions"),
        width="stretch",
    )
    st.caption(
        "iso = Isolation Forest · lof = Local Outlier Factor · maha = robust Mahalanobis distance. "
        "The ensemble averages all three."
    )
    st.markdown("**Most anomalous accounts by ML ensemble**")
    top_ml = res.ml_scores.sort_values("ml_score", ascending=False).head(12)[["iso", "lof", "maha", "ml_score"]]
    st.dataframe(top_ml.round(3), width="stretch")


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
with tabs[5]:
    if not truth:
        st.info("Evaluation requires ground-truth labels — available with the synthetic generator.")
    else:
        st.subheader("Detection quality vs injected ground truth")
        colA, colB = st.columns(2)
        m_med = evaluate(results, truth, alert_band=("MEDIUM", "HIGH", "CRITICAL"))
        m_high = evaluate(results, truth, alert_band=("HIGH", "CRITICAL"))
        with colA:
            st.markdown("**Full alert queue (MEDIUM+)**")
            st.json({k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in m_med.items() if k in ["precision", "recall", "f1", "roc_auc"]})
            cm = pd.DataFrame(
                [[m_med["tp"], m_med["fp"]], [m_med["fn"], m_med["tn"]]],
                index=["truth: suspicious", "truth: clean"],
                columns=["pred: alert", "pred: clear"],
            )
            st.plotly_chart(
                px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix (MEDIUM+)"),
                width="stretch",
            )
        with colB:
            st.markdown("**High-confidence tier (HIGH/CRITICAL)**")
            st.json({k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in m_high.items() if k in ["precision", "recall", "f1", "roc_auc"]})
            try:
                from sklearn.metrics import roc_curve
                y = results.index.isin(set(truth)).astype(int)
                fpr, tpr, _ = roc_curve(y, results["risk_score"])
                figr = go.Figure()
                figr.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC",
                                          line=dict(color="#2DD4BF", width=3)))
                figr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          line=dict(dash="dash", color="#64748B"), name="chance"))
                figr.update_layout(title=f"ROC curve (AUC = {m_med['roc_auc']:.3f})",
                                   xaxis_title="false positive rate", yaxis_title="true positive rate",
                                   height=420)
                st.plotly_chart(figr, width="stretch")
            except Exception as e:
                st.caption(f"ROC unavailable: {e}")

st.markdown("---")
st.caption("FinSentry · built with Python, scikit-learn, NetworkX, SciPy & Streamlit · "
           "synthetic data only — no real customer information.")
