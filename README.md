# 🛡️ FinSentry — Financial Transaction Anomaly & AML Detection

An end-to-end **transaction-monitoring system** that flags suspicious financial
activity the way a bank's AML team does: it scores every account for money-laundering
risk by combining an **unsupervised machine-learning ensemble**, an **AML typology
rule engine**, and **money-flow network analysis**, then ranks the results into an
explainable **alert queue** with case-level drill-down.

**▶️ Live demo:** _deploy on Streamlit Community Cloud (see below)_
**Stack:** Python · scikit-learn · NetworkX · SciPy · pandas · Plotly · Streamlit

---

## Why it exists

Real transaction-monitoring platforms (Actimize, SAS AML, etc.) don't rely on a single
model — they blend deterministic rules, statistical/ML anomaly detection, and link
analysis, and every alert must be **explainable** to an investigator. FinSentry is a
compact, working implementation of that exact pattern on synthetic data.

## How it works

```
ledger ─► feature engineering ─► ┌─ AML rule engine ───┐
                                 ├─ ML anomaly ensemble ─┤─► composite risk score ─► alert queue ─► case file
                                 └─ money-flow graph ────┘        (0–100, banded)        (ranked)     (export)
```

**1. AML typology rule engine** — deterministic detectors for the classic laundering
patterns, each producing a weighted, human-readable reason code:
`STRUCTURING` (deposits just under the reporting threshold), `SMURFING`,
`RAPID_MOVEMENT` (pass-through / layering), `CIRCULAR_FLOW` (round-tripping),
`HIGH_VELOCITY`, `CASH_INTENSIVE`, `CROSS_BORDER_HIGH_RISK`, `ROUND_AMOUNTS`, `ODD_HOUR`.

**2. ML anomaly ensemble** — `IsolationForest` + `LocalOutlierFactor` + robust
`Mahalanobis` distance over standardized behavioural features, averaged into a single
anomaly score, with a per-account explanation of the top deviating features.

**3. Money-flow graph analysis** — a directed value graph (NetworkX) detects circular
flows (round-tripping), fan-in/fan-out hubs (collector / mule patterns), and
betweenness (pass-through intermediaries) over *material* (high-value) flows.

**4. Composite risk scoring** — the three signals are blended into a `0–100` score,
banded `LOW / MEDIUM / HIGH / CRITICAL`, with reason codes attached for every alert.

## Detection performance

Measured against injected ground-truth labels on the synthetic generator
(600 accounts, ~8,400 transactions, 6% suspicious):

| Alert band | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| MEDIUM+ (full queue) | **0.93** | **0.90** | **0.91** | **0.96** |
| HIGH / CRITICAL (high-confidence) | **1.00** | 0.41 | 0.59 | 0.96 |

Reproduce with `python evaluate.py`.

## The dashboard

- **Overview** — risk distribution, accounts by band, typologies triggered, flagged value over time
- **Alert queue** — sortable, filterable, exportable to CSV
- **Case detail** — risk gauge, reason codes, key features, transaction timeline, SAR/STR-style JSON export
- **Network** — interactive money-flow graph with alerts highlighted + detected circular flows
- **Model insights** — per-model anomaly distributions and the most anomalous accounts
- **Evaluation** — precision/recall, confusion matrix, ROC curve vs ground truth

Upload your own ledger (CSV) or generate synthetic data with adjustable size, time span,
and suspicious-rate; tune the rule/ML/graph weights live.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py          # dashboard at http://localhost:8501
python evaluate.py            # CLI performance report
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. Go to **share.streamlit.io → New app**, pick the repo, set the main file to `app.py`.
3. Deploy — you get a public `https://<name>.streamlit.app` URL.

A `Dockerfile` is included for container deployment (Render / Fly / Cloud Run).

## Data & ethics

All data is **synthetic** — generated locally with reproducible seeds. No real customer
or transaction information is used. The thresholds and typologies are illustrative of
real AML controls, not a production compliance system.

## Project layout

```
finsentry/
  datagen.py    synthetic ledger + injected laundering typologies (+ ground truth)
  features.py   account-level behavioural feature engineering
  rules.py      AML typology rule engine (weighted reason codes)
  models.py     IsolationForest + LOF + Mahalanobis ensemble
  graph.py      money-flow network: cycles, hubs, betweenness
  scoring.py    composite risk score + banding + reason codes
  pipeline.py   orchestration + evaluation metrics
app.py          Streamlit dashboard
evaluate.py     CLI performance report
```
