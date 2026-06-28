"""Money-flow network analysis.

Builds a directed graph of value flow between accounts and surfaces structural
laundering signals: circular flows (round-tripping / layering), fan-in and
fan-out hubs (collector / mule patterns), and betweenness centrality
(pass-through intermediaries).

Cycle detection is run only over *material* flows (high-value edges) and with a
bounded cycle length, so it stays fast and focuses on the large, layered
movements that matter for AML rather than incidental small-value loops.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(txns: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    agg = txns.groupby(["src", "dst"])["amount"].agg(["sum", "count"]).reset_index()
    for _, r in agg.iterrows():
        g.add_edge(r["src"], r["dst"], amount=float(r["sum"]), count=int(r["count"]))
    return g


def _material_subgraph(g: nx.DiGraph, min_amount: float) -> nx.DiGraph:
    keep = [(u, v, d) for u, v, d in g.edges(data=True) if d["amount"] >= min_amount]
    h = nx.DiGraph()
    h.add_edges_from(keep)
    return h


def graph_signals(txns: pd.DataFrame, max_cycle_len: int = 6, amount_percentile: float = 0.85):
    """Return (signals_df indexed by account, list_of_cycles)."""
    g = build_graph(txns)
    accounts = list(g.nodes())
    sig = pd.DataFrame(index=pd.Index(accounts, name="account"))

    sig["fan_in"] = pd.Series(dict(g.in_degree()))
    sig["fan_out"] = pd.Series(dict(g.out_degree()))

    # Focus structural analysis on material (large) flows.
    edge_amounts = np.array([d["amount"] for _, _, d in g.edges(data=True)]) if g.number_of_edges() else np.array([0.0])
    min_amount = max(float(np.quantile(edge_amounts, amount_percentile)), 15_000.0)
    h = _material_subgraph(g, min_amount)

    # betweenness on the (smaller) material graph; sample sources if still large
    if h.number_of_nodes() > 2:
        k = min(h.number_of_nodes(), 150)
        try:
            bc = nx.betweenness_centrality(h, k=k, weight="amount", normalized=True, seed=7)
        except Exception:
            bc = {n: 0.0 for n in h.nodes()}
    else:
        bc = {}
    sig["betweenness"] = pd.Series(bc)

    # circular flows over material edges only, bounded length
    in_cycle = {a: 0 for a in accounts}
    cycles: list[list[str]] = []
    if h.number_of_edges():
        try:
            for cyc in nx.simple_cycles(h, length_bound=max_cycle_len):
                if len(cyc) >= 2:
                    cycles.append(cyc)
                    for a in cyc:
                        in_cycle[a] = 1
                if len(cycles) >= 200:
                    break
        except Exception:
            pass
    sig["in_cycle"] = pd.Series(in_cycle)

    sig = sig.fillna(0.0)
    bmax = sig["betweenness"].max() or 1.0
    fan = sig[["fan_in", "fan_out"]].max(axis=1)
    fmax = fan.max() or 1.0
    sig["graph_score"] = (
        0.5 * sig["in_cycle"]
        + 0.3 * (sig["betweenness"] / bmax)
        + 0.2 * (fan / fmax)
    ).clip(0, 1)
    return sig, cycles
