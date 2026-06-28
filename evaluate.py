"""Run the FinSentry pipeline on synthetic data and print detection metrics.

Usage:  python evaluate.py
"""
from finsentry.datagen import generate_dataset
from finsentry.pipeline import run_pipeline, evaluate


def main():
    txns, truth = generate_dataset(n_accounts=600, n_days=45, fraud_rate=0.06, seed=7)
    print(f"Generated {len(txns):,} transactions across "
          f"{txns[['src','dst']].stack().nunique():,} accounts.")
    print(f"Injected {len(truth)} suspicious accounts.\n")

    res = run_pipeline(txns)

    for band_label, bands in [("MEDIUM+ (full alert queue)", ("MEDIUM", "HIGH", "CRITICAL")),
                              ("HIGH/CRITICAL (high-confidence)", ("HIGH", "CRITICAL"))]:
        m = evaluate(res.results, truth, alert_band=bands)
        print(f"=== Detection performance — alert band = {band_label} ===")
        for k in ["precision", "recall", "f1", "roc_auc"]:
            v = m.get(k)
            print(f"  {k:>10}: {v:.3f}" if isinstance(v, float) else f"  {k:>10}: {v}")
        print(f"  alerts: {m['n_alerts']}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})  of {m['n_truth']} suspicious\n")

    print("=== Top 10 alerts ===")
    cols = ["risk_score", "risk_band", "n_rules", "throughput"]
    top = res.results.head(10)[cols].copy()
    top["risk_score"] = top["risk_score"].round(1)
    top["throughput"] = top["throughput"].round(0)
    print(top.to_string())
    print(f"\nCircular flows detected: {len(res.cycles)}")


if __name__ == "__main__":
    main()
