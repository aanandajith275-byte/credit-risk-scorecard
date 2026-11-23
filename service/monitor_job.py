"""Scheduled monitoring job (e.g. Cloud Scheduler -> Cloud Run job, or EventBridge -> Lambda), daily.

    python service/monitor_job.py --log /tmp/prediction_log.jsonl --reference results/train_predictions_taiwan.csv

Reads the API's prediction log, compares the period's score distribution with the development sample
(PSI) and writes a JSON report. If an outcomes file is given (defaults known months later), it also checks
calibration and AUC. Exits with code 2 on an ALERT, so the scheduler can page someone.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from creditrisk.monitoring import performance_report, population_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--reference", required=True, help="CSV with a 'scorecard' column of development-sample PDs")
    ap.add_argument("--outcomes", help="CSV with decision_id, y (1 = defaulted)")
    ap.add_argument("--out", default="monitoring_report.json")
    a = ap.parse_args()
    log = pd.read_json(a.log, lines=True)
    ref = pd.read_csv(a.reference)["scorecard"].to_numpy()
    report = {"n_decisions": len(log), "approval_rate": float((log["decision"] == "approve").mean()),
              **population_report(ref, log["pd"].to_numpy())}
    if "shadow" in log:
        sh = log["shadow"].dropna().map(lambda d: d["pd"])
        report["shadow_mean_pd"] = float(sh.mean())
        report["champion_challenger_pd_corr"] = float(log.loc[sh.index, "pd"].corr(sh))
    if a.outcomes:
        o = pd.read_csv(a.outcomes).merge(log, on="decision_id")
        report["performance"] = performance_report(o["y"], o["pd"])
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))
    alert = report["population_status"] == "ALERT" or report.get("performance", {}).get("performance_status") == "ALERT"
    sys.exit(2 if alert else 0)


if __name__ == "__main__":
    main()
