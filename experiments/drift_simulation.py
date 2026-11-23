"""Twelve simulated months of production scoring, monitored the way a scheduled cloud job would.

    python experiments/drift_simulation.py       (~1 min; needs run_benchmark.py results)

Each month, 3,000 applicants are drawn from the Taiwan hold-out set (never used in training):
  covariate shift  the mix moves towards customers with high card utilisation (sampling weights
                   exp(g * z_utilisation), g rising 0 -> 1). The link between features and default is
                   unchanged, so a good model should stay right.
  concept drift    the mix is unchanged, but a downturn makes some good customers default (each good
                   becomes bad with probability rising 0 -> 10%). The model's inputs look identical,
                   but its PDs are now too low.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from creditrisk.data import load_taiwan  # noqa: E402
from creditrisk.monitoring import performance_report, population_report  # noqa: E402

RES = os.path.join(ROOT, "results")
X, _, _ = load_taiwan()
P = pd.read_csv(os.path.join(RES, "holdout_predictions_taiwan.csv"))
T = pd.read_csv(os.path.join(RES, "train_predictions_taiwan.csv"))
Xte = X.iloc[P["row"].to_numpy()].reset_index(drop=True)
Xtr = X.iloc[T["row"].to_numpy()].reset_index(drop=True)
CHARS = ["PAY_1", "LIMIT_BAL", "utilisation_avg6", "PAY_AMT1", "AGE", "EDUCATION", "max_delinquency_6"]
MODEL = "scorecard"
rng = np.random.default_rng(0)
z = (Xte["utilisation_avg6"] - Xte["utilisation_avg6"].mean()) / Xte["utilisation_avg6"].std()

rows = []
for scenario in ["covariate shift", "concept drift"]:
    for month in range(13):
        s = month / 12
        if scenario == "covariate shift":
            w = np.exp(1.0 * s * z.clip(-3, 3).to_numpy())
            idx = rng.choice(len(P), 3000, replace=True, p=w / w.sum())
            y = P["y"].to_numpy()[idx]
        else:
            idx = rng.choice(len(P), 3000, replace=True)
            y = P["y"].to_numpy()[idx].copy()
            flip = (y == 0) & (rng.random(len(y)) < 0.10 * s)
            y[flip] = 1
        pdv = P[MODEL].to_numpy()[idx]
        pop = population_report(T[MODEL].to_numpy(), pdv, Xtr, Xte.iloc[idx], CHARS)
        perf = performance_report(y, pdv)
        rows.append({"scenario": scenario, "month": month, "score_psi": pop["score_psi"],
                     "worst_characteristic": pop["worst_characteristic"], "worst_csi": pop["worst_csi"],
                     "population_status": pop["population_status"], **perf})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "drift_simulation.csv"), index=False)
print(df.round(3).to_string(index=False))
