"""Turn AUC differences into money and decisions.

    python experiments/business_value.py        (needs run_benchmark.py results; ~1 min)

Taiwan (credit cards):
  * strategy curves: approval rate vs bad rate among approved applicants, for every model
  * credit loss at a fixed 80% approval rate: EAD = September balance, LGD = 75% (assumed, typical for
    unsecured cards), with a paired bootstrap CI for champion - challenger
  * fairness: approval rate by sex (sex is NOT a model input) and the adverse impact ratio
German (instalment loans):
  * the dataset's 5:1 cost matrix: Bayes-optimal cut-off p* = 1/(1+5) is only optimal if the PDs
    are calibrated, so the cost achieved at p* measures calibration in money
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from creditrisk.data import load_taiwan  # noqa: E402

RES = os.path.join(ROOT, "results")
LGD = 0.75
APPROVAL = 0.80
TWD_PER_USD = 32.0          # approximate; only used to state the break-even in dollars
out = {}

# ---------------------------------------------------------------- Taiwan
P = pd.read_csv(os.path.join(RES, "holdout_predictions_taiwan.csv"))
X, _, _ = load_taiwan()
P["ead"] = X["BILL_AMT1"].clip(lower=0).to_numpy()[P["row"]]
models = [c for c in P.columns if c not in ("row", "y", "exposure", "sex", "ead")]

curve = []
for m in models:
    order = np.argsort(P[m].to_numpy())                    # lowest PD approved first
    yb = P["y"].to_numpy()[order]
    for a in np.linspace(0.3, 1.0, 71):
        k = int(round(a * len(P)))
        curve.append({"model": m, "approval_rate": a, "bad_rate_approved": yb[:k].mean()})
pd.DataFrame(curve).to_csv(os.path.join(RES, "strategy_curves_taiwan.csv"), index=False)


def loss_at(df, m, approval=APPROVAL):
    k = int(round(approval * len(df)))
    approved = np.argsort(df[m].to_numpy())[:k]
    a = df.iloc[approved]
    return float((a["y"] * a["ead"] * LGD).sum()), int(a["y"].sum())


rows = []
for m in models:
    loss, bads = loss_at(P, m)
    rows.append({"model": m, "bads_approved": bads, "bad_rate_approved": bads / round(APPROVAL * len(P)),
                 "loss_ntd": loss, "loss_per_applicant_ntd": loss / len(P)})
loss_df = pd.DataFrame(rows)
loss_df.to_csv(os.path.join(RES, "credit_loss_taiwan.csv"), index=False)

rng = np.random.default_rng(0)
champ = loss_df.set_index("model").loc["scorecard"]
out["taiwan"] = {"approval_rate": APPROVAL, "lgd": LGD, "n_test": len(P),
                 "avg_loss_per_approved_bad_ntd": float(champ.loss_ntd / champ.bads_approved), "challengers": {}}
boots = [P.iloc[rng.integers(0, len(P), len(P))] for _ in range(1000)]
for chal in ["gbm", "gbm_monotone", "stack"]:
    c = loss_df.set_index("model").loc[chal]
    d_loss, d_bads = [], []
    for s in boots:
        lc, bc = loss_at(s, "scorecard")
        lx, bx = loss_at(s, chal)
        d_loss.append((lc - lx) / len(P))
        d_bads.append((bc - bx) / len(P) * 1e4)
    saving = (champ.loss_ntd - c.loss_ntd) / len(P)
    out["taiwan"]["challengers"][chal] = {
        "bads_avoided_per_10k": (champ.bads_approved - c.bads_approved) / len(P) * 1e4,
        "bads_avoided_ci95": [float(np.quantile(d_bads, 0.025)), float(np.quantile(d_bads, 0.975))],
        "loss_saving_per_applicant_ntd": saving,
        "loss_saving_ci95_ntd": [float(np.quantile(d_loss, 0.025)), float(np.quantile(d_loss, 0.975))],
        "loss_saving_per_applicant_usd": saving / TWD_PER_USD}

fair = []
for m in models:
    k = int(round(APPROVAL * len(P)))
    approved = np.zeros(len(P), bool)
    approved[np.argsort(P[m].to_numpy())[:k]] = True
    rates = pd.Series(approved).groupby(P["sex"].to_numpy()).mean()
    bad_by = P.assign(a=approved)[approved].groupby("sex")["y"].mean()
    fair.append({"model": m, "approval_female": rates["female"], "approval_male": rates["male"],
                 "adverse_impact_ratio": rates.min() / rates.max(),
                 "bad_rate_approved_female": bad_by["female"], "bad_rate_approved_male": bad_by["male"]})
pd.DataFrame(fair).to_csv(os.path.join(RES, "fairness_taiwan.csv"), index=False)

# ---------------------------------------------------------------- German: cost matrix
G = pd.read_csv(os.path.join(RES, "holdout_predictions_german.csv"))
gm = [c for c in G.columns if c not in ("row", "y", "exposure", "sex")]
p_star = 1 / (1 + 5)
cost_rows = []
for m in gm:
    p, y = G[m].to_numpy(), G["y"].to_numpy()
    reject = p >= p_star
    cost = (5 * ((~reject) & (y == 1)).sum() + 1 * (reject & (y == 0)).sum()) / len(y)
    # the best cut-off in hindsight (on the test set itself: an optimistic reference)
    best_c = min((5 * ((p < t) & (y == 1)).sum() + ((p >= t) & (y == 0)).sum()) / len(y) for t in np.unique(p))
    cost_rows.append({"model": m, "cost_at_bayes_cutoff": cost, "best_possible_cost": best_c,
                      "reject_rate": reject.mean(), "mean_pd": p.mean()})
pd.DataFrame(cost_rows).to_csv(os.path.join(RES, "cost_matrix_german.csv"), index=False)
out["german"] = {"bayes_cutoff": p_star, "cost_reject_all": float((G['y'] == 0).mean()),
                 "cost_accept_all": float(5 * G['y'].mean())}
json.dump(out, open(os.path.join(RES, "business_value.json"), "w"), indent=1)
print(loss_df.round(1).to_string(index=False))
print(pd.DataFrame(cost_rows).round(3).to_string(index=False))
print(pd.DataFrame(fair).round(3).to_string(index=False))
print(json.dumps(out, indent=1))
