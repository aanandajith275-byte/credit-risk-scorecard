"""How long must a challenger run in production before its advantage can be confirmed?

    python experiments/ab_power.py               (seconds; needs run_benchmark.py results)

Uses the DeLong variance of (AUC_challenger - AUC_champion) measured on the Taiwan hold-out set and
scales it with sample size (variance ~ 1/n).

  shadow mode  both models score every applicant; only the champion's decision is used.
               The comparison is PAIRED: the two models' errors on the same applicant are correlated,
               which cancels most of the noise.
  A/B split    each applicant is scored by one model only (50/50). UNPAIRED: the variances add.

A further, unavoidable problem: outcomes are only observed for APPROVED applicants (the bank never
learns whether a rejected applicant would have repaid). The comparison is also re-run on the
champion-approved 80% only, which is what a live evaluation would actually see.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from creditrisk.metrics import auc_se, delong  # noqa: E402

RES = os.path.join(ROOT, "results")
P = pd.read_csv(os.path.join(RES, "holdout_predictions_taiwan.csv"))
H = pd.read_csv(os.path.join(RES, "holdout_taiwan.csv"))
chal = "gbm_monotone"            # the challenger deployed in shadow mode (fixed in advance, not picked on test data)
y = P["y"].to_numpy()
a_ch, a_c, se_pair, p = delong(y, P[chal], P["scorecard"])
n = len(y)
se_unpaired = np.sqrt(auc_se(y, P[chal]) ** 2 + auc_se(y, P["scorecard"]) ** 2)
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
delta = a_ch - a_c


def n_needed(se_at_n, delta, n_ref, split=1.0):
    # var(n) = se_at_n^2 * n_ref / n ; for an A/B split each arm only sees split * n applicants
    return n_ref * (z * se_at_n / delta) ** 2 / split


approved = P["scorecard"].to_numpy() <= np.quantile(P["scorecard"], 0.80)
b_ch, b_c, se_sel, p_sel = delong(y[approved], P[chal][approved], P["scorecard"][approved])

out = {"challenger": chal, "n_test": n, "auc_challenger": a_ch, "auc_champion": a_c, "delta_auc": delta,
       "se_paired": se_pair, "se_unpaired": se_unpaired, "delong_p": p,
       "labelled_needed_shadow": n_needed(se_pair, delta, n),
       "labelled_needed_ab": n_needed(se_unpaired, delta, n, split=0.5) * 1.0,
       "approved_only": {"auc_challenger": b_ch, "auc_champion": b_c, "delta_auc": b_ch - b_c, "se_paired": se_sel,
                         "delong_p": p_sel, "n": int(approved.sum()),
                         "labelled_needed_shadow": n_needed(se_sel, b_ch - b_c, int(approved.sum()))}}
out["efficiency_ratio_ab_over_shadow"] = out["labelled_needed_ab"] / out["labelled_needed_shadow"]
rows = []
for d in [0.005, 0.01, 0.02, 0.03, 0.05]:
    for vol in [2000, 10000, 50000]:
        sh = n_needed(se_pair, d, n)
        ab = n_needed(se_unpaired, d, n, 0.5)
        rows.append({"delta_auc": d, "applications_per_month": vol,
                     "months_shadow": sh / (vol * 0.8), "months_ab": ab / (vol * 0.8)})
pd.DataFrame(rows).to_csv(os.path.join(RES, "ab_power.csv"), index=False)
json.dump(out, open(os.path.join(RES, "ab_power.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
