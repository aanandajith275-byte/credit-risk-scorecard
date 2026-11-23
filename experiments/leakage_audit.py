"""How three common validation mistakes inflate a credit model's reported AUC.

    python experiments/leakage_audit.py          (~4 min)

For each mistake we compare the AUC the analyst would REPORT with the AUC the model actually achieves
on data that played no part in building or choosing it.

  A. post-decision information:  a field recorded after the outcome (simulated "sent to collections")
  B. preprocessing before the split: WoE bins and WoE values computed on all rows, then cross-validated
  C. choosing the model on the test set: best of N candidates by test AUC (the winner's curse)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from creditrisk.binning import WoETransformer  # noqa: E402
from creditrisk.data import LOADERS  # noqa: E402
from creditrisk.scorecard import Scorecard  # noqa: E402

RES = os.path.join(ROOT, "results")
rows = []

for name in ["german", "taiwan"]:
    X, y, meta = LOADERS[name]()
    cat = meta["categorical"]
    dev_idx, fresh_idx = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=7)
    Xd, yd, Xf, yf = X.iloc[dev_idx], y[dev_idx], X.iloc[fresh_idx], y[fresh_idx]

    # ---------------------------------------------------------------- A. post-decision field
    rng = np.random.default_rng(1)
    flag = np.where(y == 1, rng.random(len(y)) < 0.6, rng.random(len(y)) < 0.03).astype(int)
    XL = X.assign(collections_flag=flag)
    honest = Scorecard(categorical=tuple(cat)).fit(Xd, yd)
    leaky = Scorecard(categorical=tuple(cat)).fit(XL.iloc[dev_idx], yd)
    # reported: evaluated on a hold-out that also has the field; real: at decision time the field does not
    # exist yet, so the deployed model sees "not flagged" for every applicant
    rep = roc_auc_score(yf, leaky.predict_proba(XL.iloc[fresh_idx])[:, 1])
    real = roc_auc_score(yf, leaky.predict_proba(XL.iloc[fresh_idx].assign(collections_flag=0))[:, 1])
    rows.append({"dataset": name, "mistake": "A. post-decision field", "reported_auc": rep, "real_auc": real,
                 "honest_model_auc": roc_auc_score(yf, honest.predict_proba(Xf)[:, 1])})

    # ---------------------------------------------------------------- B. WoE fitted before the split
    def woe_cv(fit_outside, max_bins=40, min_share=0.005):
        cv = StratifiedKFold(5, shuffle=True, random_state=3)
        aucs = []
        if fit_outside:
            W_all = WoETransformer(cat, max_bins, min_share, monotone=False).fit_transform(Xd, yd)
        for tr, te in cv.split(Xd, yd):
            if fit_outside:
                Wtr, Wte = W_all.iloc[tr], W_all.iloc[te]
            else:
                wt = WoETransformer(cat, max_bins, min_share, monotone=False).fit(Xd.iloc[tr], yd[tr])
                Wtr, Wte = wt.transform(Xd.iloc[tr]), wt.transform(Xd.iloc[te])
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(Wtr, yd[tr])
            aucs.append(roc_auc_score(yd[te], lr.predict_proba(Wte)[:, 1]))
        return float(np.mean(aucs))

    wt = WoETransformer(cat, 40, 0.005, monotone=False).fit(Xd, yd)
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(wt.transform(Xd), yd)
    fresh_auc = roc_auc_score(yf, lr.predict_proba(wt.transform(Xf))[:, 1])
    rows.append({"dataset": name, "mistake": "B. WoE fitted before CV split", "reported_auc": woe_cv(True),
                 "real_auc": fresh_auc, "honest_model_auc": woe_cv(False)})

    # ---------------------------------------------------------------- C. winner's curse
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan).fit(Xd[cat])

    def mat(D):
        return np.column_stack([enc.transform(D[cat]), D.drop(columns=cat).to_numpy(float)])

    tr_i, pool_i = train_test_split(np.arange(len(yd)), test_size=0.4, stratify=yd, random_state=5)
    cand = []
    crng = np.random.default_rng(11)
    for _ in range(30):
        hgb = HistGradientBoostingClassifier(learning_rate=float(10 ** crng.uniform(-1.7, -0.5)),
                                             max_leaf_nodes=int(crng.integers(3, 64)),
                                             min_samples_leaf=int(crng.integers(5, 200)),
                                             max_iter=int(crng.integers(50, 400)), early_stopping=False,
                                             categorical_features=list(range(len(cat))), random_state=0)
        hgb.fit(mat(Xd.iloc[tr_i]), yd[tr_i])
        cand.append((hgb.predict_proba(mat(Xd.iloc[pool_i]))[:, 1], hgb.predict_proba(mat(Xf))[:, 1]))
    fresh_aucs = np.array([roc_auc_score(yf, f) for _, f in cand])
    for n_test in [100, 200, 500, 1000, 2000]:
        if n_test > len(pool_i):
            continue
        reported, real = [], []
        for rep_i in range(300):
            s = crng.choice(len(pool_i), n_test, replace=False)
            if yd[pool_i][s].min() == yd[pool_i][s].max():
                continue
            test_aucs = [roc_auc_score(yd[pool_i][s], p[s]) for p, _ in cand]
            best = int(np.argmax(test_aucs))
            reported.append(test_aucs[best])
            real.append(fresh_aucs[best])
        rows.append({"dataset": name, "mistake": f"C. best of 30 on a {n_test}-row test set",
                     "reported_auc": float(np.mean(reported)), "real_auc": float(np.mean(real)),
                     "honest_model_auc": float(fresh_aucs.max()), "n_test": n_test})
    print(pd.DataFrame([r for r in rows if r["dataset"] == name]).round(4).to_string(index=False), flush=True)

df = pd.DataFrame(rows)
df["optimism"] = df.reported_auc - df.real_auc
df.to_csv(os.path.join(RES, "leakage_audit.csv"), index=False)
