"""Champion vs challengers on both datasets.

    python experiments/run_benchmark.py            (~25 min on 2 CPU cores)

1. Cross-validation (German: 5x repeated 5-fold, Taiwan: 5-fold), every model refitted and tuned
   inside each training fold. Differences vs the champion are tested with the Nadeau-Bengio
   corrected resampled t-test, because CV folds overlap and the naive t-test is over-confident.
2. A fixed 80/20 stratified hold-out on each dataset: test metrics, bootstrap CIs, DeLong tests,
   and the saved predictions / fitted models that every later experiment uses.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from creditrisk import metrics as M  # noqa: E402
from creditrisk.data import LOADERS  # noqa: E402
from creditrisk.models import MODELS, build  # noqa: E402

RES = os.path.join(ROOT, "results")
ART = os.path.join(ROOT, "models")
SVC = os.path.join(ROOT, "service", "model")
for d in (RES, ART, SVC):
    os.makedirs(d, exist_ok=True)
FAST = os.environ.get("FAST") == "1"          # quick smoke run for CI


def cross_validate(name, X, y, meta):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if (FAST or name == "taiwan") else 5, random_state=0)
    rows = []
    for k, (tr, te) in enumerate(cv.split(X, y)):
        for m in MODELS:
            if FAST and m in ("stack", "random_forest"):
                continue
            t = time.time()
            est = build(m, name, meta["categorical"], meta["numeric"], seed=k, gbm_iter=4 if FAST else 12)
            est.fit(X.iloc[tr], y[tr])
            p = est.predict_proba(X.iloc[te])[:, 1]
            rows.append({"fold": k, "model": m, "auc": roc_auc_score(y[te], p), "ks": M.ks(y[te], p),
                         "brier": float(np.mean((p - y[te]) ** 2)), "seconds": time.time() - t})
        print(f"  [{name}] fold {k}: " + ", ".join(f"{r['model']} {r['auc']:.3f}" for r in rows if r["fold"] == k), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"cv_folds_{name}.csv"), index=False)
    n_te = len(y) // 5
    champ = df[df.model == "scorecard"].set_index("fold").auc
    out = []
    for m, g in df.groupby("model", sort=False):
        d = g.set_index("fold").auc - champ
        t, p = M.corrected_resampled_ttest(d.values, len(y) - n_te, n_te) if m != "scorecard" else (np.nan, np.nan)
        out.append({"model": m, "auc_mean": g.auc.mean(), "auc_sd": g.auc.std(), "ks_mean": g.ks.mean(),
                    "brier_mean": g.brier.mean(), "delta_auc_vs_champion": d.mean(), "corrected_t": t,
                    "p_value": p, "folds": len(g), "fit_seconds": g.seconds.mean()})
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(RES, f"cv_summary_{name}.csv"), index=False)
    print(out.round(4).to_string(index=False))


def holdout(name, X, y, meta):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    preds = pd.DataFrame({"row": te, "y": yte, "exposure": meta["exposure"][te], "sex": meta["protected"][te]})
    train_preds = pd.DataFrame({"row": tr, "y": ytr})
    rows, fitted = [], {}
    for m in MODELS:
        if FAST and m in ("stack", "random_forest"):
            continue
        t = time.time()
        est = build(m, name, meta["categorical"], meta["numeric"], seed=0, gbm_iter=4 if FAST else 12).fit(Xtr, ytr)
        fit_s = time.time() - t
        preds[m] = est.predict_proba(Xte)[:, 1]
        train_preds[m] = est.predict_proba(Xtr)[:, 1]
        fitted[m] = est
        r = {"model": m, **M.summary(yte, preds[m].values), "fit_seconds": fit_s}
        r["auc_lo"], r["auc_hi"] = M.bootstrap_ci(yte, preds[m].values, B=300 if FAST else 1000)
        r["auc_se_delong"] = M.auc_se(yte, preds[m].values)
        if m != "scorecard":
            _, _, se, p = M.delong(yte, preds[m].values, preds["scorecard"].values)
            r["delta_auc_vs_champion"], r["delta_se"], r["delong_p"] = r["auc"] - rows[0]["auc"], se, p
        r["train_auc"] = roc_auc_score(ytr, train_preds[m])
        rows.append(r)
        print(f"  [{name} holdout] {m}: AUC {r['auc']:.4f} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]  KS {r['ks']:.3f}  "
              f"ECE {r['ece']:.4f}  ({fit_s:.1f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"holdout_{name}.csv"), index=False)
    preds.to_csv(os.path.join(RES, f"holdout_predictions_{name}.csv"), index=False)
    train_preds.to_csv(os.path.join(RES, f"train_predictions_{name}.csv"), index=False)
    sc = fitted["scorecard"]
    sc.table().to_csv(os.path.join(RES, f"scorecard_table_{name}.csv"), index=False)
    sc.woe_.iv_.rename("information_value").to_csv(os.path.join(RES, f"information_value_{name}.csv"))
    json.dump({"features": sc.features_, "C": sc.C_, "dropped_for_sign": sc.dropped_for_sign_,
               "factor": sc.factor_, "offset": sc.offset_}, open(os.path.join(RES, f"scorecard_meta_{name}.json"), "w"), indent=1)
    for m, est in fitted.items():
        joblib.dump(est.best_estimator_ if hasattr(est, "best_estimator_") else est, os.path.join(ART, f"{name}_{m}.joblib"))
    if name == "taiwan":                              # the deployed pair: champion JSON + challenger in shadow
        sc.to_json(os.path.join(SVC, "champion_scorecard.json"))
        joblib.dump(fitted["gbm_monotone"].best_estimator_, os.path.join(SVC, "challenger_gbm_monotone.joblib"))
        json.dump({"columns": list(X.columns), "categorical": meta["categorical"],
                   "example": Xte.iloc[0].to_dict()}, open(os.path.join(SVC, "schema.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    which = sys.argv[1:] or ["german", "taiwan"]
    for name in which:
        X, y, meta = LOADERS[name]()
        print(f"== {meta['name']}: {X.shape[0]} applicants, {X.shape[1]} characteristics, bad rate {y.mean():.3f}")
        holdout(name, X, y, meta)
        cross_validate(name, X, y, meta)
