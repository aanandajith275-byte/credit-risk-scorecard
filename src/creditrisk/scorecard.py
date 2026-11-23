"""The champion: a points-based logistic-regression scorecard on WoE-binned characteristics.

Pipeline (all fitted on training data only):
  1. monotone WoE binning of every characteristic             (binning.py)
  2. keep characteristics with information value >= min_iv
  3. L2 logistic regression on the WoE values, C chosen by stratified CV
  4. drop any characteristic whose coefficient has the wrong sign (a symptom of collinearity:
     with positive WoE = safer, every coefficient on P(bad) must be negative), refit
  5. scale to points:  score = offset + factor * ln(odds of good),  factor = PDO / ln 2
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from .binning import WoETransformer


class Scorecard(BaseEstimator, ClassifierMixin):
    def __init__(self, categorical=(), min_iv=0.02, max_bins=10, min_share=0.05, pdo=20.0,
                 base_score=600.0, base_odds=50.0, Cs=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0), random_state=0):
        self.categorical, self.min_iv, self.max_bins, self.min_share = categorical, min_iv, max_bins, min_share
        self.pdo, self.base_score, self.base_odds, self.Cs, self.random_state = pdo, base_score, base_odds, Cs, random_state

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y):
        y = np.asarray(y, int)
        self.classes_ = np.array([0, 1])
        self.woe_ = WoETransformer(self.categorical, self.max_bins, self.min_share).fit(X, y)
        W = self.woe_.transform(X)
        feats = [c for c, iv in self.woe_.iv_.items() if iv >= self.min_iv]
        cv = StratifiedKFold(5, shuffle=True, random_state=self.random_state)
        self.dropped_for_sign_ = []
        while True:
            lr = LogisticRegressionCV(Cs=list(self.Cs), cv=cv, scoring="neg_log_loss", max_iter=5000)
            lr.fit(W[feats], y)
            wrong = [f for f, b in zip(feats, lr.coef_[0]) if b > 0]
            if not wrong:
                break
            worst = min(wrong, key=lambda f: self.woe_.iv_[f])           # drop the weakest offender first
            feats.remove(worst)
            self.dropped_for_sign_.append(worst)
        self.features_, self.C_ = feats, float(lr.C_[0])
        self.coef_ = lr.coef_[0].copy()
        self.intercept_ = float(lr.intercept_[0])
        self._build_points()
        return self

    def _build_points(self):
        self.factor_ = self.pdo / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds)
        n = len(self.features_)
        self.points_ = {}
        for f, b in zip(self.features_, self.coef_):
            woe = self.woe_.binners_[f].woe_
            self.points_[f] = -self.factor_ * b * woe + (self.offset_ - self.factor_ * self.intercept_) / n

    # ------------------------------------------------------------------ scoring
    def score(self, X, y=None):                       # noqa: D401  (credit score, not sklearn's accuracy)
        total = np.zeros(len(X))
        for f in self.features_:
            idx = self.woe_.binners_[f].bin_index(X[f].to_numpy())
            pts = self.points_[f]
            # unseen category -> WoE 0 -> the neutral point value
            neutral = (self.offset_ - self.factor_ * self.intercept_) / len(self.features_)
            total += np.where(idx >= 0, pts[np.clip(idx, 0, None)], neutral)
        return total

    def predict_proba(self, X):
        s = self.score(X)
        p_bad = 1.0 / (1.0 + np.exp((s - self.offset_) / self.factor_))
        return np.column_stack([1 - p_bad, p_bad])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def reason_codes(self, X, k=4):
        """Adverse-action style reasons: the k characteristics where the applicant lost the most points
        relative to the best attribute of that characteristic."""
        out = []
        lost = {}
        for f in self.features_:
            idx = np.clip(self.woe_.binners_[f].bin_index(X[f].to_numpy()), 0, None)
            lost[f] = self.points_[f].max() - self.points_[f][idx]
        L = pd.DataFrame(lost)
        for i in range(len(X)):
            row = L.iloc[i].sort_values(ascending=False)
            out.append([(f, float(v)) for f, v in row.head(k).items() if v > 0.5])
        return out

    # ------------------------------------------------------------------ export
    def table(self):
        rows = []
        for f in self.features_:
            b = self.woe_.binners_[f]
            for lab, n, bad, woe, pts in zip(b.labels(), b.n_, b.bad_, b.woe_, self.points_[f]):
                rows.append({"characteristic": f, "attribute": lab, "n": int(n), "bad_rate": bad / max(n, 1),
                             "woe": woe, "iv": b.iv_, "points": pts})
        return pd.DataFrame(rows)

    def to_json(self, path=None):
        card = {"format": "points-scorecard/v1", "pdo": self.pdo, "base_score": self.base_score,
                "base_odds": self.base_odds, "factor": self.factor_, "offset": self.offset_,
                "neutral_points": (self.offset_ - self.factor_ * self.intercept_) / len(self.features_),
                "characteristics": []}
        for f in self.features_:
            b = self.woe_.binners_[f]
            d = b.to_dict()
            d.update({"name": f, "points": [float(p) for p in self.points_[f]], "labels": b.labels()})
            card["characteristics"].append(d)
        s = json.dumps(card, indent=1)
        if path:
            open(path, "w").write(s)
        return card
