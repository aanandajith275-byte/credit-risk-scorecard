"""Unit tests: metrics against reference implementations, scorecard maths, runtime equivalence."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from creditrisk.binning import NumericBinner  # noqa: E402
from creditrisk.data import load_german  # noqa: E402
from creditrisk.metrics import delong, ks, psi  # noqa: E402
from creditrisk.runtime import ScorecardRuntime  # noqa: E402
from creditrisk.scorecard import Scorecard  # noqa: E402

rng = np.random.default_rng(0)


def test_ks_matches_scipy():
    y = rng.integers(0, 2, 400)
    p = y + rng.normal(0, 1.2, 400)
    assert ks(y, p) == pytest.approx(ks_2samp(p[y == 1], p[y == 0]).statistic)


def test_delong_auc_and_se():
    y = rng.integers(0, 2, 600)
    p1, p2 = y + rng.normal(0, 1.5, 600), y + rng.normal(0, 2.0, 600)
    a1, a2, se, _ = delong(y, p1, p2)
    assert a1 == pytest.approx(roc_auc_score(y, p1))
    boot = []
    for _ in range(1500):
        i = rng.integers(0, 600, 600)
        boot.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    assert se == pytest.approx(np.std(boot), rel=0.15)


def test_psi_zero_for_same_distribution_and_positive_for_shift():
    a = rng.normal(0, 1, 20000)
    assert psi(a, rng.normal(0, 1, 20000)) < 0.01
    assert psi(a, rng.normal(0.5, 1, 20000)) > 0.2


def test_monotone_binning():
    x = rng.uniform(0, 1, 5000)
    y = (rng.random(5000) < 0.1 + 0.5 * x).astype(int)
    b = NumericBinner(max_bins=10).fit(x, y)
    rate = b.bad_ / b.n_
    assert np.all(np.diff(rate) > 0)            # risk rises with x
    assert np.all(b.n_ >= 0.05 * 5000)


def test_scorecard_points_and_runtime_agree():
    X, y, meta = load_german()
    sc = Scorecard(categorical=tuple(meta["categorical"])).fit(X, y)
    # points add up to the logistic regression's log-odds
    s = sc.score(X)
    lp = sc.intercept_ + sc.woe_.transform(X)[sc.features_].to_numpy() @ sc.coef_
    assert np.allclose(s, sc.offset_ - sc.factor_ * lp)
    # PDO: 20 more points = odds of good doubled
    assert sc.factor_ * np.log(2) == pytest.approx(20)
    rt = ScorecardRuntime(sc.to_json())
    assert np.allclose([rt.score_one(r)[0] for r in X.head(200).to_dict("records")], s[:200])
    assert all(c < 0 for c in sc.coef_)          # sign check: positive WoE (safer) lowers P(bad)
