"""Discrimination, calibration, stability and significance tests used throughout the project."""
import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def ks(y, p):
    """Kolmogorov-Smirnov: max distance between the score CDFs of bads and goods."""
    order = np.argsort(p)
    y = np.asarray(y)[order]
    cb = np.cumsum(y) / y.sum()
    cg = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cb - cg)))


def ece(y, p, bins=10):
    """Expected calibration error with equal-count bins: mean |observed - predicted| default rate."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) * abs(np.mean(np.asarray(y)[c]) - np.mean(p[c])) for c in chunks) / len(p))


def calibration_slope_intercept(y, p):
    """Logistic recalibration y ~ a + b * logit(p). Perfect calibration: a = 0, b = 1.
    b < 1 means predictions are too extreme (over-fitting), b > 1 too timid."""
    from sklearn.linear_model import LogisticRegression
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def summary(y, p):
    a = roc_auc_score(y, p)
    a0, b0 = calibration_slope_intercept(y, p)
    return {"auc": a, "gini": 2 * a - 1, "ks": ks(y, p), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p), "ece": ece(y, p), "cal_intercept": a0, "cal_slope": b0}


# ---------------------------------------------------------------- DeLong test for correlated AUCs
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong(y, p1, p2):
    """DeLong, DeLong & Clarke-Pearson (1988), fast version of Sun & Xu (2014).
    Returns (auc1, auc2, se of the difference, two-sided p-value) for two models scored on the same cases."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        p = np.asarray(p, float)
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        auc = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
        aucs.append(auc)
        v10.append((tz[:m] - tx) / n)               # structural components
        v01.append(1.0 - (tz[m:] - ty) / m)
    S10, S01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    se = float(np.sqrt(max(var, 1e-18)))
    z = (aucs[0] - aucs[1]) / se
    return float(aucs[0]), float(aucs[1]), se, float(2 * stats.norm.sf(abs(z)))


def auc_se(y, p):
    """DeLong standard error of a single AUC."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    p = np.asarray(p, float)
    tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    return float(np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n))


def bootstrap_ci(y, p, fn=roc_auc_score, B=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003): k-fold CV folds overlap, so the naive t-test is anti-conservative.
    The variance is inflated by (1/k + n_test/n_train)."""
    d = np.asarray(diffs, float)
    k = len(d)
    var = d.var(ddof=1) * (1.0 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


# ---------------------------------------------------------------- stability
def psi(expected, actual, bins=10, edges=None):
    """Population stability index between a reference sample and a new one, on reference-quantile bins.
    Rules of thumb: < 0.10 stable, 0.10-0.25 investigate, > 0.25 significant shift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if edges is None:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected, side="left"), minlength=len(edges) + 1) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual, side="left"), minlength=len(edges) + 1) / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
