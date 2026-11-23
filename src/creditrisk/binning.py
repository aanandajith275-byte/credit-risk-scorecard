"""Weight-of-evidence (WoE) binning with monotonicity, the standard first step of a credit scorecard.

For a bin b:   WoE_b = ln( (good_b / good_total) / (bad_b / bad_total) )
               IV    = sum_b (good_b/good_total - bad_b/bad_total) * WoE_b

Positive WoE = safer than average. Counts get +0.5 (Laplace-style) so empty cells stay finite.
"""
import numpy as np
import pandas as pd


def _woe_iv(good, bad, eps=0.5):
    g = (np.asarray(good, float) + eps) / (np.sum(good) + eps * len(good))
    b = (np.asarray(bad, float) + eps) / (np.sum(bad) + eps * len(bad))
    woe = np.log(g / b)
    return woe, float(np.sum((g - b) * woe))


class NumericBinner:
    """Quantile fine-classing, then merge adjacent bins until (1) each has >= min_share of rows and
    (2) the bad rate is monotone in the direction of the overall Spearman correlation."""

    def __init__(self, max_bins=10, min_share=0.05, monotone=True):
        self.max_bins, self.min_share, self.monotone = max_bins, min_share, monotone

    def fit(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, int)
        qs = np.unique(np.quantile(x, np.linspace(0, 1, self.max_bins + 1)[1:-1]))
        edges = list(qs)                                    # bin i = (edges[i-1], edges[i]]
        while True:
            idx = np.searchsorted(edges, x, side="left")
            n = np.bincount(idx, minlength=len(edges) + 1)
            bad = np.bincount(idx, weights=y, minlength=len(edges) + 1)
            if len(edges) == 0:
                break
            rate = (bad + 0.5) / (n + 1.0)
            small = np.where(n < self.min_share * len(x))[0]
            if len(small):                                   # merge the smallest bin into its smaller neighbour
                i = small[np.argmin(n[small])]
                if i == 0:
                    del edges[0]                             # bins 0 and 1
                elif i == len(edges):
                    del edges[-1]                            # last two bins
                elif n[i - 1] > n[i + 1]:
                    del edges[i]                             # bins i and i+1
                else:
                    del edges[i - 1]                         # bins i-1 and i
            elif self.monotone:
                sign = np.sign(pd.Series(x).corr(pd.Series(y), method="spearman")) or 1.0
                viol = np.where(np.diff(rate) * sign < 0)[0]
                if not len(viol):
                    break
                i = viol[np.argmin(np.abs(np.diff(rate)[viol]))]   # gentlest violation first
                del edges[i]                                       # bins i and i+1
            else:
                break
        self.edges_ = np.array(edges)
        idx = np.searchsorted(self.edges_, x, side="left")
        k = len(self.edges_) + 1
        self.n_ = np.bincount(idx, minlength=k)
        self.bad_ = np.bincount(idx, weights=y, minlength=k)
        self.woe_, self.iv_ = _woe_iv(self.n_ - self.bad_, self.bad_)
        return self

    def bin_index(self, x):
        return np.searchsorted(self.edges_, np.asarray(x, float), side="left")

    def transform(self, x):
        return self.woe_[self.bin_index(x)]

    def labels(self):
        e = list(self.edges_)
        lo = ["-inf"] + [f"{v:.4g}" for v in e]
        hi = [f"{v:.4g}" for v in e] + ["inf"]
        return [f"({a}, {b}]" for a, b in zip(lo, hi)]

    def to_dict(self):
        return {"type": "numeric", "edges": [float(v) for v in self.edges_]}


class CategoricalBinner:
    """One bin per category; categories rarer than min_share are pooled with the category whose bad
    rate is closest. Unseen categories at scoring time get WoE 0 (= average risk)."""

    def __init__(self, min_share=0.05):
        self.min_share = min_share

    def fit(self, x, y):
        s = pd.Series(np.asarray(x).astype(str))
        y = pd.Series(np.asarray(y, int))
        groups = {c: [c] for c in s.unique()}
        while True:
            key = {c: g for g, members in groups.items() for c in members}
            gb = y.groupby(s.map(key)).agg(["count", "sum"])
            gb["rate"] = (gb["sum"] + 0.5) / (gb["count"] + 1)
            small = gb[gb["count"] < self.min_share * len(s)]
            if small.empty or len(gb) <= 2:
                break
            g = small["count"].idxmin()
            others = gb.drop(index=g)
            tgt = (others["rate"] - gb.loc[g, "rate"]).abs().idxmin()
            groups[tgt] += groups.pop(g)
        self.map_ = {c: i for i, (g, members) in enumerate(sorted(groups.items())) for c in members}
        self.groups_ = [members for _, members in sorted(groups.items())]
        idx = s.map(self.map_).to_numpy()
        k = len(self.groups_)
        self.n_ = np.bincount(idx, minlength=k)
        self.bad_ = np.bincount(idx, weights=y.to_numpy(), minlength=k)
        self.woe_, self.iv_ = _woe_iv(self.n_ - self.bad_, self.bad_)
        return self

    def bin_index(self, x):
        return pd.Series(np.asarray(x).astype(str)).map(self.map_).fillna(-1).astype(int).to_numpy()

    def transform(self, x):
        idx = self.bin_index(x)
        return np.where(idx >= 0, self.woe_[np.clip(idx, 0, None)], 0.0)

    def labels(self):
        return [" / ".join(sorted(m)) for m in self.groups_]

    def to_dict(self):
        return {"type": "categorical", "groups": [sorted(m) for m in self.groups_]}


class WoETransformer:
    """Fits one binner per column; transform() returns the WoE-encoded frame."""

    def __init__(self, categorical, max_bins=10, min_share=0.05, monotone=True):
        self.categorical = set(categorical)
        self.max_bins, self.min_share, self.monotone = max_bins, min_share, monotone

    def fit(self, X, y):
        self.binners_ = {}
        for c in X.columns:
            b = CategoricalBinner(self.min_share) if c in self.categorical else \
                NumericBinner(self.max_bins, self.min_share, self.monotone)
            self.binners_[c] = b.fit(X[c].to_numpy(), y)
        self.iv_ = pd.Series({c: b.iv_ for c, b in self.binners_.items()}).sort_values(ascending=False)
        return self

    def transform(self, X):
        return pd.DataFrame({c: self.binners_[c].transform(X[c].to_numpy()) for c in self.binners_}, index=X.index)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
