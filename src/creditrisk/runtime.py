"""Dependency-free scorecard runtime: the JSON scorecard plus ~40 lines of standard-library Python.

This is what the cloud service runs. It needs no numpy, pandas or scikit-learn, which is why the
champion has a tiny deployment package and a fast cold start (see experiments/cloud_benchmark.py).
"""
import bisect
import json
import math


class ScorecardRuntime:
    def __init__(self, card):
        self.card = card if isinstance(card, dict) else json.load(open(card))
        self._lookup = []
        for ch in self.card["characteristics"]:
            if ch["type"] == "categorical":
                m = {cat: i for i, grp in enumerate(ch["groups"]) for cat in grp}
                self._lookup.append((ch["name"], "c", m, ch["points"], ch["labels"]))
            else:
                self._lookup.append((ch["name"], "n", ch["edges"], ch["points"], ch["labels"]))

    def score_one(self, applicant):
        """applicant: dict of raw characteristic values. Returns (score, pd, [(name, points, max, label)])."""
        total, detail = 0.0, []
        for name, kind, spec, pts, labels in self._lookup:
            v = applicant[name]
            if kind == "c":
                i = spec.get(str(v), -1)
            else:
                i = bisect.bisect_left(spec, float(v))       # same convention as numpy searchsorted(side="left")
            p = pts[i] if i >= 0 else self.card["neutral_points"]
            total += p
            detail.append((name, p, max(pts), labels[i] if i >= 0 else "unseen"))
        pd_bad = 1.0 / (1.0 + math.exp((total - self.card["offset"]) / self.card["factor"]))
        return total, pd_bad, detail

    def reason_codes(self, detail, k=4):
        lost = sorted(((mx - p, name, label) for name, p, mx, label in detail), reverse=True)
        return [{"characteristic": n, "attribute": lab, "points_lost": round(l, 1)} for l, n, lab in lost[:k] if l > 0.5]
