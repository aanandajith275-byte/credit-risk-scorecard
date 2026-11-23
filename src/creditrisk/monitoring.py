"""Model monitoring checks, written to run as a scheduled batch job over a prediction log.

Two different questions, which need two different checks:
  * Has the applicant population changed?   PSI on the score, CSI on each characteristic.
    Needs no outcomes, so it can run the day after scoring.
  * Is the model still right?                Observed vs predicted default rate (calibration-in-the-large,
    binomial z-test) and AUC. Needs outcomes, which arrive months later.
A population shift does not imply the model is wrong, and a wrong model need not show a population shift.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from .metrics import psi

PSI_WARN, PSI_ALERT = 0.10, 0.25


def population_report(ref_scores, new_scores, ref_X=None, new_X=None, characteristics=()):
    out = {"score_psi": psi(ref_scores, new_scores)}
    csi = {}
    for c in characteristics:
        a, b = ref_X[c], new_X[c]
        if a.dtype.kind in "OUS" or str(a.dtype) in ("str", "string", "category"):
            cats = sorted(set(a.astype(str)) | set(b.astype(str)))
            pa = a.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
            pb = b.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
            csi[c] = float(((pb - pa) * np.log(pb / pa)).sum())
        else:
            csi[c] = psi(a.to_numpy(float), b.to_numpy(float))
    out["csi"] = csi
    if csi:
        worst = max(csi, key=csi.get)
        out["worst_characteristic"], out["worst_csi"] = worst, csi[worst]
    worst_csi = out.get("worst_csi", 0.0)
    out["population_status"] = ("ALERT" if out["score_psi"] > PSI_ALERT else
                                 "WARN" if (out["score_psi"] > PSI_WARN or worst_csi > PSI_ALERT) else "OK")
    return out


def performance_report(y, pd_pred, alpha=0.01):
    y, p = np.asarray(y), np.asarray(pd_pred)
    obs, exp = y.mean(), p.mean()
    # variance of the observed default count under the model: sum p(1-p)
    z = (y.sum() - p.sum()) / np.sqrt((p * (1 - p)).sum())
    pval = float(2 * stats.norm.sf(abs(z)))
    return {"observed_bad_rate": float(obs), "predicted_bad_rate": float(exp), "calibration_z": float(z),
            "calibration_p": pval, "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
            "performance_status": "ALERT" if pval < alpha else "OK"}
