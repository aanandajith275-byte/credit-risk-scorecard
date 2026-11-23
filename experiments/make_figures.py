"""Draws every figure in figures/ from results/.   python experiments/make_figures.py"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import style  # noqa: E402
from style import AQUA, BLUE, INK, INK2, MAGENTA, MUTED, ORANGE, VIOLET, YELLOW, subtitle  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
style.apply()
R = lambda f: pd.read_csv(os.path.join(RES, f))            # noqa: E731
J = lambda f: json.load(open(os.path.join(RES, f)))         # noqa: E731
NAMES = {"scorecard": "WoE scorecard (champion)", "logistic": "Logistic, one-hot", "random_forest": "Random forest",
         "gbm": "Gradient boosting", "gbm_monotone": "Monotone boosting", "mlp": "Neural net (MLP)",
         "stack": "Stack: scorecard + boosting"}
ORDER = ["scorecard", "logistic", "random_forest", "gbm", "gbm_monotone", "mlp", "stack"]
COL = {"scorecard": INK, "logistic": MUTED, "random_forest": YELLOW, "gbm": BLUE, "gbm_monotone": AQUA,
       "mlp": MAGENTA, "stack": VIOLET}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. headline benchmark
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=False)
for ax, name, title in [(axes[0], "german", "German credit: 1,000 loans"), (axes[1], "taiwan", "Taiwan credit cards: 30,000 accounts")]:
    folds = R(f"cv_folds_{name}.csv")
    summ = R(f"cv_summary_{name}.csv").set_index("model")
    y = np.arange(len(ORDER))[::-1]
    champ = summ.loc["scorecard", "auc_mean"]
    for yi, m in zip(y, ORDER):
        f = folds[folds.model == m].auc
        ax.scatter(f, np.full(len(f), yi) + np.random.default_rng(0).uniform(-0.15, 0.15, len(f)), s=10, color=COL[m], alpha=0.35)
        ax.plot([summ.loc[m, "auc_mean"]] * 2, [yi - 0.3, yi + 0.3], color=COL[m], lw=3)
        p = summ.loc[m, "p_value"]
        lab = f"{summ.loc[m, 'auc_mean']:.3f}" + ("" if m == "scorecard" else f"  ({summ.loc[m, 'delta_auc_vs_champion']:+.3f}, p={p:.2g})")
        ax.text(1.02, yi, lab, va="center", fontsize=8.5, color=INK, transform=ax.get_yaxis_transform())
    ax.axvline(champ, color=INK, lw=0.8, ls=":")
    ax.set_yticks(y, [NAMES[m] for m in ORDER], fontsize=9)
    ax.set_xlabel("cross-validated AUC (each dot = one test fold; bar = mean)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=11.5, pad=10)
axes[0].set_xlim(0.55, 0.9)
axes[1].set_xlim(0.735, 0.80)
fig.subplots_adjust(wspace=1.25)
fig.suptitle("On 1,000 loans no challenger beats the scorecard; on 30,000 accounts gradient boosting does (+0.014 AUC)",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.08)
fig.text(0.01, 1.01, "Mean AUC, difference vs the champion and Nadeau-Bengio corrected p-value (folds overlap, so the naive t-test is over-confident)",
         fontsize=9.5, color=INK2)
save(fig, "01_champion_vs_challengers.png")

# ---------------------------------------------------------------- 2. the scorecard
T = R("scorecard_table_taiwan.csv")
IV = R("information_value_taiwan.csv").rename(columns={"Unnamed: 0": "feature"}).head(14)
meta = J("scorecard_meta_taiwan.json")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]
yv = np.arange(len(IV))[::-1]
used = set(meta["features"])
ax.barh(yv, IV.information_value, color=[BLUE if f in used else MUTED for f in IV.feature], height=0.65)
for yi, (f, v) in zip(yv, zip(IV.feature, IV.information_value)):
    ax.text(v + 0.01, yi, f"{v:.2f}" + ("" if f in used else "  dropped: wrong sign"), va="center", fontsize=8, color=INK2)
ax.set_yticks(yv, IV.feature, fontsize=8.5)
ax.axvline(0.02, color=MUTED, ls=":", lw=1)
ax.set_xlabel("information value (> 0.3 = strong predictor)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 1.15)
ax.set_title("Which characteristics carry information", fontsize=11.5, pad=10)
ax = axes[1]
show = ["PAY_1", "max_delinquency_6", "LIMIT_BAL", "utilisation_avg6"]
rows = T[T.characteristic.isin(show)].copy()
rows["label"] = rows.characteristic + "  " + rows.attribute
yy = np.arange(len(rows))[::-1]
cols = {c: col for c, col in zip(show, [BLUE, ORANGE, AQUA, VIOLET])}
ax.barh(yy, rows.points, color=[cols[c] for c in rows.characteristic], height=0.7)
for yi, (pts, br) in zip(yy, zip(rows.points, rows.bad_rate)):
    ax.text(pts + 0.5, yi, f"{pts:.0f} pts   (bad rate {br:.0%})", va="center", fontsize=7.8, color=INK2)
ax.set_yticks(yy, rows.label, fontsize=7.8)
ax.set_xlim(rows.points.min() - 5, rows.points.max() + 28)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("points awarded (more points = safer)")
ax.grid(axis="y", visible=False)
ax.set_title("Part of the points table (20 points doubles the odds of repaying)", fontsize=11.5, pad=10)
fig.suptitle(f"The champion: {len(meta['features'])} characteristics, monotone bins, points that add up to a score",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.04)
save(fig, "02_scorecard.png")

# ---------------------------------------------------------------- 3. calibration
P = R("holdout_predictions_taiwan.csv")
H = R("holdout_taiwan.csv").set_index("model")
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
for m in ["scorecard", "gbm_monotone", "random_forest", "mlp"]:
    order = np.argsort(P[m].to_numpy())
    chunks = np.array_split(order, 15)
    ax.plot([P[m].to_numpy()[c].mean() for c in chunks], [P["y"].to_numpy()[c].mean() for c in chunks], marker="o", ms=4,
            color=COL[m], label=f"{NAMES[m]}  (ECE {H.loc[m, 'ece']:.3f}, slope {H.loc[m, 'cal_slope']:.2f})")
ax.set_xlabel("predicted probability of default (bin mean)")
ax.set_ylabel("observed default rate")
ax.set_xlim(0, 0.85)
ax.set_ylim(0, 0.85)
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("All models are close to calibrated on 6,000 unseen accounts")
subtitle(ax, "Equal-count bins. A PD of 0.3 should mean 30% default: needed for pricing, provisioning and cut-offs")
save(fig, "03_calibration.png")

# ---------------------------------------------------------------- 4. leakage audit
L = R("leakage_audit.csv")
L = L[~L.mistake.str.contains("500|1000|2000")]
fig, ax = plt.subplots(figsize=(10, 4.8))
labs = [f"{'German' if d == 'german' else 'Taiwan'}: {m[3:]}" for d, m in zip(L.dataset, L.mistake)]
yv = np.arange(len(L))[::-1]
for yi, (_, r) in zip(yv, L.iterrows()):
    ax.plot([r.real_auc, r.reported_auc], [yi, yi], color=MUTED, lw=2, zorder=1)
    ax.scatter(r.real_auc, yi, color=BLUE, s=60, zorder=2)
    ax.scatter(r.reported_auc, yi, color=ORANGE, s=60, zorder=2)
    ax.text(max(r.reported_auc, r.real_auc) + 0.006, yi, f"{r.optimism:+.3f}", va="center", fontsize=8.5, color=INK)
ax.scatter([], [], color=ORANGE, label="AUC the analyst would report")
ax.scatter([], [], color=BLUE, label="AUC on data never touched")
ax.set_yticks(yv, labs, fontsize=8.8)
ax.set_xlabel("AUC")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=8.5)
ax.set_title("Three validation mistakes, measured: the smaller the data, the bigger the lie")
subtitle(ax, "Post-decision data inflates AUC by ~0.13 anywhere; WoE fitted before the split and test-set selection only bite on small data")
save(fig, "04_leakage_audit.png")

# ---------------------------------------------------------------- 5. business value
S = R("strategy_curves_taiwan.csv")
BV = J("business_value.json")["taiwan"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
for m in ["scorecard", "gbm", "gbm_monotone"]:
    s = S[S.model == m]
    ax.plot(s.approval_rate * 100, s.bad_rate_approved * 100, color=COL[m], label=NAMES[m])
ax.axvline(80, color=MUTED, ls=":", lw=1)
ax.set_xlabel("approval rate (%)")
ax.set_ylabel("default rate among approved (%)")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("Strategy curve: same approvals, fewer defaults", fontsize=11.5, pad=10)
ax = axes[1]
ch = ["gbm", "gbm_monotone", "stack"]
v1 = [BV["challengers"][c]["bads_avoided_per_10k"] for c in ch]
e1 = np.array([BV["challengers"][c]["bads_avoided_ci95"] for c in ch])
v2 = [BV["challengers"][c]["loss_saving_per_applicant_ntd"] for c in ch]
e2 = np.array([BV["challengers"][c]["loss_saving_ci95_ntd"] for c in ch])
x = np.arange(len(ch))
ax.errorbar(x - 0.12, v1, yerr=[np.array(v1) - e1[:, 0], e1[:, 1] - np.array(v1)], fmt="o", color=BLUE, capsize=4,
            label="defaults avoided per 10,000 applicants")
ax.errorbar(x + 0.12, np.array(v2) / 10, yerr=[(np.array(v2) - e2[:, 0]) / 10, (e2[:, 1] - np.array(v2)) / 10], fmt="s",
            color=ORANGE, capsize=4, label="credit loss saved per applicant (NT$ / 10)")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(x, ["Gradient\nboosting", "Monotone\nboosting", "Stack"], fontsize=8.5)
ax.set_xlim(-0.5, len(ch) - 0.5)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=8)
ax.set_title("vs the champion at 80% approval (95% CI)", fontsize=11.5, pad=10)
fig.suptitle("Boosting avoids 42 defaults per 10,000 applicants, yet saves no money: the defaults it avoids are small balances",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.04)
save(fig, "05_business_value.png")

# ---------------------------------------------------------------- 6. cloud benchmark
C = R("cloud_benchmark.csv").set_index("model").loc[ORDER]
fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
specs = [("artifact_kb", "deployment artifact (KB, log)", True), ("cold_start_ms", "cold start, one full vCPU (ms)", False),
         ("p50_ms", "warm p50 latency (ms, log)", True), ("cost_per_million_usd", "Lambda \\$ per 1M decisions", False)]
yv = np.arange(len(C))[::-1]
for ax, (col, lab, log) in zip(axes, specs):
    ax.barh(yv, C[col], color=[COL[m] for m in C.index], height=0.65)
    for yi, v in zip(yv, C[col]):
        txt = f"{v:,.0f}" if v >= 100 else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}")
        ax.text(v * (1.15 if log else 1.02) + (0 if log else C[col].max() * 0.01), yi, txt, va="center", fontsize=8, color=INK)
    if log:
        ax.set_xscale("log")
        ax.set_xlim(C[col].min() / 2, C[col].max() * 12)
    else:
        ax.set_xlim(0, C[col].max() * 1.35)
    ax.set_yticks(yv, [NAMES[m] for m in C.index] if ax is axes[0] else [""] * len(yv), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_title(lab, fontsize=10, pad=8)
fig.subplots_adjust(wspace=0.12)
sc, gm = C.loc["scorecard"], C.loc["gbm_monotone"]
fig.suptitle(f"The scorecard ships as {sc.artifact_kb:.0f} KB with no dependencies and starts in {sc.cold_start_ms:.0f} ms; "
             f"the boosted challenger needs scikit-learn and {gm.cold_start_ms / 1000:.1f} s", x=0.01, ha="left",
             fontsize=12.5, fontweight="bold", color=INK, y=1.06)
fig.text(0.01, 0.985, "Measured single-threaded on this machine; cost = \\$0.20 per 1M requests + \\$0.0000166667 per GB-s, billed per ms, "
         "at the cheapest memory setting that fits", fontsize=9.5, color=INK2)
save(fig, "06_cloud_cost.png")

# ---------------------------------------------------------------- 7. monitoring under drift
D = R("drift_simulation.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for j, sc_name in enumerate(["covariate shift", "concept drift"]):
    d = D[D.scenario == sc_name]
    ax = axes[0, j]
    ax.plot(d.month, d.score_psi, color=BLUE, marker="o", ms=4, label="score PSI")
    ax.plot(d.month, d.worst_csi, color=ORANGE, marker="o", ms=4, label="largest characteristic CSI")
    ax.axhline(0.10, color=MUTED, ls=":", lw=1)
    ax.axhline(0.25, color=MUTED, ls="--", lw=1)
    ax.text(0, 0.265, "alert 0.25", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("stability index")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(("Covariate shift: customers' utilisation rises" if j == 0 else "Concept drift: a downturn, same customers"),
                 fontsize=11.5, pad=10)
    ax = axes[1, j]
    ax.plot(d.month, d.observed_bad_rate * 100, color=INK, marker="o", ms=4, label="observed default rate")
    ax.plot(d.month, d.predicted_bad_rate * 100, color=AQUA, marker="o", ms=4, label="predicted by the model")
    al = d[d.performance_status == "ALERT"]
    ax.scatter(al.month, al.observed_bad_rate * 100 + 0.8, marker="v", color=ORANGE, s=30, label="calibration alert (p < 0.01)")
    ax.set_ylim(18, 33)
    ax.set_xlabel("month")
    ax.set_ylabel("default rate (%)")
    ax.legend(loc="upper left", fontsize=8.5)
fig.suptitle("Score PSI caught neither problem: CSI flags the mix change (harmless), only outcomes reveal the downturn (harmful)",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.01)
save(fig, "07_drift_monitoring.png")

# ---------------------------------------------------------------- 8. shadow vs A/B
A = R("ab_power.csv")
AB = J("ab_power.json")
fig, ax = plt.subplots(figsize=(8.6, 4.4))
a = A[A.applications_per_month == 10000]
ax.plot(a.delta_auc, a.months_shadow, color=AQUA, marker="o", label="shadow mode (both models score everyone; paired)")
ax.plot(a.delta_auc, a.months_ab, color=ORANGE, marker="o", label="A/B split 50/50 (unpaired)")
ax.axvline(AB["delta_auc"], color=MUTED, ls=":", lw=1)
ax.text(AB["delta_auc"] + 0.001, 30, f"observed lift\n{AB['delta_auc']:+.3f}", fontsize=8.5, color=INK2)
ax.set_yscale("log")
ax.set_xlabel("true AUC advantage of the challenger")
ax.set_ylabel("months of applications needed (log)")
ax.legend(loc="lower left", fontsize=8.5)
ax.set_title(f"Shadow scoring needs {AB['efficiency_ratio_ab_over_shadow']:.0f}x fewer outcomes than an A/B test")
subtitle(ax, "10,000 applications/month, 80% approved, 80% power at 5% significance. Add 6-12 months for defaults to be observed")
save(fig, "08_shadow_vs_ab.png")
