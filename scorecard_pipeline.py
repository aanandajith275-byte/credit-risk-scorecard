import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Configure Seaborn styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# ==============================================================================
#        DATA INGESTION 
# ==============================================================================
data = fetch_openml("credit-g", version=1, as_frame=True)
X_raw = data.data
y = (data.target == "bad").astype(int)  # 1 = Default (Bad risk), 0 = Good risk

X_encoded = pd.get_dummies(X_raw, drop_first=True)

train_size = int(len(X_encoded) * 0.8)
X_train, X_test = X_encoded.iloc[:train_size], X_encoded.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

imputer = SimpleImputer(strategy="median")
X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=X_encoded.columns)
X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=X_encoded.columns)

# ==============================================================================
 #    MODEL TRAINING & METRICS CALCULATION
# ==============================================================================

model = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
model.fit(X_train_clean, y_train)

y_probs = model.predict_proba(X_test_clean)[:, 1]

auc = roc_auc_score(y_test, y_probs)
gini = 2 * auc - 1
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
ks_stat = np.max(tpr - fpr)
ks_idx = np.argmax(tpr - fpr)


def calculate_psi(expected_probs, actual_probs, num_bins=10):
    counts_expected, bin_edges = np.histogram(expected_probs, bins=num_bins)
    counts_actual, _ = np.histogram(actual_probs, bins=bin_edges)
    e_pct = (counts_expected / len(expected_probs)) + 1e-4
    a_pct = (counts_actual / len(actual_probs)) + 1e-4
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))

train_probs = model.predict_proba(X_train_clean)[:, 1]
psi = calculate_psi(train_probs, y_probs)

print("="*50)
print("STAGE 2 OUTPUT: DISCRIMINATION METRICS SUMMARY")
print("="*50)
print(f"OOT AUC            : {auc:.2f}")
print(f"Gini Coefficient   : {gini:.2f}")
print(f"KS Statistic       : {ks_stat:.2f} (Target Benchmark > 0.30)")
print(f"Population Stability (PSI): {psi:.4f} (Target < 0.08)")
print("="*50)

# ==============================================================================
#                    VISUAL DIAGNOSTICS
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

#  Feature Correlation Heatmap (Top 10 Risk Features) 
top_features = X_train_clean.var().sort_values(ascending=False).head(10).index
corr_matrix = X_train_clean[top_features].corr()

sns.heatmap(corr_matrix, ax=axes[0, 0], annot=True, fmt=".2f", cmap="vlag", cbar=True, square=True)
axes[0, 0].set_title("Stage 1: Feature Correlation Heatmap", fontsize=12, fontweight="bold")

#  Predicted Probability KDE Density Distribution
prob_df = pd.DataFrame({"Default_Prob": y_probs, "Actual_Status": y_test.map({1: "Bad Risk (Default)", 0: "Good Risk"})})
sns.kdeplot(data=prob_df, x="Default_Prob", hue="Actual_Status", ax=axes[0, 1], fill=True, common_norm=False, palette=["green", "red"], alpha=0.4)
axes[0, 1].set_title("Stage 2: Scorecard Score Separation (KDE Density)", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Predicted Default Probability")

# KS Separation Gap Chart 
axes[1, 0].plot(thresholds, tpr, label="Cumulative Bad Risk (Default)", color="crimson", lw=2.5)
axes[1, 0].plot(thresholds, fpr, label="Cumulative Good Risk", color="forestgreen", lw=2.5)
axes[1, 0].vlines(thresholds[ks_idx], fpr[ks_idx], tpr[ks_idx], colors="navy", linestyles="--", label=f"Max KS Gap = {ks_stat:.2f}")
axes[1, 0].set_title("Stage 3: Kolmogorov-Smirnov (KS) Cumulative Separation", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Probability Threshold")
axes[1, 0].set_ylabel("Cumulative Percentage")
axes[1, 0].invert_xaxis()
axes[1, 0].legend(loc="lower left")

# Characteristic Stability Index (CSI) Feature Drift 
def calculate_csi(train_col, test_col, num_bins=5):
    counts_exp, bin_edges = np.histogram(train_col, bins=num_bins)
    counts_act, _ = np.histogram(test_col, bins=bin_edges)
    e_pct = (counts_exp / len(train_col)) + 1e-4
    a_pct = (counts_act / len(test_col)) + 1e-4
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))

csi_scores = {col: calculate_csi(X_train_clean[col], X_test_clean[col]) for col in top_features[:6]}
csi_df = pd.DataFrame(list(csi_scores.items()), columns=["Feature", "CSI_Value"])

sns.barplot(data=csi_df, x="CSI_Value", y="Feature", ax=axes[1, 1], palette="crest")
axes[1, 1].axvline(0.10, color="orange", linestyle="--", label="Stability Benchmark (0.10)")
axes[1, 1].set_title("Stage 4: Feature-Level Drift (CSI Analysis)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("CSI Metric")
axes[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()