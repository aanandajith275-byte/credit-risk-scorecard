import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# DATA INGESTION
data = fetch_openml("credit-g", version=1, as_frame=True)
X_raw = data.data
y = (data.target == "bad").astype(int)

# EDA
print("Dataset:", X_raw.shape)

plt.figure(figsize=(5, 4))
sns.countplot(x=y.map({0: "Good", 1: "Bad"}))
plt.title("Class Distribution")
plt.xlabel("Credit Risk")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "class_distribution.png"), dpi=200)
plt.close()

num_cols = X_raw.select_dtypes(include=np.number).columns

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(X_raw[col], kde=True, ax=ax)
    ax.set_title(col)
for ax in axes.flat[len(num_cols):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "numerical_distributions.png"), dpi=200)
plt.close()

corr_df = X_raw[list(num_cols)].copy()
corr_df["target"] = y.values

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "numerical_correlation.png"), dpi=200)
plt.close()

pairs = [
    ("duration", "credit_amount"),
    ("age", "credit_amount"),
    ("age", "duration"),
    ("credit_amount", "installment_commitment")
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (x_col, y_col) in zip(axes.flat, pairs):
    sns.scatterplot(
        data=X_raw.assign(target=y.values),
        x=x_col,
        y=y_col,
        hue="target",
        alpha=0.7,
        ax=ax
    )
    ax.set_title(f"{x_col} vs {y_col}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "selected_relationships.png"), dpi=200)
plt.close()

# SKEWNESS HANDLING
# Credit amount is strongly right-skewed, so log1p is tested as a targeted transformation.
X_original = X_raw.copy()
X_model = X_raw.copy()
X_model["credit_amount"] = np.log1p(X_model["credit_amount"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(X_original["credit_amount"], kde=True, ax=axes[0])
axes[0].set_title("Original Credit Amount")
axes[0].set_xlabel("Credit Amount")
axes[0].set_ylabel("Count")

sns.histplot(X_model["credit_amount"], kde=True, ax=axes[1])
axes[1].set_title("After log1p Transformation")
axes[1].set_xlabel("log1p(Credit Amount)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "credit_amount_transformation.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

X_encoded_original = pd.get_dummies(X_original, drop_first=True)
X_encoded_log = pd.get_dummies(X_model, drop_first=True)

# MODEL TRAINING
train_size = int(len(X_encoded_original) * 0.8)

X_train_original = X_encoded_original.iloc[:train_size]
X_test_original = X_encoded_original.iloc[train_size:]
X_train_log = X_encoded_log.iloc[:train_size]
X_test_log = X_encoded_log.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# Imputation is fitted only on the training portion.
imputer_original = SimpleImputer(strategy="median")
X_train_original_clean = pd.DataFrame(
    imputer_original.fit_transform(X_train_original),
    columns=X_encoded_original.columns
)
X_test_original_clean = pd.DataFrame(
    imputer_original.transform(X_test_original),
    columns=X_encoded_original.columns
)

imputer_log = SimpleImputer(strategy="median")
X_train_log_clean = pd.DataFrame(
    imputer_log.fit_transform(X_train_log),
    columns=X_encoded_log.columns
)
X_test_log_clean = pd.DataFrame(
    imputer_log.transform(X_test_log),
    columns=X_encoded_log.columns
)

# LogisticRegressionCV tests multiple C values using 5-fold CV.
# ROC-AUC is used for hyperparameter selection.
C_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]

original_model = LogisticRegressionCV(
    Cs=C_grid,
    cv=5,
    scoring="roc_auc",
    max_iter=2000,
    random_state=42,
    solver="liblinear"
)
original_model.fit(X_train_original_clean, y_train)

log_model = LogisticRegressionCV(
    Cs=C_grid,
    cv=5,
    scoring="roc_auc",
    max_iter=2000,
    random_state=42,
    solver="liblinear"
)
log_model.fit(X_train_log_clean, y_train)

original_c = float(original_model.C_[0])
log_c = float(log_model.C_[0])

# Mean cross-validation AUC at the selected C.
original_cv_scores = original_model.scores_[1].mean(axis=0)
log_cv_scores = log_model.scores_[1].mean(axis=0)
original_cv_auc = float(original_cv_scores[np.argmax(original_cv_scores)])
log_cv_auc = float(log_cv_scores[np.argmax(log_cv_scores)])

# Holdout evaluation
baseline_probs = original_model.predict_proba(X_test_original_clean)[:, 1]
log_probs = log_model.predict_proba(X_test_log_clean)[:, 1]

baseline_auc = roc_auc_score(y_test, baseline_probs)
baseline_gini = 2 * baseline_auc - 1
baseline_fpr, baseline_tpr, baseline_thresholds = roc_curve(
    y_test, baseline_probs
)
baseline_ks = np.max(baseline_tpr - baseline_fpr)

log_auc = roc_auc_score(y_test, log_probs)
log_gini = 2 * log_auc - 1
log_fpr, log_tpr, _ = roc_curve(y_test, log_probs)
log_ks = np.max(log_tpr - log_fpr)

comparison_df = pd.DataFrame({
    "Model": ["Original credit_amount", "log1p(credit_amount)"],
    "Selected_C": [original_c, log_c],
    "CV_AUC": [original_cv_auc, log_cv_auc],
    "Holdout_AUC": [baseline_auc, log_auc],
    "Gini": [baseline_gini, log_gini],
    "KS": [baseline_ks, log_ks],
    "Selected_For_Final": [
        original_cv_auc >= log_cv_auc,
        original_cv_auc < log_cv_auc
    ]
})
comparison_df.to_csv(
    os.path.join(OUT, "transformation_comparison.csv"),
    index=False
)

# Pick the transformation using CV on the training data.
# The holdout set is kept for the final evaluation.
if original_cv_auc >= log_cv_auc:
    selected_name = "Original credit_amount"
    model = original_model
    X_train_clean = X_train_original_clean
    X_test_clean = X_test_original_clean
    y_probs = baseline_probs
    auc = baseline_auc
    gini = baseline_gini
    fpr = baseline_fpr
    tpr = baseline_tpr
    thresholds = baseline_thresholds
    ks_stat = baseline_ks
    selected_c = original_c
    selected_cv_auc = original_cv_auc
else:
    selected_name = "log1p(credit_amount)"
    model = log_model
    X_train_clean = X_train_log_clean
    X_test_clean = X_test_log_clean
    y_probs = log_probs
    auc = log_auc
    gini = log_gini
    fpr, tpr, thresholds = roc_curve(y_test, log_probs)
    ks_stat = log_ks
    selected_c = log_c
    selected_cv_auc = log_cv_auc

ks_idx = np.argmax(tpr - fpr)

def calculate_psi(expected_probs, actual_probs, num_bins=10):
    counts_expected, bin_edges = np.histogram(expected_probs, bins=num_bins)
    counts_actual, _ = np.histogram(actual_probs, bins=bin_edges)
    e_pct = (counts_expected / len(expected_probs)) + 1e-4
    a_pct = (counts_actual / len(actual_probs)) + 1e-4
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))


train_probs = model.predict_proba(X_train_clean)[:, 1]
psi = calculate_psi(train_probs, y_probs)

print("=" * 50)
print("LOGISTIC REGRESSION HYPERPARAMETER SELECTION")
print("=" * 50)
print(f"Selected model        : {selected_name}")
print(f"Selected C            : {selected_c}")
print(f"5-fold CV AUC         : {selected_cv_auc:.3f}")
print(f"Other model CV AUC    : {min(original_cv_auc, log_cv_auc):.3f}")
print("=" * 50)
print("BEFORE vs AFTER log1p")
print("=" * 50)
print(comparison_df.to_string(index=False))
print("=" * 50)
print("FINAL MODEL VALIDATION METRICS")
print("=" * 50)
print(f"Holdout AUC         : {auc:.2f}")
print(f"Gini Coefficient    : {gini:.2f}")
print(f"KS Statistic        : {ks_stat:.2f}")
print(f"Population Stability: {psi:.4f}")
print("=" * 50)

metrics_df = pd.DataFrame({
    "Metric": ["Selected C", "CV AUC", "Holdout AUC", "Gini", "KS", "PSI"],
    "Value": [selected_c, selected_cv_auc, auc, gini, ks_stat, psi]
})
metrics_df.to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)

# VISUAL DIAGNOSTICS
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top_features = X_train_clean.var().sort_values(ascending=False).head(10).index
corr_matrix = X_train_clean[top_features].corr()

sns.heatmap(
    corr_matrix,
    ax=axes[0, 0],
    annot=True,
    fmt=".2f",
    cmap="vlag",
    cbar=True,
    square=True
)
axes[0, 0].set_title("Feature Correlation: Top Variance Features")

prob_df = pd.DataFrame({
    "Default_Prob": y_probs,
    "Actual_Status": y_test.map({
        1: "Bad Risk (Default)",
        0: "Good Risk"
    })
})

sns.kdeplot(
    data=prob_df,
    x="Default_Prob",
    hue="Actual_Status",
    ax=axes[0, 1],
    fill=True,
    common_norm=False,
    palette=["green", "red"],
    alpha=0.4
)
axes[0, 1].set_title("Predicted Probability Separation")
axes[0, 1].set_xlabel("Predicted Default Probability")

axes[1, 0].plot(
    thresholds,
    tpr,
    label="Cumulative Bad Risk",
    color="crimson",
    lw=2.5
)
axes[1, 0].plot(
    thresholds,
    fpr,
    label="Cumulative Good Risk",
    color="forestgreen",
    lw=2.5
)
axes[1, 0].vlines(
    thresholds[ks_idx],
    fpr[ks_idx],
    tpr[ks_idx],
    colors="navy",
    linestyles="--",
    label=f"Max KS Gap = {ks_stat:.2f}"
)
axes[1, 0].set_title("Kolmogorov-Smirnov Separation")
axes[1, 0].set_xlabel("Probability Threshold")
axes[1, 0].set_ylabel("Cumulative Percentage")
axes[1, 0].invert_xaxis()
axes[1, 0].legend(loc="lower left")


def calculate_csi(train_col, test_col, num_bins=5):
    counts_exp, bin_edges = np.histogram(train_col, bins=num_bins)
    counts_act, _ = np.histogram(test_col, bins=bin_edges)
    e_pct = (counts_exp / len(train_col)) + 1e-4
    a_pct = (counts_act / len(test_col)) + 1e-4
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))


csi_scores = {
    col: calculate_csi(X_train_clean[col], X_test_clean[col])
    for col in top_features[:6]
}
csi_df = pd.DataFrame(
    list(csi_scores.items()),
    columns=["Feature", "CSI_Value"]
)

sns.barplot(
    data=csi_df,
    x="CSI_Value",
    y="Feature",
    ax=axes[1, 1],
    palette="crest"
)
axes[1, 1].axvline(
    0.10,
    color="orange",
    linestyle="--",
    label="Reference = 0.10"
)
axes[1, 1].set_title("Feature-Level Distribution Shift")
axes[1, 1].set_xlabel("CSI Metric")
axes[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "validation_dashboard.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

csi_df.to_csv(os.path.join(OUT, "csi_scores.csv"), index=False)

print("\nOutputs saved to:")
print(os.path.abspath(OUT))
