# Credit Risk Scorecard & Model Validation Pipeline

A production-ready credit scoring and model governance framework using the German Credit dataset (`credit-g`). The pipeline trains a regularized Logistic Regression scorecard, computes core discrimination metrics (AUC, Gini, KS Statistic), and evaluates population stability using PSI and CSI.

## Key Metrics & Governance Benchmarks

* **AUC & Gini:** Evaluates the overall ranking performance between Good Risk and Bad Risk (Default).
* **KS Statistic:** Identifies maximum separation distance between cumulative bad and good risk distributions (Target > 0.30).
* **PSI (Population Stability Index):** Measures total score distribution drift across train and test splits (Target < 0.08).
* **CSI (Characteristic Stability Index):** Tracks individual input feature drift against stability thresholds (Target < 0.10).

## Diagnostic Dashboard Overview

Running the pipeline generates a 4-stage visual diagnostic grid:
1. **Stage 1: Feature Correlation Heatmap:** Audits feature collinearity to maintain stable logistic regression coefficients.
2. **Stage 2: KDE Density Separation:** Visualizes score separation between default and non-default distributions.
3. **Stage 3: KS Separation Gap:** Plotting cumulative default vs. good risk curves to identify optimal decision cutoffs.
4. **Stage 4: Feature-Level Drift (CSI):** Highlights single-variable population shifts against standard risk benchmarks.

## Quickstart

### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/your-username/credit-risk-scorecard.git](https://github.com/aanandajith275-byte/credit-risk-scorecard.git)
cd credit-risk-scorecard

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt