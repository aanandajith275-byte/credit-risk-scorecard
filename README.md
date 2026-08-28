# Credit Risk Scorecard & Model Validation Pipeline

A production-grade credit risk modeling and model risk governance framework built on the OpenML German Credit dataset (`credit-g`). This project implements an Out-of-Time (OOT) validated regularized Logistic Regression scorecard, incorporating complete discrimination metrics, stability audits, and diagnostic visualizations aligned with Federal Reserve SR 11-7 model risk management guidance.

---

## 📌 Executive Summary & Key Benchmarks

| Metric | Model Value | Regulatory / Industry Benchmark | Status | Validation Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **OOT AUC** | **0.80** | $> 0.70$ | ✅ Passed | Strong ranking ability to distinguish good vs. bad borrowers. |
| **Gini Coefficient** | **0.61** | $> 0.40$ | ✅ Passed | High discriminative power ($2 \times \text{AUC} - 1$). |
| **KS Statistic** | **0.52** | $> 0.30$ | ✅ Passed | Excellent maximum cumulative separation between risk classes. |
| **Population Stability (PSI)** | **0.0343** | $< 0.10$ | ✅ Passed | Minimal overall score distribution drift across train/OOT splits. |
| **Feature Drift (CSI)** | **< 0.06** | $< 0.10$ | ✅ Passed | Top predictive features remain stable across sample periods. |

---

## 🔬 4-Stage Model Validation Workflow

* **Stage 1: Feature Correlation Audit (Multicollinearity):** Evaluates pairwise feature correlations among top risk drivers to prevent coefficient inflation and ensure scorecard stability under macroeconomic stress.
* **Stage 2: Score Separation & Density Analysis:** Analyzes Kernel Density Estimation (KDE) distributions of predicted default probabilities ($P(\text{Default})$) across Good vs. Bad risk profiles to verify decision clarity.
* **Stage 3: Kolmogorov-Smirnov (KS) Cumulative Separation:** Locates the optimal decision cutoff threshold where the gap between cumulative bad risk profiles and good risk profiles is maximized ($\text{Max KS} = 0.52$).
* **Stage 4: Characteristic Stability Index (CSI):** Performs bin-level feature drift analysis to ensure incoming applicant attributes do not deviate significantly from baseline training distributions.

---

## 📊 Visual Diagnostic Dashboard

<details>
<summary><b>▶ Click to Expand 4-Stage Diagnostic Charts</b></summary>

<br>

![Credit Risk Diagnostic Dashboard](dashboard.png)

</details>

---

## 🏛️ Model Risk Governance & Compliance

* **SR 11-7 Alignment:** Rigorous Out-of-Time (OOT) benchmarking, drift detection, and threshold validation to satisfy regulatory model validation standards.
* **Adverse Action Explainability (FCRA / ECOA):** Utilizing linear logistic models enables direct extraction of Odds Ratios and Weight of Evidence (WOE) scores to generate compliant adverse action reason codes for rejected applicants.
* **Regularization & Generalization:** Applied L2 penalty tuning ($C=0.05$) to prevent overfitting on sparse credit attributes and ensure smooth score transformations.

---

## 📂 Repository Structure

```text
credit-risk-scorecard/
├── README.md                  # Project documentation & model risk report
├── dashboard.png              # Generated 4-stage validation dashboard
├── requirements.txt           # Environment dependencies
├── .gitignore                 # Tracked files filter
└── src/
    └── scorecard_pipeline.py  # End-to-end model ingestion & validation pipeline