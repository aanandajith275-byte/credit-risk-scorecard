"""Load the two credit datasets in a common format.

Both loaders return (X, y, meta):
  X     pandas DataFrame of decision-time features (protected attributes removed)
  y     numpy int array, 1 = bad (defaulted), 0 = good
  meta  dict with feature types, the protected attribute kept aside for fairness checks,
        and an exposure column for the loss calculations
"""
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# UCI Statlog German Credit, attribute codes -> readable labels (from german.doc)
GERMAN_COLS = ["checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
               "savings", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
               "residence_since", "property", "age", "other_installment_plans", "housing",
               "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "target"]
GERMAN_LABELS = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no account"},
    "credit_history": {"A30": "no credits / all paid", "A31": "all paid at this bank", "A32": "existing paid duly",
                       "A33": "delays in the past", "A34": "critical account"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio/TV", "A44": "appliances",
                "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business",
                "A410": "other"},
    "savings": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "unknown / none"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs", "A75": ">= 7 yrs"},
    "personal_status_sex": {"A91": "male: divorced", "A92": "female", "A93": "male: single",
                            "A94": "male: married", "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings / insurance", "A123": "car / other", "A124": "unknown / none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unskilled non-resident", "A172": "unskilled resident", "A173": "skilled", "A174": "management"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def load_german():
    df = pd.read_csv(os.path.join(DATA, "german.data"), header=None, names=GERMAN_COLS)
    for c, m in GERMAN_LABELS.items():
        df[c] = df[c].map(m)
    y = (df.pop("target") == 2).astype(int).to_numpy()            # 2 = bad in the original coding
    sex = df.pop("personal_status_sex").str.startswith("female").map({True: "female", False: "male"})
    X = df
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    meta = {"name": "German credit", "categorical": cat, "numeric": num, "protected": sex.to_numpy(),
            "exposure": X["credit_amount"].to_numpy(float), "currency": "DM",
            # the dataset's documented cost matrix: approving a bad applicant costs 5x rejecting a good one
            "cost_fn_fp": 5.0}
    return X, y, meta


def load_taiwan():
    """Default of credit card clients (Yeh & Lien, 2009): 30,000 Taiwanese card holders, Apr-Sep 2005.
    Target: default on the October 2005 payment."""
    df = pd.read_csv(os.path.join(DATA, "taiwan_credit_default.csv")).drop(columns="ID")
    y = df.pop("default.payment.next.month").to_numpy(int)
    sex = df.pop("SEX").map({1: "male", 2: "female"}).to_numpy()
    df = df.rename(columns={"PAY_0": "PAY_1"})                    # the source skips PAY_1; months 1..6 = Sep..Apr
    # undocumented codes (EDUCATION 0/5/6, MARRIAGE 0) are merged into "other", as the data description implies
    df["EDUCATION"] = df["EDUCATION"].map({1: "graduate school", 2: "university", 3: "high school"}).fillna("other")
    df["MARRIAGE"] = df["MARRIAGE"].map({1: "married", 2: "single"}).fillna("other")
    # engineered behaviour features, all computable at the decision date (end of September 2005)
    lim = df["LIMIT_BAL"].clip(lower=1)
    df["utilisation_1"] = df["BILL_AMT1"] / lim
    df["utilisation_avg6"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].mean(1) / lim
    df["pay_ratio_1"] = df["PAY_AMT1"] / df["BILL_AMT2"].clip(lower=1)           # Sep payment vs Aug bill
    df["months_delinquent_6"] = (df[[f"PAY_{i}" for i in range(1, 7)]] > 0).sum(1)
    df["max_delinquency_6"] = df[[f"PAY_{i}" for i in range(1, 7)]].max(1).clip(lower=0)
    cat = ["EDUCATION", "MARRIAGE"]
    num = [c for c in df.columns if c not in cat]
    meta = {"name": "Taiwan credit card", "categorical": cat, "numeric": num, "protected": sex,
            "exposure": df["LIMIT_BAL"].to_numpy(float), "currency": "NT$", "cost_fn_fp": None}
    return df, y, meta


LOADERS = {"german": load_german, "taiwan": load_taiwan}
