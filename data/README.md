# Data

| File | Dataset | Rows | Source |
|---|---|---|---|
| `german.data` | Statlog German Credit (Hofmann, 1994), UCI ML Repository #144 | 1,000 loans, 20 attributes, 30% bad | comma-separated copy of the original `german.data`, from [commit-live-students/Statlog-German_Credit_Data-_project](https://github.com/commit-live-students/Statlog-German_Credit_Data-_project); attribute codes decoded in `src/creditrisk/data.py` from the UCI documentation |
| `taiwan_credit_default.csv` | Default of Credit Card Clients (Yeh & Lien, 2009), UCI ML Repository #350 | 30,000 card holders, Apr-Sep 2005, 22.1% default in Oct 2005 | the widely used `UCI_Credit_Card.csv` version, from [YuChenAmberLu/Data-Science--Credit-Card-Default](https://github.com/YuChenAmberLu/Data-Science--Credit-Card-Default) |

Both are public research datasets licensed CC BY 4.0 by the UCI Machine Learning Repository.

**Protected attributes are not model inputs.** German `personal_status_sex` and Taiwan `SEX` are removed by the
loaders and kept aside only to check approval rates by sex afterwards (`experiments/business_value.py`).

**Taiwan coding notes.** `PAY_0` is renamed `PAY_1` (the source skips 1); PAY_k = repayment status k months before
October 2005 (-2 no consumption, -1 paid in full, 0 revolving credit, 1..9 months late). Undocumented EDUCATION codes
(0, 5, 6) and MARRIAGE code 0 are pooled into "other". Amounts are in NT dollars.
