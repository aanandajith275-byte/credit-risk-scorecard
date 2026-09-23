# Credit Risk Scorecard

A small credit-risk modeling project using Logistic Regression on the German Credit dataset from OpenML.

## What it does

- Checks class balance, distributions and correlations
- Tests a log1p transformation for `credit_amount`
- One-hot encodes categorical variables
- Uses median imputation fitted on training data only
- Tunes Logistic Regression regularization with `LogisticRegressionCV`
- Uses 5-fold ROC-AUC cross-validation
- Evaluates the selected model on a held-out 20% split
- Calculates AUC, Gini and KS
- Calculates PSI and CSI as stability checks

The target is encoded as `bad = 1` and `good = 0`.

## Dataset

OpenML `credit-g` (German Credit):

- 1,000 observations
- 20 original predictors
- Numerical and categorical variables
- Binary good/bad target

The dataset is downloaded automatically when the script runs.

## Modeling

Categorical variables are one-hot encoded with `drop_first=True`.

Missing values are filled with the median. The median is learned from the training data and then applied to the holdout data.

For Logistic Regression, these values of `C` are tested:

```text
[0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]
```

Five-fold cross-validation uses ROC-AUC to choose `C`. The same training-only CV comparison is used to choose between the original and `log1p(credit_amount)` versions. The holdout is evaluated after that choice.

`C` is the inverse of regularization strength:

- smaller `C` = stronger regularization
- larger `C` = weaker regularization

The model uses the `liblinear` solver and `max_iter=2000`.

## Holdout validation

The data is split 80/20 while keeping the original row order. The public dataset has no real production timestamps, so this is an ordered holdout, not a true production OOT test.

Current recorded results:

| Metric | Value |
|---|---:|
| Selected C | 0.10 |
| CV ROC-AUC | 0.789 |
| Holdout ROC-AUC | 0.796 |
| Gini | 0.592 |
| KS | 0.522 |
| PSI | 0.066 |

The current run selects the original `credit_amount` version. Full results are saved under `outputs/`.

## Stability checks

**PSI** compares model-score distributions between training and holdout data.

**CSI** checks distribution changes for selected features.

Both are stability checks, not substitutes for predictive-performance metrics.

## Outputs

The `outputs/` folder contains:

- class distribution
- numerical distributions
- credit amount transformation
- numerical correlation
- selected feature relationships
- encoded feature heatmap
- validation dashboard
- transformation comparison
- model metrics
- CSI scores

## Limitations

- Small public dataset
- No real applicant timestamps
- Ordered holdout is not true production OOT validation
- PSI/CSI are sample-based checks
- No production deployment or monitoring
- No calibration or business-threshold analysis

## Run

```bash
pip install -r requirements.txt
python scorecard_pipeline.py
```

## Project structure

```text
credit-risk-scorecard/
├── scorecard_pipeline.py
├── requirements.txt
├── outputs/
└── .github/
    └── workflows/
        └── generate_outputs.yml
```

## Tools

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

## Author

Aanand Ajith — B.Tech Mechanical Engineering, IIT Hyderabad
