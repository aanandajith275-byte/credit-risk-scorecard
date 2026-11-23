"""The challenger line-up. Every model is a scikit-learn estimator taking the raw DataFrame.

  scorecard       champion: WoE points scorecard (scorecard.py)
  logistic        L2 logistic regression on one-hot + standardised features (the old baseline)
  random_forest   bagged deep trees
  gbm             histogram gradient boosting (the LightGBM algorithm), tuned by randomised CV
  gbm_monotone    the same booster with monotonic constraints on characteristics whose direction
                  is known from credit economics: explainable "more delinquency never lowers risk"
  mlp             2-layer neural network on quantile-normalised inputs
  stack           logistic meta-model on out-of-fold PDs of the scorecard and the booster
"""
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler

from .scorecard import Scorecard

# +1: risk can only rise with the feature; -1: can only fall
MONOTONE = {
    "german": {"duration_months": 1, "credit_amount": 1, "installment_rate": 1, "age": -1},
    "taiwan": {"LIMIT_BAL": -1, **{f"PAY_{i}": 1 for i in range(1, 7)}, "max_delinquency_6": 1,
               "months_delinquent_6": 1, "utilisation_1": 1, "utilisation_avg6": 1,
               **{f"PAY_AMT{i}": -1 for i in range(1, 7)}, "pay_ratio_1": -1},
}


def _onehot_scaled(cat, num):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat),
                              ("num", StandardScaler(), num)])


def _ordinal(cat, num):
    return ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat),
                              ("num", "passthrough", num)], verbose_feature_names_out=False)


def _gbm(cat, num, monotone=None, seed=0, n_iter=12, n_jobs=2):
    cst = None
    if monotone:
        cst = [0] * len(cat) + [monotone.get(c, 0) for c in num]
    hgb = HistGradientBoostingClassifier(categorical_features=list(range(len(cat))), monotonic_cst=cst,
                                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
                                         max_iter=1000, random_state=seed)
    space = {"histgradientboostingclassifier__learning_rate": loguniform(0.02, 0.2),
             "histgradientboostingclassifier__max_leaf_nodes": randint(4, 32),
             "histgradientboostingclassifier__min_samples_leaf": randint(20, 200),
             "histgradientboostingclassifier__l2_regularization": loguniform(1e-3, 10)}
    return RandomizedSearchCV(make_pipeline(_ordinal(cat, num), hgb), space, n_iter=n_iter, scoring="roc_auc",
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed), random_state=seed, n_jobs=n_jobs)


def build(name, dataset, cat, num, seed=0, gbm_iter=12):
    if name == "scorecard":
        return Scorecard(categorical=tuple(cat), random_state=seed)
    if name == "logistic":
        return make_pipeline(_onehot_scaled(cat, num),
                             LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring="neg_log_loss", max_iter=5000))
    if name == "random_forest":
        return make_pipeline(_ordinal(cat, num),
                             RandomForestClassifier(n_estimators=400, min_samples_leaf=10, max_features="sqrt",
                                                    n_jobs=2, random_state=seed))
    if name == "gbm":
        return _gbm(cat, num, None, seed, gbm_iter)
    if name == "gbm_monotone":
        return _gbm(cat, num, MONOTONE[dataset], seed, gbm_iter)
    if name == "mlp":
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat),
                                 ("num", QuantileTransformer(n_quantiles=200, output_distribution="normal"), num)])
        return make_pipeline(pre, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                                validation_fraction=0.15, n_iter_no_change=20, max_iter=500,
                                                random_state=seed))
    if name == "stack":
        return StackingClassifier([("scorecard", Scorecard(categorical=tuple(cat), random_state=seed)),
                                   ("gbm", _gbm(cat, num, None, seed, max(4, gbm_iter // 2)))],
                                  final_estimator=LogisticRegression(), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                  stack_method="predict_proba", n_jobs=1)
    raise ValueError(name)


MODELS = ["scorecard", "logistic", "random_forest", "gbm", "gbm_monotone", "mlp", "stack"]
LABELS = {"scorecard": "WoE scorecard (champion)", "logistic": "Logistic regression (one-hot)",
          "random_forest": "Random forest", "gbm": "Gradient boosting", "gbm_monotone": "Monotone gradient boosting",
          "mlp": "Neural network (MLP)", "stack": "Stacked: scorecard + boosting"}
