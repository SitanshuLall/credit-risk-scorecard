"""LightGBM challenger model with a small randomized search and SHAP explanations."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config

PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.02, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [50, 100, 200, 400],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
    # Imbalance handling: re-weight bads (only shifts probabilities, not ranking much)
    "scale_pos_weight": [1.0, 3.0, 7.0],
}


def tune_lgbm(X: pd.DataFrame, y: pd.Series, n_iter=15, cv=3, seed=config.SEED):
    """Randomized search over PARAM_GRID with stratified CV on ROC-AUC."""
    base = LGBMClassifier(subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
    search = RandomizedSearchCV(
        base, PARAM_GRID, n_iter=n_iter, scoring="roc_auc",
        cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
        random_state=seed, n_jobs=1, refit=True,
    )
    search.fit(X, y)
    return search


def cv_results_table(search, top=5) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    cols = ["mean_test_score", "std_test_score", "params"]
    return res.sort_values("rank_test_score")[cols].head(top).reset_index(drop=True)


def shap_explanation(model, X: pd.DataFrame):
    """SHAP values (log-odds of bad) for the given rows."""
    import shap

    explainer = shap.TreeExplainer(model)
    exp = explainer(X)
    if exp.values.ndim == 3:  # some versions return one slice per class
        exp = exp[:, :, 1]
    return exp


def plot_shap_summary(exp, path=None):
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.plots.beeswarm(exp, show=False, max_display=10)
    plt.title("LightGBM challenger: SHAP summary (impact on log-odds of default)")
    if path:
        plt.savefig(path)
    return plt.gcf()


def plot_shap_waterfall(exp, i, path=None, title=None):
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.plots.waterfall(exp[i], show=False, max_display=10)
    plt.title(title or "Why this applicant scored as high risk")
    if path:
        plt.savefig(path)
    return plt.gcf()
