"""Loading, splitting and cleaning the Give Me Some Credit data.

All cleaning thresholds are learned on the training split only and then applied
unchanged to the test split (and to new applicants in the Streamlit app).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

SPECIAL_CODES = (96, 98)


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the Kaggle csv, drop its unnamed index column and rename columns."""
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns=config.RENAME)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size=config.TEST_SIZE, seed=config.SEED):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


@dataclass
class Cleaner:
    """Train-fitted outlier treatment.

    * RevolvingUtil above 1 (over-limit, and values in the thousands are data errors)
      is capped at 1.
    * Past-due counts of 96/98 are special codes, not real counts; they are set to the
      highest genuine count seen in train so they fall into the riskiest bin.
    * Age 0 is impossible and is replaced with the train median.
    * DebtRatio and MonthlyIncome are capped at their train 99th / 99.9th percentiles.
    * Missing MonthlyIncome / Dependents are left as NaN on purpose: the binning gives
      them their own "Missing" bin (their risk differs from the median borrower) and
      LightGBM handles NaN natively.
    """

    util_cap: float = 1.0
    past_due_caps: dict = field(default_factory=dict)
    age_median: float = None
    debt_ratio_cap: float = None
    income_cap: float = None

    def fit(self, X: pd.DataFrame) -> "Cleaner":
        for c in config.PAST_DUE_COLS:
            self.past_due_caps[c] = float(X.loc[~X[c].isin(SPECIAL_CODES), c].max())
        self.age_median = float(X.loc[X["Age"] > 0, "Age"].median())
        self.debt_ratio_cap = float(X["DebtRatio"].quantile(0.99))
        self.income_cap = float(X["MonthlyIncome"].quantile(0.999))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["RevolvingUtil"] = X["RevolvingUtil"].clip(0, self.util_cap)
        for c, cap in self.past_due_caps.items():
            X[c] = X[c].where(~X[c].isin(SPECIAL_CODES), cap).clip(upper=cap)
        X["Age"] = X["Age"].where(X["Age"] > 0, self.age_median)
        X["DebtRatio"] = X["DebtRatio"].clip(0, self.debt_ratio_cap)
        X["MonthlyIncome"] = X["MonthlyIncome"].clip(0, self.income_cap)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missing share and a few distribution stats."""
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "min": df.min(),
        "median": df.median(),
        "p99": df.quantile(0.99),
        "max": df.max(),
    })
    return rep


def bad_rate_by(df: pd.DataFrame, col: str, bins=None, target=config.TARGET) -> pd.DataFrame:
    """Count and bad rate of the target by a (optionally binned) column."""
    key = pd.cut(df[col], bins=bins, include_lowest=True) if bins is not None else df[col]
    out = df.groupby(key, observed=True)[target].agg(["count", "mean"])
    out.index = out.index.astype(str)
    if df[col].isna().any():
        miss = df.loc[df[col].isna(), target]
        out.loc["Missing"] = [len(miss), miss.mean()]
    out.columns = ["count", "bad_rate"]
    return out
