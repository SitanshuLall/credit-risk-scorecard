"""Generate a synthetic dataset in the exact format of Kaggle's cs-training.csv.

Use it to test the pipeline end to end without the real data. The data mimics the
real file's quirks so every cleaning step is exercised:
  * an unnamed index column first, the same 11 column names
  * ~20% missing MonthlyIncome, ~2.6% missing NumberOfDependents
  * RevolvingUtilization > 1 (including absurd values), age = 0, 96/98 past-due codes
  * DebtRatio stored as a raw debt amount when income is missing
The default flag is drawn from a logistic model of the features, calibrated to a
~6.7% bad rate, so the scorecard has real signal to find.

Usage:
    python scripts/make_dummy_data.py                      # 20,000 rows -> data/dummy/cs-training.csv
    python scripts/make_dummy_data.py --rows 50000 --out data/dummy/cs-training.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_dummy(n=20_000, bad_rate=0.067, seed=7) -> pd.DataFrame:
    """Return a DataFrame with the Kaggle column names (index added on save)."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(52, 15, n), 21, 100).round()
    income = np.exp(rng.normal(np.log(5400), 0.65, n)).round()
    dependents = np.clip(rng.poisson(0.76, n), 0, 20).astype(float)
    open_lines = rng.poisson(8.5, n)
    real_estate = rng.poisson(1.0, n)

    # Utilisation rises with a hidden "financial stress" factor
    stress = rng.normal(0, 1, n)
    util = 1 / (1 + np.exp(-(-1.3 + 1.3 * stress + rng.normal(0, 0.8, n))))
    debt_ratio = np.exp(rng.normal(np.log(0.37), 0.7, n) + 0.15 * stress)

    # Delinquency counts share the stress factor and are rarer for older borrowers
    base = np.exp(-2.6 + 0.9 * stress - 0.02 * (age - 50))
    pd30 = rng.poisson(base * 1.6)
    pd60 = rng.poisson(base * 0.6)
    pd90 = rng.poisson(base * 0.7)

    # True default model (log-odds); intercept tuned below to hit the target bad rate
    logit = (2.2 * util + 0.55 * pd30 + 0.7 * pd60 + 0.9 * pd90
             - 0.03 * (age - 50) - 0.25 * np.log(income / 5400)
             + 0.08 * dependents + 0.3 * (debt_ratio > 0.5) - 0.15 * (real_estate > 0)
             + 0.6 * stress)
    lo, hi = -15.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > bad_rate:
            hi = mid
        else:
            lo = mid
    y = rng.binomial(1, 1 / (1 + np.exp(-(logit + mid))))

    df = pd.DataFrame({
        "SeriousDlqin2yrs": y,
        "RevolvingUtilizationOfUnsecuredLines": util.round(6),
        "age": age.astype(int),
        "NumberOfTime30-59DaysPastDueNotWorse": pd30,
        "DebtRatio": debt_ratio.round(6),
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": pd90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": pd60,
        "NumberOfDependents": dependents,
    })

    # --- inject the real file's data-quality quirks ---
    miss_inc = rng.random(n) < 0.20
    df.loc[miss_inc, "MonthlyIncome"] = np.nan
    df.loc[miss_inc, "DebtRatio"] = np.exp(rng.normal(np.log(1200), 1.0, miss_inc.sum())).round()
    df.loc[rng.random(n) < 0.026, "NumberOfDependents"] = np.nan

    over = rng.random(n) < 0.02
    df.loc[over, "RevolvingUtilizationOfUnsecuredLines"] = rng.uniform(1.0, 1.5, over.sum()).round(6)
    junk = np.flatnonzero(over)[: max(1, over.sum() // 10)]
    df.loc[junk, "RevolvingUtilizationOfUnsecuredLines"] = rng.uniform(100, 50_000, len(junk)).round(1)

    codes = rng.random(n) < 0.0018
    code_vals = rng.choice([96, 98], codes.sum())
    for c in ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
              "NumberOfTime60-89DaysPastDueNotWorse"]:
        df.loc[codes, c] = code_vals
    df.loc[codes, "SeriousDlqin2yrs"] = rng.binomial(1, 0.55, codes.sum())

    df.loc[rng.choice(n, 1, replace=False), "age"] = 0
    df.index = np.arange(1, n + 1)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "dummy" / "cs-training.csv")
    args = ap.parse_args()

    df = make_dummy(args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"Wrote {len(df):,} rows to {args.out}  (bad rate {df['SeriousDlqin2yrs'].mean():.2%}, "
          f"income missing {df['MonthlyIncome'].isna().mean():.1%})")


if __name__ == "__main__":
    main()
