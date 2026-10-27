"""Discrimination, rank-ordering and stability metrics.

Conventions: `y` is 1 for bad. `pd_` is a probability of default (higher = riskier);
`score` is a scorecard score (higher = safer). Helpers accept either via `risk`.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from . import config


def ks_statistic(y, risk) -> float:
    """Max distance between the cumulative bad and good distributions."""
    fpr, tpr, _ = roc_curve(y, risk)
    return float(np.max(tpr - fpr))


def metrics(y, risk) -> dict:
    """AUC, Gini (= 2*AUC - 1) and KS for a risk measure (higher = riskier)."""
    auc = roc_auc_score(y, risk)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_statistic(y, risk)}


def gains_table(y, score, n_bands=10) -> pd.DataFrame:
    """Decile table from riskiest (lowest score) to safest band."""
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    df["band"] = pd.qcut(df["score"].rank(method="first"), n_bands, labels=False) + 1
    g = df.groupby("band").agg(min_score=("score", "min"), max_score=("score", "max"),
                               count=("y", "size"), bads=("y", "sum"))
    g["goods"] = g["count"] - g["bads"]
    g["bad_rate"] = g["bads"] / g["count"]
    g["cum_bads_pct"] = g["bads"].cumsum() / g["bads"].sum()
    g["cum_goods_pct"] = g["goods"].cumsum() / g["goods"].sum()
    g["ks"] = g["cum_bads_pct"] - g["cum_goods_pct"]
    g["lift"] = g["bad_rate"] / df["y"].mean()
    g.index.name = "decile (1 = riskiest)"
    return g


def rank_ordering_breaks(gains: pd.DataFrame) -> list:
    """Deciles where the bad rate fails to fall as the score rises."""
    br = gains["bad_rate"].to_numpy()
    return [int(gains.index[i + 1]) for i in range(len(br) - 1) if br[i + 1] > br[i]]


def calculate_psi(expected, actual, bins=10, categorical=False, eps=1e-4) -> float:
    """Population Stability Index between a baseline and a new distribution.

    PSI = sum((A% - E%) * ln(A% / E%)). Numeric inputs are bucketed on quantiles of
    the expected (baseline) sample; categorical inputs use their own categories.
    Reading: < 0.10 stable, 0.10-0.25 monitor, > 0.25 investigate.
    """
    expected, actual = np.asarray(expected), np.asarray(actual)
    if categorical:
        cats = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == c).mean() for c in cats])
        a = np.array([(actual == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_reading(psi: float) -> str:
    if psi < 0.10:
        return "Stable"
    if psi <= 0.25:
        return "Monitor"
    return "Investigate"


def csi_table(bins_expected: pd.DataFrame, bins_actual: pd.DataFrame) -> pd.DataFrame:
    """Characteristic Stability Index per feature, on scorecard bins."""
    rows = [{"Feature": f, "CSI": calculate_psi(bins_expected[f], bins_actual[f], categorical=True)}
            for f in bins_expected.columns]
    t = pd.DataFrame(rows).sort_values("CSI", ascending=False).reset_index(drop=True)
    t["Reading"] = t["CSI"].map(psi_reading)
    return t


# ---- plots -------------------------------------------------------------------
def plot_roc(curves: dict, ax=None):
    """curves: {label: (y, risk)}."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    colors = [config.GOOD_COLOR, config.BAD_COLOR]
    for (label, (y, risk)), c in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(y, risk)
        auc = roc_auc_score(y, risk)
        ax.plot(fpr, tpr, color=c, label=f"{label} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate (goods declined)")
    ax.set_ylabel("True positive rate (bads caught)")
    ax.set_title("ROC curve - test set")
    ax.legend(loc="lower right")
    return ax


def plot_ks(y, score, ax=None, title="KS plot - scorecard, test set"):
    """Cumulative % of bads and goods by score (low to high)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.2))
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score)}).sort_values("s")
    cb = df["y"].cumsum() / df["y"].sum()
    cg = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    ax.plot(df["s"], cb, color=config.BAD_COLOR, label="Cumulative % of bads")
    ax.plot(df["s"], cg, color=config.GOOD_COLOR, label="Cumulative % of goods")
    i = int(np.argmax((cb - cg).to_numpy()))
    s_at, top, bot = df["s"].iloc[i], cb.iloc[i], cg.iloc[i]
    ax.vlines(s_at, bot, top, color=config.INK_2, linewidth=1.2)
    ax.annotate(f"KS = {top - bot:.3f}\nat score {s_at:.0f}", (s_at, (top + bot) / 2),
                xytext=(8, 0), textcoords="offset points", fontsize=9, color=config.INK)
    ax.set_xlabel("Score")
    ax.set_ylabel("Cumulative share")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax
