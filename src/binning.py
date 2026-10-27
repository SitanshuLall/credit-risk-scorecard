"""Supervised WoE binning and Information Value.

optbinning needs Google OR-Tools, which has no build for Python 3.15, so binning is
implemented here with the same idea in two stages:

1. Fine classing - a shallow decision tree on the single feature proposes cut points
   (every bin must hold at least MIN_BIN_FRAC of the population).
2. Coarse classing - adjacent bins are merged until the bad rate is monotonic in the
   feature. Both directions are tried and the one keeping more IV wins.

Missing values get their own bin. WoE is defined as ln(%good / %bad), so a positive
WoE means lower risk than average.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from . import config

MISSING = "Missing"


def iv_strength(iv: float) -> str:
    """Conventional reading of an Information Value."""
    if iv < 0.02:
        return "Useless"
    if iv < 0.1:
        return "Weak"
    if iv < 0.3:
        return "Medium"
    if iv <= 0.5:
        return "Strong"
    return "Very strong (check for leakage)"


def _woe_iv(goods, bads, total_good, total_bad, eps=0.5):
    """WoE and IV contribution per bin, with a small count adjustment for empty cells."""
    dist_g = (np.asarray(goods) + eps) / total_good
    dist_b = (np.asarray(bads) + eps) / total_bad
    woe = np.log(dist_g / dist_b)
    return woe, (dist_g - dist_b) * woe


class FeatureBinner:
    """Optimal monotonic binning for one numeric feature."""

    def __init__(self, name, max_bins=config.MAX_BINS, min_bin_frac=config.MIN_BIN_FRAC,
                 monotonic=True, random_state=config.SEED):
        self.name = name
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.monotonic = monotonic
        self.random_state = random_state

    # ---- fitting -------------------------------------------------------------
    def fit(self, x: pd.Series, y: pd.Series) -> "FeatureBinner":
        x = pd.Series(x).astype(float).to_numpy()
        y = pd.Series(y).to_numpy()
        mask = ~np.isnan(x)
        self.total_good_ = float((y == 0).sum())
        self.total_bad_ = float((y == 1).sum())

        tree = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=max(int(self.min_bin_frac * len(x)), 1),
            random_state=self.random_state,
        ).fit(x[mask].reshape(-1, 1), y[mask])
        splits = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])

        if self.monotonic and len(splits) > 1:
            best = None
            for direction in (1, -1):
                cand = self._merge_monotonic(x[mask], y[mask], splits, direction)
                iv = self._iv_for(x[mask], y[mask], cand)
                if best is None or iv > best[0]:
                    best = (iv, cand, direction)
            _, splits, self.trend_ = best
        else:
            self.trend_ = 0
        self.splits_ = np.asarray(splits, dtype=float)

        idx = self._bin_index(x)
        n_bins = len(self.splits_) + 1
        goods = np.array([((idx == i) & (y == 0)).sum() for i in range(n_bins)], float)
        bads = np.array([((idx == i) & (y == 1)).sum() for i in range(n_bins)], float)
        miss_g, miss_b = float(((~mask) & (y == 0)).sum()), float(((~mask) & (y == 1)).sum())
        self.has_missing_ = (miss_g + miss_b) > 0
        if self.has_missing_:
            goods, bads = np.append(goods, miss_g), np.append(bads, miss_b)

        self.woe_, iv_parts = _woe_iv(goods, bads, self.total_good_, self.total_bad_)
        self.goods_, self.bads_ = goods, bads
        self.iv_parts_ = iv_parts
        self.iv_ = float(iv_parts.sum())
        # Unseen missing values at scoring time get a neutral WoE of 0
        self.missing_woe_ = float(self.woe_[-1]) if self.has_missing_ else 0.0
        return self

    def _merge_monotonic(self, x, y, splits, direction):
        """Drop cut points until bad rate moves in one direction across bins."""
        splits = list(splits)
        while len(splits) > 0:
            idx = np.searchsorted(splits, x, side="left")
            n = np.bincount(idx, minlength=len(splits) + 1)
            b = np.bincount(idx, weights=y, minlength=len(splits) + 1)
            br = b / np.maximum(n, 1)
            viol = np.where(direction * np.diff(br) <= 0)[0]
            if len(viol) == 0:
                break
            # Merge the violating pair with the smallest difference in bad rate
            i = viol[np.argmin(np.abs(np.diff(br)[viol]))]
            splits.pop(i)
        return np.asarray(splits)

    def _iv_for(self, x, y, splits):
        idx = np.searchsorted(splits, x, side="left")
        k = len(splits) + 1
        g = np.bincount(idx, weights=(y == 0), minlength=k)
        b = np.bincount(idx, weights=(y == 1), minlength=k)
        _, iv = _woe_iv(g, b, self.total_good_, self.total_bad_)
        return iv.sum()

    # ---- applying ------------------------------------------------------------
    def _bin_index(self, x):
        """Bin i holds values in (split[i-1], split[i]]; NaN maps to -1."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.splits_, x, side="left")
        return np.where(np.isnan(x), -1, idx)

    def bin_index(self, x):
        """Bin position for each value; the Missing bin is the last position."""
        idx = self._bin_index(x)
        miss_pos = len(self.splits_) + 1 if self.has_missing_ else -1
        return np.where(idx == -1, miss_pos, idx)

    def transform(self, x):
        """Replace raw values by the WoE of their bin."""
        idx = self._bin_index(x)
        n_num = len(self.splits_) + 1
        woe_num = self.woe_[:n_num]
        out = np.where(idx >= 0, woe_num[np.clip(idx, 0, n_num - 1)], self.missing_woe_)
        return out

    def labels(self):
        """Human-readable bin labels, matching the order of the binning table."""
        edges = [-np.inf, *self.splits_, np.inf]
        fmt = lambda v: f"{v:,.4g}" if np.isfinite(v) else ("-inf" if v < 0 else "inf")
        labs = [f"({fmt(lo)}, {fmt(hi)}]" for lo, hi in zip(edges[:-1], edges[1:])]
        if len(labs) == 1:
            labs = ["All values"]
        else:
            labs[0] = f"<= {fmt(edges[1])}"
            labs[-1] = f"> {fmt(edges[-2])}"
        if self.has_missing_:
            labs.append(MISSING)
        return labs

    def table(self) -> pd.DataFrame:
        """Binning table: counts, bad rate, WoE and IV per bin."""
        n = self.goods_ + self.bads_
        return pd.DataFrame({
            "Feature": self.name,
            "Bin": self.labels(),
            "Count": n.astype(int),
            "Count %": n / n.sum(),
            "Goods": self.goods_.astype(int),
            "Bads": self.bads_.astype(int),
            "Bad rate": self.bads_ / np.maximum(n, 1),
            "WoE": self.woe_,
            "IV": self.iv_parts_,
        })


class BinningProcess:
    """Fits a FeatureBinner per feature and exposes IV and WoE transforms."""

    def __init__(self, features, monotonic=None, **binner_kwargs):
        self.features = list(features)
        self.monotonic = monotonic or {}
        self.binner_kwargs = binner_kwargs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BinningProcess":
        self.binners_ = {}
        for f in self.features:
            b = FeatureBinner(f, monotonic=self.monotonic.get(f, True), **self.binner_kwargs)
            self.binners_[f] = b.fit(X[f], y)
        return self

    def iv_table(self) -> pd.DataFrame:
        rows = [{"Feature": f, "IV": b.iv_, "Bins": len(b.woe_)} for f, b in self.binners_.items()]
        t = pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)
        t["Strength"] = t["IV"].map(iv_strength)
        return t

    def transform(self, X: pd.DataFrame, features=None) -> pd.DataFrame:
        features = features or self.features
        return pd.DataFrame({f: self.binners_[f].transform(X[f]) for f in features}, index=X.index)

    def bin_indices(self, X: pd.DataFrame, features=None) -> pd.DataFrame:
        features = features or self.features
        return pd.DataFrame({f: self.binners_[f].bin_index(X[f]) for f in features}, index=X.index)

    def table(self, feature) -> pd.DataFrame:
        return self.binners_[feature].table()

    def plot(self, feature, ax=None):
        """Bar chart of bin population with the bad-rate line on a separate panel."""
        import matplotlib.pyplot as plt

        t = self.table(feature)
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        x = np.arange(len(t))
        ax[0].bar(x, t["Count %"] * 100, color=config.GOOD_COLOR, width=0.6)
        ax[0].set_title(f"{feature}: population by bin")
        ax[0].set_ylabel("% of applicants")
        ax[1].bar(x, t["WoE"], color=np.where(t["WoE"] >= 0, config.GOOD_COLOR, config.BAD_COLOR),
                  width=0.6)
        for xi, br in zip(x, t["Bad rate"]):
            ax[1].annotate(f"{br:.1%}", (xi, 0), textcoords="offset points",
                           xytext=(0, -12 if t["WoE"].iloc[xi] >= 0 else 4),
                           ha="center", fontsize=8, color=config.INK_2)
        ax[1].axhline(0, color=config.INK_2, linewidth=0.8)
        ax[1].set_title(f"{feature}: WoE by bin (labels = bad rate)")
        ax[1].set_ylabel("WoE  (+ = safer)")
        for a in ax:
            a.set_xticks(x)
            a.set_xticklabels(t["Bin"], rotation=30, ha="right", fontsize=8)
            a.grid(axis="x", visible=False)
        return ax
