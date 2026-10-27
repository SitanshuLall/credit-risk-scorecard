"""Logistic regression on WoE features and scaling to scorecard points.

Points scaling (higher score = lower risk):
    Factor = PDO / ln(2)
    Offset = BaseScore - Factor * ln(BaseOdds)
    Score  = Offset + Factor * ln(odds of good) = Offset - Factor * logit(PD)
With logit(PD) = b0 + sum_i b_i * WoE_i, each feature/bin gets
    points_ij = -(b_i * WoE_ij + b0 / n) * Factor + Offset / n
so the applicant's score is simply the sum of their bin points.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor for each column (computed with an intercept)."""
    Xc = sm.add_constant(X, has_constant="add")
    vals = [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])]
    return pd.Series(vals, index=X.columns, name="VIF").sort_values(ascending=False)


def select_by_vif(X: pd.DataFrame, iv: pd.Series, threshold=5.0):
    """Iteratively drop the lowest-IV feature among those with VIF above threshold."""
    cols, dropped = list(X.columns), []
    while True:
        vif = compute_vif(X[cols])
        high = vif[vif > threshold]
        if high.empty:
            return cols, dropped, vif
        worst = iv.loc[high.index].idxmin()
        cols.remove(worst)
        dropped.append(worst)


def fit_logit(X_woe: pd.DataFrame, y: pd.Series):
    """Unpenalised logistic regression on WoE features (statsmodels, for p-values)."""
    return sm.Logit(y.astype(float), sm.add_constant(X_woe, has_constant="add")).fit(disp=0)


def fit_with_checks(X_woe: pd.DataFrame, y: pd.Series, cols, p_max=0.05):
    """Refit until every coefficient is negative (business-consistent) and significant.

    With WoE = ln(%good/%bad) and the model predicting bad, a positive coefficient
    would mean a safer bin earns fewer points - that feature is removed. Features are
    dropped one at a time (wrong sign first, then highest p-value).
    """
    cols, log = list(cols), []
    while True:
        model = fit_logit(X_woe[cols], y)
        coefs, pvals = model.params.drop("const"), model.pvalues.drop("const")
        wrong_sign = coefs[coefs >= 0]
        if not wrong_sign.empty:
            f = pvals.loc[wrong_sign.index].idxmax()
            reason = f"positive coefficient {coefs[f]:+.3f} (p={pvals[f]:.3f})"
        elif (pvals > p_max).any():
            f = pvals.idxmax()
            reason = f"not significant (p={pvals[f]:.3f})"
        else:
            return model, cols, log
        cols.remove(f)
        log.append((f, reason))


def predict_pd(model, X_woe: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(sm.add_constant(X_woe, has_constant="add")))


def scaling(base_score=config.BASE_SCORE, base_odds=config.BASE_ODDS, pdo=config.PDO):
    """Return (factor, offset) for the points scale."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


class Scorecard:
    """Points table built from a fitted BinningProcess and logistic model."""

    def __init__(self, binning_process, model, features):
        self.bp = binning_process
        self.model = model
        self.features = list(features)
        self.factor, self.offset = scaling()
        self._build()

    def _build(self):
        params = self.model.params
        b0, n = params["const"], len(self.features)
        rows = []
        for f in self.features:
            t = self.bp.table(f)
            pts = -(params[f] * t["WoE"] + b0 / n) * self.factor + self.offset / n
            for pos, (lab, woe, p, cnt, br) in enumerate(
                    zip(t["Bin"], t["WoE"], pts, t["Count %"], t["Bad rate"])):
                rows.append({"Feature": f, "Bin #": pos, "Bin": lab, "Count %": cnt,
                             "Bad rate": br, "WoE": woe, "Coefficient": params[f],
                             "Points": int(round(p))})
        self.table_ = pd.DataFrame(rows)
        self._lookup = {f: g.set_index("Bin #")["Points"].to_dict()
                        for f, g in self.table_.groupby("Feature")}
        # Neutral points for a value that is missing but had no Missing bin in train
        self._neutral = {f: int(round(-(b0 / n) * self.factor + self.offset / n))
                         for f in self.features}

    def points(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        """Points per feature for each applicant (cleaned raw features in)."""
        idx = self.bp.bin_indices(X_clean, self.features)
        return pd.DataFrame(
            {f: idx[f].map(lambda i, f=f: self._lookup[f].get(i, self._neutral[f]))
             for f in self.features}, index=X_clean.index)

    def score(self, X_clean: pd.DataFrame) -> pd.Series:
        """Total scorecard points (integer)."""
        return self.points(X_clean).sum(axis=1).rename("Score")

    def model_score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Unrounded score implied directly by the logistic model."""
        pd_ = predict_pd(self.model, X_woe[self.features])
        return self.offset + self.factor * np.log((1 - pd_) / pd_)

    def score_to_pd(self, score):
        """Invert the scaling: PD implied by a score."""
        odds_good = np.exp((np.asarray(score) - self.offset) / self.factor)
        return 1 / (1 + odds_good)

    def max_points(self) -> dict:
        return self.table_.groupby("Feature")["Points"].max().to_dict()

    def reason_codes(self, X_clean: pd.DataFrame, top=3):
        """Features where the applicant lost the most points versus the best bin."""
        pts = self.points(X_clean)
        gap = pd.DataFrame({f: self.max_points()[f] - pts[f] for f in self.features})
        return [list(r.sort_values(ascending=False).head(top).items()) for _, r in gap.iterrows()]
