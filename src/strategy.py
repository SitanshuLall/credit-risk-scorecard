"""Cut-off strategy and risk bands on top of the scorecard.

Expected profit at a cut-off (approve everyone scoring >= cut-off):
    profit = goods_approved * NIM * loan - bads_approved * LGD * loan
A good loan earns 0.12 * 2,00,000 = 24,000; a bad one loses 0.60 * 2,00,000 = 1,20,000,
so a marginal applicant is worth approving while their bad rate is below
NIM / (NIM + LGD) = 16.7%.
"""
import numpy as np
import pandas as pd

from . import config

BAND_ACTIONS = {
    "A": "Auto-approve",
    "B": "Approve with lower limit",
    "C": "Manual review",
    "D": "Approve with higher pricing or collateral",
    "E": "Decline",
}


def breakeven_bad_rate(nim=config.NIM, lgd=config.LGD) -> float:
    return nim / (nim + lgd)


def cutoff_table(y, score, cutoffs=None, loan=config.AVG_LOAN, lgd=config.LGD,
                 nim=config.NIM) -> pd.DataFrame:
    """Approval rate, bad rate among approved and expected profit for each cut-off."""
    y, score = np.asarray(y), np.asarray(score)
    if cutoffs is None:
        cutoffs = np.arange(int(score.min()), int(score.max()) + 1)
    rows = []
    for c in cutoffs:
        appr = score >= c
        n_appr, bads = int(appr.sum()), int(y[appr].sum())
        goods = n_appr - bads
        rows.append({
            "cutoff": int(c),
            "approved": n_appr,
            "approval_rate": n_appr / len(y),
            "bad_rate_approved": bads / n_appr if n_appr else np.nan,
            "bads_declined_pct": 1 - bads / y.sum(),
            "profit": goods * nim * loan - bads * lgd * loan,
        })
    return pd.DataFrame(rows)


def pick_cutoffs(table: pd.DataFrame, tolerance=0.02):
    """Profit-maximising cut-off, and a conservative one: the highest cut-off whose
    profit is still within `tolerance` of the maximum."""
    best = table.loc[table["profit"].idxmax()]
    near = table[table["profit"] >= best["profit"] * (1 - tolerance)]
    conservative = near.loc[near["cutoff"].idxmax()]
    return best, conservative


def band_edges(score, cutoff) -> list:
    """E = below cut-off; A-D split the approved population into equal quarters."""
    approved = np.asarray(score)[np.asarray(score) >= cutoff]
    q = np.quantile(approved, [0.25, 0.5, 0.75])
    # Edges are lower bounds for D, C, B, A
    return [int(cutoff), int(np.ceil(q[0])), int(np.ceil(q[1])), int(np.ceil(q[2]))]


def assign_band(score, edges):
    score = np.asarray(score)
    d, c, b, a = edges
    return np.select([score >= a, score >= b, score >= c, score >= d], ["A", "B", "C", "D"], "E")


def band_table(y, score, edges) -> pd.DataFrame:
    bands = assign_band(score, edges)
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score), "band": bands})
    t = df.groupby("band").agg(min_score=("score", "min"), max_score=("score", "max"),
                               applicants=("y", "size"), bads=("y", "sum"))
    t["pct_applicants"] = t["applicants"] / len(df)
    t["bad_rate"] = t["bads"] / t["applicants"]
    t["action"] = t.index.map(BAND_ACTIONS)
    return t.sort_index()


# ---- plots -------------------------------------------------------------------
def plot_tradeoff(table, chosen, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(table["approval_rate"] * 100, table["bad_rate_approved"] * 100, color=config.GOOD_COLOR)
    ax.scatter([chosen["approval_rate"] * 100], [chosen["bad_rate_approved"] * 100], s=60,
               color=config.BAD_COLOR, zorder=3, edgecolor=config.SURFACE, linewidth=2)
    ax.annotate(f"Cut-off {int(chosen['cutoff'])}\n{chosen['approval_rate']:.1%} approved, "
                f"{chosen['bad_rate_approved']:.2%} bad",
                (chosen["approval_rate"] * 100, chosen["bad_rate_approved"] * 100),
                xytext=(-150, 25), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=config.INK_2))
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Bad rate among approved (%)")
    ax.set_title("Approval rate vs bad rate across cut-offs")
    return ax


def plot_profit(table, best, conservative, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    t = table[table["approval_rate"] > 0.3]
    ax.plot(t["cutoff"], t["profit"] / 1e7, color=config.GOOD_COLOR)
    for row, lab, side in [(best, "Profit-max", "right"), (conservative, "Conservative", "left")]:
        ax.axvline(row["cutoff"], color=config.INK_2, linewidth=0.8, linestyle="--")
        ax.text(row["cutoff"] + (-4 if side == "right" else 4), 0.08,
                f"{lab}: {int(row['cutoff'])}\nINR {row['profit'] / 1e7:,.1f} cr",
                transform=ax.get_xaxis_transform(), ha=side, fontsize=9, color=config.INK)
    ax.set_xlabel("Score cut-off (approve if score >= cut-off)")
    ax.set_ylabel("Expected profit on test set (INR crore)")
    ax.set_title("Expected profit by cut-off")
    return ax
