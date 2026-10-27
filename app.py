"""Streamlit scoring app: enter an applicant, get score, risk band, decision and reasons.

Run:  .venv\\Scripts\\streamlit.exe run app.py
Loads the artifacts saved by notebooks/credit_scorecard.ipynb; nothing is retrained here.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config, strategy

REASON_TEXT = {
    "RevolvingUtil": "High utilisation of revolving credit limits",
    "PastDue90Plus": "Previous 90+ days past-due delinquencies",
    "PastDue30_59": "Previous 30-59 days past-due delinquencies",
    "PastDue60_89": "Previous 60-89 days past-due delinquencies",
    "Age": "Younger age group has higher observed default rates",
    "MonthlyIncome": "Lower monthly income",
    "RealEstateLoans": "No real-estate loan or line on file",
    "Dependents": "Higher number of dependents",
    "DebtRatio": "High debt-to-income ratio",
}


@st.cache_resource
def load_artifacts():
    return joblib.load(config.MODELS_DIR / "scorecard_artifacts.joblib")


def main():
    st.set_page_config(page_title="Credit Scorecard", page_icon=":bar_chart:", layout="wide")
    art = load_artifacts()
    card, cleaner = art["scorecard"], art["cleaner"]

    st.title("Application credit scorecard")
    st.caption(f"Logistic regression scorecard - 600 points at 50:1 odds, PDO 20. "
               f"Approval cut-off: {art['cutoff']} points.")

    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 100, 40)
        income_known = c1.checkbox("Monthly income provided", value=True)
        income = c1.number_input("Monthly income", 0, 1_000_000, 5_000, step=500, disabled=not income_known)
        deps_known = c1.checkbox("Number of dependents provided", value=True)
        deps = c1.number_input("Dependents", 0, 20, 1, disabled=not deps_known)
        util = c2.slider("Revolving utilisation (balance / limit)", 0.0, 1.5, 0.30, 0.01)
        debt_ratio = c2.number_input("Debt ratio (monthly debt / income)", 0.0, 10_000.0, 0.35, step=0.05)
        real_estate = c2.number_input("Real-estate loans or lines", 0, 50, 1)
        open_lines = c2.number_input("Open credit lines and loans", 0, 60, 8)
        pd30 = c3.number_input("Times 30-59 days past due (last 2 yrs)", 0, 20, 0)
        pd60 = c3.number_input("Times 60-89 days past due (last 2 yrs)", 0, 20, 0)
        pd90 = c3.number_input("Times 90+ days late", 0, 20, 0)
        submitted = st.form_submit_button("Score applicant", type="primary")

    if not submitted:
        st.info("Enter the applicant's details and press **Score applicant**.")
        return

    raw = pd.DataFrame([{
        "RevolvingUtil": util, "Age": age, "PastDue30_59": pd30, "DebtRatio": debt_ratio,
        "MonthlyIncome": income if income_known else np.nan, "OpenCreditLines": open_lines,
        "PastDue90Plus": pd90, "RealEstateLoans": real_estate, "PastDue60_89": pd60,
        "Dependents": deps if deps_known else np.nan,
    }])[config.FEATURES]
    X = cleaner.transform(raw)
    points = card.points(X).iloc[0]
    score = int(points.sum())
    band = str(strategy.assign_band([score], art["band_edges"])[0])
    action = art["band_actions"][band]
    pd_ = float(card.score_to_pd(score))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score", score)
    m2.metric("Risk band", band)
    m3.metric("Implied probability of default", f"{pd_:.1%}")
    m4.metric("Decision", {"E": "Decline", "C": "Refer"}.get(band, "Approve"))
    (st.error if band == "E" else st.warning if band in ("C", "D") else st.success)(
        f"**Recommended action:** {action}")

    reasons = card.reason_codes(X, top=3)[0]
    st.subheader("Top reasons lowering the score")
    for f, lost in reasons:
        if lost > 0:
            st.write(f"- **{REASON_TEXT[f]}** ({f}: {lost} points below the best bin)")
    if all(lost == 0 for _, lost in reasons):
        st.write("No adverse factors - the applicant is in the best bin for every characteristic.")

    st.subheader("Points breakdown")
    bins = card.bp.bin_indices(X, card.features).iloc[0]
    labels = {f: card.bp.binners_[f].labels() for f in card.features}
    best = card.max_points()
    breakdown = pd.DataFrame({
        "Characteristic": card.features,
        "Value": [X.iloc[0][f] for f in card.features],
        "Bin": [labels[f][int(bins[f])] if bins[f] >= 0 else "Missing" for f in card.features],
        "Points": [int(points[f]) for f in card.features],
        "Max possible": [best[f] for f in card.features],
    })
    st.dataframe(breakdown, hide_index=True, width="stretch")
    st.caption("Band edges (lower bounds) - A: {3}, B: {2}, C: {1}, D: {0}; E below {0}.".format(*art["band_edges"]))


if __name__ == "__main__":
    main()
