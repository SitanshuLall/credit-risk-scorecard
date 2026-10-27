# Results summary

All figures come from `notebooks/credit_scorecard.ipynb` (seed 42, stratified 70/30 split, 45,000 test applicants).

## Model performance (test)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic scorecard (champion) | 0.855 | 0.710 | 0.550 |
| LightGBM (challenger) | 0.867 | 0.734 | 0.575 |

- Train vs test Gini: scorecard 0.709 vs 0.710 (no overfit); LightGBM 0.759 vs 0.734 (2.5-point gap).
- Score PSI (train vs test): **0.0003**, stable. CSI is below 0.001 for every feature.
- Rank-ordering: no breaks across deciles. The bottom decile captures 54.4% of bads (lift 5.4×).

## Strategy (cut-off chosen on train, reported on test)

| | Approval rate | Bad rate | Bads avoided | Expected profit |
|---|---|---|---|---|
| Approve all | 100% | 6.68% | 0% | ₹64.7 cr |
| **Cut-off 536 (profit-max)** | **91.0%** | **3.55%** | **51.7%** | **₹77.4 cr** |
| Cut-off 550 (conservative) | 85.3% | 2.87% | 63.4% | ₹76.3 cr |

## Model choices

- Features: 10 binned. None had IV < 0.02. `OpenCreditLines` was dropped because its coefficient was
  positive (+0.082, p = 0.095). 9 features remain in the scorecard, all with VIF < 1.4.
- Scaling: 600 points at 50:1 odds, PDO 20. Factor 28.85, offset 487.12. Test scores range from 419 to 626.

## Resume bullets

- Built an application credit scorecard on 150K borrowers using WoE/IV binning and logistic regression
  (test Gini 0.71, KS 0.55, no train-test gap). Benchmarked it against a tuned LightGBM challenger
  (Gini 0.73) and explained the challenger with SHAP.
- Designed a profit-optimised cut-off strategy (score 536) that approves 91% of applicants, cuts the
  approved bad rate from 6.7% to 3.5%, avoids 52% of defaults and lifts expected profit by about 20%.
  Translated scores into five risk bands (A–E) with approve/price/review/decline actions.
- Implemented model validation and monitoring: gains/lift tables (top decile captures 54% of bads),
  rank-ordering checks, and a reusable PSI/CSI framework. Deployed the scorecard in a Streamlit app that
  returns score, band, decision and adverse-action reason codes.
