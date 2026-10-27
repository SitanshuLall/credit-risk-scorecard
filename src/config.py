"""Project-wide constants: paths, seed, scorecard scaling and illustrative loan economics.

Two optional environment variables redirect a run (e.g. a test on dummy data):
    CREDIT_DATA  - path to the input csv (default data/cs-training.csv)
    CREDIT_OUT   - folder that receives models/ and reports/ (default: project root)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get("CREDIT_DATA", ROOT / "data" / "cs-training.csv"))
OUT_ROOT = Path(os.environ.get("CREDIT_OUT", ROOT))
MODELS_DIR = OUT_ROOT / "models"
REPORTS_DIR = OUT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

SEED = 42
TEST_SIZE = 0.30
TARGET = "SeriousDlqin2yrs"

# Shorter, readable names for the Kaggle columns
RENAME = {
    "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtil",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "PastDue30_59",
    "DebtRatio": "DebtRatio",
    "MonthlyIncome": "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans": "OpenCreditLines",
    "NumberOfTimes90DaysLate": "PastDue90Plus",
    "NumberRealEstateLoansOrLines": "RealEstateLoans",
    "NumberOfTime60-89DaysPastDueNotWorse": "PastDue60_89",
    "NumberOfDependents": "Dependents",
}
FEATURES = [v for v in RENAME.values()]
PAST_DUE_COLS = ["PastDue30_59", "PastDue60_89", "PastDue90Plus"]

# Binning
MAX_BINS = 6
MIN_BIN_FRAC = 0.02  # each bin holds at least 2% of the training population

# Scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20

# Illustrative economics (per loan)
AVG_LOAN = 200_000  # INR
LGD = 0.60
NIM = 0.12

# Plot palette (validated categorical slots 1-2: blue = good, orange = bad)
GOOD_COLOR = "#2a78d6"
BAD_COLOR = "#eb6834"
NEUTRAL = "#f0efec"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def set_plot_style():
    """Apply a clean, consistent matplotlib style to every chart."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.size": 10,
    })
