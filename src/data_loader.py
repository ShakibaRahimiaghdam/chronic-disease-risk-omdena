# -----------------------------
#       Data Loading Functions
# -----------------------------
import pandas as pd
from config import DATA_PATH


def load_data():
    """Load the merged NHANES dataset from CSV."""
    df = pd.read_csv(DATA_PATH)
    return df


def load_data_and_convert():
    """Load the dataset and convert the target 'CVD-Positive' to integer."""
    df = load_data()
    df["CVD-Positive"] = df["CVD-Positive"].astype(int)
    return df
