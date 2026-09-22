"""A quick look at the raw data.

Run it with the Run Python File button (the triangle, top right), or from the
project folder with:

    python src/check_data.py

This file is made to be run on its own with that button, so it does not use
`from src.config ...`: run that way, Python would look for `src` inside the `src`
folder and fail. It finds the data by itself.
"""

from pathlib import Path

import pandas as pd

# The project folder is one level above this file's folder (src/).
PROJECT = Path(__file__).resolve().parent.parent
CSV = PROJECT / "data" / "raw" / "lending-club.csv"


def check_data(csv=CSV):
    """Print the size of the data and the values of `home_ownership`.

    Args:
        csv (Path): The CSV file to look at.

    Raises:
        SystemExit: If the file is not there, with a message saying where to put it.
    """
    if not csv.exists():
        raise SystemExit(f"File not found: {csv}\nCopy lending-club.csv into data/raw/")

    df = pd.read_csv(csv)
    print("File   :", csv.name)
    print("Rows   :", len(df))
    print("Columns:", len(df.columns))
    print(df["home_ownership"].value_counts(dropna=False))


if __name__ == "__main__":
    check_data()
