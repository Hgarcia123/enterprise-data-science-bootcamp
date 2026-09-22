"""Raw data in, clean data out.

Run it from the project folder with:

    python -m src.dataset

The raw file is only read, never changed. The result goes to `data/processed`,
which Git ignores because this code can rebuild it at any time.
"""

import pandas as pd

from src.config import CLEAN_CSV, FEATURES, RAW_CSV, TARGET


def make_dataset():
    """Clean the raw Lending Club table and save it.

    Three things are done, each one found while profiling the data:

    1. Rows with no target are dropped: they cannot teach a model anything.
    2. `term` is stripped: `" 36 months"` becomes `"36 months"`.
    3. Only the columns the project uses are kept.

    Returns:
        pandas.DataFrame: The clean table, the same one written to `CLEAN_CSV`.

    Raises:
        FileNotFoundError: If the raw file is not in `data/raw`.
    """
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=[TARGET])
    df["term"] = df["term"].str.strip()
    df = df[FEATURES + ["term", "home_ownership", "split", TARGET]]
    df.to_csv(CLEAN_CSV, index=False)
    return df


if __name__ == "__main__":
    clean = make_dataset()
    print(f"Saved {len(clean)} rows to {CLEAN_CSV}")
