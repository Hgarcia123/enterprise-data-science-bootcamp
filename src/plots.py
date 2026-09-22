"""Charts, saved as files in `reports/generated`.

Run it from the project folder with:

    python -m src.plots
"""

import matplotlib

matplotlib.use("Agg")  # draw to a file, not to a window
import matplotlib.pyplot as plt
import pandas as pd

from src.config import GENERATED, RAW_CSV


def plot_home_ownership():
    """Draw how many loans there are for each value of `home_ownership`.

    The data dictionary in `references` says the column has five values and is empty
    in 826 rows. The chart lets you check it: documentation is a claim, and the data
    is the evidence.

    Returns:
        Path: The PNG file that was written: `reports/generated/home_ownership.png`.
    """
    counts = pd.read_csv(RAW_CSV)["home_ownership"].value_counts(dropna=False)
    ax = counts.plot.bar(title="home_ownership: loans for each value, blanks included")
    ax.set_ylabel("loans")
    plt.tight_layout()
    out = GENERATED / "home_ownership.png"
    plt.savefig(out)
    return out


if __name__ == "__main__":
    print("Saved", plot_home_ownership())
