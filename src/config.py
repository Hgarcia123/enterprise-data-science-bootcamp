"""Paths and settings of the `src` pipeline, in ONE place.

Every file of the pipeline imports what it needs from here, so a path or a setting is
never typed by hand in two places. To change where the data lives, change it
here, or in your `.env` file, and the pipeline follows. (`check_data.py` and the
notebook are run on their own, with the Run button, so they find the data by
themselves.)

About FEATURES: the model may only use what is known BEFORE the lender decides.
The five columns below come from the borrower or from the credit file. Columns
that the lender produces while deciding, such as `int_rate`, `sub_grade` and
`installment`, are left out on purpose: using them would be cheating, because
they already contain the lender's own judgment of the risk.

Attributes:
    PROJECT (Path): The project folder, the one that holds `src` and `data`.
    RAW_CSV (Path): The data as we received it. Read only.
    CLEAN_CSV (Path): The cleaned table that `src.dataset` writes.
    MODEL_FILE (Path): The trained model that `src.modeling.train` writes.
    GENERATED (Path): The folder for reports and charts that the code writes.
    SEED (int): The random seed. One number for the whole project, so that
        results can be reproduced.
    TARGET (str): The column the model learns to predict.
        0 = charged off or default, 1 = fully paid.
    FEATURES (list[str]): The columns the model learns from.
"""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def load_env(path=PROJECT / ".env"):
    """Read settings from a `.env` file into the environment variables.

    Each line of the file looks like `NAME=value`. Lines that start with `#`
    and lines without `=` are skipped. A variable that already exists in the
    environment is NOT replaced, so a value set in the terminal wins.

    Args:
        path (Path): The file to read. If it does not exist, nothing happens.

    Returns:
        None. The values are stored in `os.environ`.

    Example:
        With the line `RAW_DATA_FILE=lending-club.csv` in the file, after the
        call `os.environ["RAW_DATA_FILE"]` is `"lending-club.csv"`.
    """
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip())


load_env()

RAW_CSV = PROJECT / "data" / "raw" / os.environ.get("RAW_DATA_FILE", "lending-club.csv")
CLEAN_CSV = PROJECT / "data" / "processed" / "lending-club-clean.csv"
MODEL_FILE = PROJECT / "models" / "model.joblib"
GENERATED = PROJECT / "reports" / "generated"

SEED = 42
TARGET = "loan_status"
FEATURES = ["annual_inc", "dti", "fico_range_low", "revol_util", "open_acc"]
