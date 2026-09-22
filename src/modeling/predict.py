"""Use the saved model to estimate the risk of default.

Run it from the project folder, after `src.modeling.train`, with:

    python -m src.modeling.predict
"""

import joblib
import pandas as pd

from src.config import CLEAN_CSV, FEATURES, MODEL_FILE
from src.features import make_X

DEFAULT = 0  # in `loan_status`, 0 = charged off or default, 1 = fully paid


def predict(df):
    """Give the estimated probability of default for each row of a table.

    Only the input columns are needed, so this works on a new loan
    application, where nobody knows the outcome yet.

    Args:
        df (pandas.DataFrame): A table with the `FEATURES` columns. The target
            column is not needed, and is ignored if it is there.

    Returns:
        numpy.ndarray: One number between 0 and 1 for each row: the estimated
        probability that the loan is charged off or defaults
        (`loan_status` = 0). Higher means riskier.

    Raises:
        FileNotFoundError: If no model has been trained and saved yet.
    """
    model = joblib.load(MODEL_FILE)
    column = list(model.classes_).index(DEFAULT)
    return model.predict_proba(make_X(df))[:, column]


if __name__ == "__main__":
    # Read ONLY the input columns: the model never sees the outcome here.
    applications = pd.read_csv(CLEAN_CSV, usecols=FEATURES, nrows=5)
    print("Estimated probability of default, first five loans:")
    print(predict(applications).round(3))
