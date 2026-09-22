"""Turn a table into what a model needs: X (the inputs) and y (the target)."""

from src.config import FEATURES, TARGET


def make_X(df):
    """Keep only the model's input columns.

    This is all that is needed to PREDICT, so it works on a new loan
    application, where the outcome is not known yet.

    Args:
        df (pandas.DataFrame): A table with every column named in `FEATURES`.
            It may or may not have the `TARGET` column.

    Returns:
        pandas.DataFrame: Only the `FEATURES` columns, in that order.

    Raises:
        KeyError: If one of the `FEATURES` columns is missing from `df`.
    """
    return df[FEATURES]


def make_features(df):
    """Split a table into the model's inputs and its target.

    This is what is needed to TRAIN and to EVALUATE, where the outcome is known.

    Args:
        df (pandas.DataFrame): A table with every column named in `FEATURES`,
            and the `TARGET` column with no missing values.

    Returns:
        tuple: `(X, y)` where `X` is the result of `make_X` and `y` is a Series
        of whole numbers: 0 = charged off or default, 1 = fully paid.
        Both have one row for each row of `df`.

    Raises:
        KeyError: If a column is missing from `df`.
    """
    X = make_X(df)
    y = df[TARGET].astype(int)
    return X, y
